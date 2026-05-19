from __future__ import annotations

from itertools import accumulate
from operator import mul

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
from torch import einsum, nn

from .pq_stem import square_distance


def grouping_operation(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, C, _ = x.shape
    _, N, K = idx.shape
    idx_flat = idx.reshape(B, -1).long()
    gathered = torch.gather(
        x.transpose(1, 2).contiguous(),
        1,
        idx_flat.unsqueeze(-1).expand(-1, -1, C),
    )
    return gathered.reshape(B, N, K, C).permute(0, 3, 1, 2).contiguous()


def query_knn(
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    include_self: bool = True,
) -> torch.Tensor:
    pad = 0 if include_self else 1
    nsample = min(nsample, max(xyz.shape[1] - pad, 1))
    sqrdists = square_distance(new_xyz, xyz)
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[
        :, :, pad : nsample + pad
    ]
    return idx.long()


def get_nearest_index(
    target: torch.Tensor,
    source: torch.Tensor,
    k: int = 1,
    return_dis: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    k = min(k, source.shape[2])
    inner = torch.bmm(target.transpose(1, 2), source)
    source_norm_2 = torch.sum(source**2, dim=1)
    target_norm_2 = torch.sum(target**2, dim=1)
    dist = source_norm_2.unsqueeze(1) + target_norm_2.unsqueeze(2) - 2 * inner
    nearest_dis, nearest_index = torch.topk(
        dist,
        k=k,
        dim=-1,
        largest=False,
    )
    return (nearest_index.long(), nearest_dis) if return_dis else nearest_index.long()


def indexing_neighbor(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    B, N, K = index.size()
    batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1)
    x = x.transpose(2, 1).contiguous()
    feature = x[batch_idx, index.long()]
    return feature.permute(0, 3, 1, 2).contiguous()


def indexing_neighbor_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    B, C, _ = x.shape
    _, N, K = index.shape
    x = x.transpose(1, 2).contiguous()
    index_flat = index.reshape(B, -1).long()
    gathered = torch.gather(x, 1, index_flat.unsqueeze(-1).expand(-1, -1, C))
    return gathered.reshape(B, N, K, C).permute(0, 3, 1, 2).contiguous()


def fps_subsample(points: torch.Tensor, n_points: int) -> torch.Tensor:
    if points.shape[1] == n_points:
        return points
    if points.shape[1] < n_points:
        raise ValueError(
            f"FPS subsampling received n_points={n_points} > {points.shape[1]}."
        )
    _, idx = sample_farthest_points(
        points[..., :3].contiguous(),
        K=n_points,
        random_start_point=False,
    )
    return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None, init_weights=False):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
        if init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        return self.conv_2(torch.relu(self.conv_1(x))) + shortcut


class UpTransformer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        dim,
        n_knn=20,
        up_factor=2,
        use_upfeat=True,
        pos_hidden_dim=64,
        attn_hidden_multiplier=4,
        scale_layer=nn.Softmax,
        attn_channel=True,
    ):
        super().__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(
            in_dim=in_channel * 2,
            hidden_dim=in_channel,
            out_dim=in_channel,
        )
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = [
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
        ]
        if up_factor:
            self.attn_mlp.append(
                nn.ConvTranspose2d(
                    dim * attn_hidden_multiplier,
                    attn_out_channel,
                    (up_factor, 1),
                    (up_factor, 1),
                )
            )
        else:
            self.attn_mlp.append(
                nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1)
            )
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        self.upsample1 = (
            nn.Upsample(scale_factor=(up_factor, 1)) if up_factor else nn.Identity()
        )
        self.upsample2 = (
            nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()
        )
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        self.residual_layer = (
            nn.Conv1d(in_channel, out_channel, 1)
            if in_channel != out_channel
            else nn.Identity()
        )

    def forward(self, pos, key, query, upfeat):
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        B, _, N = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((B, -1, N, 1)) - key
        pos_rel = pos.reshape((B, -1, N, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)

        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat)
            upfeat_rel = upfeat.reshape((B, -1, N, 1)) - grouping_operation(
                upfeat, idx_knn
            )
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel)
        attention = self.scale(attention)

        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel
        value = self.upsample1(value)
        agg = einsum("b c i j, b c i j -> b c i", attention, value)
        y = self.conv_end(agg)

        identity = self.residual_layer(identity)
        identity = self.upsample2(identity)
        return y + identity


class UpLayer(nn.Module):
    def __init__(
        self,
        dim,
        seed_dim,
        up_factor=2,
        i=0,
        radius=1,
        n_knn=20,
        interpolate="three",
        attn_channel=True,
    ):
        super().__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[dim, dim])
        self.uptrans1 = UpTransformer(
            dim,
            dim,
            dim=64,
            n_knn=self.n_knn,
            use_upfeat=True,
            up_factor=None,
        )
        self.uptrans2 = UpTransformer(
            dim,
            dim,
            dim=64,
            n_knn=self.n_knn,
            use_upfeat=True,
            attn_channel=attn_channel,
            up_factor=self.up_factor,
        )
        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.ip_factor = 4
        self.mlp_delta_feature = MLP_Res(in_dim=dim * 2, hidden_dim=dim, out_dim=dim)
        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        if self.interpolate == "nearest":
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3)
        elif self.interpolate == "three":
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True)
            dist_recip = 1.0 / (dis.clamp_min(0.0) + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            feat_upsample = torch.sum(
                indexing_neighbor_gather(seed_feat, idx) * weight.unsqueeze(1),
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown Interpolation: {self.interpolate}")

        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat(
                    (1, 1, feat_1.size(2))
                ),
                feat_upsample,
            ],
            1,
        )
        Q = self.mlp_2(feat_1)
        H = self.uptrans1(
            pcd_prev,
            K_prev if K_prev is not None else Q,
            Q,
            upfeat=feat_upsample,
        )
        feat_child = self.uptrans2(
            pcd_prev,
            K_prev if K_prev is not None else H,
            H,
            upfeat=feat_upsample,
        )

        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i
        pcd_new = self.upsample(pcd_prev) + delta
        return pcd_new, K_curr


class PQStemTransUpHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_seed: int = 512,
        up_factors: tuple[int, ...] = (1, 4, 4),
        n_knn: int = 16,
        radius: float = 1.0,
        interpolate: str = "three",
        attn_channel: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_seed = num_seed
        self.up_factors = tuple(up_factors)
        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    dim=embed_dim,
                    seed_dim=embed_dim,
                    up_factor=factor,
                    i=i,
                    n_knn=n_knn,
                    radius=radius,
                    interpolate=interpolate,
                    attn_channel=attn_channel,
                )
                for i, factor in enumerate(self.up_factors)
            ]
        )

    @property
    def output_sizes(self) -> tuple[int, ...]:
        factors = tuple(accumulate(self.up_factors, mul))
        return (self.num_seed,) + tuple(self.num_seed * factor for factor in factors)

    @staticmethod
    def interpolate_seed_features(
        seed_points: torch.Tensor,
        token_centers: torch.Tensor,
        token_features: torch.Tensor,
        k: int = 3,
    ) -> torch.Tensor:
        seed = seed_points.transpose(1, 2).contiguous()
        centers = token_centers.transpose(1, 2).contiguous()
        features = token_features.transpose(1, 2).contiguous()
        idx, dis = get_nearest_index(seed, centers, k=k, return_dis=True)
        dist_recip = 1.0 / (dis.clamp_min(0.0) + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
        return torch.sum(
            indexing_neighbor_gather(features, idx) * weight.unsqueeze(1),
            dim=-1,
        )

    def forward(
        self,
        points: torch.Tensor,
        token_centers: torch.Tensor,
        token_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        seed_points = fps_subsample(points[..., :3].contiguous(), self.num_seed)
        seed = seed_points.transpose(1, 2).contiguous()
        seed_feat = self.interpolate_seed_features(
            seed_points,
            token_centers,
            token_features,
        )

        pred_pcds = [seed_points]
        pcd = seed
        k_prev = None
        for layer in self.up_layers:
            pcd, k_prev = layer(pcd, seed, seed_feat, k_prev)
            pred_pcds.append(pcd.transpose(1, 2).contiguous())
        return pred_pcds

    def reconstruction_loss(
        self,
        points: torch.Tensor,
        token_centers: torch.Tensor,
        token_features: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        target = points[..., :3].contiguous()
        pred_pcds = self(points, token_centers, token_features)
        losses = []
        for pred in pred_pcds:
            losses.append(self.chamfer_loss(pred, target, norm=1))
        return torch.stack(losses).sum(), pred_pcds

    @staticmethod
    def chamfer_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        norm: int = 1,
    ) -> torch.Tensor:
        if pred.shape[1] == target.shape[1]:
            target_i = target
        else:
            target_i = fps_subsample(target, pred.shape[1])
        loss, _ = chamfer_distance(
            pred.float(),
            target_i.float(),
            norm=norm,
            batch_reduction="mean",
            point_reduction="mean",
        )
        return loss

    def pqdt_up_layers_state_dict(
        self,
        prefix: str = "up_layers.",
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        state_dict = self.up_layers.state_dict()
        if cpu:
            return {
                f"{prefix}{key}": value.detach().cpu()
                for key, value in state_dict.items()
            }
        return {f"{prefix}{key}": value for key, value in state_dict.items()}
