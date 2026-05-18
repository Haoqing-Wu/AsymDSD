from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch import nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).unsqueeze(-1)
    dist += torch.sum(dst**2, -1).unsqueeze(1)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(
        sqrdists, nsample, dim=-1, largest=False, sorted=False
    )
    return group_idx


def get_knn_index(
    coor_q: torch.Tensor,
    coor_k: torch.Tensor | None = None,
    k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    coor_k = coor_k if coor_k is not None else coor_q
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)
    k = min(k, num_points_k)

    with torch.no_grad():
        idx_b = knn_point(
            k,
            coor_k.transpose(-1, -2).contiguous(),
            coor_q.transpose(-1, -2).contiguous(),
        )
        idx = idx_b.transpose(-1, -2).contiguous()
        idx_base = (
            torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1)
            * num_points_k
        )
        idx = (idx + idx_base).view(-1)

    del num_points
    return idx, idx_b


def get_knn_cross_index(
    coor_q: torch.Tensor,
    coor_k: torch.Tensor,
    k1: int,
    k2: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, num_points_q = coor_q.size()
    _, _, num_points_k = coor_k.size()
    num_points = num_points_q + num_points_k

    with torch.no_grad():
        coor_k_t = coor_k.transpose(1, 2).contiguous()
        coor_q_t = coor_q.transpose(1, 2).contiguous()
        k1 = min(k1, num_points_q, num_points_k)
        k2 = min(k2, num_points_q, num_points_k)

        qk_group_idx = knn_point(k1, coor_k_t, coor_q_t) + coor_k_t.size(1)
        qq_group_idx = knn_point(k1, coor_q_t, coor_q_t)
        kq_group_idx = knn_point(k2, coor_q_t, coor_k_t)
        kk_group_idx = knn_point(k2, coor_k_t, coor_k_t) + coor_k_t.size(1)

        q_group_idx = torch.cat([qq_group_idx, qk_group_idx], dim=-1)
        k_group_idx = torch.cat([kq_group_idx, kk_group_idx], dim=-1)
        group_idx = torch.cat([q_group_idx, k_group_idx], dim=1)
        idx_base = (
            torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1)
            * num_points
        )
        group_idx_f = group_idx.transpose(1, 2).contiguous() + idx_base
        group_idx_f = group_idx_f.view(-1)

    return group_idx_f, group_idx


def gather_feature(
    x: torch.Tensor,
    group_idx: torch.Tensor,
    x_q: torch.Tensor | None = None,
) -> torch.Tensor:
    B, N_src, C = x.size()
    _, N, K = group_idx.size()
    group_idx_flat = group_idx.view(B, -1)
    gathered = torch.gather(
        x,
        1,
        group_idx_flat.unsqueeze(-1).expand(-1, -1, C),
    ).view(B, N, K, C)

    if x_q is None:
        x_q = x
    if x_q.size(1) == N_src:
        x_center = torch.gather(
            x_q,
            1,
            group_idx[..., 0].unsqueeze(-1).expand(-1, -1, C),
        )
    else:
        x_center = x_q
    x_center = x_center.unsqueeze(2)
    return torch.cat(
        (gathered - x_center, x_center.expand(-1, -1, K, -1)), dim=-1
    )


def gather_operation(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SkipFeatureGather(nn.Module):
    def __init__(self, dim, tri=False):
        super().__init__()
        self.tri = tri
        self.map1 = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        if tri:
            self.map2 = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.merge_map = nn.Linear(dim * 3, dim)
        else:
            self.merge_map = nn.Linear(dim * 2, dim)

    def forward(
        self,
        prev_f: torch.Tensor,
        f: torch.Tensor,
        group_idx: torch.Tensor,
        f_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev_f_g = gather_feature(prev_f, group_idx, f_q)
        prev_f_g = self.map1(prev_f_g).max(dim=2, keepdim=False)[0]
        if self.tri:
            f_g = gather_feature(f, group_idx)
            f_g = self.map2(f_g).max(dim=2, keepdim=False)[0]
            f = torch.cat([f, prev_f_g, f_g], dim=-1)
        else:
            f = torch.cat([f, prev_f_g], dim=-1)
        return self.merge_map(f)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"Sinusoidal positional encoding with odd d_model: {d_model}")
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, emb_indices):
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)
        embeddings = embeddings.view(*input_shape, self.d_model)
        return embeddings.detach()


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a="max"):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.reduction_a = reduction_a
        if self.reduction_a not in ["max", "mean"]:
            raise ValueError(f"Unsupported reduction mode: {self.reduction_a}.")

    @torch.no_grad()
    def get_embedding_indices(self, points):
        batch_size, num_point, _ = points.shape
        dist_map = torch.sqrt(square_distance(points, points).clamp_min(0.0))
        d_indices = dist_map / self.sigma_d

        k = min(self.angle_k, max(num_point - 1, 1))
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)
        expanded_points = points.unsqueeze(1).expand(
            batch_size, num_point, num_point, 3
        )
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)
        ref_vectors = knn_points - points.unsqueeze(2)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)
        ref_vectors = ref_vectors.unsqueeze(2).expand(
            batch_size, num_point, num_point, k, 3
        )
        anc_vectors = anc_vectors.unsqueeze(3).expand(
            batch_size, num_point, num_point, k, 3
        )
        sin_values = torch.linalg.norm(
            torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1
        )
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)
        a_indices = torch.atan2(sin_values, cos_values) * self.factor_a
        return d_indices, a_indices

    def forward(self, points, group_idx=None):
        d_indices, a_indices = self.get_embedding_indices(points)
        if group_idx is not None:
            d_indices = torch.gather(d_indices, 2, group_idx)
            a_indices = torch.gather(
                a_indices,
                2,
                group_idx.unsqueeze(-1).expand(-1, -1, -1, a_indices.size(-1)),
            )
        d_embeddings = self.proj_d(self.embedding(d_indices))
        a_embeddings = self.proj_a(self.embedding(a_indices))
        if self.reduction_a == "max":
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        return d_embeddings + a_embeddings


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return (x, attn) if return_attention else x


class GEGroupMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"`d_model` ({d_model}) must be a multiple of `num_heads` ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_vp = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.f_gather = SkipFeatureGather(d_model, tri=False)

    def forward(
        self,
        q_in,
        k_in,
        v_in,
        embed_qk,
        group_idx,
        key_weights=None,
        key_masks=None,
        attention_factors=None,
    ):
        B, N, C = k_in.size()
        _, _, K = group_idx.size()
        q = self.proj_q(q_in)
        k = self.proj_k(k_in)
        v = self.proj_v(v_in)
        p = self.proj_p(embed_qk)
        vp = self.proj_vp(embed_qk)

        group_idx_flat = group_idx.view(B, -1)
        k_g = torch.gather(
            k, 1, group_idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).view(B, N, K, C)
        v_g = torch.gather(
            v, 1, group_idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).view(B, N, K, C)

        q = q.view(B, N, self.num_heads, self.d_model_per_head)
        k = k_g.view(B, N, K, self.num_heads, self.d_model_per_head).permute(
            0, 1, 3, 2, 4
        )
        v = v_g.view(B, N, K, self.num_heads, self.d_model_per_head).permute(
            0, 1, 3, 2, 4
        )
        p = p.view(B, N, K, self.num_heads, self.d_model_per_head).permute(
            0, 1, 3, 2, 4
        )
        vp = vp.view(B, N, K, self.num_heads, self.d_model_per_head).permute(
            0, 1, 3, 2, 4
        )

        attention_scores_p = torch.einsum("BNhc,BNhkc->BNhk", q, p)
        attention_scores_e = torch.einsum("BNhc,BNhkc->BNhk", q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / (
            self.d_model_per_head**0.5
        )
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(
                key_masks.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attention_scores = self.dropout(nn.functional.softmax(attention_scores, dim=-1))
        x = torch.einsum("BNhk,BNhkc->BNhc", attention_scores, v + vp)
        x = x.reshape(B, N, C)
        x = self.f_gather(v_in, x, group_idx)
        return x, attention_scores


class GEEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn="ge_attn",
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_class = attn
        if attn == "ge_attn":
            self.attn = GEGroupMultiHeadAttention(dim, num_heads=num_heads, dropout=attn_drop)
        elif attn == "attn":
            self.attn = Attention(dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attn}")
        self.drop_path = nn.Identity()
        if drop_path > 0.0:
            try:
                from timm.layers import DropPath

                self.drop_path = DropPath(drop_path)
            except Exception:
                self.drop_path = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, group_idx, geo_pos, return_attention=False):
        norm_x = self.norm1(x)
        attn_weights = None
        if self.attn_class == "ge_attn":
            x_1, _ = self.attn(norm_x, norm_x, norm_x, geo_pos, group_idx)
        else:
            attn_out = self.attn(norm_x, return_attention=return_attention)
            if return_attention:
                x_1, attn_weights = attn_out
            else:
                x_1 = attn_out
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attn_weights) if return_attention else x


class GEEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_cls,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pos_add=True,
        cross_knn=False,
    ):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, dim, 1),
        )
        self.pos_embed_geo = GeometricStructureEmbedding(
            dim, sigma_d=0.2, sigma_a=15, angle_k=3
        )
        self.encoder = nn.ModuleList(
            [
                GEEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn=attn,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for attn in attn_cls
            ]
        )
        self.pos_add = pos_add
        self.cross_knn = cross_knn

    def forward(self, coor, f, return_attention=False):
        if self.cross_knn:
            _, _, n = coor.shape
            _, group_idx = get_knn_cross_index(
                coor[:, :, : n // 2], coor[:, :, n // 2 :], k1=6, k2=6
            )
        else:
            _, group_idx = get_knn_index(coor, k=12)
        pos_geo_emb = self.pos_embed_geo(coor.transpose(1, 2).contiguous(), group_idx)
        pos_emb = self.pos_embed(coor).transpose(1, 2)
        if self.pos_add:
            f = f + pos_emb
        attn_weights = [] if return_attention else None
        for blk in self.encoder:
            if return_attention:
                f, attn = blk(f, group_idx, pos_geo_emb, return_attention=True)
                if attn is not None:
                    attn_weights.append(attn)
            else:
                f = blk(f, group_idx, pos_geo_emb)
        return (f, attn_weights) if return_attention else f


class GeoEdgeConv(nn.Module):
    def __init__(
        self,
        channels,
        fps,
        num_heads,
        depth,
        mlp_ratio=4.0,
        drop_rate=0.0,
        k=8,
        use_attn=False,
    ):
        super().__init__()
        self.fps = fps
        self.k = k
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=False),
            nn.GroupNorm(4, channels[1]),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=1, bias=False),
            nn.GroupNorm(4, channels[2]),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.attn = (
            GEEncoder(
                channels[2],
                num_heads,
                attn_cls=["attn"] * depth,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                pos_add=False,
            )
            if use_attn and depth > 0
            else None
        )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()
        _, fps_idx = sample_farthest_points(
            xyz, K=min(num_group, xyz.shape[1]), random_start_point=False
        )
        new_coor = gather_operation(coor, fps_idx)
        new_x = gather_operation(x, fps_idx)
        return new_coor, new_x

    def forward(self, coor, f):
        _, group_idx = get_knn_index(coor, k=self.k)
        f = gather_feature(f.transpose(1, 2), group_idx).transpose(1, 2).transpose(1, 3)
        f = self.layer1(f).max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, self.fps)
        _, group_idx = get_knn_index(coor_q, coor, k=self.k)
        f = gather_feature(
            f.transpose(1, 2), group_idx, f_q.transpose(1, 2)
        ).transpose(1, 2).transpose(1, 3)
        f = self.layer2(f).max(dim=-1, keepdim=False)[0]

        if self.attn is not None:
            f = self.attn(coor_q, f.transpose(1, 2)).transpose(1, 2)
        return coor_q, f


class Transdown(nn.Module):
    def __init__(
        self,
        in_dim=3,
        fps=(2048, 512, 128),
        dims=(64, 256, 1024),
        num_heads=(1, 4, 8),
        sa_depth=(2, 2, 2),
        k=(16, 16, 16),
        use_attn=False,
    ):
        super().__init__()
        self.dims = [8] + list(dims)
        if isinstance(use_attn, bool):
            use_attn = [use_attn] * len(fps)
        else:
            use_attn = list(use_attn)
        if len(use_attn) != len(fps):
            raise ValueError("use_attn must be a bool or have one value per FPS layer.")
        self.input_trans = nn.Conv1d(in_dim, self.dims[0], 1)
        self.down_layers = nn.ModuleList(
            [
                GeoEdgeConv(
                    channels=[self.dims[i] * 2, self.dims[i + 1] // 2, self.dims[i + 1]],
                    fps=fps[i],
                    num_heads=num_heads[i],
                    depth=sa_depth[i],
                    k=k[i],
                    use_attn=use_attn[i],
                )
                for i in range(len(fps))
            ]
        )

    def forward(self, coor):
        coor = coor.transpose(1, 2).contiguous()
        f = self.input_trans(coor)
        coors, fs = [coor], [f]
        for layer in self.down_layers:
            coor, f = layer(coor, f)
            coors.append(coor)
            fs.append(f)
        return coors, fs


@dataclass
class PQStemOutput:
    coors: list[torch.Tensor]
    features: list[torch.Tensor]
    coor_c: torch.Tensor
    x1: torch.Tensor


class PQStemEncoder(nn.Module):
    """Local PQDT-compatible stem: Transdown -> input_proj -> encoder_1."""

    def __init__(
        self,
        in_chans=256,
        embed_dim=384,
        num_heads=6,
        enc_attn=("ge_attn", "attn", "attn", "attn"),
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        transdown_fps=(512, 128),
        transdown_dims=(64, 256),
        transdown_num_heads=(1, 4),
        transdown_sa_depth=(3, 3),
        transdown_k=(16, 16),
        transdown_use_attn=False,
    ):
        super().__init__()
        self.transdown = Transdown(
            in_dim=3,
            fps=list(transdown_fps),
            dims=list(transdown_dims),
            num_heads=list(transdown_num_heads),
            sa_depth=list(transdown_sa_depth),
            k=list(transdown_k),
            use_attn=transdown_use_attn,
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )
        self.encoder_1 = GEEncoder(
            embed_dim,
            num_heads,
            attn_cls=enc_attn,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

    def down(self, points):
        return self.transdown(points)

    def encode_pyramid(self, coors, features, mask_fn=None):
        coor_c = coors[-1]
        f_x = self.input_proj(features[-1]).transpose(1, 2)
        if mask_fn is not None:
            coor_c, f_x = mask_fn(coor_c, f_x)
        x1 = self.encoder_1(coor_c, f_x)
        return PQStemOutput(coors=coors, features=features, coor_c=coor_c, x1=x1)

    def forward(self, points, mask_fn=None):
        coors, features = self.down(points)
        return self.encode_pyramid(coors, features, mask_fn=mask_fn)

    def forward_dual(self, source_points, sketch_points, mask_fn=None):
        coors_src, features_src = self.down(source_points)
        coors_skt, features_skt = self.down(sketch_points)
        coors = [
            torch.cat([coors_src[i], coors_skt[i]], dim=2)
            for i in range(len(coors_src))
        ]
        features = [
            torch.cat([features_src[i], features_skt[i]], dim=2)
            for i in range(len(features_src))
        ]
        return self.encode_pyramid(coors, features, mask_fn=mask_fn)
