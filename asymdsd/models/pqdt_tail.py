from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from timm.layers import DropPath
from torch import nn

from .pq_stem import Attention, GEEncoder, GEGroupMultiHeadAttention, get_knn_index
from .pq_transup import UpLayer, fps_subsample


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, _ = q.shape
        C = self.out_dim
        NK = v.size(1)

        q = (
            self.q_map(q)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_map(v)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_map(v)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class GEDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn: str = "ge_attn",
        dim_q: int | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float | None = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_class = attn
        if attn == "ge_attn":
            self.self_attn = GEGroupMultiHeadAttention(
                dim,
                num_heads=num_heads,
                dropout=attn_drop,
            )
        elif attn == "attn":
            self.self_attn = Attention(dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attn}")

        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop or 0.0,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        from .pq_stem import Mlp

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        group_idx_q: torch.Tensor,
        geo_pos_q: torch.Tensor,
    ) -> torch.Tensor:
        norm_q = self.norm1(q)
        if self.attn_class == "ge_attn":
            q_1, _ = self.self_attn(norm_q, norm_q, norm_q, geo_pos_q, group_idx_q)
        else:
            q_1 = self.self_attn(norm_q)
        q = q + self.drop_path(q_1)

        q_2 = self.cross_attn(self.norm_q(q), self.norm_v(v))
        q = q + self.drop_path(q_2)
        return q + self.drop_path(self.mlp(self.norm2(q)))


class GEDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_cls: tuple[str, ...],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        pos_add: bool = True,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, dim, 1),
        )
        from .pq_stem import GeometricStructureEmbedding

        self.pos_embed_geo = GeometricStructureEmbedding(
            dim,
            sigma_d=0.2,
            sigma_a=15,
            angle_k=3,
        )
        self.decoder = nn.ModuleList(
            [
                GEDecoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn=attn,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
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

    def forward(
        self,
        coor_q: torch.Tensor,
        q: torch.Tensor,
        coor_x: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        del coor_x
        _, group_idx_q = get_knn_index(coor_q, k=12)
        pos_geo_emb_q = self.pos_embed_geo(
            coor_q.transpose(1, 2).contiguous(),
            group_idx_q,
        )
        pos_q = self.pos_embed(coor_q).transpose(1, 2)
        if self.pos_add:
            q = q + pos_q
        for blk in self.decoder:
            q = blk(q, x, group_idx_q, pos_geo_emb_q)
        return q


class GumbelTopK(nn.Module):
    def __init__(self, k: int, noise_scale: float = 1.0) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = k
        self.noise_scale = noise_scale

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be shaped (B, N).")
        B, N = logits.shape
        g = _sample_gumbel((B, N), device=logits.device, dtype=logits.dtype)
        _, idx = (logits + self.noise_scale * g).topk(self.k, dim=-1)
        return idx


class DQS(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        gf_dim: int = 1024,
        num_query: int = 384,
        tau0: float = 1.0,
        total_epochs: int = 200,
        sel: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query = num_query
        self.tau0 = tau0
        self.total_epochs = total_epochs
        self.mlp_query = nn.Sequential(
            nn.Conv1d(3 + embed_dim + gf_dim, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.sel = sel
        self.sampler = GumbelTopK(k=num_query)
        self.query_ranking = nn.Sequential(
            nn.Conv1d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(
        self,
        coor: torch.Tensor,
        f: torch.Tensor,
        f_g: torch.Tensor,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, n, _ = f.size()
        f_agg = torch.cat(
            [
                f.transpose(1, 2),
                f_g.unsqueeze(-1).expand(-1, -1, n),
                coor,
            ],
            dim=1,
        )
        f_agg = self.mlp_query(f_agg).transpose(1, 2)
        if not self.sel:
            return coor, f_agg, f

        scores = self.query_ranking(f_agg.transpose(1, 2)).squeeze(1)
        scores = (scores - scores.mean(-1, keepdim=True)) / (
            scores.std(-1, keepdim=True) + 1e-8
        )

        if self.num_query > n:
            raise ValueError(f"num_query={self.num_query} cannot exceed {n}.")
        self.sampler.noise_scale = get_tau(
            current_epoch,
            tau0=self.tau0,
            tau_min=0.0,
            total_epochs=self.total_epochs,
            mode="cosine",
        )
        idx = self.sampler(scores)
        coor = torch.gather(coor, 2, idx.unsqueeze(1).expand(-1, 3, -1))
        f_agg = torch.gather(
            f_agg,
            1,
            idx.unsqueeze(-1).expand(-1, -1, f_agg.size(-1)),
        )
        f = torch.gather(f, 1, idx.unsqueeze(-1).expand(-1, -1, f.size(-1)))
        return coor, f_agg, f


class HardBaseHead(nn.Module):
    def __init__(self, d_model: int, radius: float = 0.3) -> None:
        super().__init__()
        self.radius = radius
        self.mlp_res = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),
        )

    def forward(
        self,
        dec_out: torch.Tensor,
        seed_feat: torch.Tensor,
        seed_xyz: torch.Tensor,
        seed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del seed_feat, seed_mask
        base = seed_xyz.transpose(1, 2).contiguous()
        delta = torch.tanh(self.mlp_res(dec_out.float())) * self.radius
        xyz = base + delta
        return xyz.transpose(1, 2).contiguous()


class PQDTPseudoStage(nn.Module):
    """PQDT pseudo-seed branch used as pretraining context."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        dec_attn: tuple[str, ...] = (
            "ge_attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
        ),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_pseudo: int = 384,
    ) -> None:
        super().__init__()
        self.num_pseudo = num_pseudo

        if len(dec_attn) < 4:
            raise ValueError(
                "PQDT decoder attention schedule must contain >= 4 blocks."
            )
        self.decoder_1 = GEDecoder(
            embed_dim,
            num_heads,
            attn_cls=tuple(dec_attn[:4]),
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.mlp_query_ps = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.pseudo_pred_head = HardBaseHead(d_model=embed_dim, radius=0.3)

    def forward(
        self,
        x1_g: torch.Tensor,
        coor_ps: torch.Tensor,
        coor_c: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_ps = torch.cat(
            [x1_g.unsqueeze(-1).expand(-1, -1, self.num_pseudo), coor_ps],
            dim=1,
        )
        q_ps = self.mlp_query_ps(q_ps).transpose(1, 2)
        q_ps = self.decoder_1(coor_ps, q_ps, coor_c, x1)
        pseudo_seed_pred = self.pseudo_pred_head(q_ps, q_ps, coor_ps, seed_mask=None)
        return q_ps, pseudo_seed_pred


class PQDTQueryStage(nn.Module):
    """PQDT transformer branch reused by finetuning checkpoints."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        enc_attn: tuple[str, ...] = ("ge_attn", "attn", "attn", "attn"),
        dec_attn: tuple[str, ...] = (
            "ge_attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
        ),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_pseudo: int = 384,
        num_query: int = 512,
        tau0: float = 1.0,
        total_epochs: int = 200,
    ) -> None:
        super().__init__()
        self.num_query = num_query
        self.num_pseudo = num_pseudo
        self.encoder_2 = GEEncoder(
            embed_dim,
            num_heads,
            attn_cls=enc_attn,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.decoder_2 = GEDecoder(
            embed_dim,
            num_heads,
            attn_cls=tuple(dec_attn),
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.increase_dim_1 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.increase_dim_2 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.increase_dim_3 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.qf_1 = DQS(
            embed_dim=embed_dim,
            gf_dim=1024,
            num_query=num_pseudo,
            tau0=tau0,
            total_epochs=total_epochs,
        )
        self.qf_2 = DQS(
            embed_dim=embed_dim,
            gf_dim=1024,
            num_query=num_query,
            tau0=tau0,
            total_epochs=total_epochs,
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.reduce_map = nn.Conv1d(embed_dim + 1027, embed_dim, 1)

    def global_features(self, x1: torch.Tensor) -> torch.Tensor:
        return torch.max(self.increase_dim_1(x1), dim=1)[0]

    def encode_pseudo_context(
        self,
        coor_c: torch.Tensor,
        x1: torch.Tensor,
        x1_g: torch.Tensor,
        q_ps: torch.Tensor,
        pseudo_seed_pred: torch.Tensor,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_ps = torch.cat([x1, q_ps], dim=1)
        pseudo_seed = torch.cat([coor_c, pseudo_seed_pred], dim=-1)
        pseudo_seed_sel, x1_ps, x_res = self.qf_1(
            pseudo_seed,
            x1_ps,
            x1_g,
            current_epoch=current_epoch,
        )

        x2 = self.encoder_2(pseudo_seed_sel, x1_ps)
        x2_g = torch.max(self.increase_dim_2(x2), dim=1)[0]
        x2 = x2 + x_res
        return pseudo_seed, pseudo_seed_sel, x2, x2_g

    def decode_query_features(
        self,
        x1_g: torch.Tensor,
        pseudo_seed_sel: torch.Tensor,
        x2: torch.Tensor,
        x2_g: torch.Tensor,
        query_seed: torch.Tensor,
    ) -> torch.Tensor:
        num_query = query_seed.shape[-1]
        q = torch.cat(
            [
                x1_g.unsqueeze(-1).expand(-1, -1, num_query),
                x2_g.unsqueeze(-1).expand(-1, -1, num_query),
                query_seed,
            ],
            dim=1,
        )
        q = self.mlp_query(q).transpose(1, 2)
        q = self.decoder_2(query_seed, q, pseudo_seed_sel, x2)
        q_g = torch.max(self.increase_dim_3(q), dim=1)[0]

        f_query_seed = torch.cat(
            [
                q_g.unsqueeze(-1).expand(-1, -1, num_query),
                q.transpose(1, 2),
                query_seed,
            ],
            dim=1,
        )
        return self.reduce_map(f_query_seed)


class PQDTUpSampler(nn.Module):
    """PQDT upsampling head reused by completion finetuning."""

    def __init__(
        self,
        embed_dim: int = 384,
        up_factors: tuple[int, ...] = (1, 4, 4),
        n_knn: int = 16,
        radius: float = 1.0,
        interpolate: str = "three",
        attn_channel: bool = True,
    ) -> None:
        super().__init__()
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
                for i, factor in enumerate(up_factors)
            ]
        )

    def forward(
        self,
        query_seed: torch.Tensor,
        seed_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        seed = query_seed
        pred_pcds = [seed.transpose(1, 2).contiguous()]
        pcd = seed
        k_prev = None
        for layer in self.up_layers:
            pcd, k_prev = layer(pcd, seed, seed_features, k_prev)
            pred_pcds.append(pcd.transpose(1, 2).contiguous())
        return pred_pcds


class PQDTTail(nn.Module):
    """PQDT tail orchestrating pseudo and query stages for AsymDSD pretraining."""

    def __init__(
        self,
        in_chans: int = 256,
        embed_dim: int = 384,
        num_heads: int = 6,
        enc_attn: tuple[str, ...] = ("ge_attn", "attn", "attn", "attn"),
        dec_attn: tuple[str, ...] = (
            "ge_attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
        ),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_pseudo: int = 384,
        num_query: int = 512,
        tau0: float = 1.0,
        total_epochs: int = 200,
        r_sph: float = 0.8,
        in_q: bool = False,
    ) -> None:
        super().__init__()
        self.in_q = in_q
        self.r_sph = r_sph
        self.num_query = num_query
        self.num_pseudo = num_pseudo
        self.pseudo_stage = PQDTPseudoStage(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dec_attn=dec_attn,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_pseudo=num_pseudo,
        )
        self.query_stage = PQDTQueryStage(
            embed_dim=embed_dim,
            num_heads=num_heads,
            enc_attn=enc_attn,
            dec_attn=dec_attn,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_pseudo=num_pseudo,
            num_query=num_query,
            tau0=tau0,
            total_epochs=total_epochs,
        )

    def _pseudo_query_centers(
        self,
        coor_ref: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.in_q:
            source = coor_ref.transpose(1, 2).contiguous()
        else:
            source = sample_sphere(2048, batch_size, self.r_sph, device)
        return fps_subsample(source, self.num_pseudo).transpose(1, 2).contiguous()

    def _encode_pseudo_context(
        self,
        coor_ref: torch.Tensor,
        coor_c: torch.Tensor,
        x1: torch.Tensor,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_g = self.query_stage.global_features(x1)
        coor_ps = self._pseudo_query_centers(
            coor_ref,
            batch_size=x1.shape[0],
            device=x1.device,
        )
        q_ps, pseudo_seed_pred = self.pseudo_stage(x1_g, coor_ps, coor_c, x1)
        pseudo_seed, pseudo_seed_sel, x2, x2_g = self.query_stage.encode_pseudo_context(
            coor_c,
            x1,
            x1_g,
            q_ps,
            pseudo_seed_pred,
            current_epoch=current_epoch,
        )
        return x1_g, pseudo_seed, pseudo_seed_sel, x2, x2_g

    def forward_full(
        self,
        points: torch.Tensor,
        coor_c: torch.Tensor,
        x1: torch.Tensor,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coor_ref = points[..., :3].transpose(1, 2).contiguous()
        x1_g, pseudo_seed, pseudo_seed_sel, x2, x2_g = self._encode_pseudo_context(
            coor_ref,
            coor_c,
            x1,
            current_epoch=current_epoch,
        )
        query_seed = (
            fps_subsample(
                coor_ref.transpose(1, 2).contiguous(),
                self.num_query,
            )
            .transpose(1, 2)
            .contiguous()
        )
        f_query_seed = self.query_stage.decode_query_features(
            x1_g,
            pseudo_seed_sel,
            x2,
            x2_g,
            query_seed,
        )
        return pseudo_seed, query_seed, f_query_seed

    def forward_queries(
        self,
        points: torch.Tensor,
        coor_c: torch.Tensor,
        x1: torch.Tensor,
        query_seed: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        coor_ref = points[..., :3].transpose(1, 2).contiguous()
        x1_g, _, pseudo_seed_sel, x2, x2_g = self._encode_pseudo_context(
            coor_ref,
            coor_c,
            x1,
            current_epoch=current_epoch,
        )
        f_query_seed = self.query_stage.decode_query_features(
            x1_g,
            pseudo_seed_sel,
            x2,
            x2_g,
            query_seed,
        )
        return f_query_seed.transpose(1, 2).contiguous()

    def pqdt_flat_state_dict(
        self,
        include_pseudo_stage: bool = False,
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in self.query_stage.state_dict().items():
            state_dict[key] = value.detach().cpu() if cpu else value
        if include_pseudo_stage:
            for key, value in self.pseudo_stage.state_dict().items():
                state_dict[key] = value.detach().cpu() if cpu else value
        return state_dict


def sample_sphere(
    num_points: int,
    batch_size: int,
    scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    x = torch.randn(batch_size, num_points, 3, device=device)
    return F.normalize(x, dim=-1) * scale


def _sample_gumbel(
    shape: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-20,
) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype).clamp_(min=eps, max=1 - eps)
    return -torch.log(-torch.log(u + eps) + eps)


def get_tau(
    epoch: int,
    tau0: float = 0.7,
    tau_min: float = 0.1,
    total_epochs: int = 200,
    mode: str = "linear",
) -> float:
    if mode == "linear":
        return max(tau_min, tau0 * (1.0 - float(epoch) / float(total_epochs)))
    if mode == "cosine":
        if epoch > total_epochs:
            return tau_min
        cos_inner = math.pi * epoch / total_epochs
        return tau_min + 0.5 * (tau0 - tau_min) * (1 + math.cos(cos_inner))
    raise NotImplementedError(f"Unknown mode: {mode}")
