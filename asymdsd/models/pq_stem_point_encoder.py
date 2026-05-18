from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch
from pytorch3d.ops import sample_farthest_points
from torch import nn

from ..layers.tokenization import Tokens, TrainableToken
from .point_encoder import PointEncoderOutput
from .pq_stem import PQStemEncoder, gather_operation


class PQStemPatches(NamedTuple):
    points: torch.Tensor
    centers: list[torch.Tensor]


class PQStemPointEncoder(nn.Module):
    """AsymDSD-compatible adapter around a local PQDT-compatible PQStemEncoder.

    The exported downstream module is exactly ``stem_encoder``.  The CLS token
    and position embedding below are pretraining-only helpers for packed FAB.
    """

    def __init__(
        self,
        in_chans: int = 256,
        embed_dim: int = 384,
        num_heads: int = 6,
        enc_attn: tuple[str, ...] = ("ge_attn", "attn", "attn", "attn"),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        transdown_fps: tuple[int, ...] = (512, 128),
        transdown_dims: tuple[int, ...] = (64, 256),
        transdown_num_heads: tuple[int, ...] = (1, 4),
        transdown_sa_depth: tuple[int, ...] = (3, 3),
        transdown_k: tuple[int, ...] = (16, 16),
        transdown_use_attn: bool | tuple[bool, ...] = False,
        cls_token: TrainableToken | bool | None = True,
    ) -> None:
        super().__init__()

        self.stem_encoder = PQStemEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            enc_attn=tuple(enc_attn),
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            transdown_fps=tuple(transdown_fps),
            transdown_dims=tuple(transdown_dims),
            transdown_num_heads=tuple(transdown_num_heads),
            transdown_sa_depth=tuple(transdown_sa_depth),
            transdown_k=tuple(transdown_k),
            transdown_use_attn=transdown_use_attn,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transdown_fps = tuple(transdown_fps)

        # Packed FAB expects ``point_encoder.encoder.config.num_heads``.
        self.encoder = self.stem_encoder.encoder_1
        self.encoder.config = SimpleNamespace(embed_dim=embed_dim, num_heads=num_heads)

        if isinstance(cls_token, bool):
            cls_token = TrainableToken(embed_dim=embed_dim) if cls_token else None
        self.cls_token = cls_token

        self.position_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def compute_centers(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the final PQStem token centers using Transdown's FPS chain."""
        coor = points[..., :3].transpose(1, 2).contiguous()
        for layer in self.stem_encoder.transdown.down_layers:
            xyz = coor.transpose(1, 2).contiguous()
            _, fps_idx = sample_farthest_points(
                xyz, K=min(layer.fps, xyz.shape[1]), random_start_point=False
            )
            coor = gather_operation(coor, fps_idx)
        return coor.transpose(1, 2).contiguous()

    def patch_embedding(self, stem_patches: PQStemPatches) -> Tokens:
        coors, features = self.stem_encoder.down(stem_patches.points)
        centers = coors[-1].transpose(1, 2).contiguous()
        embeddings = self.stem_encoder.input_proj(features[-1]).transpose(1, 2)
        pos_embeddings = self.position_embedding(centers)
        return Tokens(embeddings, pos_embeddings, None, centers)

    def transformer_encoder_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        *,
        self_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_hidden_states: bool = False,
        token_centers: torch.Tensor | None = None,
        attn_bias_scale: float = 1.0,
    ) -> PointEncoderOutput:
        if self_mask is not None or self_key_padding_mask is not None:
            raise ValueError("PQStemPointEncoder does not support attention masks.")
        del pos_enc, attn_bias_scale

        if token_centers is None:
            raise ValueError("PQStemPointEncoder requires token_centers.")

        B = x.shape[0]
        centers = token_centers
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, 1, -1)
            cls_center = torch.zeros(
                B, 1, centers.shape[-1], device=centers.device, dtype=centers.dtype
            )
            x = torch.cat((cls_token, x), dim=1)
            centers = torch.cat((cls_center, centers), dim=1)

        encoder_out = self.stem_encoder.encoder_1(
            centers.transpose(1, 2).contiguous(),
            x,
            return_attention=return_attention,
        )
        if return_attention:
            x_out, attn_weights = encoder_out
        else:
            x_out = encoder_out
            attn_weights = None

        if self.cls_token is not None:
            cls_features = x_out[:, 0]
            patch_features = x_out[:, 1:]
        else:
            cls_features = None
            patch_features = x_out

        hidden_states = [x_out] if return_hidden_states else None
        return PointEncoderOutput(
            patch_features=patch_features,
            cls_features=cls_features,
            attn_weights=attn_weights,
            hidden_states=hidden_states,
        )

    def forward(self, points: torch.Tensor | PQStemPatches) -> PointEncoderOutput:
        if isinstance(points, PQStemPatches):
            stem_patches = points
        else:
            stem_patches = PQStemPatches(points=points, centers=[])
        tokens = self.patch_embedding(stem_patches)
        return self.transformer_encoder_forward(
            tokens.embeddings,
            tokens.pos_embeddings,
            token_centers=tokens.centers,
        )

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def enable_gradient_checkpointing(self) -> None:
        # PQDT's stem does not expose the same checkpointing contract.
        return None
