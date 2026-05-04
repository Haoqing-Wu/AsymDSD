from typing import NamedTuple

import torch
from jsonargparse import lazy_instance
from torch import nn

from ..components.common_types import OptionalListTensor, OptionalTensor
from ..components.utils import init_lazy_defaults
from ..defaults import DEFAULT_PATCH_EMBEDDING, DEFAULT_TRANSFORMER_ENC_CONFIG
from ..layers import (
    MultiPointPatchify,
    PatchEmbedding,
    PatchEmbeddingConfig,
    ToMultiPatches,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from ..layers.patchify import (
    PatchPoints,
)
from ..layers.tokenization import (
    Tokens,
    TrainableToken,
)
from ..layers.transformer import TransformerOutput


class PointEncoderOutput(NamedTuple):
    patch_features: torch.Tensor
    cls_features: OptionalTensor
    attn_weights: OptionalListTensor
    hidden_states: OptionalListTensor


class PointEncoder(nn.Module):
    @init_lazy_defaults
    def __init__(
        self,
        patchify: MultiPointPatchify | ToMultiPatches | None = None,
        cls_token: TrainableToken | bool | None = None,
        patch_embedding: PatchEmbedding
        | PatchEmbeddingConfig = DEFAULT_PATCH_EMBEDDING,
        encoder: TransformerEncoder
        | TransformerEncoderConfig = DEFAULT_TRANSFORMER_ENC_CONFIG,
    ) -> None:
        super().__init__()

        self.patchify = patchify or ToMultiPatches()

        self.patch_embedding: PatchEmbedding = patch_embedding  # type: ignore
        self.patch_embedding = (
            patch_embedding
            if isinstance(patch_embedding, PatchEmbedding)
            else patch_embedding.instantiate()
        )

        self.encoder = (
            encoder
            if isinstance(encoder, TransformerEncoder)
            else TransformerEncoder(encoder)
        )

        self.embed_dim = self.encoder.config.embed_dim

        if isinstance(cls_token, bool):
            if cls_token:
                cls_token = TrainableToken(embed_dim=self.embed_dim)
            else:
                cls_token = None

        self.cls_token = cls_token

        self._gradient_checkpointing = False

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
        B = x.shape[0]

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, 1, -1)
            x = torch.cat((cls_token, x), dim=1)

            pos_enc = torch.cat((torch.zeros_like(cls_token), pos_enc), dim=1)

            if token_centers is not None:
                cls_center = torch.zeros(
                    B,
                    1,
                    token_centers.shape[-1],
                    device=token_centers.device,
                    dtype=token_centers.dtype,
                )
                token_centers = torch.cat((cls_center, token_centers), dim=1)

        out: TransformerOutput = self.encoder(
            x,
            pos_enc,
            self_mask=self_mask,
            self_key_padding_mask=self_key_padding_mask,
            return_attention=return_attention,
            return_hidden_states=return_hidden_states,
            token_centers=token_centers,
            attn_bias_scale=attn_bias_scale,
        )

        if self.cls_token is not None:
            cls_features = out.x[:, 0]
            patch_features = out.x[:, 1:]
        else:
            cls_features = None
            patch_features = out.x

        return PointEncoderOutput(
            patch_features=patch_features,
            cls_features=cls_features,
            attn_weights=out.attn_weights,
            hidden_states=out.hidden_states,
        )

    def forward(
        self,
        patch_points: PatchPoints,
        *,
        self_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_hidden_states: bool = False,
        attn_bias_scale: float = 1.0,
    ) -> PointEncoderOutput:
        multi_patches = self.patchify(patch_points)

        tokens: Tokens = self.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers

        out = self.transformer_encoder_forward(
            x,
            pos_enc,
            self_mask=self_mask,
            self_key_padding_mask=self_key_padding_mask,
            return_attention=return_attention,
            return_hidden_states=return_hidden_states,
            token_centers=token_centers,
            attn_bias_scale=attn_bias_scale,
        )

        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True
        self.encoder.enable_gradient_checkpointing()
        if hasattr(
            self.patch_embedding.point_embedding, "enable_gradient_checkpointing"
        ):
            self.patch_embedding.point_embedding.enable_gradient_checkpointing()


DEFAULT_POINT_ENCODER = lazy_instance(PointEncoder)
