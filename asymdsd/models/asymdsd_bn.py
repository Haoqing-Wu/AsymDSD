"""AsymDSD with a lightweight bottleneck predictor.

Instead of the default 6-layer transformer decoder predictor, this variant
uses a simple MLP bottleneck between the encoder and the target prediction:

    encoder output → project down (e.g. 384 → 64) → activate → project up
                     (64 → 384) → patch projection head → loss

The hypothesis: the full transformer predictor has enough capacity to
"cheat" by memorizing spatial relationships rather than learning semantic
features.  A low-dimensional bottleneck forces the encoder to produce
representations that are **compressible** — i.e. semantically rich — since
the predictor cannot carry spatial details through the narrow channel.

Inspired by BYOL/SimSiam which showed a 2-layer MLP predictor is enough
to prevent representation collapse.

All losses remain unchanged; only the predictor architecture differs.
"""

from __future__ import annotations

import torch
from torch import nn

from ..components import *
from ..components.common_types import FloatMayCall
from ..components.utils import init_lazy_defaults
from ..defaults import *
from ..layers import *
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode

logger = get_default_logger()


# ------------------------------------------------------------------
# Bottleneck predictor module
# ------------------------------------------------------------------


class BottleneckPredictor(nn.Module):
    """Lightweight MLP bottleneck that replaces the transformer predictor.

    Interface matches the encoder-style predictor path in ``AsymDSD``:
    ``forward(x, pos_enc) -> (output,)`` where the output has the same
    sequence length as the input.

    Architecture::

        input + pos_enc
            → LayerNorm
            → Linear(embed_dim → bottleneck_dim)
            → GELU
            → Linear(bottleneck_dim → bottleneck_dim)
            → GELU
            → Linear(bottleneck_dim → embed_dim)
            → LayerNorm

    The two hidden layers (project down, transform, project up) form the
    information bottleneck.  The surrounding LayerNorms stabilize training.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        bottleneck_dim: int = 64,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act1 = act_layer()
        self.mid = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.act2 = act_layer()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Forward pass matching the encoder-style predictor interface.

        Args:
            x: (B, S, D) token embeddings (visible context + mask tokens)
            pos_enc: (B, S, D) positional encodings

        Returns:
            Tuple of (output,) with shape (B, S, D).
        """
        h = self.norm_in(x + pos_enc)
        h = self.act1(self.down(h))
        h = self.act2(self.mid(h))
        h = self.norm_out(self.up(h))
        return (h,)

    def enable_gradient_checkpointing(self):
        """No-op — bottleneck is cheap enough to not need checkpointing."""
        pass


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


class BottleneckAsymDSD(AsymDSD):
    """AsymDSD with a bottleneck MLP predictor instead of a transformer.

    New parameters (everything else is forwarded to ``AsymDSD``):

    ``bottleneck_dim``
        Dimensionality of the bottleneck layer.  Lower values force more
        compression (and more semantic encoding).  Default 64.
    """

    @init_lazy_defaults
    def __init__(
        self,
        # --- bottleneck-specific ---
        bottleneck_dim: int = 64,
        # --- everything below is forwarded to AsymDSD ---
        max_epochs: int | None = None,
        max_steps: int | None = None,
        steps_per_epoch: int | None = None,
        optimizer: OptimizerSpec = DEFAULT_OPTIMIZER,
        training_mode: TraingingMode = TraingingMode.CLS_MASK,
        patchify: MultiPointPatchify | None = None,
        local_patchify: MultiPointPatchify | None = None,
        norm_transform: NormalizationTransform = DEFAULT_NORM_TRANSFORM,
        aug_transform: AugmentationTransform = DEFAULT_AUG_TRANSFORM,
        mask_generator: MaskGenerator = DEFAULT_MASKING_GENERATOR,
        patch_embedding: PatchEmbeddingConfig = DEFAULT_PATCH_EMBEDDING_CFG,
        encoder_config: TransformerEncoderConfig = DEFAULT_TRANSFORMER_ENC_CONFIG,
        predictor_config: TransformerEncoderConfig
        | TransformerDecoderConfig
        | None = DEFAULT_TRANSFORMER_PROJ_CONFIG,
        projection_head_config: ProjectionHeadConfig = DEFAULT_PROJECTION_HEAD_CONFIG,
        classification_head_config: ClassificationHeadConfig | None = None,
        num_point_features: int = 3,
        batch_size: int = AsymDSD.DEFAULT_BATCH_SIZE,
        init_weight_scale: float = 0.02,
        shared_projection_head: bool = False,
        cls_teacher_temp: FloatMayCall = 0.05,
        cls_student_temp: FloatMayCall = 0.1,
        patch_teacher_temp: FloatMayCall = 0.05,
        patch_student_temp: FloatMayCall = 0.1,
        cls_centering_momentum: FloatMayCall | None = None,
        patch_centering_momentum: FloatMayCall | None = None,
        cls_centering_power_law_tau: float | None = None,
        patch_centering_power_law_tau: float | None = None,
        ema_decay: FloatMayCall = DEFAULT_EMA_DECAY,
        mask_pos_noise: float | None = None,
        me_max_weight: float | None = None,
        koleo_loss_weight: float | None = None,
        classification_loss_weight: float | None = None,
        classification_label_smoothing: float | None = 0.2,
        regression_loss_weight: float | None = None,
        regression_loss_beta: float | None = None,
        mask_probability: float | None = 0.5,
        cls_predictor: ClsPredictor = ClsPredictor.DISABLED,
        add_unmasked_global_cls: bool = False,
        patch_instance_norm: bool = False,
        disable_projection: bool = False,
        gradient_checkpointing: bool = False,
        modules_ckpt_path: str | None = None,
    ) -> None:
        super().__init__(
            max_epochs=max_epochs,
            max_steps=max_steps,
            steps_per_epoch=steps_per_epoch,
            optimizer=optimizer,
            training_mode=training_mode,
            patchify=patchify,
            local_patchify=local_patchify,
            norm_transform=norm_transform,
            aug_transform=aug_transform,
            mask_generator=mask_generator,
            patch_embedding=patch_embedding,
            encoder_config=encoder_config,
            predictor_config=predictor_config,
            projection_head_config=projection_head_config,
            classification_head_config=classification_head_config,
            num_point_features=num_point_features,
            batch_size=batch_size,
            init_weight_scale=init_weight_scale,
            shared_projection_head=shared_projection_head,
            cls_teacher_temp=cls_teacher_temp,
            cls_student_temp=cls_student_temp,
            patch_teacher_temp=patch_teacher_temp,
            patch_student_temp=patch_student_temp,
            cls_centering_momentum=cls_centering_momentum,
            patch_centering_momentum=patch_centering_momentum,
            cls_centering_power_law_tau=cls_centering_power_law_tau,
            patch_centering_power_law_tau=patch_centering_power_law_tau,
            ema_decay=ema_decay,
            mask_pos_noise=mask_pos_noise,
            me_max_weight=me_max_weight,
            koleo_loss_weight=koleo_loss_weight,
            classification_loss_weight=classification_loss_weight,
            classification_label_smoothing=classification_label_smoothing,
            regression_loss_weight=regression_loss_weight,
            regression_loss_beta=regression_loss_beta,
            mask_probability=mask_probability,
            cls_predictor=cls_predictor,
            add_unmasked_global_cls=add_unmasked_global_cls,
            patch_instance_norm=patch_instance_norm,
            disable_projection=disable_projection,
            gradient_checkpointing=gradient_checkpointing,
            modules_ckpt_path=modules_ckpt_path,
        )

        if not self.do_predict:
            raise ValueError(
                "BottleneckAsymDSD requires a predictor_config to be set "
                "(it will be replaced with the bottleneck). "
                "Set predictor_config to any TransformerEncoderConfig."
            )

        self.bottleneck_dim = bottleneck_dim

        # Replace the transformer predictor with the lightweight bottleneck.
        # The parent __init__ created self.student["predictor"] from
        # predictor_config (possibly wrapped in a ProjectionWrapper if
        # enc_dim != pred_dim).  We replace the entire thing.
        bottleneck = BottleneckPredictor(
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
        )
        self.student["predictor"] = bottleneck

        # Force encoder-style predictor path (not decoder-style) since
        # our bottleneck uses the simpler (x, pos_enc) -> (output,) interface.
        self.do_predict = True
        self.decoder_style_predictor = False
        self.concat_tgt_memory = False

        logger.info(
            "Replaced transformer predictor with BottleneckPredictor "
            "(embed_dim=%d, bottleneck_dim=%d)",
            self.embed_dim,
            bottleneck_dim,
        )
