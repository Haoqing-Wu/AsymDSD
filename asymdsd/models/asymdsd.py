from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asymdsd.data import PointCloudDataModule

from enum import StrEnum, auto
from functools import partial

import lightning as L
import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.checkpointing_utils import load_module_from_checkpoint
from ..components.common_types import FloatMayCall, OptionalTensor
from ..components.scheduling import Schedule
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
    lengths_to_mask,
    sequentialize_transform,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import MultiPatches, PatchPoints
from ..layers.tokenization import Tokens, TrainableToken
from ..loggers import get_default_logger
from ..loss import (
    ClsLoss,
    ClsRegressionLoss,
    KoLeoLoss,
    MeanEntropyLoss,
    # MemEfficientPatchLoss,
    PatchLoss,
)
from .point_encoder import PointEncoder

logger = get_default_logger()


class TraingingMode(StrEnum):
    MASK = auto()  # Masked Point Cloud Modeling
    CLS = auto()
    CLS_MASK = auto()

    @property
    def do_cls(self) -> bool:
        return self != TraingingMode.MASK

    @property
    def do_mask(self) -> bool:
        return self != TraingingMode.CLS


class ClsPredictor(StrEnum):
    DISABLED = auto()
    MASK_ONLY = auto()
    ALWAYS = auto()

    @property
    def is_enabled(self) -> bool:
        return self != ClsPredictor.DISABLED


class AsymDSD(L.LightningModule):
    DEFAULT_BATCH_SIZE = 128

    @init_lazy_defaults
    def __init__(
        self,
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
        batch_size: int = DEFAULT_BATCH_SIZE,
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
        attn_bias_scale: FloatMayCall = 1.0,
        modules_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs if max_epochs and max_epochs > 0 else None
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be specified.")

        self.steps_per_epoch = steps_per_epoch

        self.num_point_features = num_point_features
        self.batch_size = batch_size
        self.act_layer = encoder_config.act_layer
        self.embed_dim = encoder_config.embed_dim
        self.n_prototypes = projection_head_config.out_dim

        self.mode = training_mode

        self.norm_transform = norm_transform or IdentityMultiArg()
        self.aug_transform: nn.Module = (
            sequentialize_transform(aug_transform)
            if aug_transform
            else IdentityMultiArg()
        )

        self.patchify = self.global_patchify = patchify or ToMultiPatches()
        self.local_patchify = local_patchify or ToMultiPatches()
        # if self.local_patchify is None and self.mode.do_cls:
        #     logger.warning("Local patchify is None, but CLS mode is enabled.")

        self.optimizer_spec = optimizer

        # Tokenizer is not used, as embedding layer should be considered by EMA.

        projection_head_config.in_dim = self.embed_dim
        projection_head = partial(ProjectionHead, **vars(projection_head_config))

        self.mask_generator = mask_generator
        self.multi_mask = self.mask_generator.multi_mask
        self.multi_block = self.mask_generator.multi_block
        self.mask_probability = mask_probability
        self.add_unmasked_global_cls = add_unmasked_global_cls

        self.masked_pos_noise = mask_pos_noise

        self.init_weight_scale = init_weight_scale
        self.return_patches = False

        if predictor_config is not None:
            self.do_predict = True
            self.decoder_style_predictor = predictor_config.CLS == TransformerDecoder
            self.concat_tgt_memory = (
                True
                if self.decoder_style_predictor and predictor_config.concat_tgt_memory  # type: ignore
                else False
            )
        else:
            self.do_predict = False
            self.decoder_style_predictor = False
            self.concat_tgt_memory = False

        self.cls_predictor = cls_predictor  # TODO: Warn when not do_cls
        self.disable_projection = disable_projection
        self.patch_instance_norm = patch_instance_norm

        if not self.mode.do_cls and self.cls_predictor.is_enabled:
            logger.warning(
                "CLS predictor is enabled, but CLS mode is not enabled. "
                "Set cls_predictor=False to disable."
            )
            if self.decoder_style_predictor:
                raise ValueError(
                    "CLS predictor is only supported with a transformer encoder-style predictor. "
                    "Set predictor_config to a TransformerEncoderConfig, or set cls_predictor=False."
                )

        if not self.do_predict and self.cls_predictor.is_enabled:
            if self.mode.do_cls:
                raise ValueError(
                    "CLS predictor is enabled, but do_predict is False. "
                    "Set do_predict=True to enable."
                )

        patch_embedding.position_embedding.embed_dim = self.embed_dim
        patch_embedding.point_embedding.embed_dim = self.embed_dim

        def point_encoder():
            return PointEncoder(
                patchify=self.patchify,
                cls_token=TrainableToken(self.embed_dim) if self.mode.do_cls else None,
                patch_embedding=patch_embedding.instantiate(),
                encoder=encoder_config.instantiate(),
            )

        self.mask_token = TrainableToken(self.embed_dim)

        # Both student and teacher always have an identical encoder
        self.student = nn.ModuleDict({"point_encoder": point_encoder()})
        self.teacher = nn.ModuleDict({"point_encoder": point_encoder()})

        if self.mode.do_mask:
            self.student["patch_projection_head"] = projection_head()
            self.teacher["patch_projection_head"] = projection_head()

            self.teacher["patch_centering"] = (
                Centering(self.n_prototypes, patch_centering_power_law_tau)
                if patch_centering_momentum is not None
                else IdentityMultiArg()
            )

        if (
            self.mode.do_mask or self.cls_predictor.is_enabled
        ) and predictor_config is not None:
            # Additional modules for prediction (mendatory for MPM)
            student_predictor = predictor_config.instantiate()  # type: ignore
            self.student["predictor"] = student_predictor

            enc_dim = self.embed_dim
            pred_dim = predictor_config.embed_dim  # type: ignore
            if pred_dim != enc_dim:
                self.student["predictor"] = ProjectionWrapper(
                    student_predictor,
                    enc_dim,
                    pred_dim,
                    project_kwargs=["memory"] if self.decoder_style_predictor else None,
                )

        if self.mode.do_cls:
            # Additional modules for global CLS
            if shared_projection_head:
                self.student["cls_projection_head"] = self.student[
                    "patch_projection_head"
                ]
                self.teacher["cls_projection_head"] = self.teacher[
                    "patch_projection_head"
                ]
            else:
                self.student["cls_projection_head"] = projection_head()
                self.teacher["cls_projection_head"] = projection_head()

            self.teacher["cls_centering"] = (
                Centering(self.n_prototypes, cls_centering_power_law_tau)
                if cls_centering_momentum is not None
                else IdentityMultiArg()
            )

        self.ema = EMA(self.student, self.teacher)

        if gradient_checkpointing:
            self.student["point_encoder"].enable_gradient_checkpointing()
            if self.mode.do_mask and self.do_predict:
                student_predictor.enable_gradient_checkpointing()

        self.patch_loss = PatchLoss()  # TODO: Update
        self.cls_loss = ClsLoss()
        self.koleo_loss = KoLeoLoss(input_is_normalized=False)
        self.me_max_loss = MeanEntropyLoss(dim=self.n_prototypes)
        self.me_max_weight = me_max_weight or 0.0
        self.koleo_loss_weight = (
            koleo_loss_weight if (koleo_loss_weight and self.mode.do_cls) else 0.0
        )
        self.regression_loss_weight = (
            regression_loss_weight if regression_loss_weight else 0.0
        )
        self.classification_loss_weight = (
            classification_loss_weight if classification_loss_weight else 0.0
        )

        self.do_regression = self.regression_loss_weight > 0.0
        self.do_koleo = self.koleo_loss_weight > 0.0
        self.do_classification = self.classification_loss_weight > 0.0

        if self.do_regression:
            self.patch_regression_loss = (
                nn.SmoothL1Loss(beta=regression_loss_beta)
                if regression_loss_beta
                else nn.MSELoss()
            )
            self.cls_regression_loss = ClsRegressionLoss(beta=regression_loss_beta)

        if self.do_classification:
            if self.mode.do_mask and self.mask_probability != 1.0:
                raise ValueError(
                    "Classification loss is only supported with mask prob 1.0."
                )
            self.classification_loss = nn.CrossEntropyLoss(
                label_smoothing=classification_label_smoothing or 0.0
            )
            classification_head_config.embed_dim = self.embed_dim  # type: ignore
            self.classification_head = classification_head_config.instantiate()  # type: ignore

        self.schedules = {
            "lr": self.optimizer_spec.lr,
            "wd": self.optimizer_spec.wd,
            "ema_decay": ema_decay,
            "cls_teacher_temp": cls_teacher_temp,
            "cls_student_temp": cls_student_temp,
            "patch_teacher_temp": patch_teacher_temp,
            "patch_student_temp": patch_student_temp,
            "patch_centering_momentum": patch_centering_momentum,  # Might be None
            "cls_centering_momentum": cls_centering_momentum,  # Might be None
            "attn_bias_scale": attn_bias_scale,
        }

        self.modules_ckpt_path = modules_ckpt_path
        self.loaded_from_checkpoint = False
        self.validation_epoch = 0
        # NOTE: Below is not needed due to CLI callback that calls save_hyperparameters
        # self.save_hyperparameters(ignore=['norm_transform', 'aug_transform'])

    def init_weights(self):
        std = self.init_weight_scale
        if self.mode != TraingingMode.MASK:
            nn.init.trunc_normal_(self.teacher.point_encoder.cls_token, std=std)
            nn.init.trunc_normal_(self.student.point_encoder.cls_token, std=std)
        nn.init.trunc_normal_(self.mask_token, std=std)

        # Might want to use a more standard approach based on (embed_dim) ** -0.5)
        # So to be determined per module.
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # For LayerNorm the standard initialization is used.

        self.apply(_init_weights)

    def setup(self, stage: str | None = None):
        if not self.steps_per_epoch:
            try:
                datamodule: "PointCloudDataModule" = self.trainer.datamodule  # type: ignore
                if datamodule.len_train_dataset is None:
                    raise AttributeError
                self.steps_per_epoch = (
                    datamodule.len_train_dataset // self.batch_size  # type: ignore
                )
            except AttributeError:
                raise ValueError(
                    "steps_per_epoch must be specified if the length of the train dataset is not known."
                )

        real_schedule: list[Schedule] = [
            s for s in self.schedules.values() if isinstance(s, Schedule)
        ]

        for schedule in real_schedule:
            max_epochs = self.max_epochs or (self.max_steps / self.steps_per_epoch)  # type: ignore
            schedule.set_default_max_epochs(max_epochs)  # type: ignore
            schedule.set_steps_per_epoch(self.steps_per_epoch)

        # For ema decay scaling, now disabled
        # decay_multiplier = partial(
        #     compute_decay_fractional_update,
        #     update_size=self.batch_size,
        #     original_update_size=AsymDSD.DEFAULT_BATCH_SIZE
        # )

        # ema_decay = self.schedules['ema_decay']

        # self.schedules['ema_decay'] = lambda x: decay_multiplier(ema_decay(x)) if callable(
        #     ema_decay) else decay_multiplier(ema_decay)

        # lr and wd are scheduled by its own lrscheduler
        self.schedules.pop("lr", None)
        self.schedules.pop("wd", None)

        self.scheduler = Scheduler(**self.schedules)

        if self.modules_ckpt_path is not None:
            load_module_from_checkpoint(
                self.modules_ckpt_path,
                self,
                device=self.device,
                strict=False,
            )
            logger.info(f"Loaded modules from checkpoint {self.modules_ckpt_path}.")

        self.validation_epoch = 0

    def _multi_mask_repeat(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(self.multi_mask, dim=0)

    def _create_attn_mask(self, tgt_len: int, src_len: int) -> torch.Tensor:
        # Can create attn_mask such that local masked regions may attend each other.
        attn_mask = torch.zeros(
            (tgt_len, src_len), dtype=torch.bool, device=self.device
        )
        attn_mask[-tgt_len:, -tgt_len:] = ~torch.eye(
            tgt_len, dtype=torch.bool, device=self.device
        )
        return attn_mask

    def _flatten_to_batch(self, patch_points: PatchPoints) -> PatchPoints:
        patch_points.points = patch_points.points.flatten(0, 1)
        num_points = patch_points.num_points
        if num_points is not None:
            patch_points.num_points = num_points.flatten(0, 1)
        patches_idx = patch_points.patches_idx
        if patches_idx is not None:
            patch_points.patches_idx = [x.flatten(0, 1) for x in patches_idx]
        centers_idx = patch_points.centers_idx
        if centers_idx is not None:
            patch_points.centers_idx = [x.flatten(0, 1) for x in centers_idx]
        return patch_points

    def _extract_patches(
        self,
        patch_points: PatchPoints,
        patchify: MultiPointPatchify | ToMultiPatches,
    ) -> MultiPatches:
        crops = patch_points.points
        num_points = patch_points.num_points

        mask = (
            lengths_to_mask(num_points, crops.size(1))
            if num_points is not None
            else None
        )

        crops = self.norm_transform(crops, mask=mask)
        crops = self.aug_transform(crops)

        # Crops is reference to the original tensor
        patch_points.points = crops

        return patchify(patch_points)

    @torch.no_grad()
    def forward_teacher(
        self,
        multi_patches: MultiPatches,
        indices_masked_crops: OptionalTensor = None,
        mask: OptionalTensor = None,
        block_idx: OptionalTensor = None,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        point_encoder: PointEncoder = self.teacher.point_encoder

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers

        attn_bias_scale = self.scheduler.value["attn_bias_scale"]

        out_dict: dict[str, OptionalTensor] = {
            "x_cls_logits": None,
            "x_cls_embedding": None,
            "x_patch_logits": None,
            "x_patch_embedding": None,
        }

        # x is normalized
        pe_out = point_encoder.transformer_encoder_forward(
            x,
            pos_enc,
            token_centers=token_centers,
            attn_bias_scale=attn_bias_scale,
        )
        x_cls = pe_out.cls_features  # type: ignore
        x_patch = pe_out.patch_features

        # ------- Invariance Learning (CLS) -------
        if self.mode.do_cls:
            if return_embeddings:
                # (B*multi_mask, embed_dim)
                out_dict["x_cls_embedding"] = x_cls

            x_cls: torch.Tensor = self.teacher.cls_projection_head(x_cls)[0]
            centering_momentum = self.scheduler.value["cls_centering_momentum"]
            out_dict["x_cls_logits"] = self.teacher.cls_centering(
                x_cls.unsqueeze(1), momentum=centering_momentum
            ).squeeze(1)

        # ------- Masked Point Modeling (MPM) -------
        if self.mode.do_mask and mask is not None:
            x_patch = x_patch[indices_masked_crops]

            if self.patch_instance_norm:
                x_patch = torch.nn.functional.instance_norm(x_patch.mT).mT

            x_patch_logits: torch.Tensor = self.teacher.patch_projection_head(x_patch)[
                0
            ]
            centering_momentum = self.scheduler.value["patch_centering_momentum"]
            x_patch_logits = self.teacher.patch_centering(
                x_patch_logits, momentum=centering_momentum
            )

            # Gather the masked patch_features according to mask (which is multi-mask)
            x_patch_logits = self._multi_mask_repeat(x_patch_logits)
            if (
                block_idx is not None
            ):  # TODO: Factor out this entire if else with shows up in both student and teacher.
                batch_indices = (
                    torch.arange(x_patch_logits.size(0))
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, block_idx.size(1), block_idx.size(2))
                )
                # (B*multi_mask*n_blocks, num_masks, F)
                x_patch_logits = x_patch_logits[batch_indices, block_idx].flatten(0, 1)

            else:
                x_patch_logits = gather_masked(x_patch_logits, mask)
            # (B*multi_mask*num_masks, n_prototypes)
            out_dict["x_patch_logits"] = x_patch_logits.flatten(0, 1)

            if return_embeddings:
                x_patch = self._multi_mask_repeat(x_patch)
                if block_idx is not None:
                    x_patch = x_patch[batch_indices, block_idx].flatten(0, 1)
                else:
                    x_patch = gather_masked(x_patch, mask)

                # (B*multi_mask*num_masks, embed_dim)
                out_dict["x_patch_embedding"] = x_patch.flatten(0, 1)

        # Could make exception for multi-mask=1, or potentially low mask_ratio
        # Consider preparing according to mask here
        # (B*C, n_prototypes), (B*multi_mask*num_masks, n_prototypes)
        return out_dict

    def forward_student(
        self,
        multi_patches: MultiPatches,
        indices_masked_crops: OptionalTensor = None,
        indices_unmasked_crops: OptionalTensor = None,
        mask: OptionalTensor = None,
        block_idx: OptionalTensor = None,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        point_encoder: PointEncoder = self.student.point_encoder

        # Patch embedding is performed only once (faster due to multi-mask)
        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers

        attn_bias_scale = self.scheduler.value["attn_bias_scale"]

        out_dict: dict[str, OptionalTensor] = {
            "x_cls_logits": None,
            "x_cls_logits_masked": None,
            "x_cls_embedding": None,
            "x_cls_embedding_masked": None,
            "x_patch_logits": None,
            "x_patch_embedding": None,
            "x_patch_proj_norm": None,
            "classification_logits": None,
            "classification_logits_masked": None,
        }

        # ------- Invariance Learning (CLS) -------
        def forward_cls(
            x: torch.Tensor,
            pos_enc: torch.Tensor,
            centers: torch.Tensor | None = None,
        ) -> None:
            pe_out = point_encoder.transformer_encoder_forward(
                x,
                pos_enc,
                token_centers=centers,
                attn_bias_scale=attn_bias_scale,
            )
            x_cls: torch.Tensor = pe_out.cls_features  # type: ignore
            x_patch = pe_out.patch_features
            if self.cls_predictor == ClsPredictor.ALWAYS:
                x_src = torch.concat((x_cls.unsqueeze(1), x_patch), dim=1)  # type: ignore
                pos_enc = torch.concat(
                    (torch.zeros_like(x_cls).unsqueeze(1), pos_enc), dim=1
                )
                x_cls = self.student.predictor(x_src, pos_enc)[0][:, 0]
                # [0][0] to grab cls token.

            if self.do_classification:
                logits = self.classification_head(x_cls, x_patch)
                out_dict["classification_logits"] = logits

            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls
            out_dict["x_cls_logits"] = self.student.cls_projection_head(x_cls)[0]

        if self.mode == TraingingMode.CLS or mask is None:
            # Only when CLS is enabled, and no MPM.
            forward_cls(x, pos_enc, token_centers)
            return out_dict

        # Unmasked global crops, if there are any in CLS_MASK mode
        # This is expensive for global crops
        # (so preferably all global patches are masked)
        if self.mode.do_cls and indices_unmasked_crops is not None:
            x_unmasked_batch = x[indices_unmasked_crops]
            pos_enc_unmasked_batch = pos_enc[indices_unmasked_crops]
            centers_unmasked_batch = (
                token_centers[indices_unmasked_crops]
                if token_centers is not None
                else None
            )
            forward_cls(
                x_unmasked_batch, pos_enc_unmasked_batch, centers_unmasked_batch
            )

        # ------- Masked Point Modeling -------
        x_mpm = x[indices_masked_crops]  # (B, P, F)
        pos_enc_mpm = pos_enc[indices_masked_crops]  # (B, P, F)
        centers_mpm = (
            token_centers[indices_masked_crops] if token_centers is not None else None
        )

        x_mpm = self._multi_mask_repeat(x_mpm)  # (B*multi_mask, P, F)
        pos_enc_mpm = self._multi_mask_repeat(pos_enc_mpm)  # (B*multi_mask, P, F)
        if centers_mpm is not None:
            centers_mpm = self._multi_mask_repeat(centers_mpm)

        # --- Visible context ---
        inv_mask = ~mask
        # (B*multi_mask, num_vis, F)
        x_visible = gather_masked(x_mpm, inv_mask)
        pos_enc_visible = gather_masked(pos_enc_mpm, inv_mask)
        centers_visible = (
            gather_masked(centers_mpm, inv_mask) if centers_mpm is not None else None
        )

        # --- Target positions queries ---
        multi_block = block_idx is not None  # == self.multi_block

        if multi_block:
            # Alternative to expand and gather.
            num_blocks = block_idx.size(1)
            batch_indices = (
                torch.arange(pos_enc_mpm.size(0))
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, num_blocks, block_idx.size(2))
            )
            # (B*multi_mask*n_blocks, num_masks, F)
            pos_enc_masked = pos_enc_mpm[batch_indices, block_idx].flatten(0, 1)

        else:
            # (B*multi_mask, num_masks, F)
            pos_enc_masked = gather_masked(pos_enc_mpm, mask)

        # Optional noisy queries (normal noise to pos_enc_masked)
        if self.masked_pos_noise is not None:
            pos_enc_masked += self.masked_pos_noise * torch.randn_like(pos_enc_masked)

        # --- Mask queries ---
        num_masks = pos_enc_masked.shape[1]
        num_vis = pos_enc_visible.shape[1]
        mask_tokens = self.mask_token.expand(x_mpm.shape[0], num_masks, -1)

        if self.do_predict:
            # --- Encoder ---
            pe_out = point_encoder.transformer_encoder_forward(
                x_visible,
                pos_enc_visible,
                token_centers=centers_visible,
                attn_bias_scale=attn_bias_scale,
            )

            # --- Encoded visible context ---
            if self.mode.do_cls:
                x_cls = pe_out.cls_features
                x_patch = pe_out.patch_features
                x_context = torch.concat((x_cls.unsqueeze(1), x_patch), dim=1)  # type: ignore
            else:
                x_context = pe_out.patch_features

            # --- Compute prediction ---
            if self.decoder_style_predictor:
                # --- Transformer Decoder-style predictor ---
                if multi_block:
                    x_context = x_context.repeat_interleave(num_blocks, dim=0)
                    mask_tokens = mask_tokens.repeat_interleave(num_blocks, dim=0)

                if self.concat_tgt_memory and not multi_block:
                    # When concatentating target and memory,
                    # target should only allow attention to memory and ITSELF.
                    # Except for multi-block, where it can attend to all blocks.
                    memory_len = num_vis + int(self.mode.do_cls)
                    memory_mask = self._create_attn_mask(
                        tgt_len=num_masks, src_len=memory_len + num_masks
                    )
                else:
                    memory_mask = None

                x_patch = self.student.predictor(
                    mask_tokens,
                    pos_enc_masked,
                    memory=x_context,
                    memory_mask=memory_mask,
                )[0]
                if self.do_classification:
                    x_pred = torch.concat((x_context, x_patch), dim=1)
            else:
                # --- Transformer Encoder-style predictor ---
                # Concatenate embeddings (x_context includes cls if do_cls)
                x_pred = torch.concat((x_context, mask_tokens), dim=1)
                if multi_block:
                    x_pred = x_pred.repeat_interleave(num_blocks, dim=0)
                    pos_enc_visible = pos_enc_visible.repeat_interleave(
                        num_blocks, dim=0
                    )

                # Concatenate positional encodings
                pos_enc_tuple = (pos_enc_visible, pos_enc_masked)
                if self.mode.do_cls:
                    x_cls_expanded = x_cls.unsqueeze(1)  # type: ignore
                    if multi_block:
                        x_cls_expanded = x_cls_expanded.repeat_interleave(
                            num_blocks, dim=0
                        )
                    pos_enc_tuple = (x_cls_expanded,) + pos_enc_tuple
                pos_enc_full = torch.concat(pos_enc_tuple, dim=1)

                x_pred = self.student.predictor(x_pred, pos_enc_full)[0]
                # (B*multi_mask, num_masks, F)
                x_patch = x_pred[:, -num_masks:]

        else:
            # --- Encoder ---
            x_input = torch.concat((x_visible, mask_tokens), dim=1)
            pos_enc_input = torch.concat((pos_enc_visible, pos_enc_masked), dim=1)
            if centers_visible is not None and centers_mpm is not None:
                centers_masked = gather_masked(centers_mpm, mask)
                centers_input = torch.concat((centers_visible, centers_masked), dim=1)
            else:
                centers_input = None
            pe_out = point_encoder.transformer_encoder_forward(
                x_input,
                pos_enc_input,
                token_centers=centers_input,
                attn_bias_scale=attn_bias_scale,
            )

            x_cls = pe_out.cls_features
            x_patch = pe_out.patch_features[:, -num_masks:]

        # --- CLS embeddings and logits ---
        if self.mode.do_cls:
            if self.cls_predictor.is_enabled:
                x_cls = x_pred[:, 0]
            # Otherwise c_xls is simply the encoder cls token.
            if return_embeddings:
                # (B*multi_mask*n_blocks, embed_dim)
                out_dict["x_cls_embedding_masked"] = x_cls
            # Note with multi-mask extra cls tokens are generated (might want to balance loss)
            out_dict["x_cls_logits_masked"] = self.student.cls_projection_head(x_cls)[0]

        if self.do_classification:
            logits = self.classification_head(
                x_cls, x_pred[:, -(num_masks + num_vis) :]
            )
            out_dict["classification_logits_masked"] = logits

        # --- MPM embeddings and logits ---
        if return_embeddings:
            # (B*multi_mask*n_blocks*num_masks, embed_dim)
            out_dict["x_patch_embedding"] = x_patch.flatten(0, 1)

        x_patch_proj: ProjectionOutput = self.student.patch_projection_head(
            x_patch, return_x_norm=True
        )
        # (B*multi_mask*num_masks, n_prototypes)
        out_dict["x_patch_logits"] = x_patch_proj.x.flatten(0, 1)
        out_dict["x_patch_proj_norm"] = x_patch_proj.x_norm.flatten(0, 1)  # type: ignore

        return out_dict

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        # If no nested key,
        global_crops_dict = batch.get("global_crops") or batch
        local_crops_dict = batch.get("local_crops")

        global_patch_points = PatchPoints(
            points=global_crops_dict["points"],  # type: ignore
            num_points=global_crops_dict.get("num_points"),  # type: ignore
            patches_idx=global_crops_dict.get("patches_idx"),  # type: ignore
            centers_idx=global_crops_dict.get("centers_idx"),  # type: ignore
        )

        if "global_crops" not in batch:
            # If there are no global crops, simply take the single entire point cloud.
            B, N, F = global_patch_points.points.shape
            C = 1  # Only one crop (which is the full view)
        else:
            B, C, N, F = global_patch_points.points.shape
            # Merge crops into batch dimension
            global_patch_points = self._flatten_to_batch(global_patch_points)

        multi_patches = self._extract_patches(
            global_patch_points,
            self.global_patchify,
        )  # (B*C, P, K, F>=3)

        # These are the centers of the super patches
        global_centers = multi_patches.centers[-1]

        indices_masked_crops = indices_unmasked_crops = mask = block_idx = None

        if self.mode.do_mask:
            if self.mode == TraingingMode.MASK or self.mask_probability is None:
                # When only masked point modeling is enabled select all crops
                # Note C=1 in the case of MASK
                indices_masked_crops = torch.arange(
                    0, B * C, step=C, device=global_centers.device
                )  # (B,)
            elif self.mask_probability == 1.0:
                # Use all crops for cls loss
                indices_masked_crops = torch.arange(
                    0, B * C, device=global_centers.device
                )
                if self.add_unmasked_global_cls:
                    # Use crop 0 for non-masked cls loss
                    indices_unmasked_crops = indices_masked_crops[::C]
            else:
                # num_mask_batches = round(self.mask_probability * B * C)
                # indices_masked_crops = torch.randperm(
                #     B * C, device=global_centers.device
                # )[:num_mask_batches]

                # Either mask all crops for a single input, or mask no of the crops.
                num_masks = round(self.mask_probability * B)
                indices_batch = torch.randperm(B, device=global_centers.device)

                indices_masked_batch = indices_batch[:num_masks]
                indices_unmasked_batch = indices_batch[num_masks:]

                def repeat_crop_indices(batch_indices: torch.Tensor) -> torch.Tensor:
                    return C * batch_indices.repeat_interleave(C) + torch.arange(
                        0, C, device=global_centers.device
                    ).repeat(num_masks)

                indices_masked_crops = repeat_crop_indices(indices_masked_batch)
                indices_unmasked_crops = repeat_crop_indices(indices_unmasked_batch)

            mask, block_idx = self.mask_generator(
                global_centers[indices_masked_crops]
            )  # (B*multi_mask, <=P)

        targets = self.forward_teacher(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            mask=mask,
            block_idx=block_idx,
            return_embeddings=self.do_regression,
        )

        preds = self.forward_student(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            mask=mask,
            block_idx=block_idx,
            # TODO: This requires better implementation of training_step
            # return_embeddings=self.do_koleo or self.do_regression,
            return_embeddings=True,
        )

        loss = 0.0
        cls_loss = patch_loss = koleo_loss = classification_loss = me_max = None
        total_terms = 0
        cls_terms = regression_terms = classification_terms = 0
        regression_loss = 0.0 if self.do_regression else None

        if self.mode.do_mask:
            if not self.disable_projection:
                patch_loss = checkpoint(
                    self.patch_loss,
                    # (B*multi_mask*num_masks, n_prototypes)
                    preds["x_patch_logits"],
                    targets["x_patch_logits"],
                    self.scheduler.value["patch_teacher_temp"],
                    self.scheduler.value["patch_student_temp"],
                )
                loss = loss + patch_loss  # type: ignore
                total_terms += 1

            if self.do_regression:
                regression_loss = self.patch_regression_loss(
                    preds["x_patch_embedding"],
                    targets["x_patch_embedding"],
                )
                loss = loss + self.regression_loss_weight * regression_loss
                total_terms += self.regression_loss_weight

        # (B*C, n_prototypes)
        cls_targets: torch.Tensor = targets["x_cls_logits"]  # type: ignore
        # (B*(C - 1), n_prototypes)
        global_cls_preds: torch.Tensor = preds["x_cls_logits"]  # type: ignore
        # (B*multi_mask, n_prototypes)
        global_cls_preds_masked: torch.Tensor = preds["x_cls_logits_masked"]  # type: ignore
        cls_preds = None

        # For Koleo and regression loss
        # (B*C, F)
        cls_embedding_targets: torch.Tensor = targets["x_cls_embedding"]  # type: ignore
        # (B*C, F)
        global_cls_embedding_preds: torch.Tensor = preds["x_cls_embedding"]  # type: ignore
        # (B*multi_mask, F)
        global_cls_embedding_preds_masked: torch.Tensor = preds[
            "x_cls_embedding_masked"
        ]  # type: ignore

        # Do local crops only if cls is needed
        if self.mode.do_cls:
            cls_loss = 0.0
            dim_0_shape = (B, -1)

            # (B, C, n_prototypes)
            cls_targets = cls_targets.unflatten(0, dim_0_shape)
            cls_target_probs = self.cls_loss.compute_target_probs(
                cls_targets,
                teacher_temp=self.scheduler.value["cls_teacher_temp"],
            )

            if self.do_regression:
                cls_embedding_targets = cls_embedding_targets.unflatten(0, dim_0_shape)

            if local_crops_dict is not None:
                local_patch_points = PatchPoints(
                    points=local_crops_dict["points"],  # type: ignore
                    num_points=local_crops_dict.get("num_points"),  # type: ignore
                    patches_idx=local_crops_dict.get("patches_idx"),  # type: ignore
                    centers_idx=local_crops_dict.get("centers_idx"),  # type: ignore
                )

                local_patch_points = self._flatten_to_batch(local_patch_points)

                multi_patches = self._extract_patches(
                    local_patch_points,
                    self.local_patchify,
                )
                # (B*C_l, P, K, F>=3)

                # No masking only cls
                # (B*C_l, n_prototypes)
                local_preds: dict[str, torch.Tensor] = self.forward_student(
                    multi_patches, return_embeddings=True
                )  # type: ignore
                # TODO: Clean this up

                local_cls_preds = local_preds["x_cls_logits"].unflatten(0, dim_0_shape)

                cls_loss += self.cls_loss(
                    local_cls_preds,
                    cls_target_probs,
                    student_temp=self.scheduler.value["cls_student_temp"],
                )
                cls_terms += 1

                if self.do_regression:
                    local_cls_embedding_preds = local_preds[
                        "x_cls_embedding"
                    ].unflatten(0, dim_0_shape)

                    regression_loss += self.cls_regression_loss(
                        local_cls_embedding_preds,
                        cls_embedding_targets,
                    )
                    regression_terms += 1

            # Merge global and local cls predictions along crop dimension
            if self.mode.do_mask:
                dim_0_shape_unmasked = (-1, 1)
                if indices_unmasked_crops is not None:
                    global_cls_preds = global_cls_preds.unflatten(
                        0, dim_0_shape_unmasked
                    )
                    global_cls_embedding_preds = global_cls_embedding_preds.unflatten(
                        0, dim_0_shape_unmasked
                    )

                dim_0_shape_masked = (-1, self.multi_mask)
                global_cls_preds_masked = global_cls_preds_masked.unflatten(
                    0, dim_0_shape_masked
                )
                global_cls_embedding_preds_masked = (
                    global_cls_embedding_preds_masked.unflatten(0, dim_0_shape_masked)
                )

                if self.mask_probability is None:
                    # Legacy functionality:
                    # Special case in which all multi-mask cls tokens are concatenated
                    # (B, C-1 + multi_mask, n_prototypes)
                    cls_preds = torch.cat(
                        (global_cls_preds, global_cls_preds_masked),
                        dim=1,
                    )
                    cls_embedding_preds = global_cls_embedding_preds

                    # Could also divide the masked loss by num_masks
                    cls_loss += self.cls_loss(
                        cls_preds,
                        cls_targets,
                        student_temp=self.scheduler.value["cls_student_temp"],
                    )
                    cls_terms += 1

                elif self.mask_probability == 1.0:
                    # Special case in which only the masked cls tokens are used
                    # Due to shape determinism all cls tokens are used in cls loss

                    cls_preds = global_cls_preds_masked.reshape(
                        *dim_0_shape, global_cls_preds_masked.shape[-1]
                    )
                    cls_embedding_preds = global_cls_embedding_preds_masked.reshape(
                        *dim_0_shape, global_cls_embedding_preds_masked.shape[-1]
                    )

                    cls_loss += self.cls_loss(
                        cls_preds,
                        cls_target_probs,
                        student_temp=self.scheduler.value["cls_student_temp"],
                    )
                    cls_terms += 1

                    if self.add_unmasked_global_cls:
                        cls_loss += self.cls_loss(
                            global_cls_preds,
                            cls_target_probs[
                                :, 1:
                            ],  # Compare to all the other crop's cls
                            student_temp=self.scheduler.value["cls_student_temp"],
                        )

                        cls_embedding_preds = torch.cat(
                            (global_cls_embedding_preds, cls_embedding_preds),
                            dim=1,
                        )

                        cls_terms += 1
                else:
                    cls_preds = torch.empty(
                        B * C,
                        1,
                        self.n_prototypes,
                        device=self.device,
                        dtype=global_cls_preds.dtype,
                    )

                    # Taking cls of only the first masked crop (multi-mask)
                    global_cls_preds_masked = global_cls_preds_masked[:, :1]

                    cls_preds[indices_unmasked_crops] = global_cls_preds
                    cls_preds[indices_masked_crops] = global_cls_preds_masked

                    cls_preds = cls_preds.reshape(*dim_0_shape, cls_preds.shape[-1])

                    cls_embedding_preds = torch.empty(
                        B * C,
                        1,
                        global_cls_embedding_preds.shape[-1],
                        device=self.device,
                        dtype=global_cls_embedding_preds.dtype,
                    )

                    global_cls_embedding_preds_masked = (
                        global_cls_embedding_preds_masked[:, :1]
                    )

                    cls_embedding_preds[indices_unmasked_crops] = (
                        global_cls_embedding_preds
                    )
                    cls_embedding_preds[indices_masked_crops] = (
                        global_cls_embedding_preds_masked
                    )

                    cls_embedding_preds = cls_embedding_preds.reshape(
                        *dim_0_shape, cls_embedding_preds.shape[-1]
                    )

                    # Could also divide the masked loss by num_masks
                    cls_loss += self.cls_loss(
                        cls_preds,
                        cls_targets,
                        # teacher_temp=self.scheduler.value["cls_teacher_temp"],
                        student_temp=self.scheduler.value["cls_student_temp"],
                    )
                    cls_terms += 1

            else:
                cls_preds = global_cls_preds.unflatten(0, dim_0_shape)
                cls_embedding_preds = global_cls_embedding_preds.unflatten(
                    0, dim_0_shape
                )

                cls_loss += self.cls_loss(
                    cls_preds,
                    cls_target_probs,
                    student_temp=self.scheduler.value["cls_student_temp"],
                )
                cls_terms += 1

            cls_loss = cls_loss / cls_terms
            loss = loss + cls_loss
            total_terms += 1

            if self.do_koleo:
                # TODO: Apply koleo loss to masked global crops
                koleo_loss = self.koleo_loss(cls_embedding_preds)
                loss = loss + self.koleo_loss_weight * koleo_loss
                total_terms += self.koleo_loss_weight

            if self.do_regression:
                regression_loss += self.cls_regression_loss(
                    cls_embedding_preds,
                    cls_embedding_targets,
                )
                regression_terms += 1
                regression_loss = regression_loss / regression_terms
                loss = loss + self.regression_loss_weight * regression_loss
                total_terms += self.regression_loss_weight

            if self.me_max_weight > 0.0:
                cls_student_temp = self.scheduler.value["cls_student_temp"]
                me_max = self.me_max_loss(cls_preds / cls_student_temp)
                loss = loss + self.me_max_weight * me_max
                total_terms += self.me_max_weight

        if self.do_classification:

            def compute_loss(
                logits: torch.Tensor, labels: torch.Tensor
            ) -> torch.Tensor:
                labels = labels.repeat_interleave(logits.shape[0] // B)

                return self.classification_loss(logits, labels)

            labels: torch.Tensor = batch["cloud_label"]  # type: ignore
            classification_loss = 0

            if preds["classification_logits"] is not None:
                classification_loss += compute_loss(
                    preds["classification_logits"],
                    labels,
                )
                classification_terms += 1

            if preds["classification_logits_masked"] is not None:
                classification_loss += compute_loss(
                    preds["classification_logits_masked"],
                    labels,
                )
                classification_terms += 1
            classification_loss = classification_loss / classification_terms
            loss = loss + self.classification_loss_weight * classification_loss
            total_terms += self.classification_loss_weight

        loss = loss / total_terms

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "cls_preds": cls_preds,
            "cls_targets": cls_targets,
            "patch_loss": patch_loss,
            "patch_preds": preds["x_patch_logits"],
            "patch_targets": targets["x_patch_logits"],
            "me_max": me_max,
            "koleo_loss": koleo_loss,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            # "patches": global_patches,
            "centers": global_centers,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_fit_start(self) -> None:
        # This is after checkpoint loading
        if self.modules_ckpt_path is None:
            if not self.loaded_from_checkpoint:
                self.init_weights()
            self.ema.init_weights()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.ema.update_parameters(self.scheduler.value["ema_decay"])
            self.scheduler.step()

        log_conditions = {
            "train/loss": True,
            "train/cls_loss": self.mode.do_cls,
            "train/patch_loss": self.mode.do_mask and not self.disable_projection,
            "train/me_max": self.me_max_weight > 0.0,
            "train/koleo_loss": self.do_koleo,
            "train/regression_loss": self.do_regression,
            "train/classification_loss": self.do_classification,
        }

        for log_key, condition in log_conditions.items():
            if condition:
                self.log(
                    log_key,
                    outputs[log_key.split("/")[-1]],
                    on_step=True,
                    on_epoch=log_key == "train/loss",
                    prog_bar=log_key
                    in [
                        "train/loss",
                        "train/cls_loss",
                        "train/patch_loss",
                        "train/regression_loss",
                    ],
                )

        self._log_schedules()

    def _log_schedules(self) -> None:
        self.log_dict(
            {k: v for k, v in self.scheduler.value.items() if v is not None},
            on_step=True,
        )

    def on_validation_end(self) -> None:
        self.validation_epoch += 1

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["scheduler"] = self.scheduler.state_dict()
        checkpoint["validation_epoch"] = self.validation_epoch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.validation_epoch = checkpoint["validation_epoch"]
        self.loaded_from_checkpoint = True

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        # Needs to overwrite to support scheduler that is not LRScheduler
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)  # Also works for wd_schedule

    def configure_optimizers(self):
        # lr_multiplier = self.batch_size / AsymDSD.DEFAULT_BATCH_SIZE
        lr_multiplier = 1.0  # TODO: Consider multi-mask and possibly multi-block

        optimizer = self.optimizer_spec.get_optim(self.parameters(), lr_multiplier)
        lr_scheduler = self.optimizer_spec.get_lr_scheduler(optimizer)
        weight_decay_scheduler = self.optimizer_spec.get_wd_scheduler(optimizer)

        optimizers = [optimizer]
        schedules = []

        if lr_scheduler is not None:
            schedules.append(
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "name": "lr_schedule",
                }
            )

        if weight_decay_scheduler is not None:
            schedules.append(
                {
                    "scheduler": weight_decay_scheduler,
                    "interval": "step",
                    "name": "wd_schedule",
                }
            )

        return optimizers, schedules
