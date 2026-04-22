"""AsymDSD with teacher-attention-guided masking.

Instead of masking patches randomly or by spatial blocks, this variant
uses the EMA teacher's own CLS-to-patch attention to decide *what* to
mask.  Patches the teacher attends to most are the most informative; by
hiding them the student is forced to reconstruct hard targets from less
informative context.

The masking strategy is:
    1. Run teacher forward with ``return_attention=True`` (no extra cost —
       the teacher forward already happens once per step).
    2. Aggregate the last-layer CLS→patch attention across heads.
    3. Sample a mask biased towards high-attention patches (controlled by
       ``attn_mask_temperature``).
    4. Use that mask for the normal student forward + MPM loss.

All other losses (CLS invariance, KoLeo, ME-max, …) remain unchanged.

Early in training the teacher's attention is near-uniform, so the mask
behaves like random masking.  As the teacher sharpens, the mask
automatically focuses on semantically important regions — a natural
curriculum with no extra hyperparameters.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall, OptionalTensor
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import MultiPatches, PatchPoints
from ..layers.tokenization import Tokens
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .point_encoder import PointEncoder, PointEncoderOutput

logger = get_default_logger()


class AttentionGuidedAsymDSD(AsymDSD):
    """AsymDSD that uses the teacher's attention to guide masking.

    New parameters (everything else is forwarded to ``AsymDSD``):

    ``attn_mask_temperature``
        Controls how strongly the mask is biased towards high-attention
        patches.  ``temperature → ∞``  ≈ uniform random masking;
        ``temperature → 0``  ≈ deterministic top-k masking.  Can be a
        ``Schedule`` for curriculum (e.g. start high, anneal low).

    ``attn_mask_top_k``
        If ``True``, use hard top-k instead of Gumbel sampling.
        Simpler but loses stochasticity.

    ``attn_layer_index``
        Which encoder layer's attention to use (default -1 = last layer).
    """

    @init_lazy_defaults
    def __init__(
        self,
        # --- attention-guided masking specific ---
        attn_mask_temperature: FloatMayCall = 1.0,
        attn_mask_top_k: bool = False,
        attn_layer_index: int = -1,
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

        if not self.mode.do_mask:
            raise ValueError(
                "AttentionGuidedAsymDSD requires training_mode that enables masking."
            )
        if not self.mode.do_cls:
            raise ValueError(
                "AttentionGuidedAsymDSD requires CLS mode (needs CLS→patch attention)."
            )

        self.attn_mask_top_k = attn_mask_top_k
        self.attn_layer_index = attn_layer_index

        # Inject temperature schedule
        self.schedules["attn_mask_temperature"] = attn_mask_temperature

    # ------------------------------------------------------------------
    # Attention-guided mask generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_attention_mask(
        self,
        attn_weights: list[torch.Tensor],
        num_masks: int,
    ) -> torch.Tensor:
        """Generate a boolean mask biased towards high-attention patches.

        Args:
            attn_weights: list of per-layer attention tensors, each
                          (B, num_heads, seq_len, seq_len) where seq_len
                          includes CLS at position 0.
            num_masks: number of patches to mask per sample.

        Returns:
            mask: (B, P) boolean tensor, True = masked.
        """
        # Take the selected layer's attention
        attn = attn_weights[self.attn_layer_index]  # (B, H, S, S)

        # CLS token is at position 0, patches at 1..P
        # Average across heads: CLS → each patch
        cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (B, P)

        temperature: float = self.scheduler.value["attn_mask_temperature"]

        if self.attn_mask_top_k:
            # Deterministic: mask the top-k most-attended patches
            _, top_indices = cls_to_patch.topk(num_masks, dim=-1)
            mask = torch.zeros_like(cls_to_patch, dtype=torch.bool)
            mask.scatter_(-1, top_indices, True)
        else:
            # Stochastic: Gumbel-softmax sampling biased by attention
            # Higher temperature → more uniform; lower → more deterministic
            log_probs = (cls_to_patch + 1e-8).log() / temperature
            gumbel_noise = -(-torch.rand_like(log_probs).clamp(1e-8).log()).log()
            scores = log_probs + gumbel_noise

            # Select top-k from noisy scores
            _, top_indices = scores.topk(num_masks, dim=-1)
            mask = torch.zeros_like(cls_to_patch, dtype=torch.bool)
            mask.scatter_(-1, top_indices, True)

        return mask

    # ------------------------------------------------------------------
    # Override forward_teacher to return attention weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_teacher(
        self,
        multi_patches: MultiPatches,
        indices_masked_crops: OptionalTensor = None,
        mask: OptionalTensor = None,
        block_idx: OptionalTensor = None,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        """Like parent, but also runs teacher with return_attention=True
        and stores attention weights for mask generation."""
        point_encoder: PointEncoder = self.teacher.point_encoder  # type: ignore

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings

        out_dict: dict[str, OptionalTensor] = {
            "x_cls_logits": None,
            "x_cls_embedding": None,
            "x_patch_logits": None,
            "x_patch_embedding": None,
        }

        # Run teacher encoder with attention
        pe_out: PointEncoderOutput = point_encoder.transformer_encoder_forward(
            x, pos_enc, return_attention=True
        )
        x_cls = pe_out.cls_features
        x_patch = pe_out.patch_features

        # Store for reuse in training_step (avoid running encoder twice)
        self._teacher_attn_weights = pe_out.attn_weights
        self._teacher_x_patch = x_patch

        # ------- CLS (same as parent) -------
        if self.mode.do_cls:
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls

            x_cls_proj: torch.Tensor = self.teacher.cls_projection_head(x_cls)[0]  # type: ignore
            centering_momentum = self.scheduler.value["cls_centering_momentum"]
            out_dict["x_cls_logits"] = self.teacher.cls_centering(  # type: ignore
                x_cls_proj.unsqueeze(1), momentum=centering_momentum
            ).squeeze(1)

        # ------- MPM targets (same as parent) -------
        if self.mode.do_mask and mask is not None:
            x_patch_crop = x_patch[indices_masked_crops]

            if self.patch_instance_norm:
                x_patch_crop = torch.nn.functional.instance_norm(x_patch_crop.mT).mT

            x_patch_logits: torch.Tensor = self.teacher.patch_projection_head(x_patch_crop)[0]  # type: ignore
            centering_momentum = self.scheduler.value["patch_centering_momentum"]
            x_patch_logits = self.teacher.patch_centering(  # type: ignore
                x_patch_logits, momentum=centering_momentum
            )

            x_patch_logits = self._multi_mask_repeat(x_patch_logits)
            if block_idx is not None:
                batch_indices = (
                    torch.arange(x_patch_logits.size(0))
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, block_idx.size(1), block_idx.size(2))
                )
                x_patch_logits = x_patch_logits[batch_indices, block_idx].flatten(0, 1)
            else:
                x_patch_logits = gather_masked(x_patch_logits, mask)
            out_dict["x_patch_logits"] = x_patch_logits.flatten(0, 1)

            if return_embeddings:
                x_patch_crop = self._multi_mask_repeat(x_patch_crop)
                if block_idx is not None:
                    x_patch_crop = x_patch_crop[batch_indices, block_idx].flatten(0, 1)
                else:
                    x_patch_crop = gather_masked(x_patch_crop, mask)
                out_dict["x_patch_embedding"] = x_patch_crop.flatten(0, 1)

        return out_dict

    # ------------------------------------------------------------------
    # Override training_step: teacher first → attn mask → student
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        # ---- Extract and patchify (same as parent) ----
        global_crops_dict = batch.get("global_crops") or batch
        local_crops_dict = batch.get("local_crops")

        global_patch_points = PatchPoints(
            points=global_crops_dict["points"],  # type: ignore
            num_points=global_crops_dict.get("num_points"),  # type: ignore
            patches_idx=global_crops_dict.get("patches_idx"),  # type: ignore
            centers_idx=global_crops_dict.get("centers_idx"),  # type: ignore
        )

        if "global_crops" not in batch:
            B = global_patch_points.points.shape[0]
            C = 1
        else:
            B, C = global_patch_points.points.shape[:2]
            global_patch_points = self._flatten_to_batch(global_patch_points)

        multi_patches = self._extract_patches(
            global_patch_points,
            self.global_patchify,
        )
        global_centers = multi_patches.centers[-1]

        # ---- Step 1: Teacher forward (with attention, before masking) ----
        # We need attention from the full unmasked teacher to generate masks.
        # Run teacher first on ALL crops without any mask.
        # This replaces the parent's teacher forward which happens after mask gen.
        indices_all_crops = torch.arange(
            0, B * C, device=global_centers.device
        )

        # Teacher forward on full input → gets attention weights + features.
        # CLS logits (with centering) are computed here; we reuse them in
        # Step 3 to avoid running projection/centering twice.
        with torch.no_grad():
            teacher_out_step1 = self.forward_teacher(
                multi_patches,
                indices_masked_crops=indices_all_crops,
                mask=None,  # No mask yet — just get embeddings + attention
                return_embeddings=self.do_regression,
            )

        # ---- Step 2: Generate attention-guided mask ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None, (
            "Teacher forward did not return attention weights. "
            "This should not happen."
        )

        mask_ratio = self.mask_generator.sample_mask_ratio()
        P = global_centers.shape[1]
        num_masks_per_sample = round(mask_ratio * P)

        if num_masks_per_sample == 0:
            # Fallback: no masking this step (very rare)
            return super().training_step(batch, batch_idx)

        # Select which crops to mask (same logic as parent for mask_probability=1.0)
        if self.mask_probability == 1.0:
            indices_masked_crops = indices_all_crops
            indices_unmasked_crops = (
                indices_all_crops[::C] if self.add_unmasked_global_cls else None
            )
        elif self.mask_probability is None or self.mode == TraingingMode.MASK:
            indices_masked_crops = torch.arange(
                0, B * C, step=C, device=global_centers.device
            )
            indices_unmasked_crops = None
        else:
            num_mask_batch = round(self.mask_probability * B)  # type: ignore
            indices_batch = torch.randperm(B, device=global_centers.device)
            indices_masked_batch = indices_batch[:num_mask_batch]
            indices_masked_crops = indices_masked_batch * C
            indices_unmasked_crops = indices_batch[num_mask_batch:] * C

        # Generate the attention-guided mask for masked crops
        # Attention weights are for ALL crops (B*C), sub-select for masked crops
        masked_attn_weights = [
            aw[indices_masked_crops] for aw in attn_weights
        ]

        # multi_mask handling: generate multi_mask independent masks
        attn_masks = []
        for _ in range(self.multi_mask):
            m = self._generate_attention_mask(masked_attn_weights, num_masks_per_sample)
            attn_masks.append(m)
        # (B_masked * multi_mask, P)
        attn_mask = torch.cat(attn_masks, dim=0)

        # ---- Step 3: Gather teacher targets from cached features ----
        # Reuse x_patch / x_cls computed in forward_teacher (Step 1) —
        # no need to run the teacher encoder a second time.
        with torch.no_grad():
            x_patch_teacher = self._teacher_x_patch
            # Free cached tensors to release GPU memory
            del self._teacher_attn_weights, self._teacher_x_patch

            out_targets: dict[str, OptionalTensor] = {
                "x_cls_logits": None,
                "x_cls_embedding": None,
                "x_patch_logits": None,
                "x_patch_embedding": None,
            }

            # CLS targets — reuse from Step 1 (already projected + centered)
            if self.mode.do_cls:
                out_targets["x_cls_logits"] = teacher_out_step1["x_cls_logits"]
                if self.do_regression:
                    out_targets["x_cls_embedding"] = teacher_out_step1["x_cls_embedding"]

            # Patch targets at attention-guided mask positions
            x_patch_crop = x_patch_teacher[indices_masked_crops]
            if self.patch_instance_norm:
                x_patch_crop = torch.nn.functional.instance_norm(x_patch_crop.mT).mT

            x_patch_logits_t: torch.Tensor = self.teacher.patch_projection_head(x_patch_crop)[0]  # type: ignore
            centering_momentum = self.scheduler.value["patch_centering_momentum"]
            x_patch_logits_t = self.teacher.patch_centering(  # type: ignore
                x_patch_logits_t, momentum=centering_momentum
            )

            x_patch_logits_t = self._multi_mask_repeat(x_patch_logits_t)
            out_targets["x_patch_logits"] = gather_masked(
                x_patch_logits_t, attn_mask
            ).flatten(0, 1)

            if self.do_regression:
                x_patch_crop = self._multi_mask_repeat(x_patch_crop)
                out_targets["x_patch_embedding"] = gather_masked(
                    x_patch_crop, attn_mask
                ).flatten(0, 1)

        # ---- Step 4: Student forward with attention-guided mask ----
        preds = self.forward_student(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            mask=attn_mask,
            block_idx=None,
            return_embeddings=True,
        )

        # ---- Step 5: Compute losses (same structure as parent) ----
        loss = 0.0
        cls_loss = patch_loss = koleo_loss = me_max = None
        classification_loss = None
        total_terms = 0
        cls_terms = 0
        regression_loss = 0.0 if self.do_regression else None

        targets = out_targets

        if self.mode.do_mask:
            if not self.disable_projection:
                patch_loss = checkpoint(
                    self.patch_loss,
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

        cls_targets_logits: torch.Tensor = targets["x_cls_logits"]  # type: ignore
        global_cls_preds_masked: torch.Tensor = preds["x_cls_logits_masked"]  # type: ignore

        if self.mode.do_cls:
            cls_loss = 0.0
            dim_0_shape = (B, -1)

            cls_targets_reshaped = cls_targets_logits.unflatten(0, dim_0_shape)
            cls_target_probs = self.cls_loss.compute_target_probs(
                cls_targets_reshaped,
                teacher_temp=self.scheduler.value["cls_teacher_temp"],
            )
            student_temp = self.scheduler.value["cls_student_temp"]

            # Local crops
            if local_crops_dict is not None:
                local_patch_points = PatchPoints(
                    points=local_crops_dict["points"],  # type: ignore
                    num_points=local_crops_dict.get("num_points"),  # type: ignore
                    patches_idx=local_crops_dict.get("patches_idx"),  # type: ignore
                    centers_idx=local_crops_dict.get("centers_idx"),  # type: ignore
                )
                local_patch_points = self._flatten_to_batch(local_patch_points)
                local_multi_patches = self._extract_patches(
                    local_patch_points, self.local_patchify
                )
                local_preds = self.forward_student(
                    local_multi_patches, return_embeddings=True
                )
                local_cls_logits: torch.Tensor = local_preds["x_cls_logits"]  # type: ignore
                local_cls_logits = local_cls_logits.unflatten(0, dim_0_shape)
                cls_loss += self.cls_loss(
                    local_cls_logits,
                    cls_target_probs,
                    student_temp=student_temp,
                )
                cls_terms += 1

            # Masked global CLS
            if self.mask_probability == 1.0:
                dim_0_mask = (-1, self.multi_mask)
                cls_preds = global_cls_preds_masked.unflatten(0, dim_0_mask)
                cls_preds = cls_preds.reshape(*dim_0_shape, cls_preds.shape[-1])
                cls_loss += self.cls_loss(
                    cls_preds,
                    cls_target_probs,
                    student_temp=student_temp,
                )
                cls_terms += 1

                if self.add_unmasked_global_cls:
                    global_cls_preds: torch.Tensor = preds["x_cls_logits"]  # type: ignore
                    global_cls_preds = global_cls_preds.unflatten(0, (-1, 1))
                    cls_loss += self.cls_loss(
                        global_cls_preds,
                        cls_target_probs[:, 1:],
                        student_temp=student_temp,
                    )
                    cls_terms += 1
            else:
                global_cls_preds_unmask: torch.Tensor = preds["x_cls_logits"]  # type: ignore
                if global_cls_preds_unmask is not None:
                    global_cls_preds_unmask = global_cls_preds_unmask.unflatten(
                        0, dim_0_shape
                    )
                    cls_loss += self.cls_loss(
                        global_cls_preds_unmask,
                        cls_target_probs,
                        student_temp=student_temp,
                    )
                    cls_terms += 1

            if cls_terms > 0:
                cls_loss = cls_loss / cls_terms
                loss = loss + cls_loss
                total_terms += 1

            if self.do_koleo:
                cls_emb = preds.get("x_cls_embedding_masked") or preds.get("x_cls_embedding")
                if cls_emb is not None:
                    cls_emb = cls_emb.unflatten(0, dim_0_shape)
                    koleo_loss = self.koleo_loss(cls_emb)
                    loss = loss + self.koleo_loss_weight * koleo_loss
                    total_terms += self.koleo_loss_weight

            if self.me_max_weight > 0.0 and cls_preds is not None:  # type: ignore
                cls_student_temp_val = self.scheduler.value["cls_student_temp"]
                me_max = self.me_max_loss(cls_preds / cls_student_temp_val)  # type: ignore
                loss = loss + self.me_max_weight * me_max
                total_terms += self.me_max_weight

        loss = loss / total_terms  # type: ignore

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "cls_preds": preds.get("x_cls_logits_masked"),
            "cls_targets": targets.get("x_cls_logits"),
            "patch_loss": patch_loss,
            "patch_preds": preds.get("x_patch_logits"),
            "patch_targets": targets.get("x_patch_logits"),
            "me_max": me_max,
            "koleo_loss": koleo_loss,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            "centers": global_centers,
        }
