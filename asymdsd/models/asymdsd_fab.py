"""AsymDSD with fused attention-block masking.

Combines teacher-attention-guided masking with spatial block structure:

1. **Block component** — select block centers via Gumbel sampling, then
   KNN-expand each center into a dense spatial cluster.  By default,
   centers are biased toward HIGH teacher CLS→patch attention.  With
   ``invert_block_attention=True``, centers are biased toward LOW
   attention instead, masking boring geometry while keeping
   discriminative patches visible for the encoder.

2. **Sparse component** — from the remaining (non-block) patches, sample
   additional individual patches biased toward HIGH attention to reach
   the target mask ratio.

Temperature annealing controls randomness:
    τ high (early) → block centers ≈ random, sparse ≈ random
    τ low  (late)  → attention-focused masking

All losses remain unchanged; only the mask generation differs.
"""

from __future__ import annotations

from typing import Any

import torch
from pytorch3d.ops import knn_points
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall, OptionalTensor
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_ag import AttentionGuidedAsymDSD

logger = get_default_logger()


class FusedAttnBlockAsymDSD(AttentionGuidedAsymDSD):
    """AsymDSD with fused attention-guided block + sparse masking.

    New parameters (on top of ``AttentionGuidedAsymDSD``):

    ``block_ratio``
        Fraction of patches per block (KNN neighborhood size).
        E.g. 0.1 means each center expands to ~6 patches (64 * 0.1).

    ``num_block_centers``
        Number of spatial block centers to sample per mask.
        If ``None``, computed automatically so block patches ≈ 40% of
        total masked patches.

    ``sparse_ratio``
        Fraction of the mask quota filled by sparse (non-block) patches.
        E.g. 0.4 means 40% sparse, 60% from blocks.  If ``None``,
        computed as whatever remains after block patches.

    ``invert_block_attention``
        If ``True``, block centers are biased toward LOW-attention patches
        (masking boring geometry) while the sparse component still targets
        HIGH-attention patches.  This keeps discriminative patches visible
        for the encoder.  If ``False`` (default), both components bias
        toward high attention (original behavior).

    ``num_visible_blocks``
        Number of high-attention blocks to keep *visible* (protected from
        masking).  These are sampled from the same attention-biased
        distribution as the masked blocks, but excluded from the mask.
        E.g. with 5 block centers and ``num_visible_blocks=1``, 4 blocks
        are masked and 1 high-attention block stays visible so the encoder
        can learn to represent discriminative regions.  ``0`` (default)
        disables this.
    """

    @init_lazy_defaults
    def __init__(
        self,
        # --- fused masking specific ---
        block_ratio: float = 0.1,
        num_block_centers: int | None = None,
        sparse_ratio: float | None = None,
        invert_block_attention: bool = False,
        num_visible_blocks: int = 0,
        # --- inherited from AttentionGuidedAsymDSD ---
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
            attn_mask_temperature=attn_mask_temperature,
            attn_mask_top_k=attn_mask_top_k,
            attn_layer_index=attn_layer_index,
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

        self.block_ratio = block_ratio
        self.num_block_centers = num_block_centers
        self.sparse_ratio = sparse_ratio
        self.invert_block_attention = invert_block_attention
        self.num_visible_blocks = num_visible_blocks

    # ------------------------------------------------------------------
    # Fused attention-block mask generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_fused_mask(
        self,
        attn_weights: list[torch.Tensor],
        centers: torch.Tensor,
        num_masks: int,
    ) -> torch.Tensor:
        """Generate a mask combining attention-guided blocks + sparse patches.

        Args:
            attn_weights: per-layer attention, each (B, H, S, S).
            centers: patch center coordinates (B, P, 3).
            num_masks: total number of patches to mask per sample.

        Returns:
            mask: (B, P) boolean tensor, True = masked.
        """
        B, P, _ = centers.shape
        device = centers.device

        attn = attn_weights[self.attn_layer_index]  # (B, H, S, S)
        cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (B, P)

        temperature: float = self.scheduler.value["attn_mask_temperature"]

        # --- Block component ---
        block_size = max(1, round(self.block_ratio * P))

        if self.num_block_centers is not None:
            num_centers = self.num_block_centers
        else:
            # Auto: aim for ~60% of mask from blocks (before overlap)
            target_block_patches = round(0.6 * num_masks)
            num_centers = max(1, round(target_block_patches / block_size))

        # Gumbel-sample block centers; optionally bias toward LOW attention
        total_centers = num_centers + self.num_visible_blocks
        block_attn = cls_to_patch
        if self.invert_block_attention:
            block_attn = 1.0 / (cls_to_patch + 1e-8)
        log_probs = (block_attn + 1e-8).log() / temperature
        gumbel_noise = -(-torch.rand_like(log_probs).clamp(1e-8).log()).log()
        center_scores = log_probs + gumbel_noise

        _, all_center_indices = center_scores.topk(total_centers, dim=-1)  # (B, K+V)

        # Split: first num_centers are masked, last num_visible_blocks are visible
        center_indices = all_center_indices[:, :num_centers]  # (B, K)

        # KNN expand masked block centers
        selected_centers = torch.gather(
            centers, 1, center_indices.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, K, 3)

        knn_res = knn_points(
            selected_centers, centers, K=block_size, return_sorted=False
        )
        block_indices = knn_res.idx.flatten(-2, -1)  # (B, K*block_size)

        block_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, block_indices, True)

        # KNN expand visible block centers → protected from masking
        protected_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        if self.num_visible_blocks > 0:
            visible_center_indices = all_center_indices[:, num_centers:]  # (B, V)
            visible_selected = torch.gather(
                centers,
                1,
                visible_center_indices.unsqueeze(-1).expand(-1, -1, 3),
            )  # (B, V, 3)
            vis_knn = knn_points(
                visible_selected, centers, K=block_size, return_sorted=False
            )
            vis_indices = vis_knn.idx.flatten(-2, -1)  # (B, V*block_size)
            protected_mask.scatter_(-1, vis_indices, True)
            # Protected patches cannot be in the block mask
            block_mask &= ~protected_mask

        num_block_masked = block_mask.sum(dim=-1)  # (B,) — may vary due to overlap

        # --- Sparse component ---
        # From non-block, non-protected patches, sample additional patches
        # biased by attention to fill the remaining mask quota
        num_sparse_needed = (num_masks - num_block_masked).clamp(min=0)  # (B,)
        max_sparse = num_sparse_needed.max().item()

        if max_sparse > 0:
            # Suppress block and protected patches from sparse sampling
            excluded = block_mask | protected_mask
            sparse_log_probs = (cls_to_patch + 1e-8).log() / temperature
            sparse_log_probs = sparse_log_probs.masked_fill(excluded, float("-inf"))
            sparse_gumbel = -(
                -torch.rand_like(sparse_log_probs).clamp(1e-8).log()
            ).log()
            sparse_scores = sparse_log_probs + sparse_gumbel

            # Sort by score descending — take up to max_sparse candidates
            _, sparse_order = sparse_scores.topk(
                min(int(max_sparse), P), dim=-1
            )  # (B, max_sparse)

            # Each sample needs a different number of sparse patches
            # Create a range mask: positions < num_sparse_needed[b] are selected
            range_idx = torch.arange(sparse_order.shape[1], device=device).unsqueeze(0)
            sparse_select = range_idx < num_sparse_needed.unsqueeze(1)

            sparse_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            # Only scatter where sparse_select is True
            selected_sparse = sparse_order.clone()
            selected_sparse[~sparse_select] = 0  # dummy index for non-selected
            sparse_mask.scatter_(-1, selected_sparse, sparse_select)
        else:
            sparse_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        # --- Union ---
        fused_mask = block_mask | sparse_mask

        # --- Ensure exactly num_masks per sample ---
        # Due to block overlaps or rounding, count might not be exact.
        # Use the same argsort-to-exact-count trick as InverseBlockPatchMasking.
        current_count = fused_mask.sum(dim=-1)  # (B,)
        target = num_masks

        needs_adjustment = (current_count != target).any().item()
        if needs_adjustment:
            rand_uniform = torch.rand(B, P, device=device)
            # Masked patches get negative scores (keep them first in sort),
            # unmasked get positive (remove them first if we need fewer).
            # Protected patches get +inf so they are never selected.
            flip_scores = torch.where(fused_mask, -rand_uniform, rand_uniform)
            flip_scores = flip_scores.masked_fill(protected_mask, float("inf"))
            adjusted_indices = flip_scores.argsort(dim=-1)[:, :target]
            fused_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            fused_mask.scatter_(-1, adjusted_indices, True)

        return fused_mask

    # ------------------------------------------------------------------
    # Override training_step to use fused mask with centers
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        # ---- Extract and patchify (same as parent ag) ----
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
        global_centers = multi_patches.centers[-1]  # (B*C, P, 3)

        # ---- Step 1: Teacher forward (with attention) ----
        indices_all_crops = torch.arange(0, B * C, device=global_centers.device)

        with torch.no_grad():
            teacher_out_step1 = self.forward_teacher(
                multi_patches,
                indices_masked_crops=indices_all_crops,
                mask=None,
                return_embeddings=self.do_regression,
            )

        # ---- Step 2: Generate fused attention-block masks ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        mask_ratio = self.mask_generator.sample_mask_ratio()
        P = global_centers.shape[1]
        num_masks_per_sample = round(mask_ratio * P)

        if num_masks_per_sample == 0:
            return AsymDSD.training_step(self, batch, batch_idx)

        # Select which crops to mask
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

        # Sub-select attention and centers for masked crops
        masked_attn_weights = [aw[indices_masked_crops] for aw in attn_weights]
        masked_centers = global_centers[indices_masked_crops]  # (B_masked, P, 3)

        # Generate multi_mask independent fused masks
        attn_masks = []
        for _ in range(self.multi_mask):
            m = self._generate_fused_mask(
                masked_attn_weights, masked_centers, num_masks_per_sample
            )
            attn_masks.append(m)
        attn_mask = torch.cat(attn_masks, dim=0)  # (B_masked * multi_mask, P)

        # ---- Step 3: Gather teacher targets from cached features ----
        with torch.no_grad():
            x_patch_teacher = self._teacher_x_patch
            del self._teacher_attn_weights, self._teacher_x_patch

            out_targets: dict[str, OptionalTensor] = {
                "x_cls_logits": None,
                "x_cls_embedding": None,
                "x_patch_logits": None,
                "x_patch_embedding": None,
            }

            if self.mode.do_cls:
                out_targets["x_cls_logits"] = teacher_out_step1["x_cls_logits"]
                if self.do_regression:
                    out_targets["x_cls_embedding"] = teacher_out_step1[
                        "x_cls_embedding"
                    ]

            x_patch_crop = x_patch_teacher[indices_masked_crops]
            if self.patch_instance_norm:
                x_patch_crop = torch.nn.functional.instance_norm(x_patch_crop.mT).mT

            x_patch_logits_t: torch.Tensor = self.teacher.patch_projection_head(
                x_patch_crop
            )[0]  # type: ignore
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

        # ---- Step 4: Student forward ----
        preds = self.forward_student(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            mask=attn_mask,
            block_idx=None,
            return_embeddings=True,
        )

        # ---- Step 5: Compute losses (identical to ag parent) ----
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
                cls_emb = preds.get("x_cls_embedding_masked")
                if cls_emb is None:
                    cls_emb = preds.get("x_cls_embedding")
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
