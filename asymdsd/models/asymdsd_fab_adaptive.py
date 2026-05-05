"""AsymDSD FAB with per-sample adaptive masking ratio.

Extends ``FusedAttnBlockAsymDSD`` by computing a per-sample mask ratio
based on the teacher's attention entropy.  Samples where the teacher is
confident (low entropy, concentrated attention) receive a higher mask
ratio (harder reconstruction task), while uncertain samples (high
entropy, diffuse attention) receive a lower ratio (easier task).

Because variable mask counts per sample break the rectangular tensor
assumption of ``gather_masked``, this variant pads visible/masked
sequences to the batch maximum and propagates ``key_padding_mask``
through the encoder and predictor transformers.

Key parameters:
    ``adaptive_mask_ratio_min`` — floor ratio (default 0.6)
    ``adaptive_mask_ratio_max`` — ceiling ratio (default 0.75)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from pytorch3d.ops import knn_points

from ..components import *
from ..components.common_types import FloatMayCall, OptionalTensor
from ..components.utils import (
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab import FusedAttnBlockAsymDSD
from .point_encoder import PointEncoder

logger = get_default_logger()


def gather_masked_padded(
    x: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather positions where mask=True, padding shorter samples to max count.

    Args:
        x: (B, P, F)
        mask: (B, P) bool — variable True counts per row

    Returns:
        gathered: (B, max_count, F) — padded with zeros
        padding_mask: (B, max_count) bool — True = valid, False = padding
    """
    B, P, F = x.shape
    counts = mask.sum(dim=-1)  # (B,)
    max_count = counts.max().item()

    sorted_indices = mask.float().argsort(dim=-1, descending=True)  # (B, P)
    sorted_indices_trunc = sorted_indices[:, :max_count]  # (B, max_count)

    gathered = torch.gather(
        x, 1, sorted_indices_trunc.unsqueeze(-1).expand(-1, -1, F)
    )  # (B, max_count, F)

    pos_range = torch.arange(max_count, device=x.device).unsqueeze(0)
    padding_mask = pos_range < counts.unsqueeze(1)  # (B, max_count) True=valid

    gathered = gathered * padding_mask.unsqueeze(-1)

    return gathered, padding_mask


class AdaptiveFusedAttnBlockAsymDSD(FusedAttnBlockAsymDSD):
    """FAB with per-sample adaptive mask ratio driven by teacher attention entropy.

    ``adaptive_mask_ratio_min``
        Minimum mask ratio (applied to uncertain/diffuse-attention samples).

    ``adaptive_mask_ratio_max``
        Maximum mask ratio (applied to confident/concentrated-attention samples).
    """

    @init_lazy_defaults
    def __init__(
        self,
        # --- adaptive masking specific ---
        adaptive_mask_ratio_min: float = 0.6,
        adaptive_mask_ratio_max: float = 0.75,
        # --- inherited from FusedAttnBlockAsymDSD ---
        block_ratio: float = 0.1,
        num_block_centers: int | None = None,
        sparse_ratio: float | None = None,
        num_inverse_masks: int = 0,
        num_random_masks: int = 0,
        random_mask_block_ratio: float = 0.1,
        random_mask_adjust_ratio: float = 0.1,
        select_visible: bool = False,
        num_select_visible_masks: int = 0,
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
        attn_bias_scale: FloatMayCall = 1.0,
        modules_ckpt_path: str | None = None,
    ) -> None:
        super().__init__(
            block_ratio=block_ratio,
            num_block_centers=num_block_centers,
            sparse_ratio=sparse_ratio,
            num_inverse_masks=num_inverse_masks,
            num_random_masks=num_random_masks,
            random_mask_block_ratio=random_mask_block_ratio,
            random_mask_adjust_ratio=random_mask_adjust_ratio,
            select_visible=select_visible,
            num_select_visible_masks=num_select_visible_masks,
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
            attn_bias_scale=attn_bias_scale,
            modules_ckpt_path=modules_ckpt_path,
        )

        self.adaptive_mask_ratio_min = adaptive_mask_ratio_min
        self.adaptive_mask_ratio_max = adaptive_mask_ratio_max

    # ------------------------------------------------------------------
    # Per-sample adaptive mask ratio
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_per_sample_num_masks(
        self,
        attn_weights: list[torch.Tensor],
        P: int,
    ) -> torch.Tensor:
        """Compute per-sample mask count based on teacher attention entropy.

        Returns:
            num_masks: (B,) long tensor of mask counts per sample.
        """
        attn = attn_weights[self.attn_layer_index]  # (B, H, S, S)
        cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (B, P)

        # Per-sample normalized entropy
        probs = cls_to_patch / cls_to_patch.sum(dim=-1, keepdim=True)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (B,)
        max_entropy = math.log(P)
        norm_entropy = entropy / max_entropy  # (B,) in [0, 1]

        # confidence = 1 - entropy: high confidence → higher mask ratio
        confidence = 1.0 - norm_entropy
        ratios = self.adaptive_mask_ratio_min + confidence * (
            self.adaptive_mask_ratio_max - self.adaptive_mask_ratio_min
        )

        num_masks = (ratios * P).round().long()
        num_masks = num_masks.clamp(min=1, max=P - 1)
        return num_masks

    # ------------------------------------------------------------------
    # Variable-count fused mask generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_fused_mask_adaptive(
        self,
        attn_weights: list[torch.Tensor],
        centers: torch.Tensor,
        num_masks_per_sample: torch.Tensor,
        invert_attn: bool = False,
        select_visible: bool = False,
    ) -> torch.Tensor:
        """Generate fused attention-block mask with per-sample variable counts.

        Args:
            attn_weights: per-layer attention, each (B, H, S, S).
            centers: patch center coordinates (B, P, 3).
            num_masks_per_sample: (B,) long tensor of target mask counts.
            invert_attn: bias toward low attention if True.
            select_visible: if True, blocks+sparse define visible patches.

        Returns:
            mask: (B, P) boolean tensor, True = masked.
        """
        B, P, _ = centers.shape
        device = centers.device

        attn = attn_weights[self.attn_layer_index]  # (B, H, S, S)
        cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (B, P)

        temperature: float = self.scheduler.value["attn_mask_temperature"]

        # When select_visible, blocks+sparse define visible patches.
        num_select = torch.where(
            torch.tensor(select_visible, device=device),
            P - num_masks_per_sample,
            num_masks_per_sample,
        )  # (B,)

        # --- Block component ---
        block_size = max(1, round(self.block_ratio * P))

        if self.num_block_centers is not None:
            num_centers = self.num_block_centers
        else:
            avg_select = num_select.float().mean().item()
            target_block_patches = round(0.6 * avg_select)
            num_centers = max(1, round(target_block_patches / block_size))

        block_attn = cls_to_patch
        if invert_attn:
            block_attn = 1.0 / (cls_to_patch + 1e-8)
        log_probs = (block_attn + 1e-8).log() / temperature
        gumbel_noise = -(-torch.rand_like(log_probs).clamp(1e-8).log()).log()
        center_scores = log_probs + gumbel_noise

        _, center_indices = center_scores.topk(num_centers, dim=-1)  # (B, K)

        selected_centers = torch.gather(
            centers, 1, center_indices.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, K, 3)

        knn_res = knn_points(
            selected_centers, centers, K=block_size, return_sorted=False
        )
        block_indices = knn_res.idx.flatten(-2, -1)  # (B, K*block_size)

        block_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, block_indices, True)

        num_block_selected = block_mask.sum(dim=-1)  # (B,)

        # --- Sparse component (per-sample count) ---
        num_sparse_needed = (num_select - num_block_selected).clamp(min=0)  # (B,)
        max_sparse = num_sparse_needed.max().item()

        if max_sparse > 0:
            sparse_attn = 1.0 / (cls_to_patch + 1e-8) if invert_attn else cls_to_patch
            sparse_log_probs = (sparse_attn + 1e-8).log() / temperature
            sparse_log_probs = sparse_log_probs.masked_fill(block_mask, float("-inf"))
            sparse_gumbel = -(
                -torch.rand_like(sparse_log_probs).clamp(1e-8).log()
            ).log()
            sparse_scores = sparse_log_probs + sparse_gumbel

            _, sparse_order = sparse_scores.topk(
                min(int(max_sparse), P), dim=-1
            )  # (B, max_sparse)

            range_idx = torch.arange(sparse_order.shape[1], device=device).unsqueeze(0)
            sparse_select = range_idx < num_sparse_needed.unsqueeze(1)

            sparse_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            selected_sparse = sparse_order.clone()
            selected_sparse[~sparse_select] = 0
            sparse_mask.scatter_(-1, selected_sparse, sparse_select)
        else:
            sparse_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        # --- Union → selected patches ---
        selected = block_mask | sparse_mask
        fused_mask = ~selected if select_visible else selected

        # --- Adjust each sample to its own num_masks ---
        current_count = fused_mask.sum(dim=-1)  # (B,)
        needs_adjustment = (current_count != num_masks_per_sample).any().item()
        if needs_adjustment:
            rand_uniform = torch.rand(B, P, device=device)
            flip_scores = torch.where(fused_mask, -rand_uniform, rand_uniform)
            sorted_indices = flip_scores.argsort(dim=-1)  # (B, P)

            pos_range = torch.arange(P, device=device).unsqueeze(0)  # (1, P)
            fused_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            fused_mask.scatter_(
                -1,
                sorted_indices,
                pos_range < num_masks_per_sample.unsqueeze(1),
            )

        return fused_mask  # (B, P) — variable True counts per row

    @torch.no_grad()
    def _generate_inverse_block_mask_adaptive(
        self,
        centers: torch.Tensor,
        num_masks_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Generate inverse-block random mask with per-sample variable counts."""
        B, P, _ = centers.shape
        device = centers.device

        block_size = max(1, round(self.random_mask_block_ratio * P))
        # Use average ratio to determine block centers
        avg_mask_ratio = num_masks_per_sample.float().mean().item() / P
        vis_ratio = 1.0 - avg_mask_ratio
        frac = (vis_ratio + self.random_mask_adjust_ratio) / block_size
        num_centers = max(1, round(P * frac))

        rand = torch.rand(B, P, device=device)
        center_idx = rand.argsort(dim=-1)[:, :num_centers]
        selected = torch.gather(centers, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))
        knn_res = knn_points(selected, centers, K=block_size, return_sorted=False)
        vis_idx = knn_res.idx.flatten(-2, -1)

        mask = torch.ones(B, P, dtype=torch.bool, device=device)
        mask.scatter_(-1, vis_idx, False)

        # Adjust to per-sample num_masks
        flip = torch.where(
            mask, -torch.rand(B, P, device=device), torch.rand(B, P, device=device)
        )
        sorted_indices = flip.argsort(dim=-1)
        pos_range = torch.arange(P, device=device).unsqueeze(0)
        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask.scatter_(-1, sorted_indices, pos_range < num_masks_per_sample.unsqueeze(1))
        return mask

    # ------------------------------------------------------------------
    # Student forward with padding masks
    # ------------------------------------------------------------------

    def _forward_student_adaptive(
        self,
        multi_patches: "MultiPatches",
        indices_masked_crops: torch.Tensor,
        indices_unmasked_crops: torch.Tensor | None,
        mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        """Student forward that handles variable-length masked/visible sequences.

        Uses padded gather + key_padding_mask to handle per-sample mask counts.
        """
        point_encoder: PointEncoder = self.student.point_encoder

        tokens: "Tokens" = point_encoder.patch_embedding(multi_patches)
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

        # ------- CLS on unmasked crops -------
        if self.mode.do_cls and indices_unmasked_crops is not None:
            x_unmasked = x[indices_unmasked_crops]
            pos_enc_unmasked = pos_enc[indices_unmasked_crops]
            centers_unmasked = (
                token_centers[indices_unmasked_crops]
                if token_centers is not None
                else None
            )
            pe_out = point_encoder.transformer_encoder_forward(
                x_unmasked,
                pos_enc_unmasked,
                token_centers=centers_unmasked,
                attn_bias_scale=attn_bias_scale,
            )
            x_cls_unmask: torch.Tensor = pe_out.cls_features  # type: ignore
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls_unmask
            out_dict["x_cls_logits"] = self.student.cls_projection_head(x_cls_unmask)[0]

        # ------- Masked Point Modeling -------
        x_mpm = x[indices_masked_crops]  # (B_m, P, F)
        pos_enc_mpm = pos_enc[indices_masked_crops]  # (B_m, P, F)
        centers_mpm = (
            token_centers[indices_masked_crops] if token_centers is not None else None
        )

        x_mpm = self._multi_mask_repeat(x_mpm)  # (B_m*multi_mask, P, F)
        pos_enc_mpm = self._multi_mask_repeat(pos_enc_mpm)
        if centers_mpm is not None:
            centers_mpm = self._multi_mask_repeat(centers_mpm)

        # --- Padded visible context ---
        inv_mask = ~mask
        x_visible, vis_pad_mask = gather_masked_padded(x_mpm, inv_mask)
        pos_enc_visible, _ = gather_masked_padded(pos_enc_mpm, inv_mask)
        centers_visible = None
        if centers_mpm is not None:
            centers_visible, _ = gather_masked_padded(centers_mpm, inv_mask)

        # --- Padded masked queries ---
        pos_enc_masked, masked_pad_mask = gather_masked_padded(pos_enc_mpm, mask)
        num_masks_max = pos_enc_masked.shape[1]

        mask_tokens = self.mask_token.expand(x_mpm.shape[0], num_masks_max, -1)
        mask_tokens = mask_tokens * masked_pad_mask.unsqueeze(-1)

        # Optional noisy queries
        if self.masked_pos_noise is not None:
            noise = self.masked_pos_noise * torch.randn_like(pos_enc_masked)
            pos_enc_masked = pos_enc_masked + noise * masked_pad_mask.unsqueeze(-1)

        if self.do_predict:
            # --- Encoder on visible tokens (with padding mask) ---
            # key_padding_mask: True = ignore position (PyTorch convention)
            # Prepend False for CLS token position
            encoder_pad_mask = ~vis_pad_mask  # (B, num_vis_max) True=ignore
            if self.mode.do_cls:
                cls_pad = torch.zeros(
                    encoder_pad_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=encoder_pad_mask.device,
                )
                encoder_pad_mask_with_cls = torch.cat(
                    [cls_pad, encoder_pad_mask], dim=1
                )
            else:
                encoder_pad_mask_with_cls = encoder_pad_mask

            pe_out = point_encoder.transformer_encoder_forward(
                x_visible,
                pos_enc_visible,
                token_centers=centers_visible,
                attn_bias_scale=attn_bias_scale,
                self_key_padding_mask=encoder_pad_mask_with_cls,
            )

            # --- Encoded visible context ---
            if self.mode.do_cls:
                x_cls = pe_out.cls_features
                x_patch = pe_out.patch_features
                x_context = torch.concat((x_cls.unsqueeze(1), x_patch), dim=1)  # type: ignore
                # memory key_padding_mask: True=ignore
                # CLS is always valid (never ignore) → False, padding → True
                cls_valid = torch.zeros(
                    vis_pad_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=vis_pad_mask.device,
                )
                memory_key_padding_mask = torch.cat(
                    [cls_valid, ~vis_pad_mask], dim=1
                )  # True=ignore
            else:
                x_context = pe_out.patch_features
                memory_key_padding_mask = ~vis_pad_mask  # True=ignore

            # --- Predictor ---
            if self.decoder_style_predictor:
                # Decoder-style: cross-attend to visible context
                predictor_self_pad = ~masked_pad_mask  # True=ignore
                x_patch_pred = self.student.predictor(
                    mask_tokens,
                    pos_enc_masked,
                    memory=x_context,
                    self_key_padding_mask=predictor_self_pad,
                    memory_key_padding_mask=memory_key_padding_mask,
                )[0]
            else:
                # Encoder-style predictor
                x_pred = torch.concat((x_context, mask_tokens), dim=1)
                pos_enc_tuple = (pos_enc_visible, pos_enc_masked)
                if self.mode.do_cls:
                    x_cls_expanded = x_cls.unsqueeze(1)  # type: ignore
                    pos_enc_tuple = (x_cls_expanded,) + pos_enc_tuple
                pos_enc_full = torch.concat(pos_enc_tuple, dim=1)

                # Combined padding mask: context + masked queries
                if self.mode.do_cls:
                    pred_pad = torch.cat(
                        [cls_pad, ~vis_pad_mask, ~masked_pad_mask], dim=1
                    )
                else:
                    pred_pad = torch.cat([~vis_pad_mask, ~masked_pad_mask], dim=1)

                x_pred = self.student.predictor(
                    x_pred, pos_enc_full, self_key_padding_mask=pred_pad
                )[0]
                x_patch_pred = x_pred[:, -num_masks_max:]
        else:
            # No predictor — full encoder with mask tokens
            x_input = torch.concat((x_visible, mask_tokens), dim=1)
            pos_enc_input = torch.concat((pos_enc_visible, pos_enc_masked), dim=1)
            if centers_visible is not None and centers_mpm is not None:
                centers_masked_padded, _ = gather_masked_padded(centers_mpm, mask)
                centers_input = torch.concat(
                    (centers_visible, centers_masked_padded), dim=1
                )
            else:
                centers_input = None

            full_pad = torch.cat([~vis_pad_mask, ~masked_pad_mask], dim=1)
            if self.mode.do_cls:
                cls_pad = torch.zeros(
                    full_pad.shape[0], 1, dtype=torch.bool, device=full_pad.device
                )
                full_pad = torch.cat([cls_pad, full_pad], dim=1)

            pe_out = point_encoder.transformer_encoder_forward(
                x_input,
                pos_enc_input,
                token_centers=centers_input,
                attn_bias_scale=attn_bias_scale,
                self_key_padding_mask=full_pad,
            )

            x_cls = pe_out.cls_features
            x_patch_pred = pe_out.patch_features[:, -num_masks_max:]

        # --- CLS embeddings and logits (from masked crops) ---
        if self.mode.do_cls:
            if return_embeddings:
                out_dict["x_cls_embedding_masked"] = x_cls
            out_dict["x_cls_logits_masked"] = self.student.cls_projection_head(x_cls)[0]

        # --- MPM embeddings and logits ---
        # Only compute loss on valid (non-padded) masked positions
        valid_mask = masked_pad_mask  # (B_total, num_masks_max) True=valid
        x_patch_valid = x_patch_pred[valid_mask]  # (N_valid, F)

        if return_embeddings:
            out_dict["x_patch_embedding"] = x_patch_valid

        x_patch_proj: "ProjectionOutput" = self.student.patch_projection_head(
            x_patch_valid, return_x_norm=True
        )
        out_dict["x_patch_logits"] = x_patch_proj.x
        out_dict["x_patch_proj_norm"] = x_patch_proj.x_norm  # type: ignore

        return out_dict

    # ------------------------------------------------------------------
    # Override training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        # ---- Extract and patchify ----
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

        # ---- Step 2: Compute per-sample adaptive mask counts ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        P = global_centers.shape[1]

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

        # Per-sample num_masks based on teacher attention entropy
        masked_attn_weights = [aw[indices_masked_crops] for aw in attn_weights]
        masked_centers = global_centers[indices_masked_crops]  # (B_masked, P, 3)

        num_masks_per_sample = self._compute_per_sample_num_masks(
            masked_attn_weights, P
        )  # (B_masked,)

        # Log mean adaptive ratio for monitoring
        mean_mask_ratio = num_masks_per_sample.float().mean().item() / P

        # ---- Step 3: Generate multi_mask independent masks ----
        # Repeat num_masks for multi_mask iterations
        attn_masks = []

        if self.num_select_visible_masks > 0:
            num_vis = self.num_select_visible_masks
            num_attn_guided = self.multi_mask - self.num_random_masks
            num_masked = num_attn_guided - num_vis
            vis_normal = (num_vis + 1) // 2
            masked_normal = (num_masked + 1) // 2

            for i in range(self.multi_mask):
                if i >= num_attn_guided:
                    m = self._generate_inverse_block_mask_adaptive(
                        masked_centers, num_masks_per_sample
                    )
                elif i < num_vis:
                    invert = i >= vis_normal
                    m = self._generate_fused_mask_adaptive(
                        masked_attn_weights,
                        masked_centers,
                        num_masks_per_sample,
                        invert_attn=invert,
                        select_visible=True,
                    )
                else:
                    j = i - num_vis
                    invert = j >= masked_normal
                    m = self._generate_fused_mask_adaptive(
                        masked_attn_weights,
                        masked_centers,
                        num_masks_per_sample,
                        invert_attn=invert,
                        select_visible=False,
                    )
                attn_masks.append(m)
        else:
            num_normal = (
                self.multi_mask - self.num_inverse_masks - self.num_random_masks
            )
            for i in range(self.multi_mask):
                if i >= num_normal + self.num_inverse_masks:
                    m = self._generate_inverse_block_mask_adaptive(
                        masked_centers, num_masks_per_sample
                    )
                else:
                    invert = i >= num_normal
                    m = self._generate_fused_mask_adaptive(
                        masked_attn_weights,
                        masked_centers,
                        num_masks_per_sample,
                        invert_attn=invert,
                        select_visible=self.select_visible,
                    )
                attn_masks.append(m)

        attn_mask = torch.cat(attn_masks, dim=0)  # (B_masked * multi_mask, P)

        # ---- Step 4: Gather teacher targets ----
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

            # Gather teacher targets with padding then extract valid only
            x_patch_logits_t = self._multi_mask_repeat(x_patch_logits_t)
            x_patch_targets_padded, target_pad_mask = gather_masked_padded(
                x_patch_logits_t, attn_mask
            )
            out_targets["x_patch_logits"] = x_patch_targets_padded[target_pad_mask]

            if self.do_regression:
                x_patch_crop_rep = self._multi_mask_repeat(x_patch_crop)
                x_patch_emb_padded, _ = gather_masked_padded(
                    x_patch_crop_rep, attn_mask
                )
                out_targets["x_patch_embedding"] = x_patch_emb_padded[target_pad_mask]

        # ---- Step 5: Student forward (adaptive) ----
        preds = self._forward_student_adaptive(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            mask=attn_mask,
            return_embeddings=True,
        )

        # ---- Step 6: Compute losses ----
        loss = 0.0
        cls_loss = patch_loss = koleo_loss = me_max = None
        classification_loss = None
        total_terms = 0
        cls_terms = 0
        regression_loss = 0.0 if self.do_regression else None

        targets = out_targets

        if self.mode.do_mask:
            if not self.disable_projection:
                patch_loss = self.patch_loss(
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
            "mask_components": None,
            "adaptive_mask_ratio": mean_mask_ratio,
        }

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if outputs.get("adaptive_mask_ratio") is not None:
            self.log(
                "train/adaptive_mask_ratio",
                outputs["adaptive_mask_ratio"],
                on_step=True,
            )
