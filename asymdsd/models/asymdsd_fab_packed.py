"""AsymDSD FAB with packed-sequence encoder for mixed mask ratios.

Pairs complementary masks into a single packed sequence:
  - Sub-sequence A: select_visible=True  → few visible patches (e.g. 19)
  - Sub-sequence B: select_visible=False → many visible patches (e.g. 45)

Since A_visible + B_visible = P, the packed sequence is always exactly P
tokens (plus 2 CLS tokens), enabling a single encoder call with a
block-diagonal attention mask that prevents cross-attention between the
two sub-sequences.

This gives all 4 mask strategies meaningful block-coherent structure
without wasting compute on padding, and requires only 2 packed samples
per step (one normal + one inverse attention pair).
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
from ..layers.patchify import PatchPoints
from ..layers.tokenization import Tokens
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab import FusedAttnBlockAsymDSD
from .point_encoder import PointEncoder, PointEncoderOutput

logger = get_default_logger()


class PackedFusedAttnBlockAsymDSD(FusedAttnBlockAsymDSD):
    """FAB variant using sequence packing for complementary mask pairs.

    Instead of requiring uniform visible-set sizes across all multi-mask
    iterations, this variant pairs each ``select_visible=True`` mask
    (few visible patches) with a ``select_visible=False`` mask (many
    visible patches) into a single packed sequence of length P (+ 2 CLS).

    A block-diagonal attention mask ensures the two sub-sequences don't
    attend to each other.  After encoding, the packed output is split
    back for independent predictor/loss computation.

    The ``multi_mask`` budget is split into pairs:
      - ``multi_mask=4``: 2 packed pairs (normal + inverse attention)
      - Pair 0: vis_high_attn (19 vis) packed with masked_high_attn (45 vis)
      - Pair 1: vis_low_attn  (19 vis) packed with masked_low_attn  (45 vis)

    ``vis_mask_ratio``
        Mask ratio for the select_visible=True sub-sequence (default 0.7).
        The select_visible=False sub-sequence automatically uses
        (1 - vis_mask_ratio) as its mask ratio.
    """

    @init_lazy_defaults
    def __init__(
        self,
        vis_mask_ratio: float = 0.7,
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

        self.vis_mask_ratio = vis_mask_ratio

    # ------------------------------------------------------------------
    # Packed encoder forward
    # ------------------------------------------------------------------

    def _packed_encoder_forward(
        self,
        x_a: torch.Tensor,
        pos_enc_a: torch.Tensor,
        centers_a: torch.Tensor | None,
        x_b: torch.Tensor,
        pos_enc_b: torch.Tensor,
        centers_b: torch.Tensor | None,
        point_encoder: PointEncoder,
        attn_bias_scale: float = 1.0,
    ) -> tuple[PointEncoderOutput, PointEncoderOutput]:
        """Run encoder on packed sequence [CLS_A, A..., CLS_B, B...].

        A and B are two sub-sequences with potentially different lengths.
        Block-diagonal attention prevents cross-attention between them.

        Returns split PointEncoderOutputs for A and B.
        """
        B_batch = x_a.shape[0]
        S_a = x_a.shape[1]  # num visible patches for sub-seq A
        S_b = x_b.shape[1]  # num visible patches for sub-seq B

        has_cls = point_encoder.cls_token is not None

        # Prepend CLS tokens to each sub-sequence, then concatenate
        if has_cls:
            cls_token = point_encoder.cls_token.expand(B_batch, 1, -1)
            cls_pos = torch.zeros_like(cls_token)

            x_packed = torch.cat([cls_token, x_a, cls_token, x_b], dim=1)
            pos_packed = torch.cat([cls_pos, pos_enc_a, cls_pos, pos_enc_b], dim=1)

            # Centers: CLS gets zero center
            if centers_a is not None and centers_b is not None:
                cls_center = torch.zeros(
                    B_batch, 1, 3, device=x_a.device, dtype=x_a.dtype
                )
                centers_packed = torch.cat(
                    [cls_center, centers_a, cls_center, centers_b], dim=1
                )
            else:
                centers_packed = None

            len_a = 1 + S_a  # CLS_A + patches_A
            len_b = 1 + S_b  # CLS_B + patches_B
        else:
            x_packed = torch.cat([x_a, x_b], dim=1)
            pos_packed = torch.cat([pos_enc_a, pos_enc_b], dim=1)

            if centers_a is not None and centers_b is not None:
                centers_packed = torch.cat([centers_a, centers_b], dim=1)
            else:
                centers_packed = None

            len_a = S_a
            len_b = S_b

        total_len = len_a + len_b

        # Build block-diagonal attention mask
        # Shape: (total_len, total_len), True = blocked (additive -inf)
        attn_mask = torch.zeros(
            total_len, total_len, device=x_a.device, dtype=x_a.dtype
        )
        # Block off-diagonal regions with -inf
        attn_mask[:len_a, len_a:] = float("-inf")
        attn_mask[len_a:, :len_a] = float("-inf")

        # Run encoder (bypass point_encoder.transformer_encoder_forward
        # since we handle CLS prepending ourselves)
        out = point_encoder.encoder(
            x_packed,
            pos_packed,
            self_mask=attn_mask,
            token_centers=centers_packed,
            attn_bias_scale=attn_bias_scale,
        )

        # Split output
        x_out = out.x
        x_out_a = x_out[:, :len_a]
        x_out_b = x_out[:, len_a:]

        if has_cls:
            out_a = PointEncoderOutput(
                patch_features=x_out_a[:, 1:],
                cls_features=x_out_a[:, 0],
                attn_weights=None,
                hidden_states=None,
            )
            out_b = PointEncoderOutput(
                patch_features=x_out_b[:, 1:],
                cls_features=x_out_b[:, 0],
                attn_weights=None,
                hidden_states=None,
            )
        else:
            out_a = PointEncoderOutput(
                patch_features=x_out_a,
                cls_features=None,
                attn_weights=None,
                hidden_states=None,
            )
            out_b = PointEncoderOutput(
                patch_features=x_out_b,
                cls_features=None,
                attn_weights=None,
                hidden_states=None,
            )

        return out_a, out_b

    # ------------------------------------------------------------------
    # Packed student forward
    # ------------------------------------------------------------------

    def _forward_student_packed(
        self,
        multi_patches,
        indices_masked_crops: torch.Tensor,
        indices_unmasked_crops: torch.Tensor | None,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        """Student forward with packed complementary mask pairs.

        Args:
            mask_a: (B_packed, P) bool, masks for select_visible=True
                    (few visible, many masked). True = masked.
            mask_b: (B_packed, P) bool, masks for select_visible=False
                    (many visible, few masked). True = masked.

        Each row of mask_a is paired with the same row of mask_b.
        """
        point_encoder: PointEncoder = self.student.point_encoder

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

        # ------- Unmasked CLS forward (same as parent) -------
        if self.mode.do_cls and indices_unmasked_crops is not None:
            x_unmasked = x[indices_unmasked_crops]
            pos_unmasked = pos_enc[indices_unmasked_crops]
            centers_unmasked = (
                token_centers[indices_unmasked_crops]
                if token_centers is not None
                else None
            )
            pe_out_unmask = point_encoder.transformer_encoder_forward(
                x_unmasked,
                pos_unmasked,
                token_centers=centers_unmasked,
                attn_bias_scale=attn_bias_scale,
            )
            x_cls_unmask = pe_out_unmask.cls_features
            out_dict["x_cls_logits"] = self.student.cls_projection_head(x_cls_unmask)[0]
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls_unmask

        # ------- Masked Point Modeling with packed sequences -------
        x_mpm = x[indices_masked_crops]  # (B_masked, P, F)
        pos_enc_mpm = pos_enc[indices_masked_crops]
        centers_mpm = (
            token_centers[indices_masked_crops] if token_centers is not None else None
        )

        B_packed = mask_a.shape[0]  # num packed pairs

        # Repeat to match packed pairs (B_masked → B_packed)
        # B_packed = B_masked * num_pairs_per_sample
        num_pairs = B_packed // x_mpm.shape[0]
        x_mpm = x_mpm.repeat_interleave(num_pairs, dim=0)
        pos_enc_mpm = pos_enc_mpm.repeat_interleave(num_pairs, dim=0)
        if centers_mpm is not None:
            centers_mpm = centers_mpm.repeat_interleave(num_pairs, dim=0)

        # Gather visible patches for each sub-sequence
        inv_mask_a = ~mask_a  # (B_packed, P) — few True (visible)
        inv_mask_b = ~mask_b  # (B_packed, P) — many True (visible)

        x_vis_a = gather_masked(x_mpm, inv_mask_a)
        pos_vis_a = gather_masked(pos_enc_mpm, inv_mask_a)
        centers_vis_a = (
            gather_masked(centers_mpm, inv_mask_a) if centers_mpm is not None else None
        )

        x_vis_b = gather_masked(x_mpm, inv_mask_b)
        pos_vis_b = gather_masked(pos_enc_mpm, inv_mask_b)
        centers_vis_b = (
            gather_masked(centers_mpm, inv_mask_b) if centers_mpm is not None else None
        )

        # --- Packed encoder forward ---
        pe_out_a, pe_out_b = self._packed_encoder_forward(
            x_vis_a,
            pos_vis_a,
            centers_vis_a,
            x_vis_b,
            pos_vis_b,
            centers_vis_b,
            point_encoder,
            attn_bias_scale=attn_bias_scale,
        )

        # --- CLS logits from both sub-sequences ---
        cls_logits_list = []
        cls_emb_list = []
        if self.mode.do_cls:
            for pe_out in (pe_out_a, pe_out_b):
                x_cls = pe_out.cls_features
                cls_logits_list.append(self.student.cls_projection_head(x_cls)[0])
                if return_embeddings:
                    cls_emb_list.append(x_cls)

            # Interleave: [a0, b0, a1, b1, ...] to match multi_mask order
            cls_logits_a, cls_logits_b = cls_logits_list
            out_dict["x_cls_logits_masked"] = torch.cat(
                [cls_logits_a, cls_logits_b], dim=0
            )
            if return_embeddings:
                out_dict["x_cls_embedding_masked"] = torch.cat(
                    [cls_emb_list[0], cls_emb_list[1]], dim=0
                )

        # --- Predictor for masked patch reconstruction ---
        # Sub-sequence A: few visible, many masked
        # Sub-sequence B: many visible, few masked
        patch_logits_list = []
        patch_emb_list = []

        for pe_out, mask, pos_full, centers_full in [
            (pe_out_a, mask_a, pos_enc_mpm, centers_mpm),
            (pe_out_b, mask_b, pos_enc_mpm, centers_mpm),
        ]:
            x_patch = pe_out.patch_features
            x_cls = pe_out.cls_features

            pos_enc_masked = gather_masked(pos_full, mask)
            num_masks = pos_enc_masked.shape[1]

            if self.masked_pos_noise is not None:
                pos_enc_masked = (
                    pos_enc_masked
                    + self.masked_pos_noise * torch.randn_like(pos_enc_masked)
                )

            mask_tokens = self.mask_token.expand(B_packed, num_masks, -1)

            if self.mode.do_cls and x_cls is not None:
                x_context = torch.cat([x_cls.unsqueeze(1), x_patch], dim=1)
            else:
                x_context = x_patch

            if self.decoder_style_predictor:
                x_pred_patch = self.student.predictor(
                    mask_tokens,
                    pos_enc_masked,
                    memory=x_context,
                )[0]
            else:
                x_pred = torch.cat([x_context, mask_tokens], dim=1)

                pos_vis = gather_masked(pos_full, ~mask)
                pos_parts = (pos_vis, pos_enc_masked)
                if self.mode.do_cls:
                    cls_pos = torch.zeros(
                        B_packed,
                        1,
                        pos_vis.shape[-1],
                        device=pos_vis.device,
                        dtype=pos_vis.dtype,
                    )
                    pos_parts = (cls_pos,) + pos_parts
                pos_full_pred = torch.cat(pos_parts, dim=1)

                x_pred = self.student.predictor(x_pred, pos_full_pred)[0]
                x_pred_patch = x_pred[:, -num_masks:]

            if return_embeddings:
                patch_emb_list.append(x_pred_patch.flatten(0, 1))

            x_patch_proj = self.student.patch_projection_head(
                x_pred_patch, return_x_norm=True
            )
            patch_logits_list.append(x_patch_proj.x.flatten(0, 1))

        out_dict["x_patch_logits"] = torch.cat(patch_logits_list, dim=0)
        if return_embeddings:
            out_dict["x_patch_embedding"] = torch.cat(patch_emb_list, dim=0)

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

        # ---- Step 2: Generate complementary mask pairs ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        P = global_centers.shape[1]
        num_masks_a = round(self.vis_mask_ratio * P)  # many masked (e.g. 45)
        num_masks_b = P - num_masks_a  # few masked (e.g. 19)

        if num_masks_a == 0 or num_masks_b == 0:
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

        # Generate paired masks:
        # multi_mask=4 → 2 packed pairs (normal-attn pair + inverse-attn pair)
        num_pairs = self.multi_mask // 2

        masks_a_list = []  # select_visible=True masks (few visible)
        masks_b_list = []  # select_visible=False masks (many visible)
        mask_components = None

        for i in range(num_pairs):
            invert = i >= (num_pairs + 1) // 2

            # Mask A: select_visible=True → attention selects VISIBLE patches
            # So it masks (P - num_select_a) patches = num_masks_a
            result = self._generate_fused_mask(
                masked_attn_weights,
                masked_centers,
                num_masks_a,
                return_components=(i == 0),
                invert_attn=invert,
                select_visible=True,
            )
            if i == 0:
                m_a, mask_components = result
            else:
                m_a = result

            # Mask B: select_visible=False → attention selects MASKED patches
            # Only num_masks_b patches are masked (block-coherent)
            m_b = self._generate_fused_mask(
                masked_attn_weights,
                masked_centers,
                num_masks_b,
                return_components=False,
                invert_attn=invert,
                select_visible=False,
            )

            masks_a_list.append(m_a)
            masks_b_list.append(m_b)

        # Stack: (B_masked * num_pairs, P)
        mask_a = torch.cat(masks_a_list, dim=0)
        mask_b = torch.cat(masks_b_list, dim=0)

        # ---- Step 3: Gather teacher targets ----
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

            # Patch targets for both mask types
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

            # Targets for mask_a (many masked patches)
            x_patch_t_a = x_patch_logits_t.repeat_interleave(num_pairs, dim=0)
            targets_a = gather_masked(x_patch_t_a, mask_a).flatten(0, 1)

            # Targets for mask_b (few masked patches)
            x_patch_t_b = x_patch_logits_t.repeat_interleave(num_pairs, dim=0)
            targets_b = gather_masked(x_patch_t_b, mask_b).flatten(0, 1)

            out_targets["x_patch_logits"] = torch.cat([targets_a, targets_b], dim=0)

            if self.do_regression:
                x_emb_crop = x_patch_teacher[indices_masked_crops]
                x_emb_a = x_emb_crop.repeat_interleave(num_pairs, dim=0)
                x_emb_b = x_emb_a.clone()
                emb_a = gather_masked(x_emb_a, mask_a).flatten(0, 1)
                emb_b = gather_masked(x_emb_b, mask_b).flatten(0, 1)
                out_targets["x_patch_embedding"] = torch.cat([emb_a, emb_b], dim=0)

        # ---- Step 4: Student forward (packed) ----
        preds = self._forward_student_packed(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            mask_a=mask_a,
            mask_b=mask_b,
            return_embeddings=True,
        )

        # ---- Step 5: Compute losses ----
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

            # Masked global CLS — 2 CLS per packed pair (A + B)
            if self.mask_probability == 1.0:
                # global_cls_preds_masked: (B_masked * num_pairs * 2, F)
                # Reshape to (B, num_pairs * 2 * C, F)
                cls_preds = global_cls_preds_masked.unflatten(0, dim_0_shape)
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

            if self.me_max_weight > 0.0 and global_cls_preds_masked is not None:
                cls_student_temp_val = self.scheduler.value["cls_student_temp"]
                cls_preds_me = global_cls_preds_masked.unflatten(0, dim_0_shape)
                me_max = self.me_max_loss(cls_preds_me / cls_student_temp_val)
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
            "mask_components": mask_components,
        }
