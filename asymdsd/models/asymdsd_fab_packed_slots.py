"""Packed FAB with auxiliary self-supervised semantic slots.

This variant keeps the packed FAB semantic/geometric mask stacks unchanged and
adds a part-aware teacher/student objective:

  - the EMA teacher encodes the full unmasked crop and produces K slot tokens;
  - the student encodes each packed masked path and produces K slot tokens;
  - slots are matched with a balanced Sinkhorn assignment, so slot identity is
    learned without labels;
  - a small diversity term discourages all slots from becoming one global CLS.

The slot objective is auxiliary.  CLS and patch losses remain active exactly as
in ``PackedFusedAttnBlockAsymDSD``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall
from ..components.exponential_moving_average import EMA
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.flash_attention import has_relative_3d_bias
from ..layers.patchify import PatchPoints
from ..layers.tokenization import Tokens
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab_packed import PackedFusedAttnBlockAsymDSD
from .point_encoder import PointEncoder


class SemanticSlotHead(nn.Module):
    """Learned slot queries that attend over encoded patch features."""

    def __init__(
        self,
        embed_dim: int,
        num_slots: int,
        num_heads: int,
        hidden_ratio: float = 2.0,
        dropout_p: float = 0.0,
        include_cls: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = round(hidden_ratio * embed_dim)

        self.include_cls = include_cls
        self.slot_tokens = nn.Parameter(torch.empty(num_slots, embed_dim))
        self.slot_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_p,
            batch_first=True,
            bias=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.slot_tokens, std=0.02)

    def forward(
        self,
        patch_features: torch.Tensor,
        cls_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = patch_features.shape[0]
        slots = self.slot_tokens.unsqueeze(0).expand(B, -1, -1)

        if self.include_cls and cls_features is not None:
            memory = torch.cat([cls_features.unsqueeze(1), patch_features], dim=1)
        else:
            memory = patch_features

        norm_memory = self.memory_norm(memory)
        attn_out = self.attn(
            self.slot_norm(slots),
            norm_memory,
            norm_memory,
            need_weights=False,
        )[0]
        slots = slots + attn_out
        slots = slots + self.ffn(slots)
        return self.out_norm(slots)


class SlotPackedFusedAttnBlockAsymDSD(PackedFusedAttnBlockAsymDSD):
    """Packed FAB with self-supervised part/slot distillation."""

    @init_lazy_defaults
    def __init__(
        self,
        # --- semantic slot objective ---
        num_semantic_slots: int = 8,
        semantic_slot_loss_weight: float = 0.1,
        semantic_slot_diversity_weight: float = 0.02,
        semantic_slot_temperature: float = 0.1,
        semantic_slot_matching_temperature: float = 0.05,
        semantic_slot_sinkhorn_iters: int = 3,
        semantic_slot_num_heads: int | None = None,
        semantic_slot_hidden_ratio: float = 2.0,
        semantic_slot_dropout_p: float = 0.0,
        semantic_slot_include_cls: bool = False,
        # --- inherited from PackedFusedAttnBlockAsymDSD ---
        vis_mask_ratio: float = 0.7,
        semantic_visible_mask_ratio: float | None = None,
        semantic_masked_mask_ratio: float = 0.5,
        sparse_visible_mask_ratio: float = 0.7,
        sparse_masked_mask_ratio: float = 0.5,
        geometric_reverse_mask_ratio: float = 0.7,
        geometric_halfspace_mask_ratio: float = 0.5,
        patch_loss_a_only: bool = False,
        use_varlen_flash_attn: bool = False,
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
        attn_num_layers: int = 1,
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
            vis_mask_ratio=vis_mask_ratio,
            semantic_visible_mask_ratio=semantic_visible_mask_ratio,
            semantic_masked_mask_ratio=semantic_masked_mask_ratio,
            sparse_visible_mask_ratio=sparse_visible_mask_ratio,
            sparse_masked_mask_ratio=sparse_masked_mask_ratio,
            geometric_reverse_mask_ratio=geometric_reverse_mask_ratio,
            geometric_halfspace_mask_ratio=geometric_halfspace_mask_ratio,
            patch_loss_a_only=patch_loss_a_only,
            use_varlen_flash_attn=use_varlen_flash_attn,
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
            attn_num_layers=attn_num_layers,
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

        if num_semantic_slots < 1:
            raise ValueError("num_semantic_slots must be >= 1.")
        if semantic_slot_loss_weight < 0.0:
            raise ValueError("semantic_slot_loss_weight must be >= 0.")
        if semantic_slot_diversity_weight < 0.0:
            raise ValueError("semantic_slot_diversity_weight must be >= 0.")
        if semantic_slot_temperature <= 0.0:
            raise ValueError("semantic_slot_temperature must be > 0.")
        if semantic_slot_matching_temperature <= 0.0:
            raise ValueError("semantic_slot_matching_temperature must be > 0.")
        if semantic_slot_sinkhorn_iters < 1:
            raise ValueError("semantic_slot_sinkhorn_iters must be >= 1.")
        if semantic_slot_num_heads is not None and semantic_slot_num_heads < 1:
            raise ValueError("semantic_slot_num_heads must be >= 1 when set.")
        if semantic_slot_hidden_ratio <= 0.0:
            raise ValueError("semantic_slot_hidden_ratio must be > 0.")

        self.num_semantic_slots = num_semantic_slots
        self.semantic_slot_loss_weight = semantic_slot_loss_weight
        self.semantic_slot_diversity_weight = semantic_slot_diversity_weight
        self.semantic_slot_temperature = semantic_slot_temperature
        self.semantic_slot_matching_temperature = semantic_slot_matching_temperature
        self.semantic_slot_sinkhorn_iters = semantic_slot_sinkhorn_iters
        self.semantic_slot_include_cls = semantic_slot_include_cls
        self.do_semantic_slot_loss = semantic_slot_loss_weight > 0.0

        if self.do_semantic_slot_loss:
            num_heads = (
                semantic_slot_num_heads
                if semantic_slot_num_heads is not None
                else self.student.point_encoder.encoder.config.num_heads
            )
            if self.embed_dim % num_heads != 0:
                raise ValueError(
                    "semantic_slot_num_heads must divide encoder_config.embed_dim."
                )

            self.student["semantic_slot_head"] = SemanticSlotHead(
                embed_dim=self.embed_dim,
                num_slots=num_semantic_slots,
                num_heads=num_heads,
                hidden_ratio=semantic_slot_hidden_ratio,
                dropout_p=semantic_slot_dropout_p,
                include_cls=semantic_slot_include_cls,
            )
            self.teacher["semantic_slot_head"] = SemanticSlotHead(
                embed_dim=self.embed_dim,
                num_slots=num_semantic_slots,
                num_heads=num_heads,
                hidden_ratio=semantic_slot_hidden_ratio,
                dropout_p=semantic_slot_dropout_p,
                include_cls=semantic_slot_include_cls,
            )

            # Parent builds EMA before subclass modules exist.  Rebuild it so
            # slot heads are copied and updated with the rest of the teacher.
            self.ema = EMA(self.student, self.teacher)

    def _semantic_slot_sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        log_assignment = logits
        for _ in range(self.semantic_slot_sinkhorn_iters):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-1, keepdim=True
            )
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-2, keepdim=True
            )
        return log_assignment.exp()

    def _semantic_slot_distill_loss(
        self,
        student_slots: torch.Tensor,
        teacher_slots: torch.Tensor,
    ) -> torch.Tensor:
        student_slots = F.normalize(student_slots, dim=-1)
        teacher_slots = F.normalize(teacher_slots.detach(), dim=-1)

        sim = student_slots @ teacher_slots.transpose(-1, -2)
        with torch.no_grad():
            assignment = self._semantic_slot_sinkhorn(
                sim / self.semantic_slot_matching_temperature
            )

        log_prob_s_to_t = F.log_softmax(
            sim / self.semantic_slot_temperature, dim=-1
        )
        log_prob_t_to_s = F.log_softmax(
            sim.transpose(-1, -2) / self.semantic_slot_temperature, dim=-1
        )

        num_slots = sim.shape[-1]
        loss_s_to_t = -(assignment * log_prob_s_to_t).sum(dim=(-2, -1))
        loss_t_to_s = -(
            assignment.transpose(-1, -2) * log_prob_t_to_s
        ).sum(dim=(-2, -1))
        return (0.5 * (loss_s_to_t + loss_t_to_s) / num_slots).mean()

    def _semantic_slot_diversity_loss(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.shape[1] < 2:
            return slots.new_zeros(())

        slots = F.normalize(slots, dim=-1)
        sim = slots @ slots.transpose(-1, -2)
        off_diag = ~torch.eye(
            slots.shape[1],
            dtype=torch.bool,
            device=slots.device,
        )
        return sim[:, off_diag].pow(2).mean()

    def _forward_student_packed(
        self,
        multi_patches,
        indices_masked_crops: torch.Tensor,
        indices_unmasked_crops: torch.Tensor | None,
        masks: list[torch.Tensor],
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        point_encoder: PointEncoder = self.student.point_encoder

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers

        attn_bias_scale = self.scheduler.value["attn_bias_scale"]

        out_dict: dict[str, Any] = {
            "x_cls_logits": None,
            "x_cls_logits_masked": None,
            "x_cls_embedding": None,
            "x_cls_embedding_masked": None,
            "x_patch_logits": None,
            "x_patch_logits_by_path": None,
            "x_patch_embedding": None,
            "x_patch_embedding_by_path": None,
            "x_patch_proj_norm": None,
            "classification_logits": None,
            "classification_logits_masked": None,
            "semantic_slots": None,
            "semantic_slots_by_path": None,
        }

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
            out_dict["x_cls_logits"] = self.student.cls_projection_head(
                x_cls_unmask
            )[0]
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls_unmask

        x_mpm = x[indices_masked_crops]
        pos_enc_mpm = pos_enc[indices_masked_crops]
        centers_mpm = (
            token_centers[indices_masked_crops] if token_centers is not None else None
        )

        B_masked = x_mpm.shape[0]
        num_paths = len(masks)

        x_vis_list: list[torch.Tensor] = []
        pos_vis_list: list[torch.Tensor] = []
        centers_vis_list: list[torch.Tensor | None] = []
        for mask in masks:
            inv_mask = ~mask
            x_vis_list.append(gather_masked(x_mpm, inv_mask))
            pos_vis_list.append(gather_masked(pos_enc_mpm, inv_mask))
            centers_vis_list.append(
                gather_masked(centers_mpm, inv_mask)
                if centers_mpm is not None
                else None
            )

        use_varlen = self.use_varlen_flash_attn and not has_relative_3d_bias(
            point_encoder.encoder
        )

        if use_varlen:
            pe_outputs = self._varlen_encoder_forward(
                x_vis_list, pos_vis_list, centers_vis_list, point_encoder
            )
        else:
            pe_outputs = []
            for x_vis, pos_vis, centers_vis in zip(
                x_vis_list, pos_vis_list, centers_vis_list, strict=True
            ):
                pe_out = point_encoder.transformer_encoder_forward(
                    x_vis,
                    pos_vis,
                    token_centers=centers_vis,
                    attn_bias_scale=attn_bias_scale,
                )
                pe_outputs.append(pe_out)

        if self.do_semantic_slot_loss:
            semantic_slots_by_path = [
                self.student.semantic_slot_head(
                    pe_out.patch_features,
                    pe_out.cls_features,
                )
                for pe_out in pe_outputs
            ]
            out_dict["semantic_slots_by_path"] = semantic_slots_by_path
            out_dict["semantic_slots"] = torch.stack(
                semantic_slots_by_path, dim=1
            ).reshape(B_masked * num_paths, self.num_semantic_slots, -1)

        cls_logits_list = []
        cls_emb_list = []
        if self.mode.do_cls:
            for pe_out in pe_outputs:
                x_cls = pe_out.cls_features
                cls_logits_list.append(self.student.cls_projection_head(x_cls)[0])
                if return_embeddings:
                    cls_emb_list.append(x_cls)

            cls_interleaved = torch.stack(cls_logits_list, dim=1).reshape(
                B_masked * num_paths, -1
            )
            out_dict["x_cls_logits_masked"] = cls_interleaved

            if return_embeddings:
                emb_interleaved = torch.stack(cls_emb_list, dim=1).reshape(
                    B_masked * num_paths, -1
                )
                out_dict["x_cls_embedding_masked"] = emb_interleaved

        patch_logits_by_path: list[torch.Tensor] = []
        patch_emb_by_path: list[torch.Tensor] = []

        predictor_paths = list(range(num_paths))
        if self.patch_loss_a_only:
            predictor_paths = list(range(0, num_paths, 2))

        for path_idx in predictor_paths:
            pe_out = pe_outputs[path_idx]
            mask = masks[path_idx]
            x_patch = pe_out.patch_features
            x_cls = pe_out.cls_features

            pos_enc_masked = gather_masked(pos_enc_mpm, mask)
            num_masks = pos_enc_masked.shape[1]

            if self.masked_pos_noise is not None:
                pos_enc_masked = (
                    pos_enc_masked
                    + self.masked_pos_noise * torch.randn_like(pos_enc_masked)
                )

            mask_tokens = self.mask_token.expand(B_masked, num_masks, -1)

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

                pos_vis = gather_masked(pos_enc_mpm, ~mask)
                pos_parts = (pos_vis, pos_enc_masked)
                if self.mode.do_cls:
                    cls_pos = torch.zeros(
                        B_masked,
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
                patch_emb_by_path.append(x_pred_patch)

            x_patch_proj = self.student.patch_projection_head(
                x_pred_patch, return_x_norm=True
            )
            patch_logits_by_path.append(x_patch_proj.x)

        out_dict["x_patch_logits_by_path"] = patch_logits_by_path
        out_dict["x_patch_logits"] = torch.cat(
            [t.flatten(0, 1) for t in patch_logits_by_path], dim=0
        )

        if return_embeddings:
            out_dict["x_patch_embedding_by_path"] = patch_emb_by_path
            out_dict["x_patch_embedding"] = torch.cat(
                [t.flatten(0, 1) for t in patch_emb_by_path], dim=0
            )

        return out_dict

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
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

        indices_all_crops = torch.arange(0, B * C, device=global_centers.device)

        with torch.no_grad():
            teacher_out_step1 = self.forward_teacher(
                multi_patches,
                indices_masked_crops=indices_all_crops,
                mask=None,
                return_embeddings=self.do_regression
                or (self.do_semantic_slot_loss and self.semantic_slot_include_cls),
            )

        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        P = global_centers.shape[1]
        use_sparse_paths = self.multi_mask == 6

        num_masks_sem_visible = round(self.semantic_visible_mask_ratio * P)
        num_masks_sem_masked = round(self.semantic_masked_mask_ratio * P)
        num_masks_geo_reverse = round(self.geometric_reverse_mask_ratio * P)
        num_masks_geo_halfspace = round(self.geometric_halfspace_mask_ratio * P)

        all_counts = [
            num_masks_sem_visible,
            num_masks_sem_masked,
            num_masks_geo_reverse,
            num_masks_geo_halfspace,
        ]

        if use_sparse_paths:
            num_masks_sparse_visible = round(self.sparse_visible_mask_ratio * P)
            num_masks_sparse_masked = round(self.sparse_masked_mask_ratio * P)
            all_counts.extend([num_masks_sparse_visible, num_masks_sparse_masked])

        if min(all_counts) == 0:
            return AsymDSD.training_step(self, batch, batch_idx)

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

        masked_attn_weights = [aw[indices_masked_crops] for aw in attn_weights]
        masked_centers = global_centers[indices_masked_crops]

        num_heads = self.student.point_encoder.encoder.config.num_heads
        step = self.global_step
        if use_sparse_paths:
            head_a = step % num_heads
            head_b = (step + num_heads // 4) % num_heads
            head_c = (step + 2 * num_heads // 4) % num_heads
            head_d = (step + 3 * num_heads // 4) % num_heads
        else:
            head_a = step % num_heads
            head_b = (step + num_heads // 2) % num_heads

        result = self._generate_fused_mask(
            masked_attn_weights,
            masked_centers,
            num_masks_sem_visible,
            return_components=True,
            invert_attn=False,
            select_visible=True,
            head_index=head_a,
        )
        sem_high_visible, mask_components = result
        mask_components["head_a"] = head_a
        mask_components["head_b"] = head_b
        mask_components["cls_to_patch_b"] = self._compute_cls_to_patch_attention(
            masked_attn_weights, head_index=head_b
        )
        sem_high_masked = self._generate_fused_mask(
            masked_attn_weights,
            masked_centers,
            num_masks_sem_masked,
            return_components=False,
            invert_attn=False,
            select_visible=False,
            head_index=head_b,
        )

        geo_reverse_block = self._generate_inverse_block_mask(
            masked_centers,
            num_masks_geo_reverse,
            self.geometric_reverse_mask_ratio,
        )
        geo_halfspace = self._generate_halfspace_mask(
            masked_centers,
            num_masks_geo_halfspace,
        )

        if use_sparse_paths:
            sparse_visible = self._generate_sparse_only_mask(
                masked_attn_weights,
                num_masks_sparse_visible,
                select_visible=True,
                head_index=head_c,
            )
            sparse_masked = self._generate_sparse_only_mask(
                masked_attn_weights,
                num_masks_sparse_masked,
                select_visible=False,
                head_index=head_d,
            )
            all_masks = [
                sem_high_visible,
                sem_high_masked,
                sparse_visible,
                sparse_masked,
                geo_reverse_block,
                geo_halfspace,
            ]
        else:
            all_masks = [
                sem_high_visible,
                sem_high_masked,
                geo_reverse_block,
                geo_halfspace,
            ]

        with torch.no_grad():
            x_patch_teacher = self._teacher_x_patch
            del self._teacher_attn_weights, self._teacher_x_patch

            out_targets: dict[str, Any] = {
                "x_cls_logits": None,
                "x_cls_embedding": None,
                "x_patch_logits": None,
                "x_patch_logits_by_path": None,
                "x_patch_embedding": None,
                "x_patch_embedding_by_path": None,
                "semantic_slots": None,
            }

            if self.mode.do_cls:
                out_targets["x_cls_logits"] = teacher_out_step1["x_cls_logits"]
                if self.do_regression:
                    out_targets["x_cls_embedding"] = teacher_out_step1[
                        "x_cls_embedding"
                    ]

            if self.do_semantic_slot_loss:
                teacher_cls = (
                    teacher_out_step1["x_cls_embedding"]
                    if self.semantic_slot_include_cls
                    else None
                )
                teacher_slots = self.teacher.semantic_slot_head(
                    x_patch_teacher,
                    teacher_cls,
                )
                out_targets["semantic_slots"] = teacher_slots[indices_masked_crops]

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

            if self.patch_loss_a_only:
                target_path_indices = list(range(0, len(all_masks), 2))
            else:
                target_path_indices = list(range(len(all_masks)))

            targets_by_path = [
                gather_masked(x_patch_logits_t, all_masks[i])
                for i in target_path_indices
            ]
            out_targets["x_patch_logits_by_path"] = targets_by_path
            out_targets["x_patch_logits"] = torch.cat(
                [t.flatten(0, 1) for t in targets_by_path], dim=0
            )

            if self.do_regression:
                x_emb_crop = x_patch_teacher[indices_masked_crops]
                emb_by_path = [
                    gather_masked(x_emb_crop, all_masks[i]) for i in target_path_indices
                ]
                out_targets["x_patch_embedding_by_path"] = emb_by_path
                out_targets["x_patch_embedding"] = torch.cat(
                    [t.flatten(0, 1) for t in emb_by_path], dim=0
                )

        preds = self._forward_student_packed(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            masks=all_masks,
            return_embeddings=True,
        )

        loss = 0.0
        cls_loss = patch_loss = koleo_loss = me_max = None
        classification_loss = None
        semantic_slot_loss = None
        semantic_slot_distill_loss = None
        semantic_slot_diversity_loss = None
        total_terms = 0
        cls_terms = 0
        regression_loss = 0.0 if self.do_regression else None

        targets = out_targets

        if self.mode.do_mask:
            if not self.disable_projection:
                patch_losses = [
                    checkpoint(
                        self.patch_loss,
                        pred_path,
                        target_path,
                        self.scheduler.value["patch_teacher_temp"],
                        self.scheduler.value["patch_student_temp"],
                    )
                    for pred_path, target_path in zip(
                        preds["x_patch_logits_by_path"],
                        targets["x_patch_logits_by_path"],
                        strict=True,
                    )
                ]
                patch_loss = torch.stack(patch_losses).mean()
                loss = loss + patch_loss  # type: ignore
                total_terms += 1

            if self.do_regression:
                regression_losses = [
                    self.patch_regression_loss(pred_path, target_path)
                    for pred_path, target_path in zip(
                        preds["x_patch_embedding_by_path"],
                        targets["x_patch_embedding_by_path"],
                        strict=True,
                    )
                ]
                regression_loss = torch.stack(regression_losses).mean()
                loss = loss + self.regression_loss_weight * regression_loss
                total_terms += self.regression_loss_weight

        if self.do_semantic_slot_loss:
            teacher_slots = targets["semantic_slots"]
            slot_losses = [
                self._semantic_slot_distill_loss(student_slots, teacher_slots)
                for student_slots in preds["semantic_slots_by_path"]
            ]
            semantic_slot_distill_loss = torch.stack(slot_losses).mean()
            semantic_slot_loss = semantic_slot_distill_loss

            if self.semantic_slot_diversity_weight > 0.0:
                diversity_losses = [
                    self._semantic_slot_diversity_loss(student_slots)
                    for student_slots in preds["semantic_slots_by_path"]
                ]
                semantic_slot_diversity_loss = torch.stack(diversity_losses).mean()
                semantic_slot_loss = (
                    semantic_slot_loss
                    + self.semantic_slot_diversity_weight
                    * semantic_slot_diversity_loss
                )

            loss = loss + self.semantic_slot_loss_weight * semantic_slot_loss
            total_terms += self.semantic_slot_loss_weight

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
            "semantic_slot_loss": semantic_slot_loss,
            "semantic_slot_distill_loss": semantic_slot_distill_loss,
            "semantic_slot_diversity_loss": semantic_slot_diversity_loss,
            "centers": global_centers,
            "mask_components": mask_components,
        }

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if not self.do_semantic_slot_loss:
            return

        for key in [
            "semantic_slot_loss",
            "semantic_slot_distill_loss",
            "semantic_slot_diversity_loss",
        ]:
            value = outputs.get(key) if isinstance(outputs, dict) else None
            if value is not None:
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=key == "semantic_slot_loss",
                    prog_bar=key == "semantic_slot_loss",
                )
