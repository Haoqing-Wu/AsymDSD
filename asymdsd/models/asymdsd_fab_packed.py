"""AsymDSD FAB with packed semantic/geometric mask stacks.

The default packed variant builds 4 or 6 mask paths per crop:

  - Semantic block+sparse (attention-guided with spatial blocks):
      A: high-attention visible (head_a)
      B: high-attention masked (head_b)
  - Semantic sparse-only (purely attention-guided, no blocks):
      C: high-attention visible (head_c)
      D: high-attention masked (head_d)
  - Geometric (no attention):
      E: random reverse-block
      F: random half-space

All paths are packed via varlen flash attention (zero wasted compute) or
run as separate encoder calls when varlen is unavailable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.flash_attention import (
    HAS_FLASH_ATTN,
    build_cu_seqlens_from_groups,
    has_relative_3d_bias,
    varlen_encoder_forward,
)
from ..layers.patchify import PatchPoints
from ..layers.tokenization import Tokens
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab import FusedAttnBlockAsymDSD
from .point_encoder import PointEncoder, PointEncoderOutput

logger = get_default_logger()


class PackedFusedAttnBlockAsymDSD(FusedAttnBlockAsymDSD):
    """FAB variant using sequence packing for multiple mask paths.

    ``multi_mask`` may be 4 or 6 and corresponds to the default paths listed
    in the module docstring. Patch loss is computed per path and averaged
    equally.
    """

    @init_lazy_defaults
    def __init__(
        self,
        # ``vis_mask_ratio`` is kept as a backward-compatible alias for the
        # semantic high-attention-visible path.
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

        self.semantic_visible_mask_ratio = (
            vis_mask_ratio
            if semantic_visible_mask_ratio is None
            else semantic_visible_mask_ratio
        )
        self.semantic_masked_mask_ratio = semantic_masked_mask_ratio
        self.sparse_visible_mask_ratio = sparse_visible_mask_ratio
        self.sparse_masked_mask_ratio = sparse_masked_mask_ratio
        self.geometric_reverse_mask_ratio = geometric_reverse_mask_ratio
        self.geometric_halfspace_mask_ratio = geometric_halfspace_mask_ratio
        self.patch_loss_a_only = patch_loss_a_only

        if self.multi_mask not in (4, 6):
            raise ValueError(
                "Packed FAB expects mask_generator.multi_mask=4 or 6. "
                "4 = sem_visible + sem_masked + geo_reverse + geo_halfspace. "
                "6 = adds sparse_visible + sparse_masked paths."
            )

        ratios_to_check = [
            ("semantic_visible_mask_ratio", self.semantic_visible_mask_ratio),
            ("semantic_masked_mask_ratio", self.semantic_masked_mask_ratio),
            ("geometric_reverse_mask_ratio", self.geometric_reverse_mask_ratio),
            ("geometric_halfspace_mask_ratio", self.geometric_halfspace_mask_ratio),
        ]
        if self.multi_mask == 6:
            ratios_to_check.extend(
                [
                    ("sparse_visible_mask_ratio", self.sparse_visible_mask_ratio),
                    ("sparse_masked_mask_ratio", self.sparse_masked_mask_ratio),
                ]
            )

        for name, ratio in ratios_to_check:
            if ratio <= 0.0 or ratio >= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        self.use_varlen_flash_attn = use_varlen_flash_attn
        if use_varlen_flash_attn and not HAS_FLASH_ATTN:
            logger.warning(
                "use_varlen_flash_attn=True but PyTorch nested_tensor_from_jagged "
                "not available (requires PyTorch >= 2.4). "
                "Falling back to block-diagonal packing."
            )
            self.use_varlen_flash_attn = False

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
    # Varlen encoder forward (all paths in one flash_attn kernel)
    # ------------------------------------------------------------------

    def _varlen_encoder_forward(
        self,
        x_list: list[torch.Tensor],
        pos_list: list[torch.Tensor],
        centers_list: list[torch.Tensor | None],
        point_encoder: PointEncoder,
    ) -> list[PointEncoderOutput]:
        """Run encoder on all paths via flash_attn_varlen_func.

        Each element of x_list is (B, S_i, D) with potentially different S_i.
        All paths are flattened into one (total_tokens, D) tensor and processed
        in a single kernel launch with zero wasted compute.

        Returns a list of PointEncoderOutput, one per path.
        """
        B = x_list[0].shape[0]
        D = x_list[0].shape[2]
        device = x_list[0].device
        num_paths = len(x_list)
        has_cls = point_encoder.cls_token is not None

        # Prepend CLS token and build flat tensors
        flat_parts_x = []
        flat_parts_pos = []
        group_lengths: list[tuple[int, int]] = []  # (count, seqlen)

        for i in range(num_paths):
            x_path = x_list[i]  # (B, S_i, D)
            pos_path = pos_list[i]  # (B, S_i, D)
            S_i = x_path.shape[1]

            if has_cls:
                cls_token = point_encoder.cls_token.expand(B, 1, -1)
                cls_pos = torch.zeros(B, 1, D, device=device, dtype=x_path.dtype)
                x_with_cls = torch.cat([cls_token, x_path], dim=1)  # (B, 1+S_i, D)
                pos_with_cls = torch.cat([cls_pos, pos_path], dim=1)
                seqlen = 1 + S_i
            else:
                x_with_cls = x_path
                pos_with_cls = pos_path
                seqlen = S_i

            # Flatten batch: (B, seqlen, D) → (B*seqlen, D)
            flat_parts_x.append(x_with_cls.reshape(-1, D))
            flat_parts_pos.append(pos_with_cls.reshape(-1, D))
            group_lengths.append((B, seqlen))

        flat_x = torch.cat(flat_parts_x, dim=0)  # (total_tokens, D)
        flat_pos = torch.cat(flat_parts_pos, dim=0)

        cu_seqlens, max_seqlen = build_cu_seqlens_from_groups(
            group_lengths, device=device
        )

        # Run through encoder
        flat_out = varlen_encoder_forward(
            point_encoder.encoder, flat_x, flat_pos, cu_seqlens, max_seqlen
        )

        # Split output back into per-path results
        outputs: list[PointEncoderOutput] = []
        offset = 0
        for i, (count, seqlen) in enumerate(group_lengths):
            total_path_tokens = count * seqlen
            path_out = flat_out[offset : offset + total_path_tokens]
            path_out = path_out.view(B, seqlen, D)
            offset += total_path_tokens

            if has_cls:
                outputs.append(
                    PointEncoderOutput(
                        patch_features=path_out[:, 1:],
                        cls_features=path_out[:, 0],
                        attn_weights=None,
                        hidden_states=None,
                    )
                )
            else:
                outputs.append(
                    PointEncoderOutput(
                        patch_features=path_out,
                        cls_features=None,
                        attn_weights=None,
                        hidden_states=None,
                    )
                )

        return outputs

    def _can_stack_encoder_paths(
        self,
        x_list: list[torch.Tensor],
        pos_list: list[torch.Tensor],
        centers_list: list[torch.Tensor | None],
    ) -> bool:
        seq_len = x_list[0].shape[1]
        has_centers = centers_list[0] is not None
        for x_path, pos_path, centers_path in zip(
            x_list, pos_list, centers_list, strict=True
        ):
            if x_path.shape[1] != seq_len or pos_path.shape[1] != seq_len:
                return False
            if (centers_path is not None) != has_centers:
                return False
            if centers_path is not None and centers_path.shape[1] != seq_len:
                return False
        return True

    def _stacked_encoder_forward(
        self,
        x_list: list[torch.Tensor],
        pos_list: list[torch.Tensor],
        centers_list: list[torch.Tensor | None],
        point_encoder: PointEncoder,
        attn_bias_scale: float,
    ) -> list[PointEncoderOutput]:
        """Run equal-length mask paths as one enlarged batch."""
        B = x_list[0].shape[0]
        num_paths = len(x_list)
        x_stacked = torch.cat(x_list, dim=0)
        pos_stacked = torch.cat(pos_list, dim=0)
        centers_stacked = (
            torch.cat(centers_list, dim=0) if centers_list[0] is not None else None
        )

        pe_out = point_encoder.transformer_encoder_forward(
            x_stacked,
            pos_stacked,
            token_centers=centers_stacked,
            attn_bias_scale=attn_bias_scale,
        )

        patch_chunks = pe_out.patch_features.split(B, dim=0)
        cls_chunks = (
            pe_out.cls_features.split(B, dim=0)
            if pe_out.cls_features is not None
            else [None] * num_paths
        )

        if pe_out.attn_weights is None:
            attn_chunks: list[list[torch.Tensor] | None] = [None] * num_paths
        else:
            attn_chunks = [[] for _ in range(num_paths)]
            for attn in pe_out.attn_weights:
                for path_idx, attn_chunk in enumerate(attn.split(B, dim=0)):
                    attn_chunks[path_idx].append(attn_chunk)

        if pe_out.hidden_states is None:
            hidden_chunks: list[list[torch.Tensor] | None] = [None] * num_paths
        else:
            hidden_chunks = [[] for _ in range(num_paths)]
            for hidden in pe_out.hidden_states:
                for path_idx, hidden_chunk in enumerate(hidden.split(B, dim=0)):
                    hidden_chunks[path_idx].append(hidden_chunk)

        return [
            PointEncoderOutput(
                patch_features=patch_chunks[path_idx],
                cls_features=cls_chunks[path_idx],
                attn_weights=attn_chunks[path_idx],
                hidden_states=hidden_chunks[path_idx],
            )
            for path_idx in range(num_paths)
        ]

    @torch.no_grad()
    def _generate_halfspace_mask(
        self,
        centers: torch.Tensor,
        num_masks: int,
    ) -> torch.Tensor:
        """Mask one random side of each object by a random 3D half-space."""
        B, P, _ = centers.shape
        direction = torch.randn(B, 3, device=centers.device, dtype=centers.dtype)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        projections = (centers * direction.unsqueeze(1)).sum(dim=-1)
        side = torch.empty(
            B, 1, device=centers.device, dtype=centers.dtype
        ).bernoulli_()
        side = side.mul_(2.0).sub_(1.0)
        scores = projections * side
        scores = scores + 1e-6 * torch.randn_like(scores)

        _, mask_indices = scores.topk(num_masks, dim=-1)
        mask = torch.zeros(B, P, dtype=torch.bool, device=centers.device)
        mask.scatter_(-1, mask_indices, True)
        return mask

    @torch.no_grad()
    def _generate_sparse_only_mask(
        self,
        attn_weights: list[torch.Tensor],
        num_masks: int,
        select_visible: bool = False,
        head_index: int | None = None,
    ) -> torch.Tensor:
        """Generate a purely attention-guided sparse mask (no block component).

        Selects patches using Gumbel-softmax sampling weighted by per-head
        CLS→patch attention scores.

        Args:
            attn_weights: per-layer attention, each (B, H, S, S).
            num_masks: total number of patches to mask per sample.
            select_visible: if True, top-attention patches define visible set.
            head_index: which head to use for attention scores.
        """
        cls_to_patch = self._compute_cls_to_patch_attention(
            attn_weights, head_index=head_index
        )
        B, P = cls_to_patch.shape
        device = cls_to_patch.device

        temperature: float = self.scheduler.value["attn_mask_temperature"]

        num_select = (P - num_masks) if select_visible else num_masks

        log_probs = (cls_to_patch + 1e-8).log() / temperature
        gumbel_noise = -(-torch.rand_like(log_probs).clamp(1e-8).log()).log()
        scores = log_probs + gumbel_noise

        _, selected_indices = scores.topk(num_select, dim=-1)

        selected = torch.zeros(B, P, dtype=torch.bool, device=device)
        selected.scatter_(-1, selected_indices, True)

        mask = ~selected if select_visible else selected
        return mask

    def _packed_mask_counts(self, num_patches: int) -> list[int]:
        counts = [
            round(self.semantic_visible_mask_ratio * num_patches),
            round(self.semantic_masked_mask_ratio * num_patches),
        ]
        if self.multi_mask == 6:
            counts.extend(
                [
                    round(self.sparse_visible_mask_ratio * num_patches),
                    round(self.sparse_masked_mask_ratio * num_patches),
                ]
            )
        counts.extend(
            [
                round(self.geometric_reverse_mask_ratio * num_patches),
                round(self.geometric_halfspace_mask_ratio * num_patches),
            ]
        )
        return counts

    def _generate_packed_masks(
        self,
        masked_attn_weights: list[torch.Tensor],
        masked_centers: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        P = masked_centers.shape[1]
        use_sparse_paths = self.multi_mask == 6

        num_masks_sem_visible = round(self.semantic_visible_mask_ratio * P)
        num_masks_sem_masked = round(self.semantic_masked_mask_ratio * P)
        num_masks_geo_reverse = round(self.geometric_reverse_mask_ratio * P)
        num_masks_geo_halfspace = round(self.geometric_halfspace_mask_ratio * P)

        # Rotate heads for mask diversity: semantic paths use different
        # heads, cycling evenly through all heads over training steps.
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

        # Semantic block+sparse paths (head_a, head_b)
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

        # Geometric paths (no attention)
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
            # Sparse-only semantic paths (head_c, head_d)
            num_masks_sparse_visible = round(self.sparse_visible_mask_ratio * P)
            num_masks_sparse_masked = round(self.sparse_masked_mask_ratio * P)
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
            path_names = [
                "semantic_visible",
                "semantic_masked",
                "sparse_visible",
                "sparse_masked",
                "geometric_reverse",
                "geometric_halfspace",
            ]
        else:
            all_masks = [
                sem_high_visible,
                sem_high_masked,
                geo_reverse_block,
                geo_halfspace,
            ]
            path_names = [
                "semantic_visible",
                "semantic_masked",
                "geometric_reverse",
                "geometric_halfspace",
            ]

        mask_components["path_masks"] = all_masks
        mask_components["path_names"] = path_names

        return all_masks, mask_components

    # ------------------------------------------------------------------
    # Packed student forward
    # ------------------------------------------------------------------

    def _after_student_patch_embedding(
        self,
        tokens: Tokens,
        point_encoder: PointEncoder,
    ) -> None:
        """Hook for subclasses that need dense student tokens before masking."""
        del tokens, point_encoder

    def _forward_student_packed(
        self,
        multi_patches,
        indices_masked_crops: torch.Tensor,
        indices_unmasked_crops: torch.Tensor | None,
        masks: list[torch.Tensor],
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Student forward with packed mask paths.

        Args:
            masks: list of (B_masked, P) bool masks, one per path.
                Each mask can have a different number of masked patches.
        """
        point_encoder: PointEncoder = self.student.point_encoder

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        self._after_student_patch_embedding(tokens, point_encoder)
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

        # ------- Masked Point Modeling with per-path sequences -------
        x_mpm = x[indices_masked_crops]  # (B_masked, P, F)
        pos_enc_mpm = pos_enc[indices_masked_crops]
        centers_mpm = (
            token_centers[indices_masked_crops] if token_centers is not None else None
        )

        B_masked = x_mpm.shape[0]
        num_paths = len(masks)

        # Gather visible patches per path (each path may have different lengths)
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

        # --- Encoder forward: varlen (zero-waste) or block-diagonal fallback ---
        use_varlen = self.use_varlen_flash_attn and not has_relative_3d_bias(
            point_encoder.encoder
        )

        if use_varlen:
            pe_outputs = self._varlen_encoder_forward(
                x_vis_list, pos_vis_list, centers_vis_list, point_encoder
            )
        elif self._can_stack_encoder_paths(x_vis_list, pos_vis_list, centers_vis_list):
            pe_outputs = self._stacked_encoder_forward(
                x_vis_list,
                pos_vis_list,
                centers_vis_list,
                point_encoder,
                attn_bias_scale,
            )
        else:
            # Fallback: run each path through the encoder separately
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

        # --- CLS logits from all paths ---
        cls_logits_list = []
        cls_emb_list = []
        if self.mode.do_cls:
            for pe_out in pe_outputs:
                x_cls = pe_out.cls_features
                cls_logits_list.append(self.student.cls_projection_head(x_cls)[0])
                if return_embeddings:
                    cls_emb_list.append(x_cls)

            # Stack and interleave: (B_masked, num_paths, D) → (B_masked * num_paths, D)
            cls_interleaved = torch.stack(cls_logits_list, dim=1).reshape(
                B_masked * num_paths, -1
            )
            out_dict["x_cls_logits_masked"] = cls_interleaved

            if return_embeddings:
                emb_interleaved = torch.stack(cls_emb_list, dim=1).reshape(
                    B_masked * num_paths, -1
                )
                out_dict["x_cls_embedding_masked"] = emb_interleaved

        # --- Predictor for masked patch reconstruction ---
        patch_logits_by_path: list[torch.Tensor] = []
        patch_emb_by_path: list[torch.Tensor] = []

        predictor_paths = list(range(num_paths))
        if self.patch_loss_a_only:
            # Legacy option: only predict the even-indexed packed paths.
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

        # ---- Step 2: Generate packed mask stacks ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        P = global_centers.shape[1]
        all_counts = self._packed_mask_counts(P)

        if min(all_counts) == 0:
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

        all_masks, mask_components = self._generate_packed_masks(
            masked_attn_weights,
            masked_centers,
        )

        # ---- Step 3: Gather teacher targets ----
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
            }

            if self.mode.do_cls:
                out_targets["x_cls_logits"] = teacher_out_step1["x_cls_logits"]
                if self.do_regression:
                    out_targets["x_cls_embedding"] = teacher_out_step1[
                        "x_cls_embedding"
                    ]

            # Patch targets: gather per-path individually (masks may differ in count)
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

            # Determine which paths to gather targets for (matches predictor_paths)
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

        # ---- Step 4: Student forward (packed) ----
        preds = self._forward_student_packed(
            multi_patches,
            indices_masked_crops=indices_masked_crops,
            indices_unmasked_crops=indices_unmasked_crops,
            masks=all_masks,
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
            # x_cls_logits_masked is (B_masked * multi_mask, D) with each
            # crop's multi_mask sub-seqs contiguous (same as parent FAB).
            if self.mask_probability == 1.0:
                dim_0_mask = (-1, self.multi_mask)
                cls_preds = global_cls_preds_masked.unflatten(0, dim_0_mask)
                # (B_masked, multi_mask, D) → (B, C*multi_mask, D)
                cls_preds = cls_preds.reshape(*dim_0_shape, cls_preds.shape[-1])
                cls_loss += self.cls_loss(
                    cls_preds,
                    cls_target_probs,
                    student_temp=student_temp,
                )
                cls_terms += 1

                if self.add_unmasked_global_cls and cls_target_probs.shape[1] > 1:
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
