"""AsymDSD FAB with packed semantic/geometric mask stacks.

The packed variant builds two block-diagonal packed stacks per crop:

  - Semantic stack:
      A: high-attention visible, 70% masked
      B: high-attention masked, 50% masked
  - Geometric stack:
      A: random reverse-block, 70% masked
      B: random half-space, 50% masked

Each packed call keeps its two sub-sequences isolated with a block-diagonal
attention mask, then splits the outputs back for CLS and patch losses.  This
keeps the packed style while removing the previous low-attention paths.
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
    """FAB variant using sequence packing for two named mask stacks.

    ``multi_mask`` must be 4 and corresponds to the four paths listed in the
    module docstring.  Patch loss is computed per path and averaged equally.
    """

    @init_lazy_defaults
    def __init__(
        self,
        # ``vis_mask_ratio`` is kept as a backward-compatible alias for the
        # semantic high-attention-visible path.
        vis_mask_ratio: float = 0.7,
        semantic_visible_mask_ratio: float | None = None,
        semantic_masked_mask_ratio: float = 0.5,
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
        self.geometric_reverse_mask_ratio = geometric_reverse_mask_ratio
        self.geometric_halfspace_mask_ratio = geometric_halfspace_mask_ratio
        self.patch_loss_a_only = patch_loss_a_only

        if self.multi_mask != 4:
            raise ValueError(
                "Packed semantic/geometric FAB expects mask_generator.multi_mask=4 "
                "for: semantic high-visible, semantic high-masked, geometric "
                "reverse-block, geometric half-space."
            )

        for name, ratio in (
            ("semantic_visible_mask_ratio", self.semantic_visible_mask_ratio),
            ("semantic_masked_mask_ratio", self.semantic_masked_mask_ratio),
            ("geometric_reverse_mask_ratio", self.geometric_reverse_mask_ratio),
            ("geometric_halfspace_mask_ratio", self.geometric_halfspace_mask_ratio),
        ):
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

    @staticmethod
    def _split_packed_path_outputs(
        x: torch.Tensor,
        num_pairs: int,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Split [all crops pair0, all crops pair1, ...] into path tensors."""
        x = x.unflatten(0, (num_pairs, batch_size))
        return [x_i.flatten(0, 1) for x_i in x]

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
    ) -> dict[str, Any]:
        """Student forward with packed semantic/geometric mask stacks.

        Args:
            mask_a: (B_packed, P) bool, first path in each packed stack.
            mask_b: (B_packed, P) bool, second path in each packed stack.

        Each row of mask_a is paired with the same row of mask_b.
        """
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
        B_masked = x_mpm.shape[0]

        # Repeat to match packed pairs (B_masked → B_packed)
        # mask_a is ordered [all_crops_pair0, all_crops_pair1, ...] from torch.cat
        # so we tile (repeat) x_mpm to match: [crops_copy0, crops_copy1, ...]
        num_pairs = B_packed // B_masked
        x_mpm = x_mpm.repeat(num_pairs, 1, 1)
        pos_enc_mpm = pos_enc_mpm.repeat(num_pairs, 1, 1)
        if centers_mpm is not None:
            centers_mpm = centers_mpm.repeat(num_pairs, 1, 1)

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

        # --- Encoder forward: varlen (zero-waste) or block-diagonal fallback ---
        use_varlen = (
            self.use_varlen_flash_attn
            and not has_relative_3d_bias(point_encoder.encoder)
        )

        if use_varlen:
            # Split mask_a and mask_b back into per-pair groups.
            # mask_a = [sem_70%, geo_70%], mask_b = [sem_50%, geo_50%].
            # x_vis_a/b are (B_packed, S_a/b, D) where B_packed = B_masked * num_pairs.
            # Split into num_pairs groups of B_masked each.
            x_vis_a_parts = x_vis_a.unflatten(0, (num_pairs, B_masked))
            pos_vis_a_parts = pos_vis_a.unflatten(0, (num_pairs, B_masked))
            x_vis_b_parts = x_vis_b.unflatten(0, (num_pairs, B_masked))
            pos_vis_b_parts = pos_vis_b.unflatten(0, (num_pairs, B_masked))

            # Build list of all 4 path tensors: [sem_a, sem_b, geo_a, geo_b]
            # ordered as pair0_a, pair0_b, pair1_a, pair1_b → interleaved
            x_list: list[torch.Tensor] = []
            pos_list: list[torch.Tensor] = []
            centers_list: list[torch.Tensor | None] = []
            for i in range(num_pairs):
                x_list.append(x_vis_a_parts[i])
                x_list.append(x_vis_b_parts[i])
                pos_list.append(pos_vis_a_parts[i])
                pos_list.append(pos_vis_b_parts[i])
                if centers_vis_a is not None and centers_vis_b is not None:
                    ca = centers_vis_a.unflatten(0, (num_pairs, B_masked))[i]
                    cb = centers_vis_b.unflatten(0, (num_pairs, B_masked))[i]
                    centers_list.extend([ca, cb])
                else:
                    centers_list.extend([None, None])

            pe_outputs = self._varlen_encoder_forward(
                x_list, pos_list, centers_list, point_encoder
            )

            # pe_outputs is [pair0_a, pair0_b, pair1_a, pair1_b, ...]
            # Recombine into pe_out_a (all A paths) and pe_out_b (all B paths)
            pe_out_a_parts = [pe_outputs[i * 2] for i in range(num_pairs)]
            pe_out_b_parts = [pe_outputs[i * 2 + 1] for i in range(num_pairs)]

            pe_out_a = PointEncoderOutput(
                patch_features=torch.cat(
                    [p.patch_features for p in pe_out_a_parts], dim=0
                ),
                cls_features=(
                    torch.cat([p.cls_features for p in pe_out_a_parts], dim=0)
                    if pe_out_a_parts[0].cls_features is not None
                    else None
                ),
                attn_weights=None,
                hidden_states=None,
            )
            pe_out_b = PointEncoderOutput(
                patch_features=torch.cat(
                    [p.patch_features for p in pe_out_b_parts], dim=0
                ),
                cls_features=(
                    torch.cat([p.cls_features for p in pe_out_b_parts], dim=0)
                    if pe_out_b_parts[0].cls_features is not None
                    else None
                ),
                attn_weights=None,
                hidden_states=None,
            )
        else:
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
        # pe_out_a/b each have shape (B_packed, ...) where B_packed = B_masked * num_pairs.
        # Order within each: [all_crops_pair0, all_crops_pair1, ...]
        # We need output ordered so each crop's sub-seqs are contiguous:
        #   [crop0_a_p0, crop0_b_p0, crop0_a_p1, crop0_b_p1, crop1_a_p0, ...]
        # This matches the multi_mask=4 convention expected by unflatten(0, (-1, multi_mask)).
        cls_logits_list = []
        cls_emb_list = []
        if self.mode.do_cls:
            for pe_out in (pe_out_a, pe_out_b):
                x_cls = pe_out.cls_features
                cls_logits_list.append(self.student.cls_projection_head(x_cls)[0])
                if return_embeddings:
                    cls_emb_list.append(x_cls)

            cls_logits_a, cls_logits_b = cls_logits_list
            B_masked = cls_logits_a.shape[0] // num_pairs

            # Reshape from [all_crops_pair0, all_crops_pair1] to (num_pairs, B_masked, D)
            cls_a_reshaped = cls_logits_a.unflatten(0, (num_pairs, B_masked))
            cls_b_reshaped = cls_logits_b.unflatten(0, (num_pairs, B_masked))
            # Interleave a,b within each pair: (num_pairs, 2, B_masked, D)
            cls_interleaved = torch.stack([cls_a_reshaped, cls_b_reshaped], dim=1)
            # Reorder to (B_masked, num_pairs*2, D) = (B_masked, multi_mask, D)
            cls_interleaved = cls_interleaved.permute(2, 0, 1, 3).reshape(
                B_masked * num_pairs * 2, -1
            )
            out_dict["x_cls_logits_masked"] = cls_interleaved

            if return_embeddings:
                emb_a = cls_emb_list[0].unflatten(0, (num_pairs, B_masked))
                emb_b = cls_emb_list[1].unflatten(0, (num_pairs, B_masked))
                emb_interleaved = torch.stack([emb_a, emb_b], dim=1)
                emb_interleaved = emb_interleaved.permute(2, 0, 1, 3).reshape(
                    B_masked * num_pairs * 2, -1
                )
                out_dict["x_cls_embedding_masked"] = emb_interleaved

        # --- Predictor for masked patch reconstruction ---
        patch_logits_packed = []
        patch_emb_packed = []

        predictor_inputs = [(pe_out_a, mask_a, pos_enc_mpm, centers_mpm)]
        if not self.patch_loss_a_only:
            predictor_inputs.append((pe_out_b, mask_b, pos_enc_mpm, centers_mpm))

        for pe_out, mask, pos_full, centers_full in predictor_inputs:
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
                patch_emb_packed.append(x_pred_patch)

            x_patch_proj = self.student.patch_projection_head(
                x_pred_patch, return_x_norm=True
            )
            patch_logits_packed.append(x_patch_proj.x)

        patch_logits_a = self._split_packed_path_outputs(
            patch_logits_packed[0], num_pairs, B_masked
        )
        if self.patch_loss_a_only:
            patch_logits_by_path = patch_logits_a
        else:
            patch_logits_b = self._split_packed_path_outputs(
                patch_logits_packed[1], num_pairs, B_masked
            )
            patch_logits_by_path = []
            for i in range(num_pairs):
                patch_logits_by_path.extend((patch_logits_a[i], patch_logits_b[i]))

        out_dict["x_patch_logits_by_path"] = patch_logits_by_path
        out_dict["x_patch_logits"] = torch.cat(patch_logits_by_path, dim=0)

        if return_embeddings:
            patch_emb_a = self._split_packed_path_outputs(
                patch_emb_packed[0], num_pairs, B_masked
            )
            if self.patch_loss_a_only:
                patch_emb_by_path = patch_emb_a
            else:
                patch_emb_b = self._split_packed_path_outputs(
                    patch_emb_packed[1], num_pairs, B_masked
                )
                patch_emb_by_path = []
                for i in range(num_pairs):
                    patch_emb_by_path.extend((patch_emb_a[i], patch_emb_b[i]))

            out_dict["x_patch_embedding_by_path"] = patch_emb_by_path
            out_dict["x_patch_embedding"] = torch.cat(patch_emb_by_path, dim=0)

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

        # ---- Step 2: Generate semantic/geometric packed stacks ----
        attn_weights = self._teacher_attn_weights
        assert attn_weights is not None

        P = global_centers.shape[1]
        num_masks_sem_visible = round(self.semantic_visible_mask_ratio * P)
        num_masks_sem_masked = round(self.semantic_masked_mask_ratio * P)
        num_masks_geo_reverse = round(self.geometric_reverse_mask_ratio * P)
        num_masks_geo_halfspace = round(self.geometric_halfspace_mask_ratio * P)

        if min(
            num_masks_sem_visible,
            num_masks_sem_masked,
            num_masks_geo_reverse,
            num_masks_geo_halfspace,
        ) == 0:
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

        # Rotate heads for mask diversity: each semantic mask uses a different
        # head, cycling through all heads over training steps.
        num_heads = self.student.point_encoder.encoder.config.num_heads
        step = self.global_step
        head_a = step % num_heads
        head_b = (step + num_heads // 2) % num_heads

        # Pair 0: semantic stack
        #   A: high-attention visible, 70% masked
        #   B: high-attention masked, 50% masked
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

        # Pair 1: geometric stack
        #   A: random reverse-block, 70% masked
        #   B: random half-space, 50% masked
        geo_reverse_block = self._generate_inverse_block_mask(
            masked_centers,
            num_masks_geo_reverse,
            self.geometric_reverse_mask_ratio,
        )
        geo_halfspace = self._generate_halfspace_mask(
            masked_centers,
            num_masks_geo_halfspace,
        )

        masks_a_list = [sem_high_visible, geo_reverse_block]
        masks_b_list = [sem_high_masked, geo_halfspace]
        num_pairs = len(masks_a_list)

        # Stack: (B_masked * 2, P), ordered [semantic crops, geometric crops]
        mask_a = torch.cat(masks_a_list, dim=0)
        mask_b = torch.cat(masks_b_list, dim=0)

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

            B_masked = x_patch_logits_t.shape[0]

            # mask_a/mask_b are [all crops semantic stack, all crops geometric stack].
            x_patch_t_a = x_patch_logits_t.repeat(num_pairs, 1, 1)
            targets_a = gather_masked(x_patch_t_a, mask_a)
            targets_a_by_path = self._split_packed_path_outputs(
                targets_a, num_pairs, B_masked
            )

            if self.patch_loss_a_only:
                targets_by_path = targets_a_by_path
            else:
                x_patch_t_b = x_patch_logits_t.repeat(num_pairs, 1, 1)
                targets_b = gather_masked(x_patch_t_b, mask_b)
                targets_b_by_path = self._split_packed_path_outputs(
                    targets_b, num_pairs, B_masked
                )
                targets_by_path = []
                for i in range(num_pairs):
                    targets_by_path.extend((targets_a_by_path[i], targets_b_by_path[i]))

            out_targets["x_patch_logits_by_path"] = targets_by_path
            out_targets["x_patch_logits"] = torch.cat(targets_by_path, dim=0)

            if self.do_regression:
                x_emb_crop = x_patch_teacher[indices_masked_crops]
                x_emb_a = x_emb_crop.repeat(num_pairs, 1, 1)
                emb_a = gather_masked(x_emb_a, mask_a)
                emb_a_by_path = self._split_packed_path_outputs(
                    emb_a, num_pairs, B_masked
                )
                if self.patch_loss_a_only:
                    emb_by_path = emb_a_by_path
                else:
                    x_emb_b = x_emb_crop.repeat(num_pairs, 1, 1)
                    emb_b = gather_masked(x_emb_b, mask_b)
                    emb_b_by_path = self._split_packed_path_outputs(
                        emb_b, num_pairs, B_masked
                    )
                    emb_by_path = []
                    for i in range(num_pairs):
                        emb_by_path.extend((emb_a_by_path[i], emb_b_by_path[i]))

                out_targets["x_patch_embedding_by_path"] = emb_by_path
                out_targets["x_patch_embedding"] = torch.cat(emb_by_path, dim=0)

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
