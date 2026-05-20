from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch3d.loss import chamfer_distance
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall
from ..components.exponential_moving_average import EMA
from ..components.utils import gather_masked, init_lazy_defaults, lengths_to_mask
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..layers.tokenization import Tokens
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab_packed import PackedFusedAttnBlockAsymDSD
from .point_encoder import PointEncoderOutput
from .pq_stem_point_encoder import PQStemPatches, PQStemPointEncoder
from .pq_transup import fps_subsample
from .pqdt_tail import PQDTTail, PQDTUpSampler

logger = get_default_logger()


class PQDTPackedFusedAttnBlockAsymDSD(PackedFusedAttnBlockAsymDSD):
    """Packed FAB pretraining with PQDT stem, transformer tail, and up layers."""

    @init_lazy_defaults
    def __init__(
        self,
        # --- PQDT stem and transformer ---
        pqdt_in_chans: int = 256,
        pqdt_stem_enc_attn: tuple[str, ...] = (
            "ge_attn",
            "attn",
            "attn",
            "attn",
        ),
        pqdt_enc_attn: tuple[str, ...] = ("ge_attn", "attn", "attn", "attn"),
        pqdt_dec_attn: tuple[str, ...] = (
            "ge_attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
            "attn",
        ),
        pqdt_mlp_ratio: float = 2.0,
        pqdt_drop_rate: float = 0.0,
        pqdt_attn_drop_rate: float = 0.0,
        pqdt_transdown_fps: tuple[int, ...] = (512, 128),
        pqdt_transdown_dims: tuple[int, ...] = (64, 256),
        pqdt_transdown_num_heads: tuple[int, ...] = (1, 4),
        pqdt_transdown_sa_depth: tuple[int, ...] = (3, 3),
        pqdt_transdown_k: tuple[int, ...] = (16, 16),
        pqdt_transdown_use_attn: bool | tuple[bool, ...] = False,
        pqdt_num_pseudo: int = 384,
        pqdt_num_query: int = 512,
        pqdt_tau0: float = 1.0,
        pqdt_total_epochs: int | None = None,
        pqdt_r_sph: float = 0.8,
        pqdt_in_q: bool = True,
        # --- PQDT up layers ---
        pqdt_up_factors: tuple[int, ...] = (1, 4, 4),
        pqdt_up_n_knn: int = 16,
        pqdt_up_radius: float = 1.0,
        pqdt_up_interpolate: str = "three",
        pqdt_up_attn_channel: bool = True,
        pqdt_cd_weight: float = 1.0,
        pqdt_cd_every_n_steps: int = 1,
        pqdt_cd_start_epoch: int = 0,
        pqdt_cd_warmup_epochs: int = 0,
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
        # --- inherited from AsymDSD ---
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
        | None = None,
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
        if use_varlen_flash_attn:
            logger.warning(
                "PQDTPackedFusedAttnBlockAsymDSD forces "
                "use_varlen_flash_attn=False because PQDT GE attention is not "
                "the AsymDSD transformer backend."
            )

        super().__init__(
            vis_mask_ratio=vis_mask_ratio,
            semantic_visible_mask_ratio=semantic_visible_mask_ratio,
            semantic_masked_mask_ratio=semantic_masked_mask_ratio,
            sparse_visible_mask_ratio=sparse_visible_mask_ratio,
            sparse_masked_mask_ratio=sparse_masked_mask_ratio,
            geometric_reverse_mask_ratio=geometric_reverse_mask_ratio,
            geometric_halfspace_mask_ratio=geometric_halfspace_mask_ratio,
            patch_loss_a_only=patch_loss_a_only,
            use_varlen_flash_attn=False,
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

        original_num_heads = self.student.point_encoder.encoder.config.num_heads
        self.pqdt_in_chans = pqdt_in_chans
        self.pqdt_stem_enc_attn = tuple(pqdt_stem_enc_attn)
        self.pqdt_enc_attn = tuple(pqdt_enc_attn)
        self.pqdt_dec_attn = tuple(pqdt_dec_attn)
        self.pqdt_mlp_ratio = pqdt_mlp_ratio
        self.pqdt_drop_rate = pqdt_drop_rate
        self.pqdt_attn_drop_rate = pqdt_attn_drop_rate
        self.pqdt_transdown_fps = tuple(pqdt_transdown_fps)
        self.pqdt_transdown_dims = tuple(pqdt_transdown_dims)
        self.pqdt_transdown_num_heads = tuple(pqdt_transdown_num_heads)
        self.pqdt_transdown_sa_depth = tuple(pqdt_transdown_sa_depth)
        self.pqdt_transdown_k = tuple(pqdt_transdown_k)
        self.pqdt_transdown_use_attn = pqdt_transdown_use_attn
        self.pqdt_num_pseudo = pqdt_num_pseudo
        self.pqdt_num_query = pqdt_num_query
        self.pqdt_tau0 = pqdt_tau0
        self.pqdt_total_epochs = pqdt_total_epochs or self.max_epochs or 200
        self.pqdt_r_sph = pqdt_r_sph
        self.pqdt_in_q = pqdt_in_q
        self.pqdt_up_factors = tuple(pqdt_up_factors)
        self.pqdt_up_n_knn = pqdt_up_n_knn
        self.pqdt_up_radius = pqdt_up_radius
        self.pqdt_up_interpolate = pqdt_up_interpolate
        self.pqdt_up_attn_channel = pqdt_up_attn_channel
        self.pqdt_cd_weight = pqdt_cd_weight
        self.pqdt_cd_every_n_steps = pqdt_cd_every_n_steps
        self.pqdt_cd_start_epoch = pqdt_cd_start_epoch
        self.pqdt_cd_warmup_epochs = pqdt_cd_warmup_epochs
        self.pqdt_gradient_checkpointing = gradient_checkpointing

        if self.pqdt_cd_every_n_steps < 1:
            raise ValueError("pqdt_cd_every_n_steps must be >= 1.")
        if self.pqdt_cd_start_epoch < 0:
            raise ValueError("pqdt_cd_start_epoch must be >= 0.")
        if self.pqdt_cd_warmup_epochs < 0:
            raise ValueError("pqdt_cd_warmup_epochs must be >= 0.")

        def point_encoder() -> PQStemPointEncoder:
            return PQStemPointEncoder(
                in_chans=self.pqdt_in_chans,
                embed_dim=self.embed_dim,
                num_heads=original_num_heads,
                enc_attn=self.pqdt_stem_enc_attn,
                mlp_ratio=self.pqdt_mlp_ratio,
                drop_rate=self.pqdt_drop_rate,
                attn_drop_rate=self.pqdt_attn_drop_rate,
                transdown_fps=self.pqdt_transdown_fps,
                transdown_dims=self.pqdt_transdown_dims,
                transdown_num_heads=self.pqdt_transdown_num_heads,
                transdown_sa_depth=self.pqdt_transdown_sa_depth,
                transdown_k=self.pqdt_transdown_k,
                transdown_use_attn=self.pqdt_transdown_use_attn,
                cls_token=self.mode.do_cls,
            )

        def pqdt_tail() -> PQDTTail:
            return PQDTTail(
                in_chans=self.pqdt_in_chans,
                embed_dim=self.embed_dim,
                num_heads=original_num_heads,
                enc_attn=self.pqdt_enc_attn,
                dec_attn=self.pqdt_dec_attn,
                mlp_ratio=self.pqdt_mlp_ratio,
                drop_rate=self.pqdt_drop_rate,
                attn_drop_rate=self.pqdt_attn_drop_rate,
                num_pseudo=self.pqdt_num_pseudo,
                num_query=self.pqdt_num_query,
                tau0=self.pqdt_tau0,
                total_epochs=self.pqdt_total_epochs,
                r_sph=self.pqdt_r_sph,
                in_q=self.pqdt_in_q,
            )

        self.student["point_encoder"] = point_encoder()
        self.teacher["point_encoder"] = point_encoder()
        self.student["pqdt_tail"] = pqdt_tail()
        self.teacher["pqdt_tail"] = pqdt_tail()
        self.ema = EMA(self.student, self.teacher)

        self.up_sampler = PQDTUpSampler(
            embed_dim=self.embed_dim,
            up_factors=self.pqdt_up_factors,
            n_knn=self.pqdt_up_n_knn,
            radius=self.pqdt_up_radius,
            interpolate=self.pqdt_up_interpolate,
            attn_channel=self.pqdt_up_attn_channel,
        )

        self._pqdt_last_points: torch.Tensor | None = None
        self._pqdt_reconstruction_teacher_cache: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None

    @property
    def up_layers(self) -> nn.ModuleList:
        return self.up_sampler.up_layers

    def _forward_student_packed(
        self,
        multi_patches: PQStemPatches,
        indices_masked_crops: torch.Tensor,
        indices_unmasked_crops: torch.Tensor | None,
        masks: list[torch.Tensor],
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        point_encoder: PQStemPointEncoder = self.student.point_encoder

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        self._after_student_patch_embedding(tokens, point_encoder)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers
        if token_centers is None:
            raise ValueError("PQDT packed pretraining requires token centers.")

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

        if self.mode.do_cls and indices_unmasked_crops is not None:
            pe_out_unmask = point_encoder.transformer_encoder_forward(
                x[indices_unmasked_crops],
                pos_enc[indices_unmasked_crops],
                token_centers=token_centers[indices_unmasked_crops],
                attn_bias_scale=attn_bias_scale,
            )
            x_cls_unmask = pe_out_unmask.cls_features
            out_dict["x_cls_logits"] = self.student.cls_projection_head(x_cls_unmask)[0]
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls_unmask

        x_mpm = x[indices_masked_crops]
        pos_enc_mpm = pos_enc[indices_masked_crops]
        centers_mpm = token_centers[indices_masked_crops]
        points_mpm = multi_patches.points[indices_masked_crops]
        B_masked = x_mpm.shape[0]
        num_paths = len(masks)

        pe_outputs = []
        query_centers_by_path: list[torch.Tensor] = []
        patch_emb_by_path: list[torch.Tensor] = []
        patch_logits_by_path: list[torch.Tensor] = []
        collect_patch_embeddings = return_embeddings and self.do_regression

        predictor_paths = list(range(num_paths))
        if self.patch_loss_a_only:
            predictor_paths = list(range(0, num_paths, 2))

        def pqdt_tail_forward(
            points: torch.Tensor,
            coor_c: torch.Tensor,
            x1: torch.Tensor,
            query_seed: torch.Tensor,
        ) -> torch.Tensor:
            return self.student.pqdt_tail.forward_queries(
                points,
                coor_c,
                x1,
                query_seed,
                current_epoch=int(self.current_epoch),
            )

        def same_shape(tensors: list[torch.Tensor]) -> bool:
            return bool(tensors) and all(
                tensor.shape == tensors[0].shape for tensor in tensors[1:]
            )

        x_vis_by_path: list[torch.Tensor] = []
        pos_vis_by_path: list[torch.Tensor] = []
        centers_vis_by_path: list[torch.Tensor] = []
        for mask in masks:
            inv_mask = ~mask
            x_vis_by_path.append(gather_masked(x_mpm, inv_mask))
            pos_vis_by_path.append(gather_masked(pos_enc_mpm, inv_mask))
            centers_vis_by_path.append(gather_masked(centers_mpm, inv_mask))

        if (
            same_shape(x_vis_by_path)
            and same_shape(pos_vis_by_path)
            and same_shape(centers_vis_by_path)
        ):
            pe_out_stacked = point_encoder.transformer_encoder_forward(
                torch.cat(x_vis_by_path, dim=0),
                torch.cat(pos_vis_by_path, dim=0),
                token_centers=torch.cat(centers_vis_by_path, dim=0),
                attn_bias_scale=attn_bias_scale,
            )
            patch_chunks = pe_out_stacked.patch_features.split(B_masked, dim=0)
            if pe_out_stacked.cls_features is None:
                cls_chunks = [None] * num_paths
            else:
                cls_chunks = pe_out_stacked.cls_features.split(B_masked, dim=0)
            pe_outputs = [
                PointEncoderOutput(
                    patch_features=patch_chunks[path_idx],
                    cls_features=cls_chunks[path_idx],
                    attn_weights=None,
                    hidden_states=None,
                )
                for path_idx in range(num_paths)
            ]
        else:
            for x_vis, pos_vis, centers_vis in zip(
                x_vis_by_path,
                pos_vis_by_path,
                centers_vis_by_path,
                strict=True,
            ):
                pe_out = point_encoder.transformer_encoder_forward(
                    x_vis,
                    pos_vis,
                    token_centers=centers_vis,
                    attn_bias_scale=attn_bias_scale,
                )
                pe_outputs.append(pe_out)

        centers_vis_t_by_pred: list[torch.Tensor] = []
        patch_features_by_pred: list[torch.Tensor] = []
        query_seed_by_path: list[torch.Tensor] = []
        for path_idx in predictor_paths:
            mask = masks[path_idx]
            centers_masked = gather_masked(centers_mpm, mask)
            if self.masked_pos_noise is not None:
                centers_masked = centers_masked + self.masked_pos_noise * (
                    torch.randn_like(centers_masked)
                )
            query_centers_by_path.append(centers_masked)
            query_seed_by_path.append(centers_masked.transpose(1, 2).contiguous())
            centers_vis_t_by_pred.append(
                centers_vis_by_path[path_idx].transpose(1, 2).contiguous()
            )
            patch_features_by_pred.append(pe_outputs[path_idx].patch_features)

        can_stack_pqdt = (
            len(predictor_paths) > 0
            and same_shape(centers_vis_t_by_pred)
            and same_shape(patch_features_by_pred)
            and same_shape(query_seed_by_path)
        )
        if can_stack_pqdt:
            points_stacked = (
                points_mpm.unsqueeze(0)
                .expand(len(predictor_paths), -1, -1, -1)
                .flatten(0, 1)
            )
            centers_vis_stacked = torch.cat(centers_vis_t_by_pred, dim=0)
            patch_features_stacked = torch.cat(patch_features_by_pred, dim=0)
            query_seed_stacked = torch.cat(query_seed_by_path, dim=0)
            if self.pqdt_gradient_checkpointing and torch.is_grad_enabled():
                x_pred_stacked = checkpoint(
                    pqdt_tail_forward,
                    points_stacked,
                    centers_vis_stacked,
                    patch_features_stacked,
                    query_seed_stacked,
                    use_reentrant=False,
                )
            else:
                x_pred_stacked = pqdt_tail_forward(
                    points_stacked,
                    centers_vis_stacked,
                    patch_features_stacked,
                    query_seed_stacked,
                )
            x_pred_by_path = list(x_pred_stacked.split(B_masked, dim=0))
        else:
            x_pred_by_path = []
            for centers_vis_t, patch_features, query_seed in zip(
                centers_vis_t_by_pred,
                patch_features_by_pred,
                query_seed_by_path,
                strict=True,
            ):
                if self.pqdt_gradient_checkpointing and torch.is_grad_enabled():
                    x_pred_patch = checkpoint(
                        pqdt_tail_forward,
                        points_mpm,
                        centers_vis_t,
                        patch_features,
                        query_seed,
                        use_reentrant=False,
                    )
                else:
                    x_pred_patch = pqdt_tail_forward(
                        points_mpm,
                        centers_vis_t,
                        patch_features,
                        query_seed,
                    )
                x_pred_by_path.append(x_pred_patch)

        for x_pred_patch in x_pred_by_path:
            if collect_patch_embeddings:
                patch_emb_by_path.append(x_pred_patch)
            x_patch_proj = self.student.patch_projection_head(
                x_pred_patch,
                return_x_norm=True,
            )
            patch_logits_by_path.append(x_patch_proj.x)

        if self.mode.do_cls:
            cls_logits_list = []
            cls_emb_list = []
            for pe_out in pe_outputs:
                x_cls = pe_out.cls_features
                cls_logits_list.append(self.student.cls_projection_head(x_cls)[0])
                if return_embeddings:
                    cls_emb_list.append(x_cls)
            out_dict["x_cls_logits_masked"] = torch.stack(
                cls_logits_list,
                dim=1,
            ).reshape(B_masked * num_paths, -1)
            if return_embeddings:
                out_dict["x_cls_embedding_masked"] = torch.stack(
                    cls_emb_list,
                    dim=1,
                ).reshape(B_masked * num_paths, -1)

        out_dict["x_patch_logits_by_path"] = patch_logits_by_path
        out_dict["x_patch_logits"] = torch.cat(
            [t.flatten(0, 1) for t in patch_logits_by_path],
            dim=0,
        )
        out_dict["x_patch_centers_by_path"] = query_centers_by_path

        if collect_patch_embeddings:
            out_dict["x_patch_embedding_by_path"] = patch_emb_by_path
            out_dict["x_patch_embedding"] = torch.cat(
                [t.flatten(0, 1) for t in patch_emb_by_path],
                dim=0,
            )

        return out_dict

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        if "global_crops" in batch or "local_crops" in batch:
            raise ValueError(
                "PQDT STAMPA pretraining expects target-only batches with "
                "`points` and optional `num_points`; global/local crop batches "
                "are intentionally disabled."
            )

        self._pqdt_last_points = None
        self._pqdt_reconstruction_teacher_cache = None
        outputs = super().training_step(batch, batch_idx)
        points = self._pqdt_last_points
        teacher_cache = self._pqdt_reconstruction_teacher_cache
        should_compute_cd = self._should_compute_pqdt_cd()
        self._pqdt_last_points = None
        self._pqdt_reconstruction_teacher_cache = None

        if should_compute_cd and points is not None:
            pqdt_cd_weight = self._pqdt_cd_effective_weight()
            pqdt_cd_loss, pred_pcds = self._pqdt_reconstruction_loss(
                points,
                token_cache=teacher_cache,
            )
            outputs["loss"] = outputs["loss"] + pqdt_cd_weight * pqdt_cd_loss
            outputs["pqdt_cd_loss"] = pqdt_cd_loss
            outputs["pqdt_cd_weight"] = pqdt_cd_weight
            outputs["pqdt_reconstructions"] = [pcd.detach() for pcd in pred_pcds]
        else:
            outputs["pqdt_cd_loss"] = None
            outputs["pqdt_cd_weight"] = 0.0
            outputs["pqdt_reconstructions"] = None
        return outputs

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        points = self._target_batch_points(batch, apply_augmentation=False)
        query_seed, f_query_seed = self._teacher_pqdt_tokens_detached(points)
        pqdt_cd_loss, pred_pcds = self._pqdt_reconstruction_loss(
            points,
            token_cache=(query_seed, f_query_seed),
        )
        pqdt_last_cd2_loss = self._chamfer_loss(
            pred_pcds[-1],
            points.detach()[..., :3].contiguous(),
            norm=2,
        )
        self.log(
            "val/pqdt_cd_loss",
            pqdt_cd_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=points.shape[0],
        )
        self.log(
            "val/pqdt_last_cd2_loss",
            pqdt_last_cd2_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=points.shape[0],
        )
        return {
            "pqdt_cd_loss": pqdt_cd_loss,
            "pqdt_last_cd2_loss": pqdt_last_cd2_loss,
            "pqdt_reconstructions": [pcd.detach() for pcd in pred_pcds],
            "transup_reconstructions": [pcd.detach() for pcd in pred_pcds],
            "gt_points": points.detach(),
            "pqdt_query_seed": query_seed.detach(),
            "eval_token_centers": query_seed.transpose(1, 2).detach(),
            "eval_mask": None,
        }

    def test_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def _should_compute_pqdt_cd(self) -> bool:
        return (
            len(self.up_layers) > 0
            and self._pqdt_cd_effective_weight() > 0.0
            and self.global_step % self.pqdt_cd_every_n_steps == 0
        )

    def _pqdt_cd_effective_weight(self) -> float:
        if self.pqdt_cd_weight <= 0.0:
            return 0.0
        epoch = float(self.current_epoch)
        start_epoch = float(self.pqdt_cd_start_epoch)
        if epoch < start_epoch:
            return 0.0
        warmup_epochs = float(self.pqdt_cd_warmup_epochs)
        if warmup_epochs <= 0.0:
            return self.pqdt_cd_weight
        scale = min((epoch - start_epoch + 1.0) / warmup_epochs, 1.0)
        return self.pqdt_cd_weight * scale

    def _after_teacher_forward_packed(
        self,
        multi_patches: Any,
        global_centers: torch.Tensor,
        x_patch_teacher: torch.Tensor,
    ) -> None:
        del multi_patches
        if not self._should_compute_pqdt_cd() or self._pqdt_last_points is None:
            return
        _, query_seed, f_query_seed = self.teacher.pqdt_tail.forward_full(
            self._pqdt_last_points.detach(),
            global_centers.transpose(1, 2).contiguous(),
            x_patch_teacher,
            current_epoch=int(self.current_epoch),
        )
        self._pqdt_reconstruction_teacher_cache = (
            query_seed.detach(),
            f_query_seed.detach(),
        )

    def _teacher_pqdt_tokens_detached(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_encoder: PQStemPointEncoder = self.teacher.point_encoder
        pqdt_tail: PQDTTail = self.teacher.pqdt_tail
        was_training_encoder = point_encoder.training
        was_training_tail = pqdt_tail.training
        point_encoder.eval()
        pqdt_tail.eval()
        try:
            with torch.no_grad():
                tokens = point_encoder.patch_embedding(
                    PQStemPatches(points=points.detach(), centers=[])
                )
                encoder_out = point_encoder.transformer_encoder_forward(
                    tokens.embeddings,
                    tokens.pos_embeddings,
                    token_centers=tokens.centers,
                )
                _, query_seed, f_query_seed = pqdt_tail.forward_full(
                    points.detach(),
                    tokens.centers.transpose(1, 2).contiguous(),
                    encoder_out.patch_features,
                    current_epoch=int(self.current_epoch),
                )
        finally:
            point_encoder.train(was_training_encoder)
            pqdt_tail.train(was_training_tail)
        return query_seed.detach(), f_query_seed.detach()

    def _pqdt_up_forward(
        self,
        query_seed: torch.Tensor,
        seed_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        return self.up_sampler(query_seed, seed_features)

    def _pqdt_reconstruction_loss(
        self,
        points: torch.Tensor,
        token_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if token_cache is None:
            query_seed, f_query_seed = self._teacher_pqdt_tokens_detached(points)
        else:
            query_seed, f_query_seed = token_cache
        pred_pcds = self._pqdt_up_forward(
            query_seed.detach(),
            f_query_seed.detach(),
        )
        target = points[..., :3].contiguous()
        losses = [self._chamfer_loss(pred, target, norm=1) for pred in pred_pcds[1:]]
        if not losses:
            losses = [self._chamfer_loss(pred_pcds[0], target, norm=1)]
        return torch.stack(losses).sum(), pred_pcds

    @staticmethod
    def _chamfer_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        norm: int = 1,
    ) -> torch.Tensor:
        if pred.shape[1] == target.shape[1]:
            target_i = target
        else:
            target_i = fps_subsample(target, pred.shape[1])
        loss, _ = chamfer_distance(
            pred.float(),
            target_i.float(),
            norm=norm,
            batch_reduction="mean",
            point_reduction="mean",
        )
        return loss

    def _target_batch_points(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        apply_augmentation: bool,
    ) -> torch.Tensor:
        if "global_crops" in batch or "local_crops" in batch:
            raise ValueError(
                "PQDT STAMPA reconstruction expects target-only batches with "
                "`points` and optional `num_points`; global/local crop batches "
                "are intentionally disabled."
            )
        return self._preprocess_target_points(
            points=batch["points"],  # type: ignore[arg-type]
            num_points=batch.get("num_points"),  # type: ignore[arg-type]
            apply_augmentation=apply_augmentation,
        )

    def _preprocess_target_points(
        self,
        points: torch.Tensor,
        num_points: torch.Tensor | None,
        apply_augmentation: bool,
    ) -> torch.Tensor:
        mask = (
            lengths_to_mask(num_points, points.size(1))
            if num_points is not None
            else None
        )
        points = self.norm_transform(points, mask=mask)
        if apply_augmentation:
            points = self.aug_transform(points)
        return points

    def _extract_patches(
        self,
        patch_points: PatchPoints,
        patchify: Any,
    ) -> PQStemPatches:
        del patchify
        crops = self._preprocess_target_points(
            patch_points.points,
            patch_points.num_points,
            apply_augmentation=True,
        )
        patch_points.points = crops
        self._pqdt_last_points = crops
        centers = self.teacher.point_encoder.compute_centers(crops)
        return PQStemPatches(points=crops, centers=[centers])

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        pqdt_cd_loss = outputs.get("pqdt_cd_loss")
        if pqdt_cd_loss is not None:
            self.log(
                "train/pqdt_cd_loss",
                pqdt_cd_loss,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                "train/pqdt_cd_weight",
                outputs.get("pqdt_cd_weight", 0.0),
                on_step=True,
                prog_bar=False,
            )

    def _prefixed_state_dict(
        self,
        module: nn.Module,
        prefix: str,
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        state_dict = module.state_dict()
        if cpu:
            return {
                f"{prefix}{key}": value.detach().cpu()
                for key, value in state_dict.items()
            }
        return {f"{prefix}{key}": value for key, value in state_dict.items()}

    def pq_stem_state_dict(
        self,
        branch: str = "teacher",
        prefix: str | None = None,
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        if branch not in {"teacher", "student"}:
            raise ValueError("branch must be 'teacher' or 'student'.")
        module = getattr(self, branch).point_encoder.stem_encoder
        if prefix is None:
            prefix = f"{branch}.stem_encoder."
        return self._prefixed_state_dict(module, prefix=prefix, cpu=cpu)

    def pqdt_tail_state_dict(
        self,
        branch: str = "teacher",
        prefix: str | None = None,
        cpu: bool = True,
        include_pseudo_stage: bool = False,
    ) -> dict[str, torch.Tensor]:
        if branch not in {"teacher", "student"}:
            raise ValueError("branch must be 'teacher' or 'student'.")
        module = getattr(self, branch).pqdt_tail
        if prefix is None:
            prefix = f"{branch}.transformer."
        return {
            f"{prefix}{key}": value
            for key, value in module.pqdt_flat_state_dict(
                include_pseudo_stage=include_pseudo_stage,
                cpu=cpu,
            ).items()
        }

    def up_layers_state_dict(
        self,
        prefix: str = "up_layers.",
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        return self._prefixed_state_dict(self.up_layers, prefix=prefix, cpu=cpu)

    def pqdt_component_state_dict(
        self,
        branch: str = "teacher",
        cpu: bool = True,
        include_up_layers: bool = True,
        include_transformer: bool = True,
        pqdt_loadable: bool = False,
        include_pseudo_stage: bool = False,
    ) -> dict[str, torch.Tensor]:
        stem_prefix = "stem_encoder." if pqdt_loadable else f"{branch}.stem_encoder."
        state_dict = self.pq_stem_state_dict(
            branch=branch,
            prefix=stem_prefix,
            cpu=cpu,
        )
        if include_transformer:
            transformer_prefix = (
                "transformer." if pqdt_loadable else f"{branch}.transformer."
            )
            state_dict.update(
                self.pqdt_tail_state_dict(
                    branch=branch,
                    prefix=transformer_prefix,
                    cpu=cpu,
                    include_pseudo_stage=include_pseudo_stage,
                )
            )
        if include_up_layers:
            state_dict.update(self.up_layers_state_dict(cpu=cpu))
        return state_dict

    def export_pqdt_components(
        self,
        path: str | Path,
        branch: str = "teacher",
        include_up_layers: bool = True,
        include_transformer: bool = True,
        pqdt_loadable: bool = True,
        include_pseudo_stage: bool = False,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.pqdt_component_state_dict(
                branch=branch,
                include_up_layers=include_up_layers,
                include_transformer=include_transformer,
                pqdt_loadable=pqdt_loadable,
                include_pseudo_stage=include_pseudo_stage,
            ),
            path,
        )
