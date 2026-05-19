from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..components import *
from ..components.common_types import FloatMayCall
from ..components.exponential_moving_average import EMA
from ..components.utils import init_lazy_defaults, lengths_to_mask
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..loggers import get_default_logger
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .asymdsd_fab_packed import PackedFusedAttnBlockAsymDSD
from .pq_stem_point_encoder import PQStemPatches, PQStemPointEncoder
from .pq_transup import PQStemTransUpHead

logger = get_default_logger()


class PQStemPackedFusedAttnBlockAsymDSD(PackedFusedAttnBlockAsymDSD):
    """Packed FAB pretraining with PQDT's reusable PQStemEncoder backbone."""

    @init_lazy_defaults
    def __init__(
        self,
        # --- PQDT stem backbone ---
        pqstem_in_chans: int = 256,
        pqstem_enc_attn: tuple[str, ...] = ("ge_attn", "attn", "attn", "attn"),
        pqstem_mlp_ratio: float = 2.0,
        pqstem_drop_rate: float = 0.0,
        pqstem_attn_drop_rate: float = 0.0,
        pqstem_transdown_fps: tuple[int, ...] = (512, 128),
        pqstem_transdown_dims: tuple[int, ...] = (64, 256),
        pqstem_transdown_num_heads: tuple[int, ...] = (1, 4),
        pqstem_transdown_sa_depth: tuple[int, ...] = (3, 3),
        pqstem_transdown_k: tuple[int, ...] = (16, 16),
        pqstem_transdown_use_attn: bool | tuple[bool, ...] = False,
        pqstem_enable_transup_reconstruction: bool = False,
        pqstem_transup_num_seed: int = 512,
        pqstem_transup_up_factors: tuple[int, ...] = (1, 4, 4),
        pqstem_transup_n_knn: int = 16,
        pqstem_transup_radius: float = 1.0,
        pqstem_transup_interpolate: str = "three",
        pqstem_transup_attn_channel: bool = True,
        pqstem_transup_cd_weight: float = 1.0,
        pqstem_transup_cd_every_n_steps: int = 1,
        # --- inherited from PackedFusedAttnBlockAsymDSD ---
        sparse_visible_mask_ratio: float = 0.7,
        sparse_masked_mask_ratio: float = 0.5,
        random_mask_ratio: float = 0.7,
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
        if use_varlen_flash_attn:
            logger.warning(
                "PQStemPackedFusedAttnBlockAsymDSD forces "
                "use_varlen_flash_attn=False because PQDT GEEncoder is not the "
                "AsymDSD transformer backend."
            )

        super().__init__(
            sparse_visible_mask_ratio=sparse_visible_mask_ratio,
            sparse_masked_mask_ratio=sparse_masked_mask_ratio,
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
        self.pqstem_in_chans = pqstem_in_chans
        self.pqstem_enc_attn = tuple(pqstem_enc_attn)
        self.pqstem_mlp_ratio = pqstem_mlp_ratio
        self.pqstem_drop_rate = pqstem_drop_rate
        self.pqstem_attn_drop_rate = pqstem_attn_drop_rate
        self.pqstem_transdown_fps = tuple(pqstem_transdown_fps)
        self.pqstem_transdown_dims = tuple(pqstem_transdown_dims)
        self.pqstem_transdown_num_heads = tuple(pqstem_transdown_num_heads)
        self.pqstem_transdown_sa_depth = tuple(pqstem_transdown_sa_depth)
        self.pqstem_transdown_k = tuple(pqstem_transdown_k)
        self.pqstem_transdown_use_attn = pqstem_transdown_use_attn
        self.pqstem_enable_transup_reconstruction = pqstem_enable_transup_reconstruction
        self.pqstem_transup_num_seed = pqstem_transup_num_seed
        self.pqstem_transup_up_factors = tuple(pqstem_transup_up_factors)
        self.pqstem_transup_n_knn = pqstem_transup_n_knn
        self.pqstem_transup_radius = pqstem_transup_radius
        self.pqstem_transup_interpolate = pqstem_transup_interpolate
        self.pqstem_transup_attn_channel = pqstem_transup_attn_channel
        self.pqstem_transup_cd_weight = pqstem_transup_cd_weight
        self.pqstem_transup_cd_every_n_steps = pqstem_transup_cd_every_n_steps
        self.random_mask_ratio = random_mask_ratio

        if self.pqstem_transup_cd_every_n_steps < 1:
            raise ValueError("pqstem_transup_cd_every_n_steps must be >= 1.")
        if self.multi_mask != 4:
            raise ValueError(
                "PQStem packed FAB expects mask_generator.multi_mask=4 for "
                "sparse_visible + sparse_masked + geometric_halfspace + random paths."
            )
        for name, ratio in [
            ("sparse_visible_mask_ratio", self.sparse_visible_mask_ratio),
            ("sparse_masked_mask_ratio", self.sparse_masked_mask_ratio),
            ("geometric_halfspace_mask_ratio", self.geometric_halfspace_mask_ratio),
            ("random_mask_ratio", self.random_mask_ratio),
        ]:
            if ratio <= 0.0 or ratio >= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.patch_loss_a_only:
            logger.warning(
                "patch_loss_a_only is disabled for PQStem packed FAB because "
                "its four mask paths are not paired semantic/geometric A/B paths."
            )
            self.patch_loss_a_only = False

        def point_encoder() -> PQStemPointEncoder:
            return PQStemPointEncoder(
                in_chans=self.pqstem_in_chans,
                embed_dim=self.embed_dim,
                num_heads=original_num_heads,
                enc_attn=self.pqstem_enc_attn,
                mlp_ratio=self.pqstem_mlp_ratio,
                drop_rate=self.pqstem_drop_rate,
                attn_drop_rate=self.pqstem_attn_drop_rate,
                transdown_fps=self.pqstem_transdown_fps,
                transdown_dims=self.pqstem_transdown_dims,
                transdown_num_heads=self.pqstem_transdown_num_heads,
                transdown_sa_depth=self.pqstem_transdown_sa_depth,
                transdown_k=self.pqstem_transdown_k,
                transdown_use_attn=self.pqstem_transdown_use_attn,
                cls_token=self.mode.do_cls,
            )

        self.student["point_encoder"] = point_encoder()
        self.teacher["point_encoder"] = point_encoder()
        self.ema = EMA(self.student, self.teacher)
        self.transup_head = (
            PQStemTransUpHead(
                embed_dim=self.embed_dim,
                num_seed=self.pqstem_transup_num_seed,
                up_factors=self.pqstem_transup_up_factors,
                n_knn=self.pqstem_transup_n_knn,
                radius=self.pqstem_transup_radius,
                interpolate=self.pqstem_transup_interpolate,
                attn_channel=self.pqstem_transup_attn_channel,
            )
            if self.pqstem_enable_transup_reconstruction
            else None
        )
        self._pqstem_last_points: torch.Tensor | None = None
        self._pqstem_transup_token_cache: tuple[torch.Tensor, torch.Tensor] | None = (
            None
        )

    def _packed_mask_counts(self, num_patches: int) -> list[int]:
        return [
            round(self.sparse_visible_mask_ratio * num_patches),
            round(self.sparse_masked_mask_ratio * num_patches),
            round(self.geometric_halfspace_mask_ratio * num_patches),
            round(self.random_mask_ratio * num_patches),
        ]

    @torch.no_grad()
    def _generate_random_patch_mask(
        self,
        centers: torch.Tensor,
        num_masks: int,
    ) -> torch.Tensor:
        B, P, _ = centers.shape
        scores = torch.rand(B, P, device=centers.device)
        _, mask_indices = scores.topk(num_masks, dim=-1)
        mask = torch.zeros(B, P, dtype=torch.bool, device=centers.device)
        mask.scatter_(-1, mask_indices, True)
        return mask

    def _generate_packed_masks(
        self,
        masked_attn_weights: list[torch.Tensor],
        masked_centers: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        P = masked_centers.shape[1]
        num_masks_sparse_visible = round(self.sparse_visible_mask_ratio * P)
        num_masks_sparse_masked = round(self.sparse_masked_mask_ratio * P)
        num_masks_geo_halfspace = round(self.geometric_halfspace_mask_ratio * P)
        num_masks_random = round(self.random_mask_ratio * P)

        num_heads = self.student.point_encoder.encoder.config.num_heads
        step = self.global_step
        head_a = step % num_heads
        head_b = (step + num_heads // 2) % num_heads

        sparse_visible = self._generate_sparse_only_mask(
            masked_attn_weights,
            num_masks_sparse_visible,
            select_visible=True,
            head_index=head_a,
        )
        sparse_masked = self._generate_sparse_only_mask(
            masked_attn_weights,
            num_masks_sparse_masked,
            select_visible=False,
            head_index=head_b,
        )
        geo_halfspace = self._generate_halfspace_mask(
            masked_centers,
            num_masks_geo_halfspace,
        )
        random_mask = self._generate_random_patch_mask(
            masked_centers,
            num_masks_random,
        )

        all_masks = [
            sparse_visible,
            sparse_masked,
            geo_halfspace,
            random_mask,
        ]
        mask_components = {
            "cls_to_patch": self._compute_cls_to_patch_attention(
                masked_attn_weights,
                head_index=head_a,
            ),
            "cls_to_patch_b": self._compute_cls_to_patch_attention(
                masked_attn_weights,
                head_index=head_b,
            ),
            "block_mask": torch.zeros_like(sparse_visible),
            "sparse_mask": ~sparse_visible,
            "select_visible": True,
            "head_a": head_a,
            "head_b": head_b,
            "path_masks": all_masks,
            "path_names": [
                "sparse_visible",
                "sparse_masked",
                "geometric_halfspace",
                "random",
            ],
        }

        return all_masks, mask_components

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        if "global_crops" in batch or "local_crops" in batch:
            raise ValueError(
                "PQStem STAMPA pretraining expects target-only batches with "
                "`points` and optional `num_points`; global/local crop batches "
                "are intentionally disabled."
            )
        self._pqstem_last_points = None
        self._pqstem_transup_token_cache = None
        outputs = super().training_step(batch, batch_idx)
        points = self._pqstem_last_points
        transup_token_cache = self._pqstem_transup_token_cache
        should_compute_transup_cd = self._should_compute_transup_cd()
        self._pqstem_last_points = None
        self._pqstem_transup_token_cache = None

        if should_compute_transup_cd and points is not None:
            transup_cd_loss, pred_pcds = self._transup_reconstruction_loss(
                points,
                token_cache=transup_token_cache,
            )
            outputs["loss"] = (
                outputs["loss"] + self.pqstem_transup_cd_weight * transup_cd_loss
            )
            outputs["transup_cd_loss"] = transup_cd_loss
            outputs["transup_reconstructions"] = [pcd.detach() for pcd in pred_pcds]
        else:
            outputs["transup_cd_loss"] = None
            outputs["transup_reconstructions"] = None

        return outputs

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        if self.transup_head is None:
            return {}
        points = self._target_batch_points(batch, apply_augmentation=False)
        token_centers, token_features = self._student_stem_tokens_detached(points)
        transup_cd_loss, pred_pcds = self.transup_head.reconstruction_loss(
            points.detach(),
            token_centers,
            token_features,
        )
        transup_last_cd2_loss = self.transup_head.chamfer_loss(
            pred_pcds[-1],
            points.detach()[..., :3].contiguous(),
            norm=2,
        )
        num_eval_masks = round(self.random_mask_ratio * token_centers.shape[1])
        eval_mask = self._generate_random_patch_mask(token_centers, num_eval_masks)
        self.log(
            "val/transup_cd_loss",
            transup_cd_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=points.shape[0],
        )
        self.log(
            "val/transup_last_cd2_loss",
            transup_last_cd2_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=points.shape[0],
        )
        return {
            "transup_cd_loss": transup_cd_loss,
            "transup_last_cd2_loss": transup_last_cd2_loss,
            "transup_reconstructions": [pcd.detach() for pcd in pred_pcds],
            "gt_points": points.detach(),
            "eval_token_centers": token_centers.detach(),
            "eval_mask": eval_mask.detach(),
        }

    def test_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def _should_compute_transup_cd(self) -> bool:
        return (
            self.transup_head is not None
            and self.pqstem_transup_cd_weight > 0.0
            and self.global_step % self.pqstem_transup_cd_every_n_steps == 0
        )

    def _after_student_patch_embedding(
        self,
        tokens: Any,
        point_encoder: Any,
    ) -> None:
        if not self._should_compute_transup_cd() or not isinstance(
            point_encoder, PQStemPointEncoder
        ):
            return
        self._pqstem_transup_token_cache = self._encode_detached_student_tokens(
            point_encoder,
            tokens,
        )

    def _encode_detached_student_tokens(
        self,
        point_encoder: PQStemPointEncoder,
        tokens: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.centers is None:
            raise RuntimeError("PQStem TransUp reconstruction requires token centers.")

        was_training = point_encoder.training
        point_encoder.eval()
        try:
            with torch.no_grad():
                token_centers = tokens.centers.detach()
                encoder_out = point_encoder.transformer_encoder_forward(
                    tokens.embeddings.detach(),
                    tokens.pos_embeddings.detach(),
                    token_centers=token_centers,
                )
        finally:
            point_encoder.train(was_training)

        return token_centers, encoder_out.patch_features.detach()

    def _student_stem_tokens_detached(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_encoder: PQStemPointEncoder = self.student.point_encoder
        was_training = point_encoder.training
        point_encoder.eval()
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
        finally:
            point_encoder.train(was_training)
        return tokens.centers.detach(), encoder_out.patch_features.detach()

    def _transup_reconstruction_loss(
        self,
        points: torch.Tensor,
        token_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.transup_head is None:
            raise RuntimeError("TransUp reconstruction is disabled.")
        if token_cache is None:
            token_centers, token_features = self._student_stem_tokens_detached(points)
        else:
            token_centers, token_features = token_cache
        return self.transup_head.reconstruction_loss(
            points.detach(),
            token_centers,
            token_features,
        )

    def _target_batch_points(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        apply_augmentation: bool,
    ) -> torch.Tensor:
        if "global_crops" in batch or "local_crops" in batch:
            raise ValueError(
                "PQStem STAMPA reconstruction expects target-only batches with "
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
        self, patch_points: PatchPoints, patchify: Any
    ) -> PQStemPatches:
        del patchify
        crops = self._preprocess_target_points(
            patch_points.points,
            patch_points.num_points,
            apply_augmentation=True,
        )
        patch_points.points = crops
        self._pqstem_last_points = crops

        centers = self.teacher.point_encoder.compute_centers(crops)
        return PQStemPatches(points=crops, centers=[centers])

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        transup_cd_loss = outputs.get("transup_cd_loss")
        if transup_cd_loss is not None:
            self.log(
                "train/transup_cd_loss",
                transup_cd_loss,
                on_step=True,
                prog_bar=True,
            )

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
        state_dict = module.state_dict()
        if cpu:
            return {
                f"{prefix}{key}": value.detach().cpu()
                for key, value in state_dict.items()
            }
        return {f"{prefix}{key}": value for key, value in state_dict.items()}

    def export_pq_stem(
        self,
        path: str | Path,
        branch: str = "teacher",
        prefix: str | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.pq_stem_state_dict(branch=branch, prefix=prefix), path)

    def pqdt_component_state_dict(
        self,
        branch: str = "teacher",
        cpu: bool = True,
        include_up_layers: bool = True,
    ) -> dict[str, torch.Tensor]:
        state_dict = self.pq_stem_state_dict(branch=branch, cpu=cpu)
        if include_up_layers:
            if self.transup_head is None:
                raise RuntimeError(
                    "Cannot export up_layers because TransUp is disabled."
                )
            state_dict.update(self.transup_head.pqdt_up_layers_state_dict(cpu=cpu))
        return state_dict

    def export_pqdt_components(
        self,
        path: str | Path,
        branch: str = "teacher",
        include_up_layers: bool = True,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.pqdt_component_state_dict(
                branch=branch,
                include_up_layers=include_up_layers,
            ),
            path,
        )
