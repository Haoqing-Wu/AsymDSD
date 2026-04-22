"""AsymDSD with dual-view asymmetric masking and cross-prediction.

Two differently-masked views of the same input go through the shared student
encoder.  Both are distilled against the same EMA teacher, plus a cross-
prediction term where each view's encoder output is used to predict the
other view's masked patch targets.

Typical configuration:
    View 1 (mask_generator):       RandomPatchMasking  (65 %, sparse global)
    View 2 (mask_generator_view2): BlockPatchMasking   (50 %, dense local)
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.common_types import FloatMayCall, OptionalTensor
from ..components.scheduling import Schedule
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import MultiPatches, PatchPoints
from ..layers.tokenization import Tokens
from ..loggers import get_default_logger
from ..loss import PatchLoss
from .asymdsd import AsymDSD, ClsPredictor, TraingingMode
from .point_encoder import PointEncoder

logger = get_default_logger()


class DualViewAsymDSD(AsymDSD):
    """AsymDSD with dual-view asymmetric masking and cross-prediction.

    When ``mask_generator_view2`` is provided the model generates two masks
    per sample (view 1 from ``mask_generator``, view 2 from
    ``mask_generator_view2``).  The student encoder runs twice (once per
    view) and a cross-prediction term is added: each encoder output predicts
    the *other* view's masked patches via the predictor, distilled against
    the shared teacher targets.
    """

    @init_lazy_defaults
    def __init__(
        self,
        # --- dual-view specific ---
        mask_generator_view2: MaskGenerator | None = None,
        cross_prediction_weight: FloatMayCall = 0.0,
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

        self.mask_generator_view2 = mask_generator_view2
        self._cross_prediction_weight = cross_prediction_weight

        if mask_generator_view2 is not None:
            if not self.mode.do_mask:
                raise ValueError(
                    "mask_generator_view2 requires training_mode that enables masking."
                )
            if not self.do_predict:
                raise ValueError(
                    "Dual-view cross-prediction requires a predictor "
                    "(set predictor_config)."
                )
            if self.multi_mask != 1:
                raise ValueError(
                    f"Dual-view mode requires multi_mask=1 on mask_generator, "
                    f"got {self.multi_mask}. Dual-view already doubles the views."
                )
            if mask_generator_view2.multi_mask != 1:
                raise ValueError(
                    f"Dual-view mode requires multi_mask=1 on mask_generator_view2, "
                    f"got {mask_generator_view2.multi_mask}."
                )

        self.cross_patch_loss = PatchLoss()

        # Inject into the parent's schedules dict so Scheduler picks it up
        self.schedules["cross_prediction_weight"] = cross_prediction_weight

    @property
    def dual_view(self) -> bool:
        return self.mask_generator_view2 is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_multi_patches(
        multi_patches: MultiPatches,
        indices: torch.Tensor,
    ) -> MultiPatches:
        """Select a subset of samples from a MultiPatches batch."""
        return MultiPatches(
            patches=multi_patches.patches[indices],
            patches_idx=[idx[indices] for idx in multi_patches.patches_idx],
            centers=[c[indices] for c in multi_patches.centers],
        )

    # ------------------------------------------------------------------
    # Dual-view student forward
    # ------------------------------------------------------------------

    def _forward_student_dual_view(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask_v1: torch.Tensor,
        mask_v2: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        """Run student encoder + predictor for both views and cross-prediction.

        Each view sees different visible tokens (determined by its mask).
        Self-prediction: each view predicts its own masked positions.
        Cross-prediction: each view predicts the *other* view's masked positions.
        """
        point_encoder: PointEncoder = self.student.point_encoder  # type: ignore

        out_dict: dict[str, OptionalTensor] = {
            "x_cls_logits_v1": None,
            "x_cls_logits_v2": None,
            "x_cls_embedding_v1": None,
            "x_cls_embedding_v2": None,
            "x_patch_logits_v1": None,
            "x_patch_logits_v2": None,
            "x_patch_embedding_v1": None,
            "x_patch_embedding_v2": None,
            "x_cross_patch_logits_1to2": None,
            "x_cross_patch_logits_2to1": None,
        }

        def encode_view(
            x_full: torch.Tensor,
            pos_enc_full: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, OptionalTensor, torch.Tensor, torch.Tensor]:
            """Encode one view: split visible/masked, run encoder.

            Returns (x_context, x_cls, pos_enc_visible, pos_enc_masked).
            """
            inv_mask = ~mask
            x_visible = gather_masked(x_full, inv_mask)
            pos_enc_visible = gather_masked(pos_enc_full, inv_mask)
            pos_enc_masked = gather_masked(pos_enc_full, mask)

            pe_out = point_encoder.transformer_encoder_forward(
                x_visible, pos_enc_visible
            )

            if self.mode.do_cls:
                x_cls = pe_out.cls_features
                x_patch = pe_out.patch_features
                x_context = torch.cat((x_cls.unsqueeze(1), x_patch), dim=1)  # type: ignore
            else:
                x_cls = None
                x_context = pe_out.patch_features

            return x_context, x_cls, pos_enc_visible, pos_enc_masked

        def predict_patches(
            x_context: torch.Tensor,
            x_cls: OptionalTensor,
            pos_enc_visible: torch.Tensor,
            pos_enc_masked: torch.Tensor,
        ) -> torch.Tensor:
            """Run predictor to produce patch predictions at masked positions."""
            num_masks = pos_enc_masked.shape[1]
            mask_tokens = self.mask_token.expand(x_context.shape[0], num_masks, -1)

            if self.decoder_style_predictor:
                x_patch = self.student.predictor(  # type: ignore
                    mask_tokens,
                    pos_enc_masked,
                    memory=x_context,
                )[0]
            else:
                x_pred = torch.cat((x_context, mask_tokens), dim=1)
                pos_enc_tuple = (pos_enc_visible, pos_enc_masked)
                if self.mode.do_cls:
                    x_cls_expanded = x_cls.unsqueeze(1)  # type: ignore
                    pos_enc_tuple = (x_cls_expanded,) + pos_enc_tuple
                pos_enc_full = torch.cat(pos_enc_tuple, dim=1)
                x_pred = self.student.predictor(x_pred, pos_enc_full)[0]  # type: ignore
                x_patch = x_pred[:, -num_masks:]

            return x_patch

        # --- Encode both views ---
        ctx_v1, cls_v1, pos_vis_v1, pos_msk_v1 = encode_view(x, pos_enc, mask_v1)
        ctx_v2, cls_v2, pos_vis_v2, pos_msk_v2 = encode_view(x, pos_enc, mask_v2)

        # --- Self-prediction (each view predicts its own masked patches) ---
        x_patch_v1 = predict_patches(ctx_v1, cls_v1, pos_vis_v1, pos_msk_v1)
        x_patch_v2 = predict_patches(ctx_v2, cls_v2, pos_vis_v2, pos_msk_v2)

        # --- Cross-prediction (each view predicts the other's masked patches) ---
        x_cross_1to2 = predict_patches(ctx_v1, cls_v1, pos_vis_v1, pos_msk_v2)
        x_cross_2to1 = predict_patches(ctx_v2, cls_v2, pos_vis_v2, pos_msk_v1)

        # --- Project patch predictions ---
        def project_patches(x_patch: torch.Tensor) -> torch.Tensor:
            proj: ProjectionOutput = self.student.patch_projection_head(  # type: ignore
                x_patch, return_x_norm=True
            )
            return proj.x.flatten(0, 1)

        out_dict["x_patch_logits_v1"] = project_patches(x_patch_v1)
        out_dict["x_patch_logits_v2"] = project_patches(x_patch_v2)
        out_dict["x_cross_patch_logits_1to2"] = project_patches(x_cross_1to2)
        out_dict["x_cross_patch_logits_2to1"] = project_patches(x_cross_2to1)

        if return_embeddings:
            out_dict["x_patch_embedding_v1"] = x_patch_v1.flatten(0, 1)
            out_dict["x_patch_embedding_v2"] = x_patch_v2.flatten(0, 1)

        # --- CLS logits ---
        if self.mode.do_cls:
            out_dict["x_cls_logits_v1"] = self.student.cls_projection_head(cls_v1)[0]  # type: ignore
            out_dict["x_cls_logits_v2"] = self.student.cls_projection_head(cls_v2)[0]  # type: ignore
            if return_embeddings:
                out_dict["x_cls_embedding_v1"] = cls_v1
                out_dict["x_cls_embedding_v2"] = cls_v2

        return out_dict

    # ------------------------------------------------------------------
    # Dual-view teacher forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_teacher_dual_view(
        self,
        multi_patches: MultiPatches,
        mask_v1: torch.Tensor,
        mask_v2: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, OptionalTensor]:
        """Run teacher on full unmasked input, gather targets per view mask."""
        point_encoder: PointEncoder = self.teacher.point_encoder  # type: ignore

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings

        out_dict: dict[str, OptionalTensor] = {
            "x_cls_logits": None,
            "x_cls_embedding": None,
            "x_patch_logits_v1": None,
            "x_patch_logits_v2": None,
        }

        pe_out = point_encoder.transformer_encoder_forward(x, pos_enc)
        x_cls = pe_out.cls_features
        x_patch = pe_out.patch_features

        # --- CLS ---
        if self.mode.do_cls:
            if return_embeddings:
                out_dict["x_cls_embedding"] = x_cls
            x_cls_proj: torch.Tensor = self.teacher.cls_projection_head(x_cls)[0]  # type: ignore
            centering_momentum = self.scheduler.value["cls_centering_momentum"]
            out_dict["x_cls_logits"] = self.teacher.cls_centering(  # type: ignore
                x_cls_proj.unsqueeze(1), momentum=centering_momentum
            ).squeeze(1)

        # --- Patch targets (gather at each view's mask positions) ---
        if self.patch_instance_norm:
            x_patch = torch.nn.functional.instance_norm(x_patch.mT).mT

        x_patch_proj: torch.Tensor = self.teacher.patch_projection_head(x_patch)[0]  # type: ignore
        centering_momentum = self.scheduler.value["patch_centering_momentum"]
        x_patch_proj = self.teacher.patch_centering(  # type: ignore
            x_patch_proj, momentum=centering_momentum
        )

        out_dict["x_patch_logits_v1"] = gather_masked(x_patch_proj, mask_v1).flatten(
            0, 1
        )
        out_dict["x_patch_logits_v2"] = gather_masked(x_patch_proj, mask_v2).flatten(
            0, 1
        )

        return out_dict

    # ------------------------------------------------------------------
    # training_step override
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        if not self.dual_view:
            return super().training_step(batch, batch_idx)

        # ---- Extract and patchify global crops ----
        global_crops_dict = batch.get("global_crops") or batch

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

        # ---- Generate two masks ----
        # Use first crop per sample (same as MASK mode).
        indices_masked_crops = torch.arange(
            0, B * C, step=C, device=global_centers.device
        )
        centers_for_mask = global_centers[indices_masked_crops]

        mask_v1, _ = self.mask_generator(centers_for_mask)  # (B, P)
        mask_v2, _ = self.mask_generator_view2(centers_for_mask)  # type: ignore  # (B, P)

        # ---- Sub-select multi_patches for masked crops ----
        masked_multi_patches = self._index_multi_patches(
            multi_patches, indices_masked_crops
        )

        # ---- Teacher forward (once, full unmasked input) ----
        targets = self._forward_teacher_dual_view(
            masked_multi_patches,
            mask_v1,
            mask_v2,
            return_embeddings=self.do_regression,
        )

        # ---- Student: embed once, then dual-view forward ----
        point_encoder: PointEncoder = self.student.point_encoder  # type: ignore
        tokens: Tokens = point_encoder.patch_embedding(masked_multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings

        preds = self._forward_student_dual_view(
            x, pos_enc, mask_v1, mask_v2, return_embeddings=True
        )

        # ---- Compute losses ----
        loss = 0.0
        total_terms = 0
        patch_loss_v1 = patch_loss_v2 = cross_loss = cls_loss = None
        koleo_loss = me_max = None

        # --- Self-prediction patch loss (averaged over both views) ---
        if not self.disable_projection:
            patch_teacher_temp = self.scheduler.value["patch_teacher_temp"]
            patch_student_temp = self.scheduler.value["patch_student_temp"]

            patch_loss_v1 = checkpoint(
                self.patch_loss,
                preds["x_patch_logits_v1"],
                targets["x_patch_logits_v1"],
                patch_teacher_temp,
                patch_student_temp,
            )
            patch_loss_v2 = checkpoint(
                self.patch_loss,
                preds["x_patch_logits_v2"],
                targets["x_patch_logits_v2"],
                patch_teacher_temp,
                patch_student_temp,
            )
            patch_loss_avg = (patch_loss_v1 + patch_loss_v2) / 2  # type: ignore
            loss = loss + patch_loss_avg
            total_terms += 1

        # --- Cross-prediction patch loss ---
        cross_weight: float = self.scheduler.value["cross_prediction_weight"]

        if cross_weight > 0:
            cross_loss_1to2 = checkpoint(
                self.cross_patch_loss,
                preds["x_cross_patch_logits_1to2"],
                targets["x_patch_logits_v2"],
                patch_teacher_temp,
                patch_student_temp,
            )
            cross_loss_2to1 = checkpoint(
                self.cross_patch_loss,
                preds["x_cross_patch_logits_2to1"],
                targets["x_patch_logits_v1"],
                patch_teacher_temp,
                patch_student_temp,
            )
            cross_loss = (cross_loss_1to2 + cross_loss_2to1) / 2  # type: ignore
            loss = loss + cross_weight * cross_loss
            total_terms += cross_weight

        # --- CLS loss (both views distilled against teacher) ---
        if self.mode.do_cls:
            cls_targets: torch.Tensor = targets["x_cls_logits"]  # type: ignore
            cls_target_probs = self.cls_loss.compute_target_probs(
                cls_targets.unsqueeze(1),  # (B, 1, D)
                teacher_temp=self.scheduler.value["cls_teacher_temp"],
            )
            student_temp = self.scheduler.value["cls_student_temp"]

            # View 1 CLS
            cls_loss_v1 = self.cls_loss(
                preds["x_cls_logits_v1"].unsqueeze(1),  # type: ignore
                cls_target_probs,
                student_temp=student_temp,
            )
            # View 2 CLS
            cls_loss_v2 = self.cls_loss(
                preds["x_cls_logits_v2"].unsqueeze(1),  # type: ignore
                cls_target_probs,
                student_temp=student_temp,
            )
            cls_loss = (cls_loss_v1 + cls_loss_v2) / 2

            # --- Local crops CLS (reuse parent's local crop pipeline) ---
            local_crops_dict = batch.get("local_crops")
            cls_terms = 1

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
                # (B*C_l, D) -> (B, C_l, D) to match cls_target_probs (B, 1, D)
                local_cls_logits = local_cls_logits.unflatten(0, (B, -1))
                cls_loss = cls_loss + self.cls_loss(
                    local_cls_logits,
                    cls_target_probs,
                    student_temp=student_temp,
                )
                cls_terms += 1

            cls_loss = cls_loss / cls_terms
            loss = loss + cls_loss
            total_terms += 1

            # --- KoLeo ---
            if self.do_koleo:
                cls_emb_v1: torch.Tensor = preds["x_cls_embedding_v1"]  # type: ignore
                cls_emb_v2: torch.Tensor = preds["x_cls_embedding_v2"]  # type: ignore
                cls_emb = torch.stack([cls_emb_v1, cls_emb_v2], dim=1)  # (B, 2, D)
                koleo_loss = self.koleo_loss(cls_emb)
                loss = loss + self.koleo_loss_weight * koleo_loss
                total_terms += self.koleo_loss_weight

            # --- ME-max ---
            if self.me_max_weight > 0.0:
                cls_student_temp = self.scheduler.value["cls_student_temp"]
                cls_logits_v1: torch.Tensor = preds["x_cls_logits_v1"]  # type: ignore
                cls_logits_v2: torch.Tensor = preds["x_cls_logits_v2"]  # type: ignore
                cls_preds_combined = torch.stack(
                    [cls_logits_v1, cls_logits_v2], dim=1
                )
                me_max = self.me_max_loss(cls_preds_combined / cls_student_temp)
                loss = loss + self.me_max_weight * me_max
                total_terms += self.me_max_weight

        loss = loss / total_terms  # type: ignore

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "cls_preds": preds.get("x_cls_logits_v1"),
            "cls_targets": targets.get("x_cls_logits"),
            "patch_loss": (
                (patch_loss_v1 + patch_loss_v2) / 2  # type: ignore
                if patch_loss_v1 is not None
                else None
            ),
            "patch_preds": preds.get("x_patch_logits_v1"),
            "patch_targets": targets.get("x_patch_logits_v1"),
            "patch_loss_v1": patch_loss_v1,
            "patch_loss_v2": patch_loss_v2,
            "cross_loss": cross_loss,
            "me_max": me_max,
            "koleo_loss": koleo_loss,
            "regression_loss": None,
            "classification_loss": None,
            "centers": global_centers,
        }

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self.dual_view:
            return super().on_train_batch_end(outputs, batch, batch_idx)

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.ema.update_parameters(self.scheduler.value["ema_decay"])
            self.scheduler.step()

        log_conditions = {
            "train/loss": True,
            "train/cls_loss": self.mode.do_cls,
            "train/patch_loss": self.mode.do_mask and not self.disable_projection,
            "train/patch_loss_v1": self.mode.do_mask and not self.disable_projection,
            "train/patch_loss_v2": self.mode.do_mask and not self.disable_projection,
            "train/cross_loss": self.dual_view,
            "train/me_max": self.me_max_weight > 0.0,
            "train/koleo_loss": self.do_koleo,
        }

        for log_key, condition in log_conditions.items():
            value = outputs[log_key.split("/")[-1]]
            if condition and value is not None:
                self.log(
                    log_key,
                    value,
                    on_step=True,
                    on_epoch=log_key == "train/loss",
                    prog_bar=log_key
                    in [
                        "train/loss",
                        "train/cls_loss",
                        "train/patch_loss",
                        "train/cross_loss",
                    ],
                )

        self._log_schedules()
