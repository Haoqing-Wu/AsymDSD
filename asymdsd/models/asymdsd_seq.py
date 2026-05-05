"""SequentialAsymDSD: fully sequential student with state-propagated glimpses.

The student never sees the full point cloud. Patches are partitioned into T
complementary groups. At each step the student encoder processes [state, patches_t],
producing a CLS output and updated state tokens. At the final step, the predictor
reconstructs patch features for all previously-seen patches using accumulated context.

Teacher sees full cloud and produces CLS + patch targets (unchanged from AsymDSD).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asymdsd.data import PointCloudDataModule

from functools import partial

import lightning as L
import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import *
from ..components.checkpointing_utils import load_module_from_checkpoint
from ..components.common_types import FloatMayCall
from ..components.scheduling import Schedule
from ..components.utils import (
    gather_masked,
    init_lazy_defaults,
    lengths_to_mask,
    sequentialize_transform,
)
from ..defaults import *
from ..layers import *
from ..layers.patchify import MultiPatches, PatchPoints
from ..layers.tokenization import Tokens, TrainableToken
from ..loggers import get_default_logger
from ..loss import (
    ClsLoss,
    KoLeoLoss,
    MeanEntropyLoss,
    PatchLoss,
)
from .asymdsd import TraingingMode
from .point_encoder import PointEncoder

logger = get_default_logger()


class SequentialAsymDSD(L.LightningModule):
    """Fully sequential self-distillation for 3D point clouds.

    The student processes T sequential glimpses with state token propagation.
    The teacher processes the full cloud. CLS loss at every step, patch
    reconstruction loss at the final step.
    """

    DEFAULT_BATCH_SIZE = 128

    @init_lazy_defaults
    def __init__(
        self,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        steps_per_epoch: int | None = None,
        optimizer: OptimizerSpec = DEFAULT_OPTIMIZER,
        patchify: MultiPointPatchify | None = None,
        norm_transform: NormalizationTransform = DEFAULT_NORM_TRANSFORM,
        aug_transform: AugmentationTransform = DEFAULT_AUG_TRANSFORM,
        patch_embedding: PatchEmbeddingConfig = DEFAULT_PATCH_EMBEDDING_CFG,
        encoder_config: TransformerEncoderConfig = DEFAULT_TRANSFORMER_ENC_CONFIG,
        predictor_config: TransformerEncoderConfig
        | TransformerDecoderConfig
        | None = DEFAULT_TRANSFORMER_PROJ_CONFIG,
        projection_head_config: ProjectionHeadConfig = DEFAULT_PROJECTION_HEAD_CONFIG,
        num_point_features: int = 3,
        batch_size: int = DEFAULT_BATCH_SIZE,
        init_weight_scale: float = 0.02,
        cls_teacher_temp: FloatMayCall = 0.05,
        cls_student_temp: FloatMayCall = 0.1,
        patch_teacher_temp: FloatMayCall = 0.05,
        patch_student_temp: FloatMayCall = 0.1,
        cls_centering_momentum: FloatMayCall | None = None,
        patch_centering_momentum: FloatMayCall | None = None,
        cls_centering_power_law_tau: float | None = None,
        patch_centering_power_law_tau: float | None = None,
        ema_decay: FloatMayCall = DEFAULT_EMA_DECAY,
        me_max_weight: float | None = None,
        koleo_loss_weight: float | None = None,
        num_sequential_steps: int = 4,
        num_state_tokens: int = 8,
        shared_projection_head: bool = False,
        patch_instance_norm: bool = False,
        gradient_checkpointing: bool = False,
        attn_bias_scale: FloatMayCall = 1.0,
        modules_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs if max_epochs and max_epochs > 0 else None
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be specified.")

        self.steps_per_epoch = steps_per_epoch
        self.num_point_features = num_point_features
        self.batch_size = batch_size
        self.embed_dim = encoder_config.embed_dim
        self.n_prototypes = projection_head_config.out_dim

        self.norm_transform = norm_transform or IdentityMultiArg()
        self.aug_transform: nn.Module = (
            sequentialize_transform(aug_transform)
            if aug_transform
            else IdentityMultiArg()
        )

        self.patchify = patchify or ToMultiPatches()
        self.optimizer_spec = optimizer

        self.num_sequential_steps = num_sequential_steps
        self.num_state_tokens = num_state_tokens
        self.init_weight_scale = init_weight_scale
        self.patch_instance_norm = patch_instance_norm

        # Compatibility with callbacks that check these attributes
        self.mode = TraingingMode.CLS_MASK
        self.disable_projection = False

        projection_head_config.in_dim = self.embed_dim
        projection_head = partial(ProjectionHead, **vars(projection_head_config))

        # State tokens: learnable initial state for sequential processing
        self.state_tokens = nn.Parameter(
            torch.empty(1, num_state_tokens, self.embed_dim)
        )

        # Student and teacher encoders
        patch_embedding.position_embedding.embed_dim = self.embed_dim
        patch_embedding.point_embedding.embed_dim = self.embed_dim

        def point_encoder():
            return PointEncoder(
                patchify=self.patchify,
                cls_token=TrainableToken(self.embed_dim),
                patch_embedding=patch_embedding.instantiate(),
                encoder=encoder_config.instantiate(),
            )

        self.student = nn.ModuleDict({"point_encoder": point_encoder()})
        self.teacher = nn.ModuleDict({"point_encoder": point_encoder()})

        # CLS projection heads
        self.student["cls_projection_head"] = projection_head()
        self.teacher["cls_projection_head"] = projection_head()

        # Patch projection heads
        if shared_projection_head:
            self.student["patch_projection_head"] = self.student["cls_projection_head"]
            self.teacher["patch_projection_head"] = self.teacher["cls_projection_head"]
        else:
            self.student["patch_projection_head"] = projection_head()
            self.teacher["patch_projection_head"] = projection_head()

        # Centering for teacher outputs
        self.teacher["cls_centering"] = (
            Centering(self.n_prototypes, cls_centering_power_law_tau)
            if cls_centering_momentum is not None
            else IdentityMultiArg()
        )
        self.teacher["patch_centering"] = (
            Centering(self.n_prototypes, patch_centering_power_law_tau)
            if patch_centering_momentum is not None
            else IdentityMultiArg()
        )

        # Predictor for patch reconstruction at final step
        self.do_predict = predictor_config is not None
        if self.do_predict:
            student_predictor = predictor_config.instantiate()  # type: ignore
            self.student["predictor"] = student_predictor

            enc_dim = self.embed_dim
            pred_dim = predictor_config.embed_dim  # type: ignore
            if pred_dim != enc_dim:
                self.student["predictor"] = ProjectionWrapper(
                    student_predictor,
                    enc_dim,
                    pred_dim,
                )

        self.ema = EMA(self.student, self.teacher)

        if gradient_checkpointing:
            self.student["point_encoder"].enable_gradient_checkpointing()
            if self.do_predict:
                student_predictor.enable_gradient_checkpointing()

        # Losses
        self.patch_loss = PatchLoss()
        self.cls_loss = ClsLoss()
        self.koleo_loss = KoLeoLoss(input_is_normalized=False)
        self.me_max_loss = MeanEntropyLoss(dim=self.n_prototypes)
        self.me_max_weight = me_max_weight or 0.0
        self.koleo_loss_weight = koleo_loss_weight or 0.0
        self.do_koleo = self.koleo_loss_weight > 0.0

        self.schedules = {
            "lr": self.optimizer_spec.lr,
            "wd": self.optimizer_spec.wd,
            "ema_decay": ema_decay,
            "cls_teacher_temp": cls_teacher_temp,
            "cls_student_temp": cls_student_temp,
            "patch_teacher_temp": patch_teacher_temp,
            "patch_student_temp": patch_student_temp,
            "patch_centering_momentum": patch_centering_momentum,
            "cls_centering_momentum": cls_centering_momentum,
            "attn_bias_scale": attn_bias_scale,
        }

        self.modules_ckpt_path = modules_ckpt_path
        self.loaded_from_checkpoint = False
        self.validation_epoch = 0

    def init_weights(self):
        std = self.init_weight_scale
        nn.init.trunc_normal_(self.teacher.point_encoder.cls_token, std=std)
        nn.init.trunc_normal_(self.student.point_encoder.cls_token, std=std)
        nn.init.trunc_normal_(self.state_tokens, std=std)

        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def setup(self, stage: str | None = None):
        if not self.steps_per_epoch:
            try:
                datamodule: "PointCloudDataModule" = self.trainer.datamodule  # type: ignore
                if datamodule.len_train_dataset is None:
                    raise AttributeError
                self.steps_per_epoch = datamodule.len_train_dataset // self.batch_size
            except AttributeError:
                raise ValueError(
                    "steps_per_epoch must be specified if the length of the train dataset is not known."
                )

        real_schedule: list[Schedule] = [
            s for s in self.schedules.values() if isinstance(s, Schedule)
        ]

        for schedule in real_schedule:
            max_epochs = self.max_epochs or (self.max_steps / self.steps_per_epoch)  # type: ignore
            schedule.set_default_max_epochs(max_epochs)  # type: ignore
            schedule.set_steps_per_epoch(self.steps_per_epoch)

        self.schedules.pop("lr", None)
        self.schedules.pop("wd", None)

        self.scheduler = Scheduler(**self.schedules)

        if self.modules_ckpt_path is not None:
            load_module_from_checkpoint(
                self.modules_ckpt_path,
                self,
                device=self.device,
                strict=False,
            )
            logger.info(f"Loaded modules from checkpoint {self.modules_ckpt_path}.")

        self.validation_epoch = 0

    # -------------------------------------------------------------------------
    # Mask generation
    # -------------------------------------------------------------------------

    def _generate_complementary_masks(
        self, num_patches: int, batch_size: int
    ) -> list[torch.Tensor]:
        """Partition patches into T complementary groups.

        Returns list of T boolean masks (B, P), True = visible at step t.
        Each patch appears in exactly one step.
        """
        T = self.num_sequential_steps
        device = self.device

        noise = torch.rand(batch_size, num_patches, device=device)
        perm = noise.argsort(dim=1)

        patches_per_step = num_patches // T

        masks = []
        for t in range(T):
            mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
            start = t * patches_per_step
            end = (t + 1) * patches_per_step if t < T - 1 else num_patches
            indices = perm[:, start:end]
            mask.scatter_(1, indices, True)
            masks.append(mask)

        return masks

    # -------------------------------------------------------------------------
    # Teacher forward
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def forward_teacher(self, multi_patches: MultiPatches) -> dict[str, torch.Tensor]:
        """Teacher: full unmasked cloud → CLS logits + all patch logits."""
        point_encoder: PointEncoder = self.teacher.point_encoder

        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings
        token_centers = tokens.centers

        attn_bias_scale = self.scheduler.value["attn_bias_scale"]

        pe_out = point_encoder.transformer_encoder_forward(
            x,
            pos_enc,
            token_centers=token_centers,
            attn_bias_scale=attn_bias_scale,
        )
        x_cls = pe_out.cls_features  # (B, D)
        x_patch = pe_out.patch_features  # (B, P, D)

        # CLS logits + centering
        x_cls_logits: torch.Tensor = self.teacher.cls_projection_head(x_cls)[0]
        centering_momentum = self.scheduler.value["cls_centering_momentum"]
        x_cls_logits = self.teacher.cls_centering(
            x_cls_logits.unsqueeze(1), momentum=centering_momentum
        ).squeeze(1)

        # Patch logits + centering
        if self.patch_instance_norm:
            x_patch = torch.nn.functional.instance_norm(x_patch.mT).mT

        x_patch_logits: torch.Tensor = self.teacher.patch_projection_head(x_patch)[0]
        centering_momentum = self.scheduler.value["patch_centering_momentum"]
        x_patch_logits = self.teacher.patch_centering(
            x_patch_logits, momentum=centering_momentum
        )

        return {
            "x_cls_logits": x_cls_logits,  # (B, n_prototypes)
            "x_patch_logits": x_patch_logits,  # (B, P, n_prototypes)
            "x_cls_embedding": x_cls,  # (B, D)
        }

    # -------------------------------------------------------------------------
    # Student forward (fully sequential)
    # -------------------------------------------------------------------------

    def forward_student_sequential(self, multi_patches: MultiPatches) -> dict[str, Any]:
        """Student: T sequential steps with state propagation.

        Returns:
            step_cls_logits: list[Tensor] — T tensors of (B, n_prototypes)
            step_cls_embeddings: list[Tensor] — T tensors of (B, D)
            patch_logits: (B, P_target, n_prototypes) predicted patches or None
            target_mask: (B, P) bool mask of predicted patch positions
        """
        point_encoder: PointEncoder = self.student.point_encoder

        # Embed all patches once
        tokens: Tokens = point_encoder.patch_embedding(multi_patches)
        x_all = tokens.embeddings  # (B, P, D)
        pos_enc_all = tokens.pos_embeddings  # (B, P, D)
        token_centers_all = tokens.centers  # (B, P, 3) or None

        B, P, D = x_all.shape
        T = self.num_sequential_steps
        N_state = self.num_state_tokens
        attn_bias_scale = self.scheduler.value["attn_bias_scale"]

        visibility_masks = self._generate_complementary_masks(P, B)

        state = self.state_tokens.expand(B, -1, -1)

        step_cls_logits = []
        step_cls_embeddings = []

        for t in range(T):
            vis_mask = visibility_masks[t]

            # Gather visible patches for this step
            x_visible = gather_masked(x_all, vis_mask)
            pos_enc_visible = gather_masked(pos_enc_all, vis_mask)
            centers_visible = (
                gather_masked(token_centers_all, vis_mask)
                if token_centers_all is not None
                else None
            )

            # Concatenate state tokens + visible patches
            state_pos = torch.zeros(B, N_state, D, device=self.device)
            x_input = torch.cat([state, x_visible], dim=1)
            pos_input = torch.cat([state_pos, pos_enc_visible], dim=1)

            if centers_visible is not None:
                state_centers = torch.zeros(
                    B, N_state, centers_visible.shape[-1], device=self.device
                )
                centers_input = torch.cat([state_centers, centers_visible], dim=1)
            else:
                centers_input = None

            # Encoder forward (CLS token prepended internally)
            pe_out = point_encoder.transformer_encoder_forward(
                x_input,
                pos_input,
                token_centers=centers_input,
                attn_bias_scale=attn_bias_scale,
            )

            x_cls = pe_out.cls_features  # (B, D)
            all_features = pe_out.patch_features  # (B, N_state + P_vis, D)

            # Split outputs
            new_state = all_features[:, :N_state]  # (B, N_state, D)
            patch_features = all_features[:, N_state:]  # (B, P_vis, D)

            # CLS logits at this step
            cls_logits = self.student.cls_projection_head(x_cls)[0]
            step_cls_logits.append(cls_logits)
            step_cls_embeddings.append(x_cls)

            # Detach state for next step (no BPTT)
            state = new_state.detach()

        # --- Patch reconstruction at final step ---
        patch_logits = None
        # Target = all patches NOT visible at the last step (steps 0..T-2)
        target_mask = ~visibility_masks[-1]  # (B, P)

        if self.do_predict:
            num_target = target_mask.sum(dim=1)[0].item()

            # Context: CLS + state + visible patches from final step
            x_context = torch.cat(
                [x_cls.unsqueeze(1), new_state, patch_features], dim=1
            )

            # Positional encodings for context
            pos_context = torch.cat(
                [
                    torch.zeros(B, 1 + N_state, D, device=self.device),
                    pos_enc_visible,
                ],
                dim=1,
            )

            # Mask tokens for target patches
            mask_tokens = torch.zeros(B, int(num_target), D, device=self.device)
            pos_enc_target = gather_masked(pos_enc_all, target_mask)

            # Predictor forward
            x_pred_input = torch.cat([x_context, mask_tokens], dim=1)
            pos_pred_input = torch.cat([pos_context, pos_enc_target], dim=1)

            x_pred = self.student.predictor(x_pred_input, pos_pred_input)[0]

            # Extract predicted patch features (last num_target tokens)
            predicted_patches = x_pred[:, -(int(num_target)) :]  # (B, P_target, D)

            # Project to logits
            patch_logits = self.student.patch_projection_head(predicted_patches)[0]

        return {
            "step_cls_logits": step_cls_logits,
            "step_cls_embeddings": step_cls_embeddings,
            "patch_logits": patch_logits,
            "target_mask": target_mask,
        }

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

    def _extract_patches(self, patch_points: PatchPoints) -> MultiPatches:
        crops = patch_points.points
        num_points = patch_points.num_points

        mask = (
            lengths_to_mask(num_points, crops.size(1))
            if num_points is not None
            else None
        )

        crops = self.norm_transform(crops, mask=mask)
        crops = self.aug_transform(crops)
        patch_points.points = crops

        return self.patchify(patch_points)

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch_idx: int = 0,
    ) -> dict[str, Any]:
        global_crops_dict = batch.get("global_crops") or batch

        global_patch_points = PatchPoints(
            points=global_crops_dict["points"],  # type: ignore
            num_points=global_crops_dict.get("num_points"),  # type: ignore
            patches_idx=global_crops_dict.get("patches_idx"),  # type: ignore
            centers_idx=global_crops_dict.get("centers_idx"),  # type: ignore
        )

        if "global_crops" not in batch:
            B, N, F = global_patch_points.points.shape
        else:
            B, C, N, F = global_patch_points.points.shape
            # Use first global crop only (student sees it sequentially)
            global_patch_points.points = global_patch_points.points[:, 0]
            num_points = global_patch_points.num_points
            if num_points is not None:
                global_patch_points.num_points = num_points[:, 0]
            patches_idx = global_patch_points.patches_idx
            if patches_idx is not None:
                global_patch_points.patches_idx = [x[:, 0] for x in patches_idx]
            centers_idx = global_patch_points.centers_idx
            if centers_idx is not None:
                global_patch_points.centers_idx = [x[:, 0] for x in centers_idx]

        multi_patches = self._extract_patches(global_patch_points)

        # --- Teacher forward (full cloud) ---
        targets = self.forward_teacher(multi_patches)

        # --- Student forward (sequential glimpses) ---
        preds = self.forward_student_sequential(multi_patches)

        # --- Losses ---
        loss = 0.0
        total_terms = 0
        T = self.num_sequential_steps

        # CLS loss: distill at every step
        cls_target_logits = targets["x_cls_logits"]  # (B, n_prototypes)
        cls_target_probs = self.cls_loss.compute_target_probs(
            cls_target_logits.unsqueeze(1),  # (B, 1, n_prototypes)
            teacher_temp=self.scheduler.value["cls_teacher_temp"],
        )

        cls_loss = 0.0
        for t in range(T):
            step_logits = preds["step_cls_logits"][t]  # (B, n_prototypes)
            step_loss = self.cls_loss(
                step_logits.unsqueeze(1),  # (B, 1, n_prototypes)
                cls_target_probs,
                student_temp=self.scheduler.value["cls_student_temp"],
            )
            cls_loss = cls_loss + step_loss

        cls_loss = cls_loss / T
        loss = loss + cls_loss
        total_terms += 1

        # Patch loss: predict patches from steps 0..T-2 using final step context
        patch_loss = None
        teacher_target_logits = None
        if preds["patch_logits"] is not None:
            target_mask = preds["target_mask"]  # (B, P)
            teacher_patch_logits = targets["x_patch_logits"]  # (B, P, n_prototypes)
            teacher_target_logits = gather_masked(teacher_patch_logits, target_mask)

            patch_loss = checkpoint(
                self.patch_loss,
                preds["patch_logits"].flatten(0, 1),
                teacher_target_logits.flatten(0, 1),
                self.scheduler.value["patch_teacher_temp"],
                self.scheduler.value["patch_student_temp"],
            )
            loss = loss + patch_loss
            total_terms += 1

        # KoLeo loss on final step CLS embedding
        koleo_loss = None
        if self.do_koleo:
            final_cls_embedding = preds["step_cls_embeddings"][-1]
            koleo_loss = self.koleo_loss(final_cls_embedding.unsqueeze(1))
            loss = loss + self.koleo_loss_weight * koleo_loss
            total_terms += self.koleo_loss_weight

        # ME-max loss
        me_max = None
        if self.me_max_weight > 0.0:
            final_cls_logits = preds["step_cls_logits"][-1]
            cls_student_temp = self.scheduler.value["cls_student_temp"]
            me_max = self.me_max_loss(final_cls_logits / cls_student_temp)
            loss = loss + self.me_max_weight * me_max
            total_terms += self.me_max_weight

        loss = loss / total_terms

        patch_targets = (
            teacher_target_logits.flatten(0, 1)
            if preds["patch_logits"] is not None
            else None
        )

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "patch_loss": patch_loss,
            "koleo_loss": koleo_loss,
            "me_max": me_max,
            "cls_targets": cls_target_logits,
            "patch_targets": patch_targets,
        }

    # -------------------------------------------------------------------------
    # Hooks & lifecycle (mirrors AsymDSD)
    # -------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_fit_start(self) -> None:
        if self.modules_ckpt_path is None:
            if not self.loaded_from_checkpoint:
                self.init_weights()
            self.ema.init_weights()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.ema.update_parameters(self.scheduler.value["ema_decay"])
            self.scheduler.step()

        log_conditions = {
            "train/loss": True,
            "train/cls_loss": True,
            "train/patch_loss": self.do_predict,
            "train/me_max": self.me_max_weight > 0.0,
            "train/koleo_loss": self.do_koleo,
        }

        for log_key, condition in log_conditions.items():
            if condition:
                value = outputs[log_key.split("/")[-1]]
                if value is not None:
                    self.log(
                        log_key,
                        value,
                        on_step=True,
                        on_epoch=log_key == "train/loss",
                        prog_bar=log_key
                        in ["train/loss", "train/cls_loss", "train/patch_loss"],
                    )

        self._log_schedules()

    def _log_schedules(self) -> None:
        self.log_dict(
            {k: v for k, v in self.scheduler.value.items() if v is not None},
            on_step=True,
        )

    def on_validation_end(self) -> None:
        self.validation_epoch += 1

    def on_save_checkpoint(self, checkpoint_dict: dict[str, Any]) -> None:
        checkpoint_dict["scheduler"] = self.scheduler.state_dict()
        checkpoint_dict["validation_epoch"] = self.validation_epoch

    def on_load_checkpoint(self, checkpoint_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(checkpoint_dict["scheduler"])
        self.validation_epoch = checkpoint_dict["validation_epoch"]
        self.loaded_from_checkpoint = True

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)

    def configure_optimizers(self):
        lr_multiplier = 1.0
        optimizer = self.optimizer_spec.get_optim(self.parameters(), lr_multiplier)
        lr_scheduler = self.optimizer_spec.get_lr_scheduler(optimizer)
        weight_decay_scheduler = self.optimizer_spec.get_wd_scheduler(optimizer)

        optimizers = [optimizer]
        schedules = []

        if lr_scheduler is not None:
            schedules.append(
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "name": "lr_schedule",
                }
            )

        if weight_decay_scheduler is not None:
            schedules.append(
                {
                    "scheduler": weight_decay_scheduler,
                    "interval": "step",
                    "name": "wd_schedule",
                }
            )

        return optimizers, schedules
