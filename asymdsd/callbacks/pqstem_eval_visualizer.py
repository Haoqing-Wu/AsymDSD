from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import numpy as np
import torch


class PQStemEvalPointCloudVisualizer(L.Callback):
    """Log PQStem validation reconstruction and mask point clouds to wandb."""

    COLOR_GT = (170, 170, 170)
    COLOR_RECON = (40, 220, 120)
    COLOR_VISIBLE = (255, 220, 50)
    COLOR_MASKED = (90, 90, 90)

    def __init__(
        self,
        sample_index: int = 0,
        max_points: int | None = 8192,
        offset_scale: float = 1.35,
        rotate_each_eval: bool = True,
    ) -> None:
        super().__init__()
        self.sample_index = sample_index
        self.max_points = max_points
        self.offset_scale = offset_scale
        self.rotate_each_eval = rotate_each_eval

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, batch
        if getattr(trainer, "sanity_checking", False):
            return
        if dataloader_idx != 0 or outputs is None:
            return
        eval_index = self._eval_index(trainer)
        if batch_idx != self._target_batch_idx(trainer, dataloader_idx, eval_index):
            return

        gt_points = outputs.get("gt_points")
        pred_pcds = outputs.get("transup_reconstructions")
        token_centers = outputs.get("eval_token_centers")
        eval_mask = outputs.get("eval_mask")
        if gt_points is None or pred_pcds is None or len(pred_pcds) == 0:
            return

        wandb_logger = None
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                wandb_logger = logger
                break
        if wandb_logger is None:
            return

        try:
            import wandb
        except ImportError:
            return

        sample_idx = self._target_sample_idx(gt_points.shape[0], eval_index)
        gt = self._to_numpy_points(gt_points[sample_idx])
        pred = self._to_numpy_points(pred_pcds[-1][sample_idx])

        log_dict: dict[str, Any] = {
            "eval/reconstruction_gt": wandb.Object3D(
                self._make_pair_cloud(gt, pred, self.COLOR_GT, self.COLOR_RECON)
            ),
        }

        if token_centers is not None and eval_mask is not None:
            centers = self._to_numpy_points(token_centers[sample_idx])
            mask = eval_mask[sample_idx].detach().cpu().numpy().astype(bool)
            log_dict["eval/masked_gt"] = wandb.Object3D(
                self._make_mask_gt_cloud(gt, centers, mask)
            )
            log_dict["eval/mask_ratio"] = float(mask.mean())

        wandb_logger.experiment.log(log_dict, step=trainer.global_step)

    def _eval_index(self, trainer: L.Trainer) -> int:
        if not self.rotate_each_eval:
            return 0
        return max(int(getattr(trainer, "current_epoch", 0)), 0)

    def _target_batch_idx(
        self,
        trainer: L.Trainer,
        dataloader_idx: int,
        eval_index: int,
    ) -> int:
        if not self.rotate_each_eval:
            return 0
        num_batches = getattr(trainer, "num_val_batches", None)
        if isinstance(num_batches, (list, tuple)):
            num_batches = (
                num_batches[dataloader_idx]
                if dataloader_idx < len(num_batches)
                else None
            )
        try:
            num_batches_int = int(num_batches)
        except (TypeError, ValueError):
            num_batches_int = 1
        if num_batches_int <= 0:
            return 0
        return eval_index % num_batches_int

    def _target_sample_idx(self, batch_size: int, eval_index: int) -> int:
        if batch_size <= 0:
            return 0
        if not self.rotate_each_eval:
            return min(self.sample_index, batch_size - 1)
        return (self.sample_index + eval_index) % batch_size

    def _to_numpy_points(self, points: torch.Tensor) -> np.ndarray:
        points_np = points[..., :3].detach().cpu().float().numpy()
        if self.max_points is not None and points_np.shape[0] > self.max_points:
            keep = np.linspace(
                0,
                points_np.shape[0] - 1,
                num=self.max_points,
                dtype=np.int64,
            )
            points_np = points_np[keep]
        return points_np.astype(np.float64)

    def _make_pair_cloud(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        gt_color: tuple[int, int, int],
        pred_color: tuple[int, int, int],
    ) -> np.ndarray:
        gt_shifted, pred_shifted = self._side_by_side(gt, pred)
        return np.concatenate(
            [
                self._with_color(gt_shifted, gt_color),
                self._with_color(pred_shifted, pred_color),
            ],
            axis=0,
        ).astype(np.float64)

    def _make_mask_gt_cloud(
        self,
        gt: np.ndarray,
        centers: np.ndarray,
        patch_mask: np.ndarray,
    ) -> np.ndarray:
        dists = np.linalg.norm(gt[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        point_masked = patch_mask[nearest]

        gt_left, gt_right = self._side_by_side(gt, gt)
        masked_rgb = np.zeros((gt.shape[0], 3), dtype=np.float64)
        masked_rgb[~point_masked] = self.COLOR_VISIBLE
        masked_rgb[point_masked] = self.COLOR_MASKED
        return np.concatenate(
            [
                self._with_color(gt_left, self.COLOR_GT),
                np.concatenate([gt_right, masked_rgb], axis=1),
            ],
            axis=0,
        ).astype(np.float64)

    def _side_by_side(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points = np.concatenate([left, right], axis=0)
        span = np.ptp(all_points, axis=0).max()
        offset = max(float(span) * self.offset_scale, 1e-3)
        shift = np.array([offset / 2.0, 0.0, 0.0], dtype=np.float64)
        return left - shift, right + shift

    @staticmethod
    def _with_color(
        points: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        rgb = np.broadcast_to(
            np.asarray(color, dtype=np.float64),
            (points.shape[0], 3),
        )
        return np.concatenate([points, rgb], axis=1)
