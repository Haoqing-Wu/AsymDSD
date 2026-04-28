from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import numpy as np
import torch


class MaskAttentionVisualizer(L.Callback):
    """Log 3D point clouds to wandb colored by attention and mask components.

    Produces two ``wandb.Object3D`` visualizations per logged step:
    1. **attention** — all points colored by their patch's teacher CLS→patch
       attention (blue=low → red=high).
    2. **mask** — all points colored by their patch's mask role:
       - Red (cat 1): block-masked patches
       - Orange (cat 2): sparse-masked patches
       - Green (cat 3): unmasked patches (visible to encoder)

    Each raw point inherits the color of its nearest patch center.
    Only logs for the first sample in the batch (index 0).

    Args:
        every_n_steps: log every N training steps (default 200).
    """

    BLOCK_MASKED = 1
    SPARSE_MASKED = 2
    UNMASKED = 3

    def __init__(self, every_n_steps: int = 200) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return

        mask_components = outputs.get("mask_components")
        centers = outputs.get("centers")
        if mask_components is None or centers is None:
            return

        wandb_logger = None
        for lg in trainer.loggers:
            if hasattr(lg, "experiment") and hasattr(lg.experiment, "log"):
                wandb_logger = lg
                break
        if wandb_logger is None:
            return

        try:
            import wandb
        except ImportError:
            return

        cls_to_patch: torch.Tensor = mask_components["cls_to_patch"]
        block_mask: torch.Tensor = mask_components["block_mask"]
        sparse_mask: torch.Tensor = mask_components["sparse_mask"]

        # Get raw points from batch (first sample, first crop)
        crops_dict = batch.get("global_crops") or batch
        raw_points = crops_dict["points"]
        if raw_points.ndim == 4:
            raw_points = raw_points[0, 0]  # (B, C, N, 3) → first sample, first crop
        elif raw_points.ndim == 3:
            raw_points = raw_points[0]  # (B, N, 3) → first sample
        raw_pts = raw_points[:, :3].detach().cpu().float().numpy()  # (N, 3)

        center_pts = centers[0].detach().cpu().float().numpy()  # (P, 3)
        attn = cls_to_patch[0].detach().cpu().float().numpy()  # (P,)
        block = block_mask[0].detach().cpu().numpy().astype(bool)
        sparse = sparse_mask[0].detach().cpu().numpy().astype(bool)

        # Assign each raw point to its nearest patch center
        # (N, 1, 3) - (1, P, 3) → (N, P)
        dists = np.linalg.norm(raw_pts[:, None, :] - center_pts[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)  # (N,)

        # --- Per-patch category ---
        patch_cat = np.full(len(center_pts), self.UNMASKED, dtype=np.float64)
        patch_cat[block] = self.BLOCK_MASKED
        patch_cat[sparse] = self.SPARSE_MASKED

        # --- Attention heatmap (xyz + rgb) for all points ---
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        point_attn = attn_norm[nearest]  # (N,)
        rgb = np.zeros((len(raw_pts), 3), dtype=np.float64)
        rgb[:, 0] = point_attn * 255
        rgb[:, 2] = (1.0 - point_attn) * 255
        attn_cloud = np.concatenate([raw_pts, rgb], axis=1).astype(np.float64)

        # --- Mask category (xyz + category) for all points ---
        point_cat = patch_cat[nearest]  # (N,)
        mask_cloud = np.concatenate([raw_pts, point_cat[:, None]], axis=1).astype(
            np.float64
        )

        step = trainer.global_step
        wandb_logger.experiment.log(
            {
                "viz/attention": wandb.Object3D(attn_cloud),
                "viz/mask": wandb.Object3D(mask_cloud),
            },
            step=step,
        )
