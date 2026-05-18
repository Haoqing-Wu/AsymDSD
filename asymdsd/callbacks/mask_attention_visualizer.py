from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import numpy as np
import torch


class MaskAttentionVisualizer(L.Callback):
    """Log 3D point clouds to wandb colored by attention and mask components.

    Produces ``wandb.Object3D`` visualizations per logged step:
    1. **attention** — all points colored by their patch's teacher CLS→patch
       attention (blue=low → red=high).
    2. **mask** — all points colored by their path-specific role:
       - Yellow: visible patches
       - Gray: masked patches

    Each raw point inherits the color of its nearest patch center.
    Only logs for the first sample in the batch (index 0).

    Args:
        every_n_steps: log every N training steps (default 200).
    """

    COLOR_VISIBLE = (255, 220, 50)  # yellow
    COLOR_MASKED = (128, 128, 128)  # gray

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

        # --- Attention heatmap (xyz + rgb) for all points ---
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        point_attn = attn_norm[nearest]  # (N,)
        rgb = np.zeros((len(raw_pts), 3), dtype=np.float64)
        rgb[:, 0] = point_attn * 255
        rgb[:, 2] = (1.0 - point_attn) * 255
        attn_cloud = np.concatenate([raw_pts, rgb], axis=1).astype(np.float64)

        step = trainer.global_step
        log_dict = {
            "viz/attention_head_a": wandb.Object3D(attn_cloud),
        }

        path_masks = mask_components.get("path_masks")
        path_names = mask_components.get("path_names")
        if path_masks is not None:
            if path_names is None:
                path_names = [f"path_{idx}" for idx in range(len(path_masks))]
            for idx, path_mask_t in enumerate(path_masks):
                path_name = path_names[idx] if idx < len(path_names) else f"path_{idx}"
                path_name = self._sanitize_path_name(path_name)
                path_mask = path_mask_t[0].detach().cpu().numpy().astype(bool)
                path_cloud = self._make_mask_cloud(raw_pts, nearest, path_mask)
                log_dict[f"viz/mask/{path_name}"] = wandb.Object3D(path_cloud)
                log_dict[f"viz/mask_ratio/{path_name}"] = float(path_mask.mean())
                if idx == 0:
                    log_dict["viz/mask"] = wandb.Object3D(path_cloud)
        else:
            # Legacy per-patch visualization. block|sparse = selected patches.
            # If select_visible, selected patches are visible; otherwise they
            # are masked, so visible = complement.
            selected = block | sparse  # (P,)
            select_visible = mask_components.get(
                "select_visible", getattr(pl_module, "select_visible", False)
            )
            patch_visible = selected if select_visible else ~selected
            point_visible = patch_visible[nearest]  # (N,)

            mask_rgb = np.zeros((len(raw_pts), 3), dtype=np.float64)
            mask_rgb[point_visible] = self.COLOR_VISIBLE
            mask_rgb[~point_visible] = self.COLOR_MASKED
            mask_cloud = np.concatenate([raw_pts, mask_rgb], axis=1).astype(np.float64)
            log_dict["viz/mask"] = wandb.Object3D(mask_cloud)

        # --- Per-head attention for head_b (if available) ---
        cls_to_patch_b = mask_components.get("cls_to_patch_b")
        if cls_to_patch_b is not None:
            attn_b = cls_to_patch_b[0].detach().cpu().float().numpy()
            attn_b_norm = (attn_b - attn_b.min()) / (attn_b.max() - attn_b.min() + 1e-8)
            point_attn_b = attn_b_norm[nearest]
            rgb_b = np.zeros((len(raw_pts), 3), dtype=np.float64)
            rgb_b[:, 0] = point_attn_b * 255
            rgb_b[:, 2] = (1.0 - point_attn_b) * 255
            attn_cloud_b = np.concatenate([raw_pts, rgb_b], axis=1).astype(np.float64)
            log_dict["viz/attention_head_b"] = wandb.Object3D(attn_cloud_b)

        # Log which heads were used
        head_a = mask_components.get("head_a")
        head_b = mask_components.get("head_b")
        if head_a is not None:
            log_dict["viz/head_a_idx"] = head_a
            log_dict["viz/head_b_idx"] = head_b

        wandb_logger.experiment.log(log_dict, step=step)

    def _make_mask_cloud(
        self,
        raw_pts: np.ndarray,
        nearest: np.ndarray,
        patch_mask: np.ndarray,
    ) -> np.ndarray:
        point_masked = patch_mask[nearest]
        mask_rgb = np.zeros((len(raw_pts), 3), dtype=np.float64)
        mask_rgb[~point_masked] = self.COLOR_VISIBLE
        mask_rgb[point_masked] = self.COLOR_MASKED
        return np.concatenate([raw_pts, mask_rgb], axis=1).astype(np.float64)

    @staticmethod
    def _sanitize_path_name(name: Any) -> str:
        name = str(name)
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
