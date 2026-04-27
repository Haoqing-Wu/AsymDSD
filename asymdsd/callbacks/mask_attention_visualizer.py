from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import numpy as np
import torch


class MaskAttentionVisualizer(L.Callback):
    """Log 3D point clouds to wandb colored by attention and mask components.

    Produces two ``wandb.Object3D`` visualizations per logged step:
    1. **attention** — patch centers colored by teacher CLS→patch attention
       (blue=low → red=high).
    2. **mask** — patch centers colored by mask role:
       - Red (cat 1): block-masked patches
       - Orange (cat 2): sparse-masked patches
       - Green (cat 3): protected/visible blocks
       - Blue (cat 4): unmasked patches (visible to encoder)

    Only logs for the first sample in the batch (index 0).

    Args:
        every_n_steps: log every N training steps (default 200).
    """

    BLOCK_MASKED = 1
    SPARSE_MASKED = 2
    PROTECTED = 3
    UNMASKED = 4

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
        protected_mask: torch.Tensor = mask_components["protected_mask"]

        pts = centers[0].detach().cpu().numpy()  # (P, 3)
        attn = cls_to_patch[0].detach().cpu().float().numpy()  # (P,)
        block = block_mask[0].detach().cpu().numpy()  # (P,)
        sparse = sparse_mask[0].detach().cpu().numpy()  # (P,)
        protected = protected_mask[0].detach().cpu().numpy()  # (P,)

        # --- Attention heatmap (xyz + rgb) ---
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        rgb = np.zeros((len(attn_norm), 3), dtype=np.float64)
        rgb[:, 0] = attn_norm * 255  # red channel
        rgb[:, 2] = (1.0 - attn_norm) * 255  # blue channel
        attn_cloud = np.concatenate([pts, rgb], axis=1).astype(np.float64)

        # --- Mask category (xyz + category) ---
        cat = np.full(len(pts), self.UNMASKED, dtype=np.float64)
        cat[block.astype(bool)] = self.BLOCK_MASKED
        cat[sparse.astype(bool)] = self.SPARSE_MASKED
        cat[protected.astype(bool)] = self.PROTECTED
        mask_cloud = np.concatenate([pts, cat[:, None]], axis=1).astype(np.float64)

        step = trainer.global_step
        wandb_logger.experiment.log(
            {
                "viz/attention": wandb.Object3D(attn_cloud),
                "viz/mask": wandb.Object3D(mask_cloud),
            },
            step=step,
        )
