from __future__ import annotations

from pathlib import Path

import lightning as L
import torch


class PQStemExportCallback(L.Callback):
    """Export PQDT-loadable PQStem weights, optionally including TransUp layers."""

    def __init__(
        self,
        path: str | Path,
        branch: str = "teacher",
        save_on_fit_end: bool = True,
        include_up_layers: bool = False,
    ) -> None:
        super().__init__()
        if branch not in {"teacher", "student"}:
            raise ValueError("branch must be 'teacher' or 'student'.")
        self.path = Path(path)
        self.branch = branch
        self.save_on_fit_end = save_on_fit_end
        self.include_up_layers = include_up_layers

    def _state_dict(self, pl_module: L.LightningModule) -> dict[str, torch.Tensor]:
        if self.include_up_layers and hasattr(pl_module, "pqdt_component_state_dict"):
            return pl_module.pqdt_component_state_dict(branch=self.branch)

        if hasattr(pl_module, "pq_stem_state_dict"):
            return pl_module.pq_stem_state_dict(branch=self.branch)

        branch_module = getattr(pl_module, self.branch)
        stem = branch_module.point_encoder.stem_encoder
        prefix = f"{self.branch}.stem_encoder."
        return {
            f"{prefix}{key}": value.detach().cpu()
            for key, value in stem.state_dict().items()
        }

    def export(self, pl_module: L.LightningModule) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(pl_module), self.path)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.save_on_fit_end and trainer.is_global_zero:
            self.export(pl_module)
