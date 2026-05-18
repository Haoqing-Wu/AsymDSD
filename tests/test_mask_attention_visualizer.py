import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _load_visualizer_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "asymdsd"
        / "callbacks"
        / "mask_attention_visualizer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mask_attention_visualizer_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MaskAttentionVisualizer


def test_mask_attention_visualizer_logs_every_path_mask(monkeypatch):
    class FakeObject3D:
        def __init__(self, data):
            self.data = data

    fake_wandb = SimpleNamespace(Object3D=FakeObject3D)
    fake_lightning = SimpleNamespace(
        Callback=object,
        Trainer=object,
        LightningModule=object,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setitem(sys.modules, "lightning", fake_lightning)

    logged = {}

    class FakeExperiment:
        def log(self, log_dict, step=None):
            logged["step"] = step
            logged["log_dict"] = log_dict

    MaskAttentionVisualizer = _load_visualizer_class()
    callback = MaskAttentionVisualizer(every_n_steps=1)
    trainer = SimpleNamespace(
        global_step=8,
        loggers=[SimpleNamespace(experiment=FakeExperiment())],
    )
    pl_module = SimpleNamespace(select_visible=False)

    raw_points = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    centers = raw_points.clone()
    path_masks = [
        torch.tensor([[False, True, False, True]]),
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[False, False, True, True]]),
        torch.tensor([[True, True, False, False]]),
    ]
    outputs = {
        "centers": centers,
        "mask_components": {
            "cls_to_patch": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "block_mask": torch.zeros(1, 4, dtype=torch.bool),
            "sparse_mask": torch.zeros(1, 4, dtype=torch.bool),
            "path_masks": path_masks,
            "path_names": [
                "sparse_visible",
                "sparse_masked",
                "geometric_halfspace",
                "random",
            ],
        },
    }

    callback.on_train_batch_end(
        trainer,
        pl_module,
        outputs,
        {"points": raw_points},
        batch_idx=0,
    )

    log_dict = logged["log_dict"]
    assert logged["step"] == 8
    assert "viz/mask" in log_dict
    for name in [
        "sparse_visible",
        "sparse_masked",
        "geometric_halfspace",
        "random",
    ]:
        assert f"viz/mask/{name}" in log_dict
        assert log_dict[f"viz/mask_ratio/{name}"] == 0.5

    sparse_visible_cloud = log_dict["viz/mask/sparse_visible"].data
    expected_rgb = np.array(
        [
            callback.COLOR_VISIBLE,
            callback.COLOR_MASKED,
            callback.COLOR_VISIBLE,
            callback.COLOR_MASKED,
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(sparse_visible_cloud[:, 3:], expected_rgb)
