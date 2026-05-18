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
        / "pqstem_eval_visualizer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pqstem_eval_visualizer_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PQStemEvalPointCloudVisualizer


def test_pqstem_eval_visualizer_logs_reconstruction_and_mask(monkeypatch):
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

    PQStemEvalPointCloudVisualizer = _load_visualizer_class()
    callback = PQStemEvalPointCloudVisualizer(max_points=None)
    trainer = SimpleNamespace(
        global_step=20,
        sanity_checking=False,
        loggers=[SimpleNamespace(experiment=FakeExperiment())],
    )

    gt_points = torch.tensor(
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
    recon = gt_points + torch.tensor([[[0.0, 1.0, 0.0]]])
    outputs = {
        "gt_points": gt_points,
        "transup_reconstructions": [recon],
        "eval_token_centers": gt_points.clone(),
        "eval_mask": torch.tensor([[False, True, False, True]]),
    }

    callback.on_validation_batch_end(
        trainer,
        SimpleNamespace(),
        outputs,
        batch={"points": gt_points},
        batch_idx=0,
    )

    log_dict = logged["log_dict"]
    assert logged["step"] == 20
    assert "eval/reconstruction_gt" in log_dict
    assert "eval/masked_gt" in log_dict
    assert log_dict["eval/mask_ratio"] == 0.5

    recon_cloud = log_dict["eval/reconstruction_gt"].data
    masked_cloud = log_dict["eval/masked_gt"].data
    assert recon_cloud.shape == (8, 6)
    assert masked_cloud.shape == (8, 6)
    np.testing.assert_array_equal(
        masked_cloud[4:, 3:],
        np.array(
            [
                callback.COLOR_VISIBLE,
                callback.COLOR_MASKED,
                callback.COLOR_VISIBLE,
                callback.COLOR_MASKED,
            ],
            dtype=np.float64,
        ),
    )


def test_pqstem_eval_visualizer_rotates_batch_and_sample(monkeypatch):
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

    PQStemEvalPointCloudVisualizer = _load_visualizer_class()
    callback = PQStemEvalPointCloudVisualizer(max_points=None)
    trainer = SimpleNamespace(
        current_epoch=5,
        global_step=50,
        sanity_checking=False,
        num_val_batches=[4],
        loggers=[SimpleNamespace(experiment=FakeExperiment())],
    )

    gt_points = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    outputs = {
        "gt_points": gt_points,
        "transup_reconstructions": [gt_points + torch.tensor([[[0.0, 1.0, 0.0]]])],
        "eval_token_centers": gt_points.clone(),
        "eval_mask": torch.tensor(
            [
                [False, True, False, True],
                [True, False, True, False],
            ]
        ),
    }

    callback.on_validation_batch_end(
        trainer,
        SimpleNamespace(),
        outputs,
        batch={"points": gt_points},
        batch_idx=0,
    )
    assert logged == {}

    callback.on_validation_batch_end(
        trainer,
        SimpleNamespace(),
        outputs,
        batch={"points": gt_points},
        batch_idx=1,
    )

    masked_cloud = logged["log_dict"]["eval/masked_gt"].data
    assert logged["step"] == 50
    np.testing.assert_array_equal(
        masked_cloud[4:, 3:],
        np.array(
            [
                callback.COLOR_MASKED,
                callback.COLOR_VISIBLE,
                callback.COLOR_MASKED,
                callback.COLOR_VISIBLE,
            ],
            dtype=np.float64,
        ),
    )
