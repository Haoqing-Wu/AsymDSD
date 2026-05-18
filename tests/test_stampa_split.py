import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_stampa_module(monkeypatch):
    fake_lightning = SimpleNamespace(LightningDataModule=object)
    monkeypatch.setitem(sys.modules, "lightning", fake_lightning)

    module_path = Path(__file__).resolve().parents[1] / "asymdsd" / "data" / "stampa.py"
    spec = importlib.util.spec_from_file_location("stampa_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stampa_fallback_eval_split_is_seeded_and_disjoint(tmp_path, monkeypatch):
    stampa = _load_stampa_module(monkeypatch)
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    points = np.zeros((8, 3), dtype=np.float32)
    for idx in range(20):
        np.savez(npz_dir / f"item_{idx:02d}.npz", coord=points)

    train = stampa.StampaAsymDSDTargetDataset(
        root_dir=tmp_path,
        version="",
        split="train",
        eval_split_ratio=0.1,
        split_seed=7,
    )
    validation = stampa.StampaAsymDSDTargetDataset(
        root_dir=tmp_path,
        version="",
        split="validation",
        eval_split_ratio=0.1,
        split_seed=7,
    )
    validation_repeat = stampa.StampaAsymDSDTargetDataset(
        root_dir=tmp_path,
        version="",
        split="validation",
        eval_split_ratio=0.1,
        split_seed=7,
    )

    assert len(train.ids) == 18
    assert len(validation.ids) == 2
    assert set(train.ids).isdisjoint(validation.ids)
    assert sorted(train.ids + validation.ids) == [
        f"item_{idx:02d}" for idx in range(20)
    ]
    assert validation.ids == validation_repeat.ids
