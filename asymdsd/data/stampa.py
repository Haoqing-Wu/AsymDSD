from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _read_split(meta_dir: Path, split: str) -> list[str] | None:
    split_file = meta_dir / f"{split}_split.txt"
    if not split_file.exists():
        return None
    return [item for item in split_file.read_text().splitlines() if item]


def _to_float_points(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


def _sample_mesh(mesh_path: Path, n_points: int, verbose: bool) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as exc:
        raise ImportError(
            "open3d is required to build STAMPA npz caches from meshes. "
            "Provide cache_dir/npz files or install open3d."
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Could not read mesh: {mesh_path}")
    if verbose:
        print(f"Caching STAMPA mesh: {mesh_path}")
    pcd = mesh.sample_points_poisson_disk(n_points)
    return _to_float_points(pcd.points)


def _surface_area(mesh_path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception:
        return np.asarray(0.0, dtype=np.float32)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return np.asarray(mesh.get_surface_area(), dtype=np.float32)


def _normalize(points: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((points - center) / scale).astype(np.float32)


def _rotation_matrix() -> np.ndarray:
    angles = np.random.random(3) * 2 * np.pi
    cz, sz = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cx, sx = np.cos(angles[2]), np.sin(angles[2])
    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return (rz @ ry @ rx).astype(np.float32)


def _scale_from_points(
    points: np.ndarray,
    center: np.ndarray,
    default_scale: float,
) -> np.ndarray:
    distances = np.linalg.norm(points - center, axis=1)
    scale = np.max(distances).astype(np.float32)
    if scale <= 0:
        return np.asarray(default_scale, dtype=np.float32)
    return scale


def _missing_cache_ids(ids: list[str], npz_dir: Path) -> list[str]:
    return [item_id for item_id in ids if not (npz_dir / f"{item_id}.npz").exists()]


def _is_eval_split(split: str) -> bool:
    return split in {"validation", "val", "eval", "test"}


def _fallback_split_ids(
    ids: list[str],
    split: str,
    eval_split_ratio: float,
    split_seed: int,
) -> list[str]:
    if split != "train" and not _is_eval_split(split):
        return ids
    if eval_split_ratio <= 0.0 or eval_split_ratio >= 1.0:
        raise ValueError("eval_split_ratio must be between 0 and 1.")
    if len(ids) < 2:
        return ids if split == "train" else []

    n_eval = round(len(ids) * eval_split_ratio)
    n_eval = min(max(n_eval, 1), len(ids) - 1)
    rng = np.random.default_rng(split_seed)
    eval_indices = set(rng.permutation(len(ids))[:n_eval].tolist())

    if _is_eval_split(split):
        return [item_id for idx, item_id in enumerate(ids) if idx in eval_indices]
    return [item_id for idx, item_id in enumerate(ids) if idx not in eval_indices]


class StampaAsymDSDTargetDataset(Dataset):
    """Target-only STAMPA dataset with no dependency on the PQDT checkout."""

    def __init__(
        self,
        root_dir: str | Path = "/home/ubuntu/dataset/parts_no_trace",
        version: str = "stampa_no_trace",
        split: str = "train",
        n_points: int = 8192,
        normalize: str = "src",
        data_augmentation: bool = True,
        verbose: bool = False,
        num_cache_workers: int = 16,
        cache_dir: str | Path | None = None,
        eval_split_ratio: float = 0.1,
        split_seed: int = 0,
    ) -> None:
        del num_cache_workers
        super().__init__()
        self.data_dir = Path(root_dir) / version if version else Path(root_dir)
        self.target_dir = self.data_dir / "target"
        self.meta_dir = self.data_dir / "metadata"
        self.split = split
        self.n_points = n_points
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.verbose = verbose
        self.configured_cache_dir = Path(cache_dir) if cache_dir else None
        self.eval_split_ratio = eval_split_ratio
        self.split_seed = split_seed
        self.ids = self._discover_ids()
        self.npz_dir = self._select_cache_dir()
        self.npz_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_npz_cache()
        self.npz_paths = [self.npz_dir / f"{item_id}.npz" for item_id in self.ids]

    def _cache_candidates(self):
        if self.configured_cache_dir is not None:
            yield self.configured_cache_dir
        yield self.data_dir / f"npz_unlabeled_{self.split}"
        yield self.data_dir / f"npz_{self.split}"
        yield self.data_dir / "npz"

    def _discover_ids(self) -> list[str]:
        split_ids = _read_split(self.meta_dir, self.split)
        if split_ids is not None:
            return split_ids
        all_ids = self._discover_all_ids()
        return _fallback_split_ids(
            all_ids,
            self.split,
            self.eval_split_ratio,
            self.split_seed,
        )

    def _discover_all_ids(self) -> list[str]:
        if self.target_dir.exists():
            return sorted(path.stem for path in self.target_dir.glob("*.obj"))
        if self.configured_cache_dir is not None and self.configured_cache_dir.exists():
            return sorted(path.stem for path in self.configured_cache_dir.glob("*.npz"))
        shared_cache_dir = self.data_dir / "npz"
        if shared_cache_dir.exists():
            return sorted(path.stem for path in shared_cache_dir.glob("*.npz"))

        cached_ids = set()
        for cache_dir in self.data_dir.glob("npz*"):
            if cache_dir.is_dir():
                cached_ids.update(path.stem for path in cache_dir.glob("*.npz"))
        if cached_ids:
            return sorted(cached_ids)

        for cache_dir in self._cache_candidates():
            if cache_dir.exists():
                return sorted(path.stem for path in cache_dir.glob("*.npz"))
        raise FileNotFoundError(
            f"No STAMPA target meshes or cache found under {self.data_dir}"
        )

    def _select_cache_dir(self) -> Path:
        for cache_dir in self._cache_candidates():
            if cache_dir.exists() and not _missing_cache_ids(self.ids, cache_dir):
                return cache_dir
        if self.configured_cache_dir is not None:
            return self.configured_cache_dir
        return self.data_dir / f"npz_unlabeled_{self.split}"

    def _ensure_npz_cache(self) -> None:
        missing_ids = _missing_cache_ids(self.ids, self.npz_dir)
        if not missing_ids:
            return
        if not self.target_dir.exists():
            raise FileNotFoundError(
                f"Missing cache and no target mesh directory: {self.target_dir}"
            )
        for item_id in missing_ids:
            mesh_path = self.target_dir / f"{item_id}.obj"
            coord = _sample_mesh(mesh_path, self.n_points, self.verbose)
            area = _surface_area(mesh_path)
            np.savez(self.npz_dir / f"{item_id}.npz", coord=coord, area=area)

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        npz_path = self.npz_paths[index]
        with np.load(npz_path) as npz:
            coord = npz["coord"] if "coord" in npz else npz["tgt_coord"]
        target = _to_float_points(coord)
        if self.data_augmentation and self.split == "train":
            target = np.dot(target, _rotation_matrix().T).astype(np.float32)
        center = target.mean(axis=0).astype(np.float32)
        scale = (
            _scale_from_points(target, center, default_scale=30.0)
            if self.normalize == "src"
            else np.asarray(30.0, dtype=np.float32)
        )
        points = torch.from_numpy(_normalize(target, center, scale))
        return {
            "points": points,
            "num_points": torch.tensor(points.shape[0], dtype=torch.long),
        }


class StampaAsymDSDDataModule(L.LightningDataModule):
    """AsymDSD datamodule for STAMPA target-only pretraining."""

    def __init__(
        self,
        root_dir: str | Path = "/home/ubuntu/dataset/parts_no_trace",
        version: str = "stampa_no_trace",
        train_split: str = "train",
        val_split: str | None = None,
        n_points: int = 8192,
        normalize: str = "src",
        data_augmentation: bool = True,
        verbose: bool = False,
        num_cache_workers: int = 16,
        cache_dir: str | Path | None = None,
        eval_split_ratio: float = 0.1,
        split_seed: int = 0,
        batch_size: int = 8,
        num_workers_train: int = 4,
        num_workers_val_test: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_kwargs = {
            "root_dir": root_dir,
            "version": version,
            "n_points": n_points,
            "normalize": normalize,
            "data_augmentation": data_augmentation,
            "verbose": verbose,
            "num_cache_workers": num_cache_workers,
            "cache_dir": cache_dir,
            "eval_split_ratio": eval_split_ratio,
            "split_seed": split_seed,
        }
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers_train = num_workers_train
        self.num_workers_val_test = num_workers_val_test
        self.pin_memory = pin_memory
        self.dataset: dict[str, Dataset] = {}

    @property
    def len_train_dataset(self) -> int | None:
        train_dataset = self.dataset.get("train")
        return len(train_dataset) if train_dataset is not None else None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.dataset["train"] = StampaAsymDSDTargetDataset(
                split=self.train_split,
                **self.dataset_kwargs,
            )
            if self.val_split:
                val_kwargs = dict(self.dataset_kwargs)
                val_kwargs["data_augmentation"] = False
                self.dataset["validation"] = StampaAsymDSDTargetDataset(
                    split=self.val_split,
                    **val_kwargs,
                )

    def train_dataloader(self, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=self.num_workers_train,
            persistent_workers=self.num_workers_train > 0,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader | list:
        if "validation" not in self.dataset:
            return []
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_val_test,
            persistent_workers=self.num_workers_val_test > 0,
            pin_memory=self.pin_memory,
        )
