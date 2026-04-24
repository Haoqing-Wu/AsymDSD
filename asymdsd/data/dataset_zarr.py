import os
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
from typing import Any, Callable

import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from ..components import FactoryConfig
from ..components.common_types import PathLike
from ..loggers import get_default_logger
from .dataset_builder import DatasetBuilder, FieldType
from .transforms import MapColumn

logger = get_default_logger()


def create_zarr_ds(
    dataset_builder: DatasetBuilder,
    dataset_save_path: PathLike | None = None,
    num_workers: int | None = None,
) -> Path:
    if dataset_save_path is None:
        dataset_save_path = (
            dataset_builder.data_path.parent / f"{dataset_builder.name}.zarr"
        )
    dataset_save_path = Path(dataset_save_path).expanduser().resolve()

    if dataset_builder.data_path:
        if not os.path.exists(dataset_builder.data_path):
            raise FileNotFoundError(
                f"Data path {dataset_builder.data_path} does not exist"
            )

    try:
        # If builder implements build method itself, use it
        dataset_builder.build(dataset_save_path, num_workers)
        return dataset_save_path
    except NotImplementedError:
        pass

    # Use append mode to support resuming incomplete dataset creation
    resume = os.path.exists(dataset_save_path)
    root = zarr.open_group(str(dataset_save_path), mode="a" if resume else "w")

    if resume:
        logger.info(f"Resuming incomplete dataset creation at {dataset_save_path}")

    class_labels = dataset_builder.class_labels
    class_label_keys = list(class_labels.keys()) if class_labels is not None else []
    data_fields = dataset_builder.data_fields

    for split in dataset_builder.splits:
        paths = []
        labels = {
            field.key: {} for field in data_fields if field.key_type != FieldType.ARRAY
        }

        if split in root:
            split_group: zarr.Group = root[split]
        else:
            split_group = root.create_group(split)

        split_iter = dataset_builder.iterate_data(split, num_workers=num_workers)

        if isinstance(split_iter, Sized):
            len_split_iter = len(split_iter)
        else:
            len_split_iter = None

        for data in tqdm(
            split_iter,
            unit="example",
            desc=f"Creating {split} split",
            total=len_split_iter,
        ):
            if data is None:
                continue

            name: str = data["name"]
            path = f"{split}/{name}"
            paths.append(path)

            for data_field in data_fields:
                key = data_field.key

                if key not in data:
                    raise ValueError(
                        f"Data field key {data_field.key} not found in data"
                    )

                if data_field.key_type == FieldType.ARRAY:
                    if resume and f"{name}/{key}" in split_group:
                        continue
                    split_group.array(
                        f"{name}/{key}",
                        data[key],
                        chunks=(None, *data[key].shape[1:]),
                    )
                elif (
                    data_field.key_type == FieldType.STRING_LABEL
                    and key in class_label_keys
                ):
                    labels[key][path] = class_labels[key].str2int(data[key])  # type: ignore
                elif (
                    data_field.key_type == FieldType.INT_LABEL
                    or data_field.key_type == FieldType.STRING_LABEL
                ):
                    labels[key][path] = data[key]
                else:
                    raise ValueError(f"Unknown key_type: {data_field.key_type}")

        split_group.attrs["paths"] = paths

        for key in labels:
            split_group.attrs[key] = labels[key]

    root.attrs["name"] = dataset_builder.name
    root.attrs["splits"] = dataset_builder.splits

    root.attrs["attr_keys"] = [
        field.key
        for field in dataset_builder.data_fields
        if field.key_type == FieldType.STRING_LABEL
        or field.key_type == FieldType.INT_LABEL
    ]
    root.attrs["array_keys"] = [
        field.key
        for field in dataset_builder.data_fields
        if field.key_type == FieldType.ARRAY
    ]

    if class_labels is not None:
        root.attrs["label_names"] = {
            key: labels.label_names for key, labels in class_labels.items()
        }
    root.attrs["complete"] = True

    return dataset_save_path


class MapMixin:
    def __init__(self):
        self.map_fns = []

    def map(
        self,
        fn: Callable,
        input_columns: str | list[str] | None = None,
        output_columns: str | list[str] | None = None,
        remove_columns: str | list[str] | None = None,
        input_as_positional_args: bool = True,
    ):
        if input_columns is None:
            if output_columns is not None or remove_columns is not None:
                raise ValueError(
                    "If input_columns is None, output_columns and remove_columns must also be None."
                )
            self.map_fns.append(fn)
        else:
            self.map_fns.append(
                MapColumn(
                    fn,
                    input_columns,
                    output_columns,
                    remove_columns,
                    input_as_positional_args=input_as_positional_args,
                )
            )

    def apply_map(self, item):
        for fn in self.map_fns:
            item = fn(item)
        return item

    def map_decorator(self, func):
        def wrapper(*args, **kwargs):
            item = func(*args, **kwargs)
            return self.apply_map(item)

        return wrapper


class BaseZarrDataset(MapMixin, Dataset, ABC):
    def __init__(
        self,
        dataset_path: PathLike,
        split: str | list[str] | None = None,
        item_paths: list[str] | None = None,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.root = zarr.open_group(str(self.dataset_path), mode="r")

        if split is None:
            split = list(self.root.keys())
        self.split = split if isinstance(split, list) else [split]

        if item_paths is not None:
            self.paths = item_paths
        else:
            self.paths = []
            for s in self.split:
                self.paths.extend(self.root[s].attrs["paths"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.get_item(idx)
        return self.apply_map(item)

    @abstractmethod
    def get_item(self, idx: int) -> dict[str, Any]:
        pass


class ZarrDataset(BaseZarrDataset):
    def __init__(
        self,
        dataset_path: PathLike,
        split: str | list[str] | None = None,
        item_paths: list[str] | None = None,
        array_keys: list[str] | None = None,
        attr_keys: list[str] | None = None,
    ):
        super().__init__(dataset_path, split, item_paths)

        if attr_keys is None:
            attr_keys = []
        if array_keys is None:
            array_keys = []

        self.array_keys = array_keys
        self.attr_keys = attr_keys

        self.path2attrs = {}

        for attr_key in self.attr_keys:
            new_map = {}
            for s in self.split:
                new_map.update(self.root[s].attrs[attr_key])
            self.path2attrs[attr_key] = new_map

    def get_item(self, idx: int):
        path = self.paths[idx]

        item = {}
        for key in self.array_keys:
            item[key] = self.root[f"{path}/{key}"][:]

        for key in self.attr_keys:
            item[key] = self.path2attrs[key][path]

        return item


class CustomZarrDatasetFactory(FactoryConfig, ABC):
    @abstractmethod
    def instantiate(
        self,
        dataset_path: PathLike,
        split: str | list[str] = "train",
        array_keys: list[str] | None = None,
        attr_keys: list[str] | None = None,
    ) -> BaseZarrDataset:
        pass
