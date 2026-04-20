# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AsymDSD (Asymmetric Dual Self-Distillation) is a framework for 3D self-supervised representation learning on point clouds. It uses a student-teacher architecture with EMA, combining CLS-token invariance learning and masked point modeling (MPM). Published at NeurIPS 2025.

**Key requirements:** Python 3.11, CUDA (nvcc), PyTorch3D (built from source), PyTorch Lightning.

## Common Commands

### Environment Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
uv pip install -e .
```

### Pre-training
```bash
sh shell_scripts/sh/train_ssrl.sh                          # AsymDSD-S on ShapeNetCore (default)
sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_mask.yaml   # MPM-only
sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_cls.yaml    # CLS-only
sh shell_scripts/sh/train_ssrl.sh --model configs/ssrl/variants/model/ssrl_model_base.yaml --data configs/data/Mixture-U.yaml  # Base on Mixture
```

### Evaluation
```bash
# Object recognition (runs linear, MLP, fine-tune across ModelNet40/ScanObjectNN)
python shell_scripts/py/train_neural_classifier_all.py --runs <N> --model.encoder_ckpt_path <ckpt>

# Semantic segmentation (ShapeNetPart)
sh shell_scripts/sh/train_semseg.sh --model.encoder_ckpt_path <ckpt>

# Single classification run
sh shell_scripts/sh/train_neural_classifier.sh --model.encoder_ckpt_path <ckpt> --data configs/data/ModelNet40-S.yaml
```

### Dataset Preparation
```bash
sh shell_scripts/sh/prepare_data_zarr.sh <dataset_name>    # e.g., Objaverse, ScannedObjects, Toys4K
```
Raw data goes in `data/`; the system auto-builds `.zarr` caches on first training run.

### Linting
```bash
ruff check .          # lint (pyflakes + pycodestyle subset + isort)
ruff format .         # format (Black-compatible)
pyright               # type checking (basic mode)
```

## Architecture

### CLI & Configuration System

All training scripts use `LightningCLI` via `TrainerCLI` (in `asymdsd/run/cli.py`). Configuration is YAML-based with `jsonargparse`, supporting class instantiation from config (`class_path`/`init_args` pattern). Config arguments are composable and overridable via CLI flags like `--model`, `--data`, `--config`.

Entry points in `asymdsd/run/`:
- `ssrl_cli.py` - Self-supervised pre-training (uses `AsymDSD` model + `PointCloudDataModule`)
- `classification_cli.py` - Supervised classification (uses `NeuralClassifier`)
- `sem_seg_cli.py` - Semantic segmentation
- `embedding_classifier_cli.py` - Embedding-based classifiers (kNN, SVM)
- `prepare_zarr_ds_cli.py` - Dataset preparation

### Config Layout (`configs/`)

- `ssrl/` - Pre-training configs. `ssrl.yaml` is the top-level entry, referencing `ssrl_model.yaml`, `ssrl_trainer.yaml`, `ssrl_optim.yaml`, `ssrl_wandb.yaml`. Variants in `ssrl/variants/model/` for different model sizes/modes.
- `classification/` - Downstream classification. Variants for linear probing, MLP probing, fine-tuning, few-shot.
- `semseg/` - Semantic segmentation configs.
- `data/` - Data module configs. Files ending in `-U` are unsupervised, `-S` are supervised. `prepare_data_zarr/` has per-dataset preparation configs.
- `compile/` - torch.compile settings.

### Core Module Structure (`asymdsd/`)

**`models/`** - Lightning modules:
- `asymdsd.py` - Main `AsymDSD` module. Student-teacher with EMA. Three training modes: `CLS` (invariance only), `MASK` (MPM only), `CLS_MASK` (both, the default). Contains `forward_student()`, `forward_teacher()`, and loss computation in `training_step()`.
- `point_encoder.py` - `PointEncoder`: patchify -> patch embedding -> transformer encoder. Shared architecture between student/teacher. Returns `PointEncoderOutput` with patch/cls features.
- `neural_classifier.py` - `NeuralClassifier`: downstream classifier with frozen/fine-tuned encoder. Supports LINEAR and MLP head types.
- `embedding_model.py` - `EmbeddingModel`: extract and cache embeddings for kNN/SVM evaluation.
- `knn_classifier.py`, `linear_svm_classifier.py` - Non-neural classifiers operating on cached embeddings.
- `semantic_segmentation.py` - Part segmentation model.

**`layers/`** - Neural network building blocks:
- `transformer.py` - `TransformerEncoder`/`TransformerDecoder` with config classes. Support gradient checkpointing, drop path, layer scale.
- `tokenization.py` - `PatchEmbedding` (combines `PointEmbedding` + `PositionEmbedding`), `MemEfficientPointMaxEmbedding` for memory-efficient point-to-token conversion.
- `patchify.py` - `MultiPointPatchify` converts raw point clouds into patch groups. `PatchPoints` and `MultiPatches` are the main data structures flowing through the model.
- `projection_head.py` - Multi-layer projection head for self-distillation.

**`components/`** - Training infrastructure:
- `exponential_moving_average.py` - `EMA` for teacher updates.
- `masking.py` - `RandomPatchMasking`, `BlockPatchMasking`, `InverseBlockPatchMasking` strategies.
- `scheduling.py` - `CosineAnnealingWarmupSchedule`, `LinearWarmupSchedule`, `Scheduler` for coordinating temperature, momentum, and other schedule-driven hyperparameters.
- `optimizer_spec.py` - `AdamWSpec`/`SGDSpec` with integrated LR/WD scheduling.
- `transforms.py` - Point cloud augmentations (rotation, scaling, translation, normalization).
- `factory_config.py` - `FactoryConfig` base class providing `instantiate()` pattern used throughout configs.

**`data/`** - Data pipeline:
- `data_module_zarr.py` - `UnsupervisedZarrPCDataModule` and `SupervisedZarrPCDataModule`. Zarr-based storage with auto-creation from raw datasets.
- `dataset_zarr.py` - `ZarrDataset` and `create_zarr_ds()` for efficient on-disk storage.
- `multi_crop.py` - Multi-crop augmentation strategy (global + local crops).
- `patchify.py` - CPU-side patchification using farthest point sampling.
- `datasets_/` - Per-dataset builders (ShapeNetCore, ModelNet40, ScanObjectNN, Objaverse, etc.) that convert raw formats into the unified zarr schema.

**`loss/`** - Loss functions: `ClsLoss` (cross-entropy distillation), `PatchLoss` (patch-level distillation), `KoLeoLoss` (uniformity), `MeanEntropyLoss`.

**`callbacks/`** - `EmbeddingClassifierEval` and `NeuralClassifierEval` run downstream evaluations during pre-training. `DefaultTrainerCheckpoint` handles checkpointing.

### Data Flow

1. Raw datasets (zip/tar/h5) in `data/` are converted to `.zarr` format by `DatasetBuilder` subclasses
2. `ZarrDataset` loads from zarr with optional multi-crop (global + local views)
3. CPU-side: subsampling -> augmentation -> patchification (FPS-based)
4. GPU-side in `AsymDSD.training_step()`: normalization -> patch embedding -> masking -> student/teacher forward passes -> loss

### Key Design Patterns

- **`FactoryConfig` + `instantiate()`**: Config dataclasses that construct their corresponding modules. Used for `TransformerEncoderConfig`, `PatchEmbeddingConfig`, `ProjectionHeadConfig`, etc.
- **`@init_lazy_defaults`**: Decorator enabling lazy default instantiation for jsonargparse compatibility.
- **Linked arguments**: `ssrl_cli.py` links args (e.g., `data.batch_size` -> `model.batch_size`, `trainer.max_epochs` -> `model.max_epochs`) so they stay in sync.
- **`EncoderBranch`**: Enum selecting student vs teacher encoder for downstream evaluation checkpoint loading.
