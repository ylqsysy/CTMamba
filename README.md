# TriScanMamba

TriScanMamba is a research codebase for hyperspectral image (HSI) classification with selective state-space modeling and mixture-of-experts routing.
The repository is organized around reproducible training and evaluation workflows that can be reused across datasets.

## Overview

The main implementation combines:

- a VSSM-style 3D backbone for spatial–spectral representation learning,
- MoE routing in deeper stages,
- fixed random-split evaluation over multiple seeds.

In addition to the main model, the project includes baseline training scripts and utilities for dataset preprocessing and split generation.

## Repository Structure

```text
configs/
  baselines/          # baseline-specific runtime settings
  datasets/           # dataset paths, keys, class metadata
  model/              # model hyperparameters
  train/              # optimizer/scheduler/training settings
scripts/
  prepare_raw_to_processed.py
  make_splits.py
  train.py
  eval.py
  run_10runs_and_mean.py
  baselines/          # baseline training entry points
src/hsi3d/            # package source code
splits/random/        # predefined random splits
tests/                # minimal import test
```

## Environment Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate triscanmamba
```

### Option B: pip

```bash
pip install -r requirements_freeze.txt
pip install -e .
```

## Data Preparation

Place raw `.mat` files under the dataset-specific `raw_dir` defined in `configs/datasets/*.yaml`.
Then convert raw files into the internal NumPy format:

```bash
python scripts/prepare_raw_to_processed.py \
  --dataset_cfg configs/datasets/pavia_university.yaml \
  --data_root data
```

## Split Generation

Create random splits for repeated experiments (example: seeds 0-9):

```bash
python scripts/make_splits.py \
  --dataset_cfg configs/datasets/pavia_university.yaml \
  --split_tag random \
  --seeds 0-9 \
  --train_ratio 0.10 \
  --val_ratio 0.10 \
  --data_root data \
  --out_dir splits
```

## Training and Evaluation

### Single run

```bash
python scripts/train.py \
  --dataset_cfg configs/datasets/pavia_university.yaml \
  --model_cfg configs/model/vssm3d_pu.yaml \
  --train_cfg configs/train/pu.yaml \
  --split_json splits/random/pavia_university_seed0.json \
  --out_dir outputs/checkpoints/pavia_university_seed0 \
  --seed 0 \
  --data_root data \
  --amp
```

```bash
python scripts/eval.py \
  --dataset_cfg configs/datasets/pavia_university.yaml \
  --model_cfg configs/model/vssm3d_pu.yaml \
  --checkpoint outputs/checkpoints/pavia_university_seed0/checkpoints/best.pt \
  --split_json splits/random/pavia_university_seed0.json \
  --out_dir outputs/checkpoints/pavia_university_seed0 \
  --data_root data
```

### 10-run protocol

```bash
python -u scripts/run_10runs_and_mean.py \
  --dataset pavia_university \
  --split_tag random \
  --amp \
  --num_workers 0 \
  --out_base outputs/checkpoints
```

## Reproducibility Notes

- Use fixed split files from `splits/random/` when comparing methods.
- Report mean and standard deviation over the same seed set.
- Record software versions together with experiment outputs.

## License

This project is released under the MIT License. See `LICENSE` for details.
