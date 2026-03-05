# TriScanMamba

TriScanMamba is a hyperspectral image (HSI) classification code release centered on a selective state-space 3D backbone with optional mixture-of-experts (MoE) routing.  
The repository is intentionally scoped to method implementation and experiment protocol logic.

## Scope of This Public Release

Included:

- core model implementation (`VSSM3DMoE`) and modules under `src/hsi3d/`
- training and evaluation entry points
- data conversion (`.mat -> .npy`) and split generation utilities

Not included:

- dataset assets (`data/`, raw or processed)
- dataset/model/train configuration files
- pre-generated split files
- baseline third-party repositories and baseline wrapper scripts
- checkpoints, logs, and output artifacts

## Repository Layout

```text
LICENSE
README.md
scripts/
  prepare_raw_to_processed.py
  make_splits.py
  train.py
  eval.py
  run_10runs_and_mean.py
src/hsi3d/
  data/
  metrics/
  models/
  modules/
  training/
  utils/
```

## Environment

Python 3.10+ is recommended.

Install dependencies manually:

```bash
pip install numpy scipy pyyaml torch h5py scikit-learn tqdm joblib
```

Since this repository is not packaged as an installable wheel, expose `src/` via `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Data Contract

Training and evaluation read processed arrays from:

- `data/processed/<dataset>/raw/cube.npy` with shape `(H, W, B)`, dtype `float32`
- `data/processed/<dataset>/raw/gt.npy` with shape `(H, W)`, dtype integer labels

`scripts/prepare_raw_to_processed.py` converts dataset `.mat` files into this format.

## Minimal Configuration Contracts

The repository does not ship dataset-specific config files.  
Create your own YAML/JSON files and pass their paths via CLI arguments.

### `dataset_cfg` (minimal example)

```yaml
dataset: pavia_university
label_offset: 1
num_classes: 9
raw_dir: /path/to/raw/pavia_university
cube_file: PaviaU.mat
gt_file: PaviaU_gt.mat
cube_key: paviaU
gt_key: paviaU_gt
```

### `model_cfg` (minimal example)

```yaml
patch_size: 15
dropout: 0.15
stages: [2, 2, 4]
stage_dims: [64, 96, 128]
spec_groups: 8
spec_layers: 3
spec_hidden: 96
moe_experts: 3
moe_topk: 1
head: ce
```

### `train_cfg` (minimal example)

```yaml
epochs: 200
batch_size: 16
eval_batch_size: 512
lr: 1.1e-4
weight_decay: 3.0e-2
augment: false
```

### `split_json` (required keys)

```json
{
  "dataset": "pavia_university",
  "seed": 0,
  "label_offset": 1,
  "num_classes": 9,
  "train_indices": [0, 1, 2],
  "val_indices": [3, 4],
  "test_indices": [5, 6, 7]
}
```

Indices are flat indices over the `H*W` raster order.

## Workflow

1. Convert raw `.mat` files:

```bash
python scripts/prepare_raw_to_processed.py \
  --dataset_cfg path/to/dataset.yaml \
  --data_root data
```

2. Generate split files:

```bash
python scripts/make_splits.py \
  --dataset_cfg path/to/dataset.yaml \
  --split_tag random \
  --seeds 0-9 \
  --train_ratio 0.10 \
  --val_ratio 0.10 \
  --data_root data \
  --out_dir splits
```

3. Train:

```bash
python scripts/train.py \
  --dataset_cfg path/to/dataset.yaml \
  --model_cfg path/to/model.yaml \
  --train_cfg path/to/train.yaml \
  --split_json splits/random/pavia_university_seed0.json \
  --out_dir outputs/checkpoints/pavia_university_seed0 \
  --seed 0 \
  --data_root data \
  --amp
```

4. Evaluate:

```bash
PYTHONPATH=src python scripts/eval.py \
  --dataset_cfg path/to/dataset.yaml \
  --model_cfg path/to/model.yaml \
  --split_json splits/random/pavia_university_seed0.json \
  --data_root data \
  --seed 0 \
  --ckpt outputs/checkpoints/pavia_university_seed0/checkpoints/best.pt \
  --ckpt_key model \
  --batch_size 512 \
  --out outputs/checkpoints/pavia_university_seed0/eval.json
```

`scripts/run_10runs_and_mean.py` is provided for repeated-seed aggregation and expects a user-prepared `configs/` and `splits/` layout.

## Reproducibility Protocol

For comparative reporting (including OA/AA/Kappa tables), the following protocol is recommended:

- use fixed train/val/test index files across all compared methods
- compute normalization statistics from training pixels only
- disable test-time augmentation at evaluation
- use identical seeds and report mean ± standard deviation across runs
- archive the exact config and split files used for final reported numbers

## License

MIT License. See `LICENSE`.
