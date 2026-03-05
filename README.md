# TriScanMamba

TriScanMamba is a compact research codebase for hyperspectral image (HSI) classification, built around a selective state-space 3D backbone with optional mixture-of-experts routing.
The repository is intentionally focused on method implementation, deterministic protocol control, and reproducible reporting.

## What This Repository Contains

```text
LICENSE
README.md
train.py
eval.py
prepare_raw_to_processed.py
make_splits.py
run_10runs_and_mean.py
models/
utils/
```

- `models/`: network definitions (`VSSM3DMoE` and core blocks).
- `utils/`: dataset loader, training engine, scheduler, metrics, I/O helpers, seed control.
- `prepare_raw_to_processed.py`: converts raw `.mat` data to standard `.npy` tensors.
- `make_splits.py`: generates seed-controlled train/val/test index files.
- `train.py`: single-run training.
- `eval.py`: checkpoint evaluation on val/test.
- `run_10runs_and_mean.py`: repeated-seed aggregation pipeline.

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install numpy scipy pyyaml torch h5py scikit-learn tqdm joblib
```

## Data Interface

All training and evaluation scripts expect processed arrays at:

- `data/processed/<dataset>/raw/cube.npy` with shape `(H, W, B)` and dtype `float32`
- `data/processed/<dataset>/raw/gt.npy` with shape `(H, W)` and integer labels

Use `prepare_raw_to_processed.py` to convert raw MATLAB files into this layout.

## Minimal Config Contracts

This repository does not ship dataset-specific config files. Provide your own YAML/JSON files at runtime.

### 1) `dataset_cfg` (YAML)

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

### 2) `model_cfg` (YAML)

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

### 3) `train_cfg` (YAML)

```yaml
batch_size: 16
eval_batch_size: 512
lr: 1.1e-4
weight_decay: 3.0e-2
max_epochs: 340
warmup_epochs: 12
augment: false
```

### 4) `split_json` (JSON)

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

Indices are flat raster indices over `H*W`.

## Workflow

### Step 1: Convert raw data

```bash
python prepare_raw_to_processed.py \
  --dataset_cfg path/to/dataset.yaml \
  --data_root data
```

### Step 2: Generate splits

```bash
python make_splits.py \
  --dataset_cfg path/to/dataset.yaml \
  --split_tag random \
  --seeds 0-9 \
  --train_ratio 0.10 \
  --val_ratio 0.10 \
  --data_root data \
  --out_dir splits
```

### Step 3: Train

```bash
python train.py \
  --dataset_cfg path/to/dataset.yaml \
  --model_cfg path/to/model.yaml \
  --train_cfg path/to/train.yaml \
  --split_json splits/random/pavia_university_seed0.json \
  --out_dir outputs/checkpoints/pavia_university_seed0 \
  --seed 0 \
  --data_root data \
  --num_workers 0 \
  --amp
```

### Step 4: Evaluate

```bash
python eval.py \
  --dataset_cfg path/to/dataset.yaml \
  --model_cfg path/to/model.yaml \
  --split_json splits/random/pavia_university_seed0.json \
  --ckpt outputs/checkpoints/pavia_university_seed0/checkpoints/best.pt \
  --ckpt_key model \
  --out outputs/checkpoints/pavia_university_seed0/eval.json \
  --seed 0 \
  --data_root data \
  --batch_size 512
```

## Evaluation Protocol Recommendation

For fair comparison and reviewer-facing reproducibility:

- keep train/val/test split indices fixed across methods
- compute normalization statistics from training pixels only
- evaluate without test-time augmentation
- report OA, AA, and Kappa
- report mean ± standard deviation across identical seed sets

## License

This project is released under the MIT License. See `LICENSE`.
