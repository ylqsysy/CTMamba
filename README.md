# TriScanMamba

PyTorch code for hyperspectral image classification.

Current `TriScanMamba` keeps the original model name and focuses on a compact, reproducible design:

- spatial selective-scan backbone
- spectral branch with cross-attention fusion
- learnable band tokenizer (optional)
- optional absolute coordinate channels (`append_coords`)

## Install

```bash
pip install numpy scipy pyyaml torch h5py scikit-learn tqdm joblib
```

## Minimal Workflow

1. Convert raw `.mat` files to `cube.npy` / `gt.npy`.
2. Generate split JSON files.
3. Train.
4. Evaluate.

Use these scripts:

- `prepare_raw_to_processed.py`
- `make_splits.py`
- `train.py`
- `eval.py`
- `run_multiseed.py`
- `run_ablations.py`

Each script provides full argument definitions:

```bash
python <script>.py --help
```

## Example

Train/eval one seed on Hanchuan:

```bash
python run_multiseed.py \
  --dataset hanchuan \
  --split_tag random \
  --seeds 0 \
  --amp \
  --data_root data \
  --out_base outputs/checkpoints
```

## Main Configs

- Model: `configs/model/vssm3d_hanchuan.yaml`
- Train: `configs/train/hanchuan.yaml`
- Dataset: `configs/datasets/hanchuan.yaml`

All released TriScanMamba configs keep `patch_size=15` (Hanchuan/Houston2013/Honghu/PU) to control runtime.
Hanchuan default training disables mixup, enables coordinate channels, and uses single-pass evaluation
(no TTA, no post-processing).

## Verified Result (hanchuan, seed0)

- Baseline checkpoint: `outputs/checkpoints/hanchuan_seed0/metrics.json`
  - TEST OA `0.9317295`, AA `0.9212653`, Kappa `0.9203114`
- Updated checkpoint (no TTA, no post-processing): `outputs/checkpoints_hanchuan_p15_coords_nomix1/hanchuan_seed0/metrics.json`
  - TEST OA `0.9581027`, AA `0.9566976`, Kappa `0.9511179`
  - Delta: OA `+0.0263732`, AA `+0.0354323`, Kappa `+0.0308065`

## License

MIT License.
