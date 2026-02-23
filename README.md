# HSI-3D VSSM + MoE + MIM (Option-1) + Evidential Uncertainty

This repository implements a **JSTARS-oriented** hyperspectral image classification pipeline:

- **Backbone**: hierarchical **VSSM-style 2D selective scanning (SS2D-like)** blocks (multi-direction mixing) in a stage-wise CNN-like hierarchy
- **Key innovation**: **Factorized 3D selective scanning (F-SS3D)**:
  - Spatial: multi-direction SS2D-like mixing
  - Spectral: bidirectional **grouped spectral scanning**
  - Cross-coupling: lightweight **FiLM / low-rank modulation** between spatial and spectral states
- **MoE**: 3-expert mixture in late stages (space-heavy / spec-heavy / coupled)
- **Uncertainty**: evidential (Dirichlet) classifier head produces calibrated confidence / uncertainty maps
- **Protocol**: **RANDOM only**, **10 runs (seed 0..9)**, **10% train / 10% val / 80% test**, report mean±std of OA/AA/Kappa
- **Self-supervised pretrain (Option-1)**: masked band modeling (**MBM**) for spectral encoder (fast & 8GB-friendly)

> Designed for **RTX 4060 Laptop (8GB VRAM)**: patch-based training, AMP, conservative defaults.

---

## 0) Put your raw data

Copy your `.mat` files into:

```
data/raw/indian_pines/*.mat
data/raw/pavia_university/*.mat
data/raw/houston2018/*.mat
data/raw/whu_hi_longkou/*.mat
```

Expected filenames (default):
- Indian Pines: `Indian_pines_corrected.mat`, `Indian_pines_gt.mat`
- PaviaU: `PaviaU.mat`, `PaviaU_gt.mat`
- Houston2018: `Houston18.mat`, `Houston18_7gt.mat`
- WHU-Hi-LongKou: `WHU_Hi_LongKou.mat`, `WHU_Hi_LongKou_gt.mat`

If your keys/filenames differ, edit `configs/datasets/*.yaml`.

---

## 1) Install

```bash
pip install -e .
```

---

## 2) Raw -> Processed (once per dataset)

Example (Indian Pines):

```bash
python scripts/prepare_raw_to_processed.py \
  --dataset_cfg configs/datasets/indian_pines.yaml \
  --data_root data
```

This writes:
- `data/processed/<dataset>/raw/cube.npy`  (H,W,B float32)
- `data/processed/<dataset>/raw/gt.npy`    (H,W int64)

---

## 3) Create RANDOM splits (10 runs)

```bash
python scripts/make_splits.py \
  --dataset_cfg configs/datasets/indian_pines.yaml \
  --split_tag random \
  --seeds 0-9 \
  --train_ratio 0.10 --val_ratio 0.10 \
  --data_root data \
  --out_dir splits
```

Outputs:
- `splits/random/indian_pines_seed0.json` ... `seed9.json`

---

## 4) (Optional) Self-supervised pretrain (Option-1: MBM)

This pretrains the **spectral encoder** only and saves `pretrain.pt`.

```bash
python -u scripts/pretrain_mim.py \
  --dataset_cfg configs/datasets/indian_pines.yaml \
  --model_cfg configs/model/vssm3d_ip.yaml \
  --pretrain_cfg configs/pretrain/mbm.yaml \
  --data_root data \
  --seed 0 \
  --out_dir outputs/pretrain/ip_mbm_seed0 \
  --amp
```

---

## 5) Train + Eval (10 runs) and report mean±std

```bash
python -u scripts/run_10runs_and_mean.py \
  --dataset indian_pines \
  --split_tag random \
  --amp --num_workers 0 \
  --out_base outputs/checkpoints
```

Outputs:
- per seed: `outputs/checkpoints/indian_pines_seed{0..9}/eval.json`
- summary: `outputs/checkpoints/indian_pines_mean10/mean_metrics.json`, `summary.csv`, `summary.md`

---

## Notes

- This code is **reviewer-safe** by default: no test-time augmentation.
- Uncertainty metrics are saved in `eval_detail.json` (entropy, evidential uncertainty, confidence histogram).

