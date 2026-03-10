# CenterTargetMamba (CTMamba)

PyTorch code for hyperspectral image classification with a fixed CTMamba architecture.

## Model Definition

CTMamba uses a fixed front-end, backbone, and back-end:

- Front-end: `1x1` spectral projection (`conv1x1`)
- Backbone: raster-route spatial selective scan blocks
- Back-end: MLP classifier head input feature fusion
- Structured adapters: `CCA + CPA + BCA + RSA`

Main model file:

- `models/ctmamba.py`

## Install

```bash
pip install numpy scipy pyyaml torch h5py scikit-learn tqdm joblib
```

## Minimal Workflow

1. Convert raw `.mat` to `cube.npy` and `gt.npy`.
2. Generate split JSON files.
3. Train.
4. Evaluate.

Scripts:

- `prepare_raw_to_processed.py`
- `make_splits.py`
- `train.py`
- `eval.py`
- `run_multiseed.py`

Help for each script:

```bash
python <script>.py --help
```

## Example

Run one seed on Pavia University:

```bash
python run_multiseed.py \
  --dataset pavia_university \
  --split_tag random \
  --seeds 0 \
  --amp \
  --data_root data \
  --out_base outputs/checkpoints
```

## Main Configs

- Model: `configs/model/ctmamba_pavia_university.yaml`
- Train: `configs/train/pu.yaml`
- Dataset: `configs/datasets/pavia_university.yaml`

All released CTMamba configs currently use `patch_size=15`.

## Output Schema

Single-seed `metrics.json` includes:

- `VAL` / `TEST`: OA, AA, Kappa (and optional class-wise outputs)
- `time_sec`
- `meta`: dataset/split/seed/config hashes/parameter counts

Multi-seed `mean_metrics.json` includes:

- `per_seed` and `per_seed_full`
- `mean` / `std`
- optional runtime and parameter summaries

## License

MIT License.
