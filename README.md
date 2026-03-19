# CenterTargetedMamba

PyTorch implementation of CenterTargetedMamba for hyperspectral image classification.

## Requirements

```bash
pip install numpy scipy pyyaml torch h5py scikit-learn tqdm joblib
```

## Data Preparation

```bash
python prepare_raw_to_processed.py --help
python make_splits.py --help
```

## Training

```bash
python run_multiseed.py \
  --dataset pavia_university \
  --split_tag random \
  --seeds 0-9 \
  --data_root data \
  --out_base outputs/checkpoints \
  --amp
```

Released configs are stored in `configs/datasets`, `configs/model`, and `configs/train`.

## Evaluation

Each seed writes `metrics.json` under `outputs/checkpoints/<dataset>_seed*/`.
The multi-seed summary is written to `outputs/checkpoints/<dataset>_mean3/mean_metrics.json`.

## License

MIT
