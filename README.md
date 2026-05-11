# Vessel Trajectory Prediction Using Deep Learning on AIS Data

**CS 109B Milestone 4, Group 24**
**Members:** Yuyang Chen, Tim Guo, Lillian Yang, Grace Yang

## Overview

This project applies deep learning to predict future vessel trajectories from historical Automatic Identification System (AIS) data. We compare three architectures, an MLP baseline, an LSTM, and a Transformer (Encoder + Query Decoder), on a large-scale West Coast AIS dataset.

**Key result:** The Transformer achieves a mean Mean Distance Error (MDE) of 0.40 km, matching the MLP baseline, while the LSTM underperforms due to error accumulation in autoregressive rollout. A 60-minute forecast error of ~0.76 km is well within the 1-5 nautical mile AIS reporting radius used by maritime traffic management.

## Repository Structure

```
cs109b_final_project/
├── cs1090b_ms4_main_group24.ipynb  # Main notebook (run this)
├── dataloader.py                   # PyTorch Dataset & DataLoader
├── dataloader.ipynb                # Dataloader exploration notebook
├── best_lstm.pt                    # Saved LSTM checkpoint
├── trained_models/                 # Saved Transformer checkpoints
├── processed/                      # Preprocessed feature stats + splits
├── lstm_results/                   # LSTM training curves
├── feature_stats_West Coast_3min.csv
├── splits_West Coast_3min.parquet
├── eda_density.png
├── eda_trip_stats.png
├── model_comparison.png
├── generate_trajectories.ipynb
├── make_splits.ipynb
├── merge_monthly.ipynb
└── README.md
```

## WARNING: Data Setup (Required Before Running)

The main preprocessed dataset file (`ais_trajectories_West_Coast_3min.parquet`, ~2 GB) is too large for GitHub. You must download it separately and place it in the correct location.

**Download link (Google Drive):** [ais_trajectories_West_Coast_3min.parquet](https://drive.google.com/file/d/1-D83AIsckyju72N6_316mcPyghoG6Vft/view?usp=sharing)

**Where to place it:**

```
cs109b_final_project/processed/ais_trajectories_West_Coast_3min.parquet
```

The `processed/` folder already contains `splits_West_Coast_3min.parquet` and `feature_stats_West_Coast_3min.csv`. Just add the downloaded file there.

## WARNING: Hardcoded Path - Update Before Running

In **Section 2.2 (Configuration)** of `cs1090b_ms4_main_group24.ipynb`, the `PROJECT_DIR` is hardcoded to the original author's machine:

```python
# Original (hardcoded - change this):
PROJECT_DIR = Path('/shared/home/liy159/DATASET_BALANCED')
```

**Change this to the path of the cloned repo on your machine.** For example:

```python
# Option 1: point directly to the repo folder
PROJECT_DIR = Path('/shared/home/YOUR_USERNAME/cs109b_final_project')

# Option 2: use a relative path (works if notebook is run from repo root)
PROJECT_DIR = Path().resolve()
```

After updating `PROJECT_DIR`, the notebook will automatically find all checkpoints and data files.

## Requirements

```
numpy >= 1.26
pandas >= 2.2
torch >= 2.10
matplotlib >= 3.8
pyarrow >= 23.0
```

Install with:

```bash
pip install numpy pandas torch matplotlib pyarrow
```

## How to Run

Open and run `cs1090b_ms4_main_group24.ipynb` in JupyterLab. The notebook checks and installs dependencies (Section 1), imports libraries (Section 2), configures paths via `PROJECT_DIR` (Section 2.2), loads preprocessed data via `dataloader.py` (Section 5), defines MLP/LSTM/Transformer models (Section 6), loads pre-trained checkpoints or trains from scratch if missing (Section 7), evaluates all three models on the test set (Section 8), and plots results (Section 9).

All models have saved checkpoints - the notebook will load weights automatically and skip training.

**GPU / CPU:** The notebook auto-detects `cuda` and falls back to `cpu`. Training from scratch on CPU will be slow; a GPU is recommended.

## Models

| Model | Architecture | Parameters | Mean MDE |
|---|---|---|---|
| MLP | Flatten -> Linear(1400->256) -> ReLU -> Linear(256->128) -> ReLU -> Linear(128->40) -> reshape(B,20,2) | ~74 k | 0.4013 km |
| LSTM | 2-layer LSTM encoder (hidden=256) + autoregressive decoder + LayerNorm + Dropout(0.2) | ~800 k | 1.1312 km |
| Transformer | 4-head, 4-layer encoder + query-decoder with learned future queries | ~varies | 0.3977 km |

## Data

The dataset is based on NOAA/USCG AIS broadcast records for the West Coast (3-minute resolution). Preprocessing steps are documented in `generate_trajectories.ipynb`, `merge_monthly.ipynb`, and `make_splits.ipynb`.

**Input features (7 columns per timestep):** `lat_rel_km`, `lon_rel_km`, `sog_norm`, `cog_cos`, `cog_sin`, `heading_cos`, `heading_sin`

**Prediction target:** `delta-lat_km`, `delta-lon_km` (relative displacement over the next 60 minutes)

**Split:** 57.6M train rows / 34.9M val rows, vessel-level assignment (no vessel appears in multiple splits)

## Results

![Model Comparison](model_comparison.png)

- **Transformer vs MLP:** -0.9% mean MDE (marginal improvement)
- - **LSTM vs MLP:** +181.9% mean MDE (significantly worse due to autoregressive error accumulation)
