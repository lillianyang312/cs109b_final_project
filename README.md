# Vessel Trajectory Prediction Using Deep Learning on AIS Data

**CS 109B Milestone 4, Group 24**
**Members:** Yuyang Chen, Tim Guo, Lillian Yang, Grace Yang

## Overview

This project applies deep learning to predict future vessel trajectories from historical Automatic Identification System (AIS) data. We compare three architectures: an MLP baseline, an LSTM, and a Transformer (Encoder + Query Decoder), on a large-scale West Coast AIS dataset.

**Key result:** The Transformer achieves a mean Mean Distance Error (MDE) of 0.40 km, matching the MLP baseline, while the LSTM underperforms due to error accumulation in autoregressive rollout. A 60-minute forecast error of ~0.76 km is well within the 1-5 nautical mile AIS reporting radius used by maritime traffic management.

## Repository Structure

```
cs109b_final_project/
├── cs1090b_ms4_main_group24.ipynb    # Main notebook (run this)
├── dataloader.py                      # PyTorch Dataset & DataLoader
├── .gitignore
├── README.md
├── eda_density.png                    # EDA visualizations
├── eda_trip_stats.png
├── model_comparison.png               # Model results comparison plot
├── model_comparison_by_sog.png        # Model results by speed-over-ground
├── MLP/
│   └── final_model.pt                 # Saved MLP checkpoint
├── trained_models/
│   ├── best_lstm (1).pt               # Saved LSTM checkpoint
│   ├── vessel_transformer_20260510_053103.pt
│   ├── vessel_transformer_20260510_053103.pth
│   ├── vessel_transformer_20260510_053103_config.json
│   ├── vessel_transformer_20260511_203652.pth
│   ├── vessel_transformer_20260511_203652_config.json
│   ├── training_history_20260510_053103.json
│   ├── training_history_20260511_203652.json
│   ├── mde_transformer_20260510_053103.npy
│   ├── mde_transformer_20260511_203652.npy
│   ├── mde_constant_velocity_SOGCOG.npy
│   └── haversine_per_horizon.csv
└── processed/
    ├── feature_stats_West_Coast_3min.csv
    ├── splits_West_Coast_3min.parquet
    └── ais_trajectories_West_Coast_3min.parquet  # Download separately (see below)
```

## Data Setup (Required Before Running)

The main preprocessed dataset file (`ais_trajectories_West_Coast_3min.parquet`, ~2 GB) is too large for GitHub. You must download it separately and place it in the correct location.

**Download link (Google Drive):** [ais_trajectories_West_Coast_3min.parquet](https://drive.google.com/file/d/1VFc9bWqFMDPpVcpL3R6xDJqzHN_tHxIk/view?usp=sharing)

**Where to place it:**

```
cs109b_final_project/processed/ais_trajectories_West_Coast_3min.parquet
```

The `processed/` folder already contains `splits_West_Coast_3min.parquet` and `feature_stats_West_Coast_3min.csv`. Just add the downloaded file there.

## Project Path — No Configuration Needed

The notebook uses a **relative path** for `PROJECT_DIR` in Section 2.2 (Configuration):

```python
# Current configuration (portable — no changes needed):
PROJECT_DIR = Path('.')
```

`Path('.')` resolves to the directory the notebook lives in (`DATASET_BALANCED/`), so all subsequent `PROJECT_DIR / 'processed' / ...` paths work automatically as long as the notebook is run from inside the repo folder (the standard JupyterLab behavior).

## How to Run the Notebook

### 1. Clone the repository
```bash
git clone https://github.com/lillianyang312/cs109b_final_project.git
cd cs109b_final_project
```

### 2. Install dependencies
```bash
pip install numpy pandas torch matplotlib pyarrow
```

Requirements:

| Package     | Version   |
|-------------|-----------|
| numpy       | >= 1.26   |
| pandas      | >= 2.2    |
| torch       | >= 2.1.0  |
| matplotlib  | >= 3.8    |
| pyarrow     | >= 14.0   |

### 3. Download the dataset

Download `ais_trajectories_West_Coast_3min.parquet` from Google Drive and place it in `processed/`.

### 4. Open and run the notebook
```bash
jupyter lab cs1090b_ms4_main_group24.ipynb
```

Then run all cells top to bottom (**Run → Run All Cells**). The notebook will:

- **Section 1** — Check and install dependencies
- - **Section 2** — Import libraries and configure paths (`PROJECT_DIR`)
  - - **Section 5** — Load preprocessed data via `dataloader.py`
    - - **Section 6** — Define MLP, LSTM, and Transformer model architectures
      - - **Section 7** — Load pre-trained checkpoints (or train from scratch if missing)
        - - **Section 8** — Evaluate all three models on the test set
          - - **Section 9** — Plot and display results
           
            - > **Note:** All models have saved checkpoints — the notebook will load weights automatically and skip training by default.
              >
              > ## GPU / CPU
              >
              > The notebook auto-detects `cuda` and falls back to `cpu`. Training from scratch on CPU will be slow; a GPU is strongly recommended.
              >
              > ## Models
              >
              > | Model       | Architecture                                                                                      | Parameters | Mean MDE   |
              > |-------------|---------------------------------------------------------------------------------------------------|------------|------------|
              > | MLP         | Flatten → Linear(1400→256) → ReLU → Linear(256→128) → ReLU → Linear(128→40) → reshape(B,20,2)  | ~74 k      | 0.4013 km  |
              > | LSTM        | 2-layer LSTM encoder (hidden=256) + autoregressive decoder + LayerNorm + Dropout(0.2)            | ~800 k     | 1.1312 km  |
              > | Transformer | 4-head, 4-layer encoder + query-decoder with learned future queries                               | ~varies    | 0.3977 km  |
              >
              > ## Data
              >
              > The dataset is based on NOAA/USCG AIS broadcast records for the West Coast (3-minute resolution). Preprocessing steps are documented in `generate_trajectories.ipynb`, `merge_monthly.ipynb`, and `make_splits.ipynb`.
              >
              > - **Input features** (7 columns per timestep): `lat_rel_km`, `lon_rel_km`, `sog_norm`, `cog_cos`, `cog_sin`, `heading_cos`, `heading_sin`
              > - - **Prediction target:** `delta-lat_km`, `delta-lon_km` (relative displacement over the next 60 minutes)
              >   - - **Split:** 57.6M train rows / 34.9M val rows, vessel-level assignment (no vessel appears in multiple splits)
              >    
              >     - ## Results
              >    
              >     - - **Transformer vs MLP:** −0.9% mean MDE (marginal improvement)
              >       - - **LSTM vs MLP:** +181.9% mean MDE (significantly worse due to autoregressive error accumulation)
