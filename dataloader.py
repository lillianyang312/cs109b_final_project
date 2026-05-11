import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# configuration

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "processed"

TIMESTEP_MIN = 3
REGION = "West_Coast"

INTERP_FILE = OUT_DIR / f"ais_trajectories_{REGION}_{TIMESTEP_MIN}min.parquet"
SPLITS_FILE = OUT_DIR / f"splits_{REGION}_{TIMESTEP_MIN}min.parquet"
STATS_FILE = OUT_DIR / f"feature_stats_{REGION}_{TIMESTEP_MIN}min.csv"


# feature registry
FEATURE_REGISTRY = {
    "lat": {"col": "latitude", "encoding": "normalize"},
    "lon": {"col": "longitude", "encoding": "normalize"},
    "sog": {"col": "sog", "encoding": "normalize"},
    "cog": {"col": "cog", "encoding": "angular"},
    "heading": {"col": "heading", "encoding": "angular"},
    "length": {"col": "length", "encoding": "normalize"},
    "width": {"col": "width", "encoding": "normalize"},
    "draft": {"col": "draft", "encoding": "normalize"},
}

DEFAULT_FEATURES = ["lat", "lon", "sog", "cog", "heading"]
DEFAULT_POSITION_ENCODING = "relative"
POSITION_FEATURES = {"lat", "lon"}
KM_PER_DEG_LAT = 111.320


# dataset

class AISDataset(Dataset):
    def __init__(
        self,
        split,
        features=None,
        input_len=20,
        target_len=20,
        stride=5,
        position_encoding=DEFAULT_POSITION_ENCODING,
    ):
        assert split in ("train", "val", "test")
        assert position_encoding in ("relative", "absolute")

        features = features or DEFAULT_FEATURES

        for f in features:
            if f not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown feature '{f}'. Available: {list(FEATURE_REGISTRY)}"
                )

        self.split = split
        self.features = features
        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride
        self.window = input_len + target_len
        self.position_encoding = position_encoding

        self._check_files_exist()

        df = pd.read_parquet(INTERP_FILE)
        splits = pd.read_parquet(SPLITS_FILE)

        keep_mmsi = splits.loc[splits["split"] == split, "mmsi"].to_numpy()
        df = df[df["mmsi"].isin(keep_mmsi)].copy()

        df = df.sort_values(["trip_id", "base_date_time"]).reset_index(drop=True)

        self.mean, self.std = self._load_stats(features, df)

        self.feature_array, self.output_names, self.relative_col_idx = (
            self._build_features(df, features)
        )

        self.n_features = self.feature_array.shape[1]
        self.windows = self._build_windows(df)

        if len(self.windows) == 0:
            raise ValueError(
                f"No valid windows for split='{split}'. "
                f"Try smaller input_len/target_len or check trip lengths."
            )

        if "lat" in features and "lon" in features:
            lat_raw = df["latitude"].to_numpy(dtype=np.float32)
            lon_raw = df["longitude"].to_numpy(dtype=np.float32)

            ref_offsets = self.windows[:, 0] + self.input_len - 1

            self._ref_lat = lat_raw[ref_offsets]
            self._ref_lon = lon_raw[ref_offsets]
            self._cos_ref_lat = np.cos(np.deg2rad(self._ref_lat)).astype(np.float32)
        else:
            self._ref_lat = None
            self._ref_lon = None
            self._cos_ref_lat = None

        self._lat_rel_col = None
        self._lon_rel_col = None

        for j, name in enumerate(self.output_names):
            if name == "lat_rel_km":
                self._lat_rel_col = j
            elif name == "lon_rel_km":
                self._lon_rel_col = j

        if self._lon_rel_col is not None and self._cos_ref_lat is None:
            raise ValueError(
                "position_encoding='relative' with 'lon' requires 'lat' too."
            )

        print(
            f"[{split}] {len(df):,} rows, "
            f"{df['trip_id'].nunique():,} trips, "
            f"{len(self.windows):,} windows | "
            f"features={self.output_names} "
            f"(n={self.n_features}) | "
            f"position_encoding={self.position_encoding}"
        )

    def _check_files_exist(self):
        missing = [
            str(p)
            for p in [INTERP_FILE, SPLITS_FILE]
            if not p.exists()
        ]

        if missing:
            files = "\n".join(missing)
            raise FileNotFoundError(
                "Required processed files are missing:\n"
                f"{files}\n\n"
                "Expected filenames are:\n"
                f"  {INTERP_FILE.name}\n"
                f"  {SPLITS_FILE.name}\n"
                f"  {STATS_FILE.name}\n\n"
                "Make sure the files are in the processed/ folder and use "
                "the same REGION/TIMESTEP_MIN naming."
            )

    def _load_stats(self, features, df):
        try:
            stats = pd.read_csv(STATS_FILE, index_col=0)
            mean = stats["mean"].to_dict()
            std = stats["std"].to_dict()
        except FileNotFoundError:
            print(
                f"WARNING: {STATS_FILE.name} not found. "
                "Computing stats from current split."
            )
            mean, std = {}, {}

        for f in features:
            cfg = FEATURE_REGISTRY[f]

            if cfg["encoding"] != "normalize":
                continue

            col = cfg["col"]

            if col not in mean:
                print(
                    f"WARNING: '{col}' not in feature stats. "
                    f"Using current {self.split} split stats."
                )
                mean[col] = df[col].mean()
                std[col] = df[col].std()

        return mean, std

    def _build_features(self, df, features):
        cols = []
        names = []
        rel_idx = []

        for f in features:
            cfg = FEATURE_REGISTRY[f]
            col = cfg["col"]

            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' for feature '{f}' is missing from {INTERP_FILE.name}."
                )

            vals = df[col].to_numpy(dtype=np.float64)

            if cfg["encoding"] == "normalize":
                if f in POSITION_FEATURES and self.position_encoding == "relative":
                    cols.append(vals)
                    names.append(f"{f}_rel_km")
                    rel_idx.append(len(cols) - 1)
                else:
                    mu = self.mean[col]
                    sd = self.std[col]
                    sd = sd if sd > 1e-9 else 1.0

                    cols.append((vals - mu) / sd)
                    names.append(f"{f}_norm")

            elif cfg["encoding"] == "angular":
                rad = np.deg2rad(vals)

                cols.append(np.cos(rad))
                cols.append(np.sin(rad))

                names.append(f"{f}_cos")
                names.append(f"{f}_sin")

            else:
                raise ValueError(f"Unknown encoding: {cfg['encoding']}")

        arr = np.stack(cols, axis=1).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        return arr, names, rel_idx

    def _build_windows(self, df):
        windows = []

        for _, idxs in df.groupby("trip_id").indices.items():
            L = len(idxs)

            if L < self.window:
                continue

            start_global = idxs[0]
            max_start = L - self.window

            for s in range(0, max_start + 1, self.stride):
                gs = start_global + s
                windows.append((gs, gs + self.window))

        return np.asarray(windows, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        start, end = self.windows[i]
        seq = self.feature_array[start:end]

        if self.relative_col_idx:
            seq = seq.copy()

            ref = seq[self.input_len - 1, self.relative_col_idx].copy()
            seq[:, self.relative_col_idx] -= ref

            if self._lat_rel_col is not None:
                seq[:, self._lat_rel_col] *= KM_PER_DEG_LAT

            if self._lon_rel_col is not None:
                seq[:, self._lon_rel_col] *= (
                    KM_PER_DEG_LAT * self._cos_ref_lat[i]
                )

        x = seq[: self.input_len]
        y = seq[self.input_len :]

        return torch.from_numpy(x), torch.from_numpy(y)

    def reference_latlon(self, i):
        if self._ref_lat is None:
            return None

        return float(self._ref_lat[i]), float(self._ref_lon[i])

    def recover_absolute(self, i, delta_km):
        if self._ref_lat is None:
            raise ValueError(
                "recover_absolute requires both 'lat' and 'lon' in features."
            )

        delta_km = np.asarray(delta_km)

        ref_lat = float(self._ref_lat[i])
        ref_lon = float(self._ref_lon[i])
        cos_ref = float(self._cos_ref_lat[i])

        abs_lat = ref_lat + delta_km[..., 0] / KM_PER_DEG_LAT
        abs_lon = ref_lon + delta_km[..., 1] / (KM_PER_DEG_LAT * cos_ref)

        return abs_lat, abs_lon


# data_loader

def get_dataloaders(
    features=None,
    input_len=20,
    target_len=20,
    stride=5,
    batch_size=64,
    num_workers=0,
    position_encoding=DEFAULT_POSITION_ENCODING,
):
    train_ds = AISDataset(
        "train",
        features,
        input_len,
        target_len,
        stride,
        position_encoding,
    )

    val_ds = AISDataset(
        "val",
        features,
        input_len,
        target_len,
        stride,
        position_encoding,
    )

    test_ds = AISDataset(
        "test",
        features,
        input_len,
        target_len,
        stride,
        position_encoding,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Using files:")
    print(f"  {INTERP_FILE}")
    print(f"  {SPLITS_FILE}")
    print(f"  {STATS_FILE}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    x, y = next(iter(train_loader))

    print("\nOne training batch:")
    print(f"  x.shape = {tuple(x.shape)}")
    print(f"  y.shape = {tuple(y.shape)}")

    names = train_loader.dataset.output_names
    print(f"  features = {names}")