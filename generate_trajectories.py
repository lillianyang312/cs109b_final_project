# %% [markdown]
# # Generate Interpolated AIS Trajectories (3-min timestep)
#
# Standalone preprocessing notebook for the group. Produces the
# **model-ready** parquet file that everyone trains on.
#
# **What this notebook does**:
# 1. Loads all raw `.csv.zst` daily AIS files
# 2. Filters to the selected region's bounding box
# 3. Cleans invalid values (sentinel 511 heading, SOG outliers, etc.)
# 4. Splits each vessel's data into trips at >30-min silence gaps
# 5. Interpolates every trip to a **regular TIMESTEP grid**
# 6. Saves a single parquet file everyone imports
#
# **No EDA here.** See `preprocessing_and_eda.ipynb` for validation plots.

# %% [markdown]
# ## 1. Configuration
#
# Change `REGION` and `TIMESTEP_MIN` below. All downstream code adapts.

# %%
import os
import warnings
import numpy as np
import pandas as pd
import zstandard as zstd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Region options (NOAA geographic divisions) ────────────────────────
REGIONS = {
    "West Coast":     {"lat": (14.06, 51.71), "lon": (-132.00, -114.00)},
    "Atlantic":       {"lat": (15.19, 45.35), "lon": (-84.10,  -60.00)},
    "Gulf of Mexico": {"lat": (18.23, 34.10), "lon": (-97.89,  -83.90)},
    "Great Lakes":    {"lat": (41.00, 49.00), "lon": (-92.40,  -72.60)},
    "Pacific":        {"lat": (13.86, 32.00), "lon": (-173.48, -148.00)},
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  >>> CHANGE THESE <<<                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
REGION = "West Coast"     # one of the keys above
TIMESTEP_MIN = 3           # interpolation timestep in minutes

# Per-month workflow (for limited storage):
#
#   Set MONTH_TAG to a label like "2024-05" when you process one month at a
#   time and plan to delete the .csv.zst before downloading the next month.
#   The output parquet is then suffixed with the tag so previous months'
#   results aren't overwritten:
#
#       processed/ais_trajectories_{REGION}_{TIMESTEP}min_{MONTH_TAG}.parquet
#
#   After all months are processed, run `merge_monthly.ipynb` to concatenate
#   the monthly parquets into the canonical file that `make_splits.ipynb`
#   consumes.
#
#   Leave MONTH_TAG = "" if you have enough storage to download every month
#   at once — the output will go to the canonical filename and you can skip
#   the merge step.
MONTH_TAG = ""

# ── Other parameters (rarely need to change) ───────────────────────────
MAX_GAP_MINUTES = 30       # split a vessel's data into trips if gap > 30 min
MIN_TRAJ_POINTS = 10       # discard trips with fewer than 10 points after interp
SOG_MAX = 50               # knots — above this is almost certainly GPS error
DATA_DIR = "."
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Derived config
LAT_MIN, LAT_MAX = REGIONS[REGION]["lat"]
LON_MIN, LON_MAX = REGIONS[REGION]["lon"]
INTERVAL = f"{TIMESTEP_MIN}min"
_tag_suffix = f"_{MONTH_TAG}" if MONTH_TAG else ""
OUTPUT_FILE = os.path.join(
    OUT_DIR,
    f"ais_trajectories_{REGION}_{TIMESTEP_MIN}min{_tag_suffix}.parquet",
)

print(f"Region:        {REGION}")
print(f"Bounding box:  lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
print(f"Timestep:      {TIMESTEP_MIN} min")
print(f"Month tag:     {MONTH_TAG or '(none — using canonical filename)'}")
print(f"Output file:   {OUTPUT_FILE}")
if MONTH_TAG:
    print(f"  → After all months are processed, run merge_monthly.ipynb.")

# %% [markdown]
# ## 2. Load Raw Daily Files
#
# Each `.csv.zst` file is ~8.5M rows nationwide. We stream-decompress and
# filter to the region bounding box on the fly.

# %%
DTYPES = {
    "mmsi": "int64", "longitude": "float64", "latitude": "float64",
    "sog": "float64", "cog": "float64", "heading": "float64",
    "vessel_name": "str", "imo": "str", "call_sign": "str",
    "vessel_type": "str", "status": "str",
    "length": "float64", "width": "float64", "draft": "float64",
    "cargo": "str", "transceiver": "str",
}

def load_day(filepath):
    """Stream-decompress a single .csv.zst file and filter to the region."""
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        reader = dctx.stream_reader(fh)
        df = pd.read_csv(
            reader, dtype=DTYPES,
            parse_dates=["base_date_time"], na_values=["", " "],
        )
    mask = (
        (df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX) &
        (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
    )
    return df.loc[mask].copy()

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv.zst"))
print(f"Found {len(files)} daily files")

dfs = []
for f in files:
    day_df = load_day(os.path.join(DATA_DIR, f))
    print(f"  {f}: {len(day_df):>9,} rows in {REGION}")
    dfs.append(day_df)

raw = pd.concat(dfs, ignore_index=True)
del dfs
print(f"\nTotal: {len(raw):,} rows, {raw['mmsi'].nunique():,} vessels")

# %% [markdown]
# ## 3. Clean

# %%
n0 = len(raw)

# Drop rows missing critical fields
raw.dropna(subset=["mmsi", "base_date_time", "latitude", "longitude"], inplace=True)

# Remove invalid coordinates
raw = raw[raw["latitude"].between(-90, 90) & raw["longitude"].between(-180, 180)]

# SOG outliers (keep NaN SOG — valid positions without speed)
raw = raw[(raw["sog"].between(0, SOG_MAX)) | raw["sog"].isna()]

# AIS sentinels: heading = 511 means "not available"
raw.loc[raw["heading"] == 511, "heading"] = np.nan
raw.loc[raw["heading"] >= 360, "heading"] = np.nan
raw.loc[raw["cog"] >= 360, "cog"] = np.nan

# vessel_type loaded as str because of blanks — convert now
raw["vessel_type"] = pd.to_numeric(raw["vessel_type"], errors="coerce")

# Sort by vessel then time (required before computing gaps)
raw.sort_values(["mmsi", "base_date_time"], inplace=True)
raw.reset_index(drop=True, inplace=True)

print(f"After cleaning: {len(raw):,} rows ({n0 - len(raw):,} removed)")

# %% [markdown]
# ## 4. Assign Trip IDs
#
# A single vessel may leave the area, drop signal, or turn off AIS. We split
# its data into separate trips whenever consecutive points are more than
# `MAX_GAP_MINUTES` apart. This prevents the model from training across
# discontinuities.

# %%
dt = raw.groupby("mmsi")["base_date_time"].diff()
gap_mask = dt > pd.Timedelta(minutes=MAX_GAP_MINUTES)
raw["trip_seg"] = gap_mask.groupby(raw["mmsi"]).cumsum().astype(int)
raw["trip_id"] = raw["mmsi"].astype(str) + "_" + raw["trip_seg"].astype(str)

# Discard trips too short to be useful
trip_lens = raw.groupby("trip_id").size()
valid = trip_lens[trip_lens >= MIN_TRAJ_POINTS].index
raw = raw[raw["trip_id"].isin(valid)].copy()
raw.reset_index(drop=True, inplace=True)

print(f"Trips: {raw['trip_id'].nunique():,}  (after dropping <{MIN_TRAJ_POINTS} pts)")

# %% [markdown]
# ## 5. Interpolate to Regular Timestep
#
# Raw AIS broadcasts arrive at irregular intervals. Sequence models need
# regularly spaced time steps — we resample every trip to one point per
# `TIMESTEP_MIN` minutes.
#
# - **Linear** interpolation for lat/lon/SOG/draft/length/width
# - **Circular** interpolation for COG/heading (handles 0°/360° wraparound)
# - **Forward-fill** for categorical fields

# %%
def circular_interp(series):
    """Interpolate angular values across the 0°/360° boundary correctly."""
    notna = series.notna()
    if notna.sum() < 2:
        return series
    rad = np.deg2rad(series.values.astype(float))
    unwrapped = np.unwrap(rad)
    s = pd.Series(unwrapped, index=series.index)
    s[~notna] = np.nan
    s = s.interpolate(method="index")
    return np.mod(np.rad2deg(s), 360)

def interpolate_trip(trip_df, interval):
    """Resample one trip onto a regular time grid."""
    trip_df = trip_df.set_index("base_date_time").sort_index()
    trip_df = trip_df[~trip_df.index.duplicated(keep="first")]

    # Ceil start / floor end so we only interpolate inside the observed range
    start = trip_df.index.min().ceil(interval)
    end = trip_df.index.max().floor(interval)
    if start >= end:
        return pd.DataFrame()
    new_idx = pd.date_range(start, end, freq=interval)

    combined = trip_df.reindex(trip_df.index.union(new_idx))

    for col in ["latitude", "longitude", "sog", "draft", "length", "width"]:
        if col in combined.columns:
            combined[col] = combined[col].interpolate(method="index")
    for col in ["cog", "heading"]:
        if col in combined.columns:
            combined[col] = circular_interp(combined[col])
    for col in ["mmsi", "vessel_name", "imo", "call_sign", "vessel_type",
                "status", "cargo", "transceiver", "trip_id", "trip_seg"]:
        if col in combined.columns:
            combined[col] = combined[col].ffill().bfill()

    result = combined.loc[new_idx].copy()
    result.index.name = "base_date_time"
    return result.reset_index()

# %%
print(f"Interpolating {raw['trip_id'].nunique():,} trips to {TIMESTEP_MIN}-min grid ...")
trips = [grp for _, grp in raw.groupby("trip_id")]
interp_list = []
for i, trip_df in enumerate(trips):
    out = interpolate_trip(trip_df, INTERVAL)
    if len(out) >= MIN_TRAJ_POINTS:
        interp_list.append(out)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(trips)} trips")

interp = pd.concat(interp_list, ignore_index=True)
print(f"\nInterpolated: {len(interp):,} rows, {interp['trip_id'].nunique():,} trips")

# %% [markdown]
# ## 6. Save

# %%
interp.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved → {OUTPUT_FILE}")
print(f"  {len(interp):,} rows")
print(f"  {interp['trip_id'].nunique():,} trips")
print(f"  {interp['mmsi'].nunique():,} vessels")
print(f"  Columns: {list(interp.columns)}")
