# %% [markdown]
# # Merge per-month trajectory parquets
#
# Run this **after** you've processed every month with `generate_trajectories.ipynb`
# (with `MONTH_TAG` set per run). It:
#
# 1. Globs all tagged monthly parquets in `processed/`.
# 2. Concatenates them.
# 3. **Redoes** trip-ID assignment globally based on `base_date_time` gaps
#    of more than `MAX_GAP_MINUTES`. This handles the (rare) case of a vessel
#    transmitting continuously across the boundary between two monthly files
#    — those tracks get merged into one trip rather than two.
# 4. Drops trips shorter than `MIN_TRAJ_POINTS` after re-segmentation.
# 5. Saves the canonical `processed/ais_trajectories_{REGION}_{TIMESTEP}min.parquet`
#    that `make_splits.ipynb` consumes.
#
# This step is unnecessary if you ran `generate_trajectories.ipynb` once with
# `MONTH_TAG = ""` (the canonical file is already in place).

# %% [markdown]
# ## 1. Configuration

# %%
import glob
import os
import numpy as np
import pandas as pd

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  >>> Must match the values used in generate_trajectories.ipynb <<<   ║
# ╚══════════════════════════════════════════════════════════════════════╝
REGION = "West Coast"
TIMESTEP_MIN = 3

OUT_DIR = "processed"
MAX_GAP_MINUTES = 30      # same threshold used in generate_trajectories.ipynb
MIN_TRAJ_POINTS = 10      # same threshold used in generate_trajectories.ipynb

CANONICAL_FILE = os.path.join(
    OUT_DIR, f"ais_trajectories_{REGION}_{TIMESTEP_MIN}min.parquet"
)
MONTHLY_GLOB = os.path.join(
    OUT_DIR, f"ais_trajectories_{REGION}_{TIMESTEP_MIN}min_*.parquet"
)

# %% [markdown]
# ## 2. Find monthly parquets

# %%
monthly_files = sorted(glob.glob(MONTHLY_GLOB))
if not monthly_files:
    raise FileNotFoundError(
        f"No monthly parquets matching {MONTHLY_GLOB!r}. Either you haven't "
        f"run generate_trajectories.ipynb with MONTH_TAG set, or the region/"
        f"timestep doesn't match. If your existing parquet has no month tag, "
        f"rename it to include a tag (e.g. `_2024-05`) before running this step."
    )

print(f"Found {len(monthly_files)} monthly parquet(s):")
for f in monthly_files:
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f}  ({size_mb:,.1f} MB)")

# %% [markdown]
# ## 3. Concatenate

# %%
parts = [pd.read_parquet(f) for f in monthly_files]
df = pd.concat(parts, ignore_index=True)
del parts
print(f"\nConcatenated: {len(df):,} rows, {df['mmsi'].nunique():,} vessels")
print(f"  Date range: {df['base_date_time'].min()} → {df['base_date_time'].max()}")

# %% [markdown]
# ## 4. Re-assign trips globally
#
# Per-month files have trip IDs that are only unique within that month
# (vessel 12345 has `12345_0`, `12345_1`, … in *every* month). After concat
# we throw those away and rebuild from gap detection.
#
# Because the per-month interpolation places points on the same global 3-min
# grid, two consecutive monthly ticks across a month boundary are exactly
# 3 minutes apart. So a vessel that transmitted continuously across the
# boundary will not show a `>MAX_GAP_MINUTES` gap and will be correctly
# merged into one trip. A vessel that had a real silence period at the
# boundary still shows a gap and will be split — which is what we want.

# %%
df = df.sort_values(["mmsi", "base_date_time"]).reset_index(drop=True)

dt = df.groupby("mmsi")["base_date_time"].diff()
gap_mask = dt > pd.Timedelta(minutes=MAX_GAP_MINUTES)
df["trip_seg"] = gap_mask.groupby(df["mmsi"]).cumsum().astype(int)
df["trip_id"] = df["mmsi"].astype(str) + "_" + df["trip_seg"].astype(str)

n_trips_before_filter = df["trip_id"].nunique()
print(f"\nTrips after global re-segmentation: {n_trips_before_filter:,}")

# %% [markdown]
# ## 5. Drop trips that are too short
#
# Re-segmentation can leave very short tracks at month boundaries (e.g., a
# vessel with one tick in May and a long gap before its next appearance).
# Apply the same `MIN_TRAJ_POINTS` cutoff that the per-month step used.

# %%
trip_lens = df.groupby("trip_id").size()
keep_trips = trip_lens[trip_lens >= MIN_TRAJ_POINTS].index
n_dropped = n_trips_before_filter - len(keep_trips)
df = df[df["trip_id"].isin(keep_trips)].reset_index(drop=True)
print(f"Dropped {n_dropped:,} trips with < {MIN_TRAJ_POINTS} points "
      f"(kept {len(keep_trips):,})")

# %% [markdown]
# ## 6. Save canonical parquet

# %%
df.to_parquet(CANONICAL_FILE, index=False)
print(f"\nSaved → {CANONICAL_FILE}")
print(f"  {len(df):,} rows")
print(f"  {df['trip_id'].nunique():,} trips")
print(f"  {df['mmsi'].nunique():,} vessels")
print(f"  Columns: {list(df.columns)}")

# %% [markdown]
# ## 7. Optional cleanup
#
# The monthly tagged parquets are no longer needed. Uncomment the loop below
# if you want to delete them automatically. Otherwise leave them in place —
# you can rerun this notebook to regenerate the canonical file at any time.

# %%
# for f in monthly_files:
#     os.remove(f)
#     print(f"removed {f}")
