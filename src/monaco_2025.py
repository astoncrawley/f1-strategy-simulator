"""
make_monaco2025_formatA.py
Fetches Monaco 2025 (Timing-only) with FastF1 and writes a Format A CSV with:
 driver, lap, base_lap_time, compound, tyre_age, delta_if_held_up, is_pit_lap, pit_loss

Behavior:
 - Pit laps: we replace the pit-lap distortion by estimating the lap driving component:
     base_driving_lap = typical outlap/inlap estimate (we use median of similar laps)
   and set pit_loss = raw_pitlap_time - base_driving_lap  (so pit_loss is the time to be added if you
   want to model pit stops separately)
 - Gap-based delta_if_held_up: for each driver & lap we compute the gap to car ahead; if gap <= GAP_THRESHOLD,
   delta_if_held_up shows the typical per-lap penalty (positive) the driver would suffer while being "held up".
   The value is computed as max( min_penalty, (driver_median_free_lap - car_ahead_median_free_lap) * penalty_factor )
   but only applied where gap <= GAP_THRESHOLD.
 - Deterministic output (no RNG). You can tweak parameters below.

Requires: fastf1, pandas, numpy
"""

import fastf1
from fastf1.core import Laps
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --------------------
# CONFIG
# --------------------
YEAR = 2025
# Monaco was Round 8 in 2025 per F1 calendars — FastF1 can load by (year, round) or event name.
# If round is different on your copy, you can set `event_name = 'Monaco'` instead.
ROUND = 8  # adjust if your FastF1 calendar differs
OUTPUT_CSV = "monaco2025_formatA_C_gap.csv"

# pit handling (Option C)
MIN_DRIVING_LAP_EST = 60.0  # minimum plausible driving lap (s) safe-guard
# Gap model
GAP_THRESHOLD = 1.2   # seconds — apply held-up penalty only when gap to car ahead <= this
MIN_PENALTY = 0.15    # seconds — minimum penalty when held up
PENALTY_FACTOR = 0.9  # scaling factor of pace diff applied as penalty

# --------------------
# Helper functions
# --------------------
def sec(td):
    # convert pandas Timedelta or string to seconds float (fastf1 returns Timedelta)
    try:
        return td.total_seconds()
    except Exception:
        return float(td)

def estimate_driving_lap_from_neighbours(laps_df, driver, lapnum):
    """
    Estimate a plausible driving lap for driver on 'lapnum' when that lap included a pit stop.
    Approach:
     - look at that driver's neighbouring non-pit laps on the same stint (outlap or inlap).
     - if none, use driver's median free lap on that compound (or overall median).
     - fallback: median of the field for that lap index.
    """
    # if lap exists and is not pit lap, return directly
    row = laps_df[(laps_df.driver==driver) & (laps_df.lap==lapnum)]
    if row.empty:
        return np.nan
    r = row.iloc[0]
    if not r['is_pit_lap']:
        return r['lap_time_s']

    # find driver median non-pit laps on same compound
    driver_nonpit = laps_df[(laps_df.driver==driver) & (~laps_df.is_pit_lap)]
    if not driver_nonpit.empty:
        med = driver_nonpit['lap_time_s'].median()
        if med >= MIN_DRIVING_LAP_EST:
            return med

    # fallback: median of all non-pit laps in race
    all_med = laps_df[~laps_df.is_pit_lap]['lap_time_s'].median()
    return max(all_med, MIN_DRIVING_LAP_EST)

# --------------------
# Load race with FastF1 (Timing-only)
# --------------------
# Note: FastF1 will try to download timing files from the upstream source; run locally.
fastf1.Cache.enable_cache('cache')  # local cache dir
# Load race session
race = fastf1.get_session(YEAR, ROUND, 'R')   # 'R' = race
race.load(laps=True, telemetry=False)         # timing only

# Extract laps table as DataFrame
laps = race.laps.copy()   # fastf1.core.Laps object
# Convert to pandas for processing
laps_df = laps[['Driver','LapNumber','LapTime','PitOutTime','Compound','IsPitLap']].to_pandas()

# Normalize columns and convert types
laps_df = laps_df.rename(columns={
    'Driver': 'driver',
    'LapNumber': 'lap',
    'LapTime': 'lap_time_td',
    'PitOutTime': 'pit_out_time_td',
    'Compound': 'compound',
    'IsPitLap': 'is_pit_lap'
})
laps_df['lap_time_s'] = laps_df['lap_time_td'].apply(lambda x: x.total_seconds() if pd.notnull(x) else np.nan)
laps_df['is_pit_lap'] = laps_df['is_pit_lap'].fillna(False).astype(bool)
# FastF1 compounds sometimes are bytes/enum; ensure string
laps_df['compound'] = laps_df['compound'].astype(str).str.lower().replace('nan', 'unknown')

# create a complete row set driver x lap (some drivers may have missing laps if retired)
drivers = sorted(laps_df['driver'].unique())
max_lap = int(laps_df['lap'].max())
rows = []
# create quick lookup for median driving lap per driver (non-pit)
driver_median = {}
for d in drivers:
    dr_nonpit = laps_df[(laps_df.driver==d) & (~laps_df.is_pit_lap)]
    driver_median[d] = dr_nonpit['lap_time_s'].median() if not dr_nonpit.empty else np.nan

# We'll need the order on each lap to compute gap to car ahead (use cumulative race time)
# First compute cumulative times by driver per lap (using lap_time_s including pit laps for ordering)
laps_df = laps_df.sort_values(['driver','lap'])
laps_df['cum_time_s'] = laps_df.groupby('driver')['lap_time_s'].cumsum()

# Build a per-lap snapshot ordering by laps completed and cumulative time
# For each lap n: consider each driver who has lap >= n (i.e., completed lap n), extract their cum_time at lap n
for lapnum in range(1, max_lap+1):
    # drivers who completed this lap
    df_lap = laps_df[laps_df.lap==lapnum].copy()
    # For drivers missing this lap (DNF before) we skip them
    # order by cum_time_s ascending
    df_lap = df_lap.sort_values('cum_time_s').reset_index(drop=True)

    # For each driver on this lap, compute gap to car ahead (in seconds)
    for idx, r in df_lap.iterrows():
        driver = r['driver']
        lap_time_s = r['lap_time_s']
        compound = r['compound']
        is_pit = bool(r['is_pit_lap'])
        # estimate driving lap if pit lap (Option C)
        if is_pit:
            est_drive = estimate_driving_lap_from_neighbours(laps_df, driver, lapnum)
            pit_loss = lap_time_s - est_drive if pd.notnull(lap_time_s) and pd.notnull(est_drive) else np.nan
            base_lap_time = est_drive
        else:
            base_lap_time = lap_time_s
            pit_loss = 0.0

        # tyre age: compute stint lap count for that driver up to this lap using IsPitLap markers
        dr_laps = laps_df[(laps_df.driver==driver) & (laps_df.lap <= lapnum)].copy()
        # Count laps since last pit (including this lap)
        if not dr_laps.empty:
            # find last pit lap index
            last_pit_idx = dr_laps[dr_laps['is_pit_lap']].last_valid_index()
            if last_pit_idx is None:
                tyre_age = len(dr_laps)
            else:
                tyre_age = int(dr_laps.index[-1] - last_pit_idx)  # approximate
            tyre_age = max(1, tyre_age)
        else:
            tyre_age = 1

        # gap to car ahead (if idx>0)
        if idx == 0:
            gap_ahead = np.nan
            delta_if_held_up = 0.0
        else:
            ahead_row = df_lap.iloc[idx-1]
            gap_ahead = r['cum_time_s'] - ahead_row['cum_time_s']
            # apply gap-based rule: if gap <= GAP_THRESHOLD, compute penalty
            if gap_ahead <= GAP_THRESHOLD:
                # compute an estimated per-lap penalty:
                # median free lap difference between driver and car ahead (positive means driver is faster)
                med_driver = driver_median.get(driver, np.nan)
                med_ahead = driver_median.get(ahead_row['driver'], np.nan)
                if pd.notnull(med_driver) and pd.notnull(med_ahead):
                    diff = med_driver - med_ahead
                    # if diff < 0 => driver is faster (negative), penalty = max(MIN_PENALTY, -diff*PENALTY_FACTOR)
                    if diff < 0:
                        delta_if_held_up = max(MIN_PENALTY, (-diff) * PENALTY_FACTOR)
                    else:
                        # if driver is slower than car ahead, they won't be held up; set 0
                        delta_if_held_up = 0.0
                else:
                    delta_if_held_up = MIN_PENALTY
            else:
                delta_if_held_up = 0.0

        rows.append({
            'driver': driver,
            'lap': int(lapnum),
            'base_lap_time': float(base_lap_time) if pd.notnull(base_lap_time) else np.nan,
            'compound': compound,
            'tyre_age': int(tyre_age),
            'delta_if_held_up': float(delta_if_held_up),
            'is_pit_lap': bool(is_pit),
            'pit_loss': float(pit_loss) if pd.notnull(pit_loss) else 0.0
        })

# Build final DataFrame
out_df = pd.DataFrame(rows)
# Ensure ordering driver, lap
out_df = out_df.sort_values(['driver','lap']).reset_index(drop=True)

# Save CSV
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV}  (rows: {len(out_df)})")
print("Columns:", list(out_df.columns))
