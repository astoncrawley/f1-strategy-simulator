import pandas as pd
import numpy as np

# --- Parameters ---
PIT_LOSS = 19.3           # seconds lost in a Monaco pit stop (drive + stop)
MIN_FOLLOW_GAP = 0.5      # minimum time gap left to a car ahead when blocked
LAPS = 78

# Compound model - use small simple model
COMPOUND_BASE = {"soft": 0.0, "medium": 0.35, "hard": 0.8}
DEGRADE = {"soft": 0.06, "medium": 0.035, "hard": 0.02}

# --- Helpers ---
def compound_extra(compound, age):
    """extra seconds added to base lap by compound + degradation of current stint"""
    return COMPOUND_BASE[compound] + DEGRADE[compound] * age

def compute_subject_free_lap(base_pace, compound, stint_age, noise=0.0):
    """Estimate free lap time for the subject driver for a lap."""
    # base_pace is driver's baseline lap time (e.g. race-mode base); noise left 0 for deterministic
    return base_pace + compound_extra(compound, stint_age) + noise

# --- Load data ---
# race_data must contain actual lap times for all drivers from the real race.
# columns: driver, lap (int), lap_time (float seconds), pit (bool), compound (str)
race_data = pd.read_csv("monaco2025_race_data.csv") 

def compute_total_time(driver, strategy, lap_data):
    """
    driver: name of driver to simulate
    strategy: list of lap numbers the driver will pit
    lap_data: dataframe filtered for that driver containing lap times per lap
    """
    total_time = 0
    pit_index = 0
    for lap, row in lap_data.iterrows():
        if pit_index < len(strategy) and row['lap'] == strategy[pit_index]:
            total_time += PIT_LOSS
            pit_index += 1
        total_time += row['lap_time']
    return total_time

# Example: simulate changing Leclerc's strategy
drivers = race_data['driver'].unique()
actual_strategies = {"Leclerc": [23, 51], ...}  # fill with real data

def simulate_new_strategy(driver, new_strategy):
    results = {}
    
    for d in drivers:
        laps = race_data[race_data['driver'] == d]
        strat = new_strategy if d == driver else actual_strategies[d]
        total_time = compute_total_time(d, strat, laps)
        results[d] = total_time
    
    # sort by finishing time
    finishing_order = sorted(results.items(), key=lambda x: x[1])
    return finishing_order

# Example run:
new_order = simulate_new_strategy("Leclerc", [18, 52])
print(new_order)











# Determine starting order from grid (or from lap 1 cumulative order)
# Here we infer starting order from min lap 1 time or supply separately.
starting_order = race_data[race_data['lap']==1].sort_values('grid_position')['driver'].tolist()
# If grid_position not available, fallback:
# starting_order = race_data[race_data.lap==1].sort_values('lap_time')['driver'].tolist()

# Build per-driver lookup for actual lap times and pit flags (others fixed)
drivers = sorted(race_data['driver'].unique())

# actual_lap_times[driver][lap] = lap_time (from real race)
actual_lap_times = {d: {} for d in drivers}
actual_pit_flags = {d: set() for d in drivers}
actual_compounds = {d: {} for d in drivers}
for _, r in race_data.iterrows():
    d = r['driver']; lap = int(r['lap'])
    actual_lap_times[d][lap] = float(r['lap_time'])
    if bool(r.get('pit', False)):
        actual_pit_flags[d].add(lap)
    actual_compounds[d][lap] = r.get('compound', None)

# Subject driver and target strategy (example)
subject = "Charles Leclerc"
# strategy: list of tuples (pit_lap, compound_to_leave_on)
subject_strategy = [(18, 'medium'), (50, 'soft')]   # example; MUST be consistent with rules

# baseline pace for subject (seconds): use subject's actual base lap (race-ish) or computed
# For deterministic run, set a base lap (this is the "race-mode baseline" without comp adjustments)
# One safe approach: compute subject's average lap time (excluding pit laps & outliers) from race_data
sub_laps = [t for lap,t in actual_lap_times[subject].items() if lap not in actual_pit_flags[subject]]
base_pace = np.mean(sub_laps) if sub_laps else 71.5   # fallback

# --- Simulation state ---
cumulative_time = {d: 0.0 for d in drivers}
completed_laps = {d: 0 for d in drivers}
current_compound = {}
stint_age = {}
next_pit_idx = {}

# initialize: everyone starts on their first-lap compound from data
for d in drivers:
    current_compound[d] = actual_compounds[d].get(1, "soft")
    stint_age[d] = 0
    # for subject, we will override strategy, so build pit list
    next_pit_idx[d] = 0

# For subject, prepare pit lap list and compounds leaving pit:
subject_pit_laps = [p for p,_ in subject_strategy]
subject_pit_compounds = {p:c for p,c in subject_strategy}

# For others, convert their actual pit flags into lists of pit laps (sorted)
other_pit_list = {d: sorted(list(actual_pit_flags[d])) for d in drivers if d != subject}

# Start simulation, track order (list) — initial order = starting_order
order = starting_order.copy()

for lap in range(1, LAPS+1):
    # We'll store lap times just computed for this lap to enable blocking logic
    lap_time_this_lap = {}
    # iterate drivers in current order (so the car ahead's lap_time is available when reaching subject)
    for pos, d in enumerate(order):
        # Determine if this driver pits this lap
        if d == subject:
            pits_this_lap = lap in subject_pit_laps
        else:
            pits_this_lap = lap in other_pit_list.get(d, [])
        # compute lap_time
        if d != subject:
            # other drivers: use their actual lap time (deterministic)
            lap_t = actual_lap_times[d].get(lap, 9999.0)  # if missing, big time
            # If they pitted and actual lap time doesn't include pit loss, ensure we add pit loss accordingly
            # (We assume actual lap_time already includes any pit loss from actual race)
            # update compound/stint if they pitted
            if pits_this_lap:
                # If data includes compounds, set compound for next laps based on actual_compounds
                current_compound[d] = actual_compounds[d].get(lap+1, current_compound[d])
                stint_age[d] = 0
            else:
                stint_age[d] += 1
        else:
            # subject: compute "free" lap time based on strategy and compound
            # figure current compound (if subject pitted this lap, they leave on subject_pit_compounds[lap])
            if lap in subject_pit_laps:
                # they perform pit *during* this lap, pit costs PIT_LOSS; we assume exit compound applied on next lap
                # Many models treat pit lap as still running some partial laptime + pit_loss; for simplicity add PIT_LOSS
                # We'll model the lap's driving time as reduced by some factor (you can refine), but simpler:
                lap_t = PIT_LOSS
                # after pit, set compound for next lap
                current_compound[d] = subject_pit_compounds[lap]
                stint_age[d] = 0
            else:
                # compute free lap using base_pace + compound/degradation
                free = compute_subject_free_lap(base_pace, current_compound[d], stint_age[d], noise=0.0)
                # blocking check: find driver directly ahead in this lap (pos-1), if any
                if pos > 0:
                    driver_ahead = order[pos-1]
                    # lap_time of driver_ahead must already be computed this lap (because we iterate in order)
                    lap_ahead = lap_time_this_lap.get(driver_ahead, actual_lap_times[driver_ahead].get(lap, 9999.0))
                    T_subject = cumulative_time[d]
                    T_ahead = cumulative_time[driver_ahead]
                    # If subject would finish the lap before the driver ahead (minus min gap), cap them
                    if (T_subject + free) < (T_ahead + lap_ahead - MIN_FOLLOW_GAP):
                        # set lap_t so subject finishes MIN_FOLLOW_GAP behind driver_ahead
                        lap_t = (T_ahead + lap_ahead + MIN_FOLLOW_GAP) - T_subject
                        # safety clamp: cannot be negative
                        lap_t = max(lap_t, 0.1)
                    else:
                        lap_t = free
                else:
                    # no car ahead (pole or currently ahead) -> free lap
                    lap_t = free
                # update stint_age if not pitting
                stint_age[d] += 1

        # add lap time to cumulative
        cumulative_time[d] += lap_t
        completed_laps[d] += 1
        lap_time_this_lap[d] = lap_t

    # end of lap: recompute order based on (laps completed desc, cumulative time asc)
    # Since all drivers complete same lap number by design, sort by cumulative time
    order = sorted(drivers, key=lambda x: ( -completed_laps[x], cumulative_time[x] ))

# After all laps, compute final finishing order and times
final = sorted(drivers, key=lambda x: (-completed_laps[x], cumulative_time[x]))
print("Final classification (no overtakes, blocking enabled):")
for i,d in enumerate(final, start=1):
    print(f"{i}. {d}  time={cumulative_time[d]:.2f}  laps={completed_laps[d]}")

