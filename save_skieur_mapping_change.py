"""Save SKIEUR mapping_change_only data from data5 to local Desktop.

Key insight: get_sessions() builds Linux paths like /auto/data6/... but on Windows
the same NAS is reachable via //129.199.81.18/data5/... — we remap the root.
"""

import numpy as np
import pandas as pd
import os
import pickle
from tqdm.auto import tqdm
from utils_load_data import get_data_spike_sorted

# ── Parameters ───────────────────────────────────────────
SHEET_ID = '1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU'
SHEET_NAME = 'SKIEUR'
DT = 0.005             # 0.005, 0.01, 0.02 available in spike_sorting/
HEADSTAGE = 0
SESSION_TYPE = 'mapping_change_only'
SPIKE_SORTED = True
REMOVE_BASELINE = False
SAVE_DIR = 'C:/Users/PenPen/Desktop/Ferret/Data/'

# Linux automount prefix → Windows SMB network prefix
WINDOWS_MOUNT_BASE = '//129.199.81.18/data5/eTheremin'

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Load session list from Google Sheets ───────────────
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"
df = pd.read_csv(url)
filtered = df[df['use'] == 'yes']
filtered = filtered[filtered['type'].isin([SESSION_TYPE])]
session_names = filtered['session'].tolist()

print(f"Google Sheet returned {len(session_names)} sessions for type '{SESSION_TYPE}'")

# ── 2. Build Windows-accessible paths ───────────────────
paths = [
    f"{WINDOWS_MOUNT_BASE}/{SHEET_NAME}/{s}/headstage_{HEADSTAGE}/"
    for s in session_names
]

# Verify first / last path
print(f"First: {paths[0]}")
print(f"Last:  {paths[-1]}")

# Quick existence check
missing = [p for p in paths if not os.path.isdir(p)]
if missing:
    print(f"WARNING: {len(missing)}/{len(paths)} paths not reachable:")
    for p in missing[:3]:
        print(f"  {p}")
else:
    print("All session paths reachable.")

# ── 3. Load spike-sorted data ────────────────────────────
n_data_s, f_data_s, neuron_ids_s = get_data_spike_sorted(
    paths, DT,
    gc_or_not=False,
    remove_baseline=REMOVE_BASELINE
)

print(f"\nLoaded {len(n_data_s)} sessions")

# ── 4. Save to local Desktop ─────────────────────────────
prefix = f"{SAVE_DIR}{SHEET_NAME}_hs_{HEADSTAGE}_{SESSION_TYPE}_{DT}"

data_path = f"{prefix}_data_ss"
with open(data_path, "wb") as fp:
    pickle.dump(n_data_s, fp)
print(f"Saved: {data_path}  ({len(n_data_s)} arrays)")

feat_path = f"{prefix}_feature_ss"
with open(feat_path, "wb") as fp:
    pickle.dump(f_data_s, fp)
print(f"Saved: {feat_path}  ({len(f_data_s)} DataFrames)")

print("\nDone!")
