# ============================================================
# Load spike-sorted Skieur data
# ============================================================

def load_pickled_ss(file_prefix, session_type, dt):
    """Load pickled spike-sorted data from NAS."""
    data_path = os.path.join(NAS, f"{file_prefix}_{session_type}_{dt}_data_ss")
    feat_path = os.path.join(NAS, f"{file_prefix}_{session_type}_{dt}_feature_ss")
    with open(data_path, "rb") as f:
        n_data = pickle.load(f)
    with open(feat_path, "rb") as f:
        f_data = pickle.load(f)
    return n_data, f_data


print("Loading hs0...")
n_data_hs0, f_data_hs0 = load_pickled_ss("SKIEUR_hs_0", SESSION_TYPE, dt)
print(f"  hs0: {len(n_data_hs0)} sessions")

print("Loading hs1...")
n_data_hs1, f_data_hs1 = load_pickled_ss("SKIEUR_hs_1", SESSION_TYPE, dt)
print(f"  hs1: {len(n_data_hs1)} sessions")

# Add Velocity_x
for f_df in f_data_hs0 + f_data_hs1:
    pos = f_df["Position"].values
    vel = np.diff(pos); vel = np.append(0, vel)
    vel = vel * 100; vel[~np.isfinite(vel)] = 0
    f_df["Velocity_x"] = vel

n_data_all_raw = list(n_data_hs0) + list(n_data_hs1)
f_data_all_raw = list(f_data_hs0) + list(f_data_hs1)
n_hs0 = len(n_data_hs0)

MIN_GC = 10
n_data_all, f_data_all = [], []
n_hs0_filtered = 0
for i, nd in enumerate(n_data_all_raw):
    if nd.shape[0] >= MIN_GC:
        n_data_all.append(nd)
        f_data_all.append(f_data_all_raw[i])
        if i < n_hs0:
            n_hs0_filtered += 1

n_hs0 = n_hs0_filtered
print(f"After gc>={MIN_GC}: {len(n_data_all)} sessions "
      f"(hs0={n_hs0}, hs1={len(n_data_all)-n_hs0})")

example = f_data_all[0]
print(f"Example: {example.shape[0]:,} tp, {n_data_all[0].shape[0]} neurons")
print(f"  Velocity_x: [{example['Velocity_x'].min():.1f}, "
      f"{example['Velocity_x'].max():.1f}]")
print(f"  Position:   [{example['Position'].min():.1f}, "
      f"{example['Position'].max():.1f}]")
print("Cell 1 -- Data loaded.")
