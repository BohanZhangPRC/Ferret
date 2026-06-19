"""
Ridge Regression Velocity Encoding: MP1 (tracking_only) vs MP2 (mapping_change).

For each pair, for each neuron, predict spike count from velocity (Speed_x + lags)
using RidgeCV with 5-fold shuffled CV. Compare per-neuron R^2 across mappings.

Output:
  - ridge_velocity_encoding_results.csv
  - ridge_velocity_encoding_scatter.png
  - ridge_velocity_encoding_hist.png
"""

import os, pickle, io, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

# ── Config ──────────────────────────────────────────────────
NAS = r"\\129.199.81.18\data5\eTheremin"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 0.005
N_LAGS = 3  # temporal lags: t-N_LAGS ... t+N_LAGS
ALPHAS = np.logspace(-2, 4, 20)
N_FOLDS = 5
RANDOM_STATE = 42

SHEET_URL = ("https://docs.google.com/spreadsheets/d/"
             "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU/"
             "gviz/tq?tqx=out:csv&sheet=SKIEUR")

MP1_TYPE = "tracking_only"
MP2_TYPE = "mapping_change"
MP1_PICKLE_DATA = f"{NAS}/SKIEUR_hs_0_Tracking_only_0.005_data_ss"
MP1_PICKLE_FEAT = f"{NAS}/SKIEUR_hs_0_Tracking_only_0.005_feature_ss"
MP2_PICKLE_DATA = f"{NAS}/SKIEUR_hs_0_mapping_change_0.005_data_ss"
MP2_PICKLE_FEAT = f"{NAS}/SKIEUR_hs_0_mapping_change_0.005_feature_ss"


# ══════════════════════════════════════════════════════════════
# 1. Load Google Sheet & build pairing
# ══════════════════════════════════════════════════════════════

def load_google_sheet():
    """Load SKIEUR sheet and return use=yes sessions with types."""
    try:
        df = pd.read_csv(SHEET_URL)
    except Exception:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "10", SHEET_URL],
            capture_output=True, text=True)
        df = pd.read_csv(io.StringIO(result.stdout))

    df = df[df["use"].astype(str).str.strip().str.lower() == "yes"].copy()
    df["session"] = df["session"].str.strip()
    df["type"] = df["type"].str.strip()
    return df


def build_pairs(df_sheet):
    """Return list of (mp1_session_name, mp2_session_name) pairs."""
    names_mp1 = df_sheet[df_sheet["type"] == MP1_TYPE]["session"].tolist()
    names_mp2 = df_sheet[df_sheet["type"] == MP2_TYPE]["session"].tolist()

    # Normalize paired_sessions column name
    paired_col = "paired_sessions" if "paired_sessions" in df_sheet.columns else "paired_session"

    pairs = []
    seen_mp1 = set()

    # Direction 1: MP1 -> paired -> MP2
    for _, row in df_sheet[df_sheet["type"] == MP1_TYPE].iterrows():
        mp1 = row["session"]
        mp2 = str(row.get(paired_col, "")).strip()
        if mp2 and mp2 != "nan" and mp2 in names_mp2:
            pairs.append((mp1, mp2))
            seen_mp1.add(mp1)

    # Direction 2: MP2 -> paired -> MP1 (reverse)
    for _, row in df_sheet[df_sheet["type"] == MP2_TYPE].iterrows():
        mp2 = row["session"]
        mp1 = str(row.get(paired_col, "")).strip()
        if mp1 and mp1 != "nan" and mp1 in names_mp1 and mp1 not in seen_mp1:
            pairs.append((mp1, mp2))
            seen_mp1.add(mp1)

    print(f"  Google Sheet: {len(names_mp1)} MP1, {len(names_mp2)} MP2 -> {len(pairs)} pairs")
    return pairs


# ══════════════════════════════════════════════════════════════
# 2. Load pickles & match to pairs
# ══════════════════════════════════════════════════════════════

def match_sessions_to_pairs(n_data_list, f_data_list, session_names_ordered, pair_session_names):
    """
    Given a pickle's data list (ordered by load_sessions), extract the index
    corresponding to each target session name. Returns reordered data or None.
    """
    indices = []
    for target_name in pair_session_names:
        found = False
        for idx, (nd, fd) in enumerate(zip(n_data_list, f_data_list)):
            # Try to match session name — may appear in neuron_ids or be embedded
            # The pickle data order matches load_sessions order from Google Sheet
            if idx < len(session_names_ordered) and session_names_ordered[idx] == target_name:
                indices.append(idx)
                found = True
                break
        if not found:
            # Fallback: match by order (paired MP1[i] <-> MP2[i] if in same sheet order)
            pass
    return indices


def load_paired_data(pairs):
    """
    Load both pickles and align data by pair.
    Returns: list of (n_data_mp1, f_data_mp1, n_data_mp2, f_data_mp2, mp1_name, mp2_name)
    """
    with open(MP1_PICKLE_DATA, "rb") as f:
        n_data_mp1_all = pickle.load(f)
    with open(MP1_PICKLE_FEAT, "rb") as f:
        f_data_mp1_all = pickle.load(f)
    with open(MP2_PICKLE_DATA, "rb") as f:
        n_data_mp2_all = pickle.load(f)
    with open(MP2_PICKLE_FEAT, "rb") as f:
        f_data_mp2_all = pickle.load(f)

    # Session ordering in pickle = Google Sheet order
    df = load_google_sheet()
    mp1_names = df[df["type"] == MP1_TYPE]["session"].tolist()
    mp2_names = df[df["type"] == MP2_TYPE]["session"].tolist()

    print(f"  Pickle MP1: {len(n_data_mp1_all)} sessions")
    print(f"  Pickle MP2: {len(n_data_mp2_all)} sessions")

    paired_data = []
    for mp1_name, mp2_name in pairs:
        idx1 = mp1_names.index(mp1_name) if mp1_name in mp1_names else None
        idx2 = mp2_names.index(mp2_name) if mp2_name in mp2_names else None

        if idx1 is None or idx2 is None:
            print(f"  WARNING: pair ({mp1_name}, {mp2_name}) not found in pickles, skipping")
            continue
        if idx1 >= len(n_data_mp1_all) or idx2 >= len(n_data_mp2_all):
            print(f"  WARNING: pair index out of range, skipping")
            continue

        nd1, fd1 = n_data_mp1_all[idx1], f_data_mp1_all[idx1]
        nd2, fd2 = n_data_mp2_all[idx2], f_data_mp2_all[idx2]

        n_neurons_1, n_neurons_2 = nd1.shape[0], nd2.shape[0]

        paired_data.append({
            "mp1_name": mp1_name, "mp2_name": mp2_name,
            "n_data_mp1": nd1, "f_data_mp1": fd1,
            "n_data_mp2": nd2, "f_data_mp2": fd2,
            "n_neurons_mp1": n_neurons_1, "n_neurons_mp2": n_neurons_2,
            "n_time_mp1": nd1.shape[1], "n_time_mp2": nd2.shape[1],
        })
        print(f"  {mp1_name} <-> {mp2_name}: {n_neurons_1}v{n_neurons_2} neurons, "
              f"{nd1.shape[1]/200:.0f}s v{nd2.shape[1]/200:.0f}s")

    return paired_data


# ══════════════════════════════════════════════════════════════
# 3. Ridge Encoding
# ══════════════════════════════════════════════════════════════

def build_features(speed_x, n_lags=N_LAGS):
    """
    Build design matrix from Speed_x with temporal lags.
    Column j corresponds to lag = j - n_lags (+/- 50ms).
    lag < 0: future (roll left), lag > 0: past (roll right).
    """
    n = len(speed_x)
    lags = range(-n_lags, n_lags + 1)
    X = np.zeros((n, len(lags)))
    for j, lag in enumerate(lags):
        shifted = np.roll(speed_x, lag)   # lag < 0 = left shift, lag > 0 = right shift
        if lag < 0:       # future: pad end
            shifted[lag:] = speed_x[-1]
        elif lag > 0:     # past: pad beginning
            shifted[:lag] = speed_x[0]
        X[:, j] = shifted
    return X


def ridge_encode_neuron(spike_counts, speed_x, alphas=ALPHAS, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """
    Ridge encoding: velocity -> spike count.
    Uses TimeSeriesSplit to avoid temporal leakage.
    Returns R^2 (cross-validated). Returns NaN if model fails.
    """
    from sklearn.model_selection import TimeSeriesSplit

    y = np.sqrt(spike_counts.astype(float))
    X = build_features(speed_x)

    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if valid.sum() < 100:
        return np.nan
    y, X = y[valid], X[valid]

    try:
        model = RidgeCV(alphas=alphas)
        tscv = TimeSeriesSplit(n_splits=n_folds)
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X[test_idx])
        r2 = r2_score(y, y_pred)
        return r2
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════
# 4. Main Analysis
# ══════════════════════════════════════════════════════════════

def run_analysis(paired_data):
    """Run ridge encoding for all pairs and collect per-neuron R^2."""
    results = []

    for pair_idx, pair in enumerate(tqdm(paired_data, desc="Pairs")):
        nd1, fd1 = pair["n_data_mp1"], pair["f_data_mp1"]
        nd2, fd2 = pair["n_data_mp2"], pair["f_data_mp2"]
        speed1 = fd1["Speed_x"].values
        speed2 = fd2["Speed_x"].values

        # Align time bins (use min length)
        min_t1 = min(len(speed1), nd1.shape[1])
        min_t2 = min(len(speed2), nd2.shape[1])
        nd1_t = nd1[:, :min_t1]
        nd2_t = nd2[:, :min_t2]
        speed1_t = speed1[:min_t1]
        speed2_t = speed2[:min_t2]

        n1, n2 = nd1_t.shape[0], nd2_t.shape[0]
        # Compare only common neuron indices (same cluster/channel across phases)
        n_common = min(n1, n2)

        for neuron_i in range(n_common):
            r2_mp1 = ridge_encode_neuron(nd1_t[neuron_i], speed1_t)
            r2_mp2 = ridge_encode_neuron(nd2_t[neuron_i], speed2_t)
            results.append({
                "pair_idx": pair_idx,
                "mp1_name": pair["mp1_name"],
                "mp2_name": pair["mp2_name"],
                "neuron_idx": neuron_i,
                "n_neurons_mp1": n1,
                "n_neurons_mp2": n2,
                "R2_mp1": r2_mp1,
                "R2_mp2": r2_mp2,
                "delta_R2": r2_mp2 - r2_mp1 if not (np.isnan(r2_mp1) or np.isnan(r2_mp2)) else np.nan,
            })

        # Extra neurons in MP2 (no MP1 counterpart)
        for neuron_i in range(n_common, n2):
            r2_mp2 = ridge_encode_neuron(nd2_t[neuron_i], speed2_t)
            results.append({
                "pair_idx": pair_idx,
                "mp1_name": pair["mp1_name"],
                "mp2_name": pair["mp2_name"],
                "neuron_idx": neuron_i,
                "n_neurons_mp1": n1,
                "n_neurons_mp2": n2,
                "R2_mp1": np.nan,
                "R2_mp2": r2_mp2,
                "delta_R2": np.nan,
            })

        # Extra neurons in MP1 (no MP2 counterpart)
        for neuron_i in range(n_common, n1):
            r2_mp1 = ridge_encode_neuron(nd1_t[neuron_i], speed1_t)
            results.append({
                "pair_idx": pair_idx,
                "mp1_name": pair["mp1_name"],
                "mp2_name": pair["mp2_name"],
                "neuron_idx": neuron_i,
                "n_neurons_mp1": n1,
                "n_neurons_mp2": n2,
                "R2_mp1": r2_mp1,
                "R2_mp2": np.nan,
                "delta_R2": np.nan,
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# 5. Statistics & Visualization
# ══════════════════════════════════════════════════════════════

def compute_stats(df):
    """Paired comparison of R^2 (only common neurons with both values valid)."""
    paired = df.dropna(subset=["R2_mp1", "R2_mp2"]).copy()
    n_total = len(df)
    n_paired = len(paired)

    if n_paired == 0:
        print("\n  WARNING: No paired neurons with valid R^2 values.")
        return {}

    r2_mp1 = paired["R2_mp1"].values
    r2_mp2 = paired["R2_mp2"].values

    # Clamp negative R^2 to 0 for summary (but keep raw for stats)
    r2_mp1_clamped = np.maximum(r2_mp1, 0)
    r2_mp2_clamped = np.maximum(r2_mp2, 0)

    # Wilcoxon signed-rank test (non-parametric paired)
    try:
        stat_w, p_w = wilcoxon(r2_mp2, r2_mp1, alternative="two-sided")
    except Exception:
        stat_w, p_w = np.nan, np.nan

    # Paired t-test
    try:
        stat_t, p_t = ttest_rel(r2_mp2, r2_mp1)
    except Exception:
        stat_t, p_t = np.nan, np.nan

    stats = {
        "n_total_neurons": n_total,
        "n_paired_neurons": n_paired,
        "mean_R2_mp1": np.mean(r2_mp1_clamped),
        "mean_R2_mp2": np.mean(r2_mp2_clamped),
        "median_R2_mp1": np.median(r2_mp1_clamped),
        "median_R2_mp2": np.median(r2_mp2_clamped),
        "mean_delta_R2": np.mean(paired["delta_R2"]),
        "median_delta_R2": np.median(paired["delta_R2"]),
        "frac_R2_mp2_gt_mp1": np.mean(r2_mp2 > r2_mp1),
        "wilcoxon_stat": stat_w,
        "wilcoxon_p": p_w,
        "ttest_stat": stat_t,
        "ttest_p": p_t,
    }

    print("\n" + "=" * 60)
    print("  STATISTICS (paired neurons only)")
    print("=" * 60)
    print(f"  Total neurons recorded: {n_total}")
    print(f"  Paired neurons (both MP1 & MP2): {n_paired}")
    print(f"  Mean R^2 MP1: {stats['mean_R2_mp1']:.4f}")
    print(f"  Mean R^2 MP2: {stats['mean_R2_mp2']:.4f}")
    print(f"  Median R^2 MP1: {stats['median_R2_mp1']:.4f}")
    print(f"  Median R^2 MP2: {stats['median_R2_mp2']:.4f}")
    print(f"  Mean dR^2 (MP2-MP1): {stats['mean_delta_R2']:.4f}")
    print(f"  Fraction R^2_mp2 > R^2_mp1: {stats['frac_R2_mp2_gt_mp1']:.2%}")
    print(f"  Wilcoxon p = {p_w:.4e}")
    print(f"  Paired t-test p = {p_t:.4e}")

    return stats


def plot_results(df, stats, output_dir):
    """Generate scatter + histogram plots."""
    paired = df.dropna(subset=["R2_mp1", "R2_mp2"])

    if len(paired) == 0:
        print("  WARNING: No paired neurons with valid R^2 values. Skipping plots.")
        return

    # Clamp for visualization
    r2_1 = np.maximum(paired["R2_mp1"].values, -0.05)
    r2_2 = np.maximum(paired["R2_mp2"].values, -0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Scatter ──
    ax = axes[0]
    ax.scatter(r2_1, r2_2, alpha=0.6, s=30, c="steelblue", edgecolors="white", linewidth=0.3)
    ax.plot([0, max(r2_1.max(), r2_2.max()) * 1.05],
            [0, max(r2_1.max(), r2_2.max()) * 1.05],
            "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("R^2 MP1 (tracking_only)")
    ax.set_ylabel("R^2 MP2 (mapping_change)")
    ax.set_title(f"Velocity Encoding R^2 (n={len(paired)} neurons)\n"
                 f"Mean: MP1={stats['mean_R2_mp1']:.3f}, MP2={stats['mean_R2_mp2']:.3f}\n"
                 f"Wilcoxon p={stats['wilcoxon_p']:.2e}")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, None)
    ax.set_ylim(-0.02, None)
    sns.despine(ax=ax)

    # ── Histogram / KDE ──
    ax = axes[1]
    r2_vals = np.maximum(paired[["R2_mp1", "R2_mp2"]].values, 0)
    bins = np.linspace(0, max(r2_vals.max(), 0.1), 30)
    ax.hist(r2_vals[:, 0], bins=bins, alpha=0.5, label=f"MP1 (mean={stats['mean_R2_mp1']:.3f})", color="steelblue")
    ax.hist(r2_vals[:, 1], bins=bins, alpha=0.5, label=f"MP2 (mean={stats['mean_R2_mp2']:.3f})", color="coral")
    ax.set_xlabel("R^2 (clamped >= 0)")
    ax.set_ylabel("Neuron count")
    ax.set_title("R^2 Distribution: MP1 vs MP2")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ridge_velocity_encoding.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: ridge_velocity_encoding.png")


def plot_per_pair(df, output_dir):
    """Per-pair summary bar plot."""
    pair_summary = df.dropna(subset=["delta_R2"]).groupby(["pair_idx", "mp1_name"]).agg(
        mean_delta=("delta_R2", "mean"),
        n_neurons=("delta_R2", "count"),
    ).reset_index().sort_values("pair_idx")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["coral" if v > 0 else "steelblue" for v in pair_summary["mean_delta"]]
    ax.bar(range(len(pair_summary)), pair_summary["mean_delta"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(pair_summary)))
    ax.set_xticklabels([n.split("_")[3] for n in pair_summary["mp1_name"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean dR^2 (MP2 - MP1)")
    ax.set_title(f"Per-Pair Velocity Encoding Change\n(positive = MP2 better encoding)")
    for i, (delta, n) in enumerate(zip(pair_summary["mean_delta"], pair_summary["n_neurons"])):
        ax.text(i, delta + (0.002 if delta >= 0 else -0.008), f"n={n}", ha="center", fontsize=6)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ridge_velocity_encoding_per_pair.png"), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: ridge_velocity_encoding_per_pair.png")


# ══════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Ridge Velocity Encoding: MP1 vs MP2")
    print("=" * 60)

    # Load & pair
    print("\n[1] Loading Google Sheet & pairing...")
    df_sheet = load_google_sheet()
    pairs = build_pairs(df_sheet)

    print("\n[2] Loading pickles & matching pairs...")
    paired_data = load_paired_data(pairs)
    print(f"  -> {len(paired_data)} valid pairs loaded")

    # Run ridge encoding
    print(f"\n[3] Running RidgeCV encoding (lags={N_LAGS}, folds={N_FOLDS})...")
    df_results = run_analysis(paired_data)
    print(f"  -> {len(df_results)} neurons analyzed")

    # Stats
    print("\n[4] Computing statistics...")
    stats = compute_stats(df_results)

    # Plots
    print("\n[5] Generating plots...")
    plot_results(df_results, stats, OUTPUT_DIR)
    plot_per_pair(df_results, OUTPUT_DIR)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "ridge_velocity_encoding_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    print("\n  Done!")
