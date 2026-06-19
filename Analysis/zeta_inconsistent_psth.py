"""
PSTH comparison for neurons with inconsistent ZETA test results
between MP1 (tracking_only) and MP2 (mapping_change).

0320 pair: MP1_only=[19,25,30], MP2_only=[26,29]
"""

import numpy as np
import pickle, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# ── Config ──────────────────────────────────────────────────
NAS = "//129.199.81.18/data5/eTheremin/SKIEUR"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIR = ("SKIEUR_20260320_SESSION_00", "SKIEUR_20260320_SESSION_01")
MP1_ONLY = [19, 25, 30]   # pass ZETA in MP1 but not MP2
MP2_ONLY = [26, 29]        # pass ZETA in MP2 but not MP1

DT = 0.005
T_PRE, T_POST = 0.3, 0.3
FS = 30000
BIN_MS = 5

# ── Helpers ─────────────────────────────────────────────────

def load_neuron_data(session, neuron_id):
    """Load binned spike counts for a specific channel from headstage_0/data_0.005.npy.
    neuron_id is a channel index (0-31), matching good_clusters.npy."""
    # Use raw channel-level data (not spike-sorted), shape=(32, n_bins)
    data = np.load(os.path.join(NAS, session, "headstage_0", "data_0.005.npy"))
    if neuron_id >= data.shape[0]:
        raise ValueError(f"Channel {neuron_id} out of range (n_channels={data.shape[0]})")
    return data[neuron_id].astype(float)


def load_tracking_triggers(session, mp_type="tracking_only"):
    """Load trigger times for tracking condition only."""
    with open(os.path.join(NAS, session, "headstage_0", "tt.pkl"), "rb") as f:
        tt = pickle.load(f)
    triggers = np.array(tt["triggers"]) / FS  # samples → seconds
    condition = np.array(tt["condition"])
    min_len = min(len(triggers), len(condition))
    triggers, condition = triggers[:min_len], condition[:min_len]

    if mp_type == "tracking_only":
        mask = condition == -1   # tracking in tracking_only
    else:
        mask = condition == 0    # tracking in mapping_change
    return triggers[mask]


def compute_psth(spike_counts, trigger_times, dt=DT, t_pre=T_PRE, t_post=T_POST):
    """
    Compute PSTH: mean firing rate aligned to triggers.
    spike_counts: 1D array of spike counts per bin (int)
    trigger_times: trigger times in seconds
    Returns: time_axis, mean_rate (spk/s), sem_rate
    """
    bins_per_sec = int(1 / dt)
    trigger_indices = (trigger_times * bins_per_sec).astype(int)
    pre_bins = int(t_pre / dt)
    post_bins = int(t_post / dt)
    window = pre_bins + post_bins

    segments = []
    for ti in trigger_indices:
        start = ti - pre_bins
        end = ti + post_bins
        if start < 0 or end > len(spike_counts):
            continue
        segments.append(spike_counts[start:end])

    if len(segments) == 0:
        return None, None, None

    segments = np.array(segments)  # (n_triggers, window_bins)
    rate = segments / dt  # spikes/s

    time_axis = np.arange(-t_pre, t_post, dt)
    mean_rate = rate.mean(axis=0)
    sem_rate = rate.std(axis=0, ddof=1) / np.sqrt(len(segments))

    return time_axis, mean_rate, sem_rate


def plot_psth_mp1_vs_mp2(neuron_id, mp1_session, mp2_session,
                          mp1_title, mp2_title, filename):
    """Plot PSTH for one neuron under MP1 and MP2, side by side."""
    # Load data
    try:
        sc1 = load_neuron_data(mp1_session, neuron_id)
        sc2 = load_neuron_data(mp2_session, neuron_id)
        trig1 = load_tracking_triggers(mp1_session, "tracking_only")
        trig2 = load_tracking_triggers(mp2_session, "mapping_change")
    except Exception as e:
        print(f"  Skipping neuron {neuron_id}: {e}")
        return

    t1, r1, s1 = compute_psth(sc1, trig1)
    t2, r2, s2 = compute_psth(sc2, trig2)

    if t1 is None or t2 is None:
        print(f"  Skipping neuron {neuron_id}: insufficient triggers")
        return

    # Smooth
    r1_s = gaussian_filter1d(r1, sigma=1.5)
    r2_s = gaussian_filter1d(r2, sigma=1.5)
    s1_s = gaussian_filter1d(s1, sigma=1.5)
    s2_s = gaussian_filter1d(s2, sigma=1.5)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 5 / 2.54),
                             constrained_layout=True)

    for ax, t, r, s, title, zeta_label in [
        (axes[0], t1, r1_s, s1_s, mp1_title, "ZETA: pass" if neuron_id in MP1_ONLY else "ZETA: fail"),
        (axes[1], t2, r2_s, s2_s, mp2_title, "ZETA: pass" if neuron_id in MP2_ONLY else "ZETA: fail"),
    ]:
        color = "#D6604D"  # red for tracking
        ax.fill_between(t, r - s, r + s, alpha=0.3, color=color, lw=0)
        ax.plot(t, r, color=color, lw=1.0)
        ax.axvline(0, color="black", linestyle="--", lw=0.6)
        ax.axhline(0, color="black", linestyle="-", lw=0.4)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Firing rate (spk/s)", fontsize=7)
        ax.set_title(f"{title}\nN{neuron_id}  {zeta_label}", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax, offset=3, trim=True)

    fig.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    mp1_s, mp2_s = PAIR

    print(f"MP1: {mp1_s}")
    print(f"MP2: {mp2_s}")
    print()

    # MP1-only neurons (pass ZETA in MP1, fail in MP2)
    for nid in MP1_ONLY:
        print(f"MP1-only neuron {nid}:")
        plot_psth_mp1_vs_mp2(
            nid, mp1_s, mp2_s,
            "MP1 (tracking_only)", "MP2 (mapping_change)",
            f"psth_0320_n{nid}_mp1only.png"
        )

    # MP2-only neurons (fail ZETA in MP1, pass in MP2)
    for nid in MP2_ONLY:
        print(f"MP2-only neuron {nid}:")
        plot_psth_mp1_vs_mp2(
            nid, mp1_s, mp2_s,
            "MP1 (tracking_only)", "MP2 (mapping_change)",
            f"psth_0320_n{nid}_mp2only.png"
        )

    print("\nDone!")
