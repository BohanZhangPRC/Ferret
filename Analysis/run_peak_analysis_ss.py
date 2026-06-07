"""Peak analysis — spike_sorted=True, session halves × within-session halves."""
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import os, pickle, warnings
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

from utils_load_data import *
from utils_trajectories import *

warnings.filterwarnings('ignore')

save_directory = r'C:\Users\PenPen\Desktop\Ferret\Code\Results'
os.makedirs(save_directory, exist_ok=True)

mpl.rcdefaults()
mpl.use('Agg')  # headless

dt = 0.005
t_pre, t_post = 0.3, 0.3
time = np.arange(-t_pre, t_post + dt, dt)

plt.rcParams.update({
    'font.size': 7, 'axes.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ── Data Loading ─────────────────────────────────────────
session_type = 'mapping_change_only'
spike_sorted = True
files = ['SKIEUR_hs_0']

n_data_all, f_data_all = [], []

for file in files:
    suffix = "_ss" if spike_sorted else ""
    path = rf"\\129.199.81.18\data5\eTheremin\{file}_{session_type}_{dt}_data{suffix}"
    with open(path, "rb") as fp:
        n_data_all.append(pickle.load(fp))
    path_feat = rf"\\129.199.81.18\data5\eTheremin\{file}_{session_type}_{dt}_feature{suffix}"
    with open(path_feat, "rb") as fp:
        f_data_all.append(pickle.load(fp))

print(f"Loaded {len(n_data_all)} file(s), {len(n_data_all[0])} sessions")

# ── Preprocessing ─────────────────────────────────────────
for i in range(len(n_data_all)):
    n_data_all[i] = remove_average(smooth_data(n_data_all[i]))

n_data_reorganised, f_data_reorganised = re_organise_data(n_data_all, f_data_all)

# Filter mismatched
filtered_n_data, filtered_f_data = [], []
for n_data, f_data in zip(n_data_reorganised, f_data_reorganised):
    diff = n_data.shape[-1] - len(f_data)
    if abs(diff) <= 2:
        filtered_n_data.append(n_data)
        filtered_f_data.append(f_data)
    else:
        print(f"REJECT session: diff={diff}")

n_data_reorganised = filtered_n_data
f_data_reorganised = filtered_f_data
print(f"After filter: {len(n_data_reorganised)} sessions")

# ── Session split: halves (beg / exp) ────────────────────
n_half = len(n_data_reorganised) // 2

n_data_a1_beg = n_data_reorganised[:n_half]
n_data_a1_exp = n_data_reorganised[n_half:2*n_half]

f_data_a1_beg = f_data_reorganised[:n_half]
f_data_a1_exp = f_data_reorganised[n_half:2*n_half]

print(f"Halves: {n_half} beg + {n_half} exp")

# ── Within-session split: halves ─────────────────────────
halves = [(0.0, 0.5), (0.5, 1.0)]
half_labels = ['H1 (0–50%)', 'H2 (50–100%)']
n_pre = int(t_pre / dt - 1)

# ── Full average PSTH ─────────────────────────────────────
all_traj_track, all_traj_pb, all_traj_mock, *_ = extract_traj(
    n_data_reorganised, f_data_reorganised,
    t_pre, t_post, dt, overlap_thresh=1.0, n_pre=n_pre, full=True
)

all_traj_tracking = np.concatenate(all_traj_track, axis=1)
mean_traj_track = np.nanmean(all_traj_tracking, axis=1)
sem_traj_track = np.nanstd(all_traj_tracking, axis=1) / np.sqrt(np.sum(~np.isnan(all_traj_tracking), axis=1))

all_traj_playback = np.concatenate(all_traj_pb, axis=1)
mean_traj_pb = np.nanmean(all_traj_playback, axis=1)
sem_traj_pb = np.nanstd(all_traj_playback, axis=1) / np.sqrt(np.sum(~np.isnan(all_traj_playback), axis=1))

# Plot full average
fig, ax = plt.subplots(figsize=(8.9/2.54, 6/2.54))
ax.fill_between(time, mean_traj_track - sem_traj_track, mean_traj_track + sem_traj_track,
                color='#D6604D', alpha=0.4)
ax.fill_between(time, mean_traj_pb - sem_traj_pb, mean_traj_pb + sem_traj_pb,
                color='#2C2C2A', alpha=0.4)
ax.plot(time, mean_traj_track, color='#D6604D', lw=1, label='Closed-loop')
ax.plot(time, mean_traj_pb, color='#2C2C2A', lw=1, label='Playback')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.axhline(0, color='gray', lw=0.3, ls='--')
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_ylabel('Firing rate (spk/s)', fontsize=7)
ax.legend(frameon=False, fontsize=6)
plt.title('Average PSTH all neurons, all triggers (spike sorted)')
sns.despine(ax=ax, offset=3, trim=True)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(save_directory, 'mean_traj_ss.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved mean_traj_ss.pdf")

# ── Per-half extraction (all sessions) ───────────────────
results_by_half = {}
for start, end in tqdm(halves, desc="half extraction (all)"):
    results_by_half[(start, end)] = extract_traj_subset(
        n_data_reorganised, f_data_reorganised,
        t_pre, t_post, dt, overlap_thresh=1.0,
        n_pre=n_pre, full=True, trial_start=start, trial_end=end
    )

# Plot halves overview
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
for ax, (start, end), label in zip(axes, halves, half_labels):
    all_traj_track, all_traj_pb, all_traj_mock, *_ = results_by_half[(start, end)]
    mean_traj_track = np.nanmean(np.concatenate(all_traj_track, axis=1), axis=1)
    mean_traj_pb = np.nanmean(np.concatenate(all_traj_pb, axis=1), axis=1)
    mean_traj_mock = np.nanmean(np.concatenate(all_traj_mock, axis=1), axis=1)
    ax.plot(time, mean_traj_track, c='red', label='Track')
    ax.plot(time, mean_traj_pb, c='black', label='Playback')
    ax.plot(time, mean_traj_mock, c='purple', label='Mock')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Mean activity")
axes[0].legend()
plt.suptitle("Average across all neurons and sessions (spike sorted)")
plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'halves_overview.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved halves_overview.pdf")

# ── Per-half × per-expertise group ───────────────────────
groups = {
    'beg': (n_data_a1_beg, f_data_a1_beg),
    'exp': (n_data_a1_exp, f_data_a1_exp),
}

results = {}
for group_name, (n_data, f_data) in tqdm(groups.items(), desc="group × half"):
    results[group_name] = {}
    for start, end in halves:
        results[group_name][(start, end)] = extract_traj_subset(
            n_data, f_data, t_pre, t_post, dt,
            overlap_thresh=1.0, n_pre=n_pre, full=True,
            trial_start=start, trial_end=end
        )

# Build traj dict
traj_by_half = {}
for q in halves:
    track_beg, pb_beg, mock_beg, *_ = results['beg'][q]
    track_exp, pb_exp, mock_exp, *_ = results['exp'][q]
    traj_by_half[q] = {
        'track': {'beg': track_beg, 'exp': track_exp},
        'pb':    {'beg': pb_beg,    'exp': pb_exp},
        'mock':  {'beg': mock_beg,  'exp': mock_exp},
    }

# ── 2×2 PSTH plot ────────────────────────────────────────
expertise_levels = ['beg', 'exp']
expertise_labels = ['Beginner', 'Expert']
C_TRACK = '#D6604D'
C_PB = 'black'

fig, axes = plt.subplots(2, 2, figsize=(8/2.54, 8/2.54), sharey='row', sharex=True)

for row, (exp, exp_label) in enumerate(zip(expertise_levels, expertise_labels)):
    for col, (q, q_label) in enumerate(zip(halves, half_labels)):
        ax = axes[row, col]
        track = traj_by_half[q]['track'][exp]
        pb = traj_by_half[q]['pb'][exp]
        mean_track = np.nanmean(np.concatenate(track, axis=1), axis=1)
        mean_pb = np.nanmean(np.concatenate(pb, axis=1), axis=1)
        cat_track = np.concatenate(track, axis=1)
        cat_pb = np.concatenate(pb, axis=1)
        sem_track = np.nanstd(cat_track, axis=1) / np.sqrt(cat_track.shape[1])
        sem_pb = np.nanstd(cat_pb, axis=1) / np.sqrt(cat_pb.shape[1])
        ax.fill_between(time, mean_track - sem_track, mean_track + sem_track,
                        color=C_TRACK, alpha=0.2)
        ax.fill_between(time, mean_pb - sem_pb, mean_pb + sem_pb,
                        color=C_PB, alpha=0.2)
        ax.plot(time, mean_pb, color=C_PB, lw=1, label='Playback')
        ax.plot(time, mean_track, color=C_TRACK, lw=1, label='Track')
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.axhline(0, color='gray', lw=0.3, ls='--')
        if row == 0: ax.set_title(q_label, fontsize=7)
        if col == 0: ax.set_ylabel(f'{exp_label}\nspk/s', fontsize=7)
        if row == 1: ax.set_xlabel('Time (s)', fontsize=7)
        sns.despine(ax=ax, offset=3, trim=True)

axes[0, 0].legend(frameon=False, fontsize=6, loc='upper left')
plt.suptitle('Track vs Playback — by expertise and session half (spike sorted)',
             fontsize=8, y=1.01)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(save_directory, 'traj_by_half_ss.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved traj_by_half_ss.pdf")

# ── AUC computation ──────────────────────────────────────
def get_diff(traj_by_half, q, group):
    track = np.concatenate(traj_by_half[q]['track'][group], axis=1)
    pb = np.concatenate(traj_by_half[q]['pb'][group], axis=1)
    return pb - track

d_beg_h1 = get_diff(traj_by_half, (0.0, 0.5), 'beg')
d_beg_h2 = get_diff(traj_by_half, (0.5, 1.0), 'beg')
d_exp_h1 = get_diff(traj_by_half, (0.0, 0.5), 'exp')
d_exp_h2 = get_diff(traj_by_half, (0.5, 1.0), 'exp')

idx_start = np.searchsorted(time, 0)
idx_end = np.searchsorted(time, 0.1)

def compute_auc(diff, time, idx_start, idx_end):
    return np.trapz(diff[idx_start:idx_end, :], x=time[idx_start:idx_end], axis=0)

auc_beg_h1 = compute_auc(d_beg_h1, time, idx_start, idx_end)
auc_beg_h2 = compute_auc(d_beg_h2, time, idx_start, idx_end)
auc_exp_h1 = compute_auc(d_exp_h1, time, idx_start, idx_end)
auc_exp_h2 = compute_auc(d_exp_h2, time, idx_start, idx_end)

# ── Violin plot ──────────────────────────────────────────
C_H1 = '#4393C3'
C_H2 = '#74C476'
ALPHA_V = 0.35
ALPHA_S = 0.5

groups_plot = ['Beginner', 'Expert']
data_all = [
    [auc_beg_h1, auc_beg_h2],
    [auc_exp_h1, auc_exp_h2],
]
conditions = ['First half', 'Second half']
n_groups = len(groups_plot)
n_conditions = len(conditions)
n_comparisons = n_groups * n_conditions + n_groups

fig, ax = plt.subplots(figsize=(5.5 / 2.54 * n_groups, 5 / 2.54 * 2.5))
offsets = [-0.18, 0.18]
colors = [C_H1, C_H2]
positions = np.arange(n_groups)

for c_idx, (cond, color, offset) in enumerate(zip(conditions, colors, offsets)):
    for g_idx, (group, gdata) in enumerate(zip(groups_plot, data_all)):
        y = gdata[c_idx]
        xpos = positions[g_idx] + offset
        vp = ax.violinplot(y, positions=[xpos], widths=0.3,
                           showmeans=False, showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color); body.set_alpha(ALPHA_V); body.set_edgecolor('none')
        jitter = np.random.uniform(-0.06, 0.06, size=len(y))
        ax.scatter(np.full(len(y), xpos) + jitter, y,
                   s=2, color=color, alpha=ALPHA_S, linewidths=0, zorder=3)
        mean = np.mean(y)
        sem = np.std(y, ddof=1) / np.sqrt(len(y))
        ax.errorbar(xpos, mean, yerr=sem, fmt='o', ms=3.5, color=color,
                    ecolor=color, elinewidth=0.8, capsize=2, capthick=0.8,
                    zorder=5, markeredgewidth=0)
        ax.plot(xpos, np.median(y), '_', color='white', ms=8, mew=1.2, zorder=6)
        _, p = wilcoxon(y)
        p_c = min(p * n_comparisons, 1.0)
        if p_c < 0.001: stars = '***'
        elif p_c < 0.01: stars = '**'
        elif p_c < 0.05: stars = '*'
        else: stars = 'n.s.'
        y_star = np.max(y) + (np.max(y) - np.min(y)) * 0.05
        ax.text(xpos, y_star, stars, ha='center', va='bottom', fontsize=6, color=color)

# Paired tests (H1 vs H2) per group
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
for g_idx, (group, gdata) in enumerate(zip(groups_plot, data_all)):
    y_h1, y_h2 = gdata
    _, p = wilcoxon(y_h1, y_h2)
    p_c = min(p * n_groups, 1.0)
    if p_c < 0.001: stars = '***'
    elif p_c < 0.01: stars = '**'
    elif p_c < 0.05: stars = '*'
    else: stars = 'n.s.'
    x1 = positions[g_idx] + offsets[0]
    x2 = positions[g_idx] + offsets[1]
    y_bracket = max(np.max(y_h1), np.max(y_h2)) + y_range * 0.12
    ax.plot([x1, x1, x2, x2],
            [y_bracket - y_range*0.02, y_bracket,
             y_bracket, y_bracket - y_range*0.02], lw=0.6, color='k')
    ax.text((x1 + x2) / 2, y_bracket + y_range * 0.005,
            stars, ha='center', va='bottom', fontsize=6)

ax.axhline(0, lw=0.5, ls='--', color='#888888', zorder=0)
ax.set_xticks(positions)
ax.set_xticklabels(groups_plot, fontsize=7)
ax.set_ylabel('AUC (0–0.1 s)', fontsize=7)
ax.set_title(r'$\Delta$ (Playback $-$ Closed-loop) spike sorted', fontsize=7, pad=4)
sns.despine(ax=ax, offset=3, trim=True)
legend_handles = [
    mpatches.Patch(facecolor=C_H1, alpha=0.8, label='First half'),
    mpatches.Patch(facecolor=C_H2, alpha=0.8, label='Second half'),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=6, loc='upper right',
          handlelength=1, handleheight=0.8)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(save_directory, 'fig_auc_ss.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_directory, 'fig_auc_ss.svg'), bbox_inches='tight')
plt.close()
print("Saved fig_auc_ss.pdf and fig_auc_ss.svg")
print("\n=== ALL DONE ===")
