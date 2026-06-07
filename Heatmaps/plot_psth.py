import matplotlib.pyplot as plt
import os
import numpy as np
from utils import *

t_pre = 0.2
t_post = 0.30
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post, bin_width)

base_path = '//129.199.81.18/data5/eTheremin/SKIEUR'
save_root = r'C:\Users\PenPen\Desktop\Ferret\Results&PLots'
save_path = os.path.join(save_root, 'psth_plots')
os.makedirs(save_path, exist_ok=True)

sessions = sorted([s for s in os.listdir(base_path) if s.startswith('SKIEUR')])

for session_name in sessions:
    headstage = os.path.join(base_path, session_name, 'headstage_0')
    data_file = os.path.join(headstage, f'data_{bin_width}.npy')
    feat_file = os.path.join(headstage, f'features_{bin_width}.npy')
    gc_file = os.path.join(headstage, 'good_clusters.npy')

    if not (os.path.exists(data_file) and os.path.exists(feat_file)):
        print(f'{session_name}: SKIP (missing data/features)')
        continue

    print(f'{session_name}: processing...')

    data = np.load(data_file, allow_pickle=True)
    features = np.load(feat_file, allow_pickle=True)

    if os.path.exists(gc_file):
        gc = np.load(gc_file, allow_pickle=True)
    else:
        gc = np.arange(0, 32)

    if len(gc) == 0:
        print(f'{session_name}: SKIP (no good clusters)')
        continue

    # auto-detect session type from conditions present
    conditions = set(f['Condition'] for f in features)
    if conditions == {0.0, 1.0}:
        session_type = 'Playback'
    elif conditions == {0.0, -1.0}:
        session_type = 'TrackingOnly'
    elif conditions == {0.0, 2.0}:
        session_type = 'MappingChange'
    elif conditions == {0.0}:
        session_type = 'Tonotopy'
    else:
        session_type = 'Playback'  # default fallback
    print(f'  detected session type: {session_type}')

    # --- Cluster plot ---
    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    num_plots, num_rows, num_columns = get_better_plot_geometry(gc)

    if session_type == 'Playback':
        tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
        playback = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback')
        tr_evoked = get_total_evoked_response(np.nanmean(tracking, axis=1), t_pre, t_post, bin_width, None, 0, len(psth_bins))
        pb_evoked = get_total_evoked_response(np.nanmean(playback, axis=1), t_pre, t_post, bin_width, None, 0, len(psth_bins))
        fig.suptitle(f'{session_name} — tracking vs playback', y=1.02)
        for n, cluster in enumerate(gc):
            if n < num_plots:
                row, col = get_plot_coords(cluster)
                axes[row, col].plot(psth_bins, np.nanmean(tracking[n], axis=0), c='red')
                axes[row, col].plot(psth_bins, np.nanmean(playback[n], axis=0), c='black')
                axes[row, col].axhline(tr_evoked[n], c='red', linestyle='--')
                axes[row, col].axhline(pb_evoked[n], c='black', linestyle='--')
                axes[row, col].axvline(0, c='grey', linestyle='--')
                axes[row, col].set_title(f'Cluster {cluster}')
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
    elif session_type == 'TrackingOnly':
        tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tail')
        fig.suptitle(f'{session_name} — tracking only', y=1.02)
        for n, cluster in enumerate(gc):
            if n < num_plots:
                row, col = get_plot_coords(cluster)
                axes[row, col].plot(psth_bins, np.nanmean(tracking[n], axis=0), c='red')
                axes[row, col].axvline(0, c='grey', linestyle='--')
                axes[row, col].set_title(f'Cluster {cluster}')
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
    elif session_type == 'PlaybackOnly':
        playback = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback')
        fig.suptitle(f'{session_name} — playback only', y=1.02)
        for n, cluster in enumerate(gc):
            if n < num_plots:
                row, col = get_plot_coords(cluster)
                axes[row, col].plot(psth_bins, np.nanmean(playback[n], axis=0), c='black')
                axes[row, col].axvline(0, c='grey', linestyle='--')
                axes[row, col].set_title(f'Cluster {cluster}')
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
    elif session_type == 'MappingChange':
        tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
        mc = get_psth(data, features, t_pre, t_post, bin_width, gc, 'mapping change')
        fig.suptitle(f'{session_name} — tracking vs mapping change', y=1.02)
        for n, cluster in enumerate(gc):
            if n < num_plots:
                row, col = get_plot_coords(cluster)
                axes[row, col].plot(psth_bins, np.nanmean(tracking[n], axis=0), c='red')
                axes[row, col].plot(psth_bins, np.nanmean(mc[cluster], axis=0), c='purple')
                axes[row, col].axvline(0, c='grey', linestyle='--')
                axes[row, col].set_title(f'Cluster {cluster}')
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
    elif session_type == 'Tonotopy':
        tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
        fig.suptitle(f'{session_name} — tonotopy', y=1.02)
        for n, cluster in enumerate(gc):
            if n < num_plots:
                row, col = get_plot_coords(cluster)
                axes[row, col].plot(psth_bins, np.nanmean(tracking[n], axis=0), c='black')
                axes[row, col].axvline(0, c='grey', linestyle='--')
                axes[row, col].set_title(f'Cluster {cluster}')
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)

    plt.savefig(os.path.join(save_path, f'{session_name}_psth_cluster.png'))
    plt.close()

    # --- Average plot ---
    if session_type == 'Playback':
        c_tracking = np.nanmean(tracking, axis=0)
        c_playback = np.nanmean(playback, axis=0)
        m_tracking = np.nanmean(c_tracking, axis=0)
        m_playback = np.nanmean(c_playback, axis=0)
        sem_tr = get_sem(c_tracking)
        sem_pb = get_sem(c_playback)

        plt.plot(psth_bins, m_tracking, c='red', label='tracking')
        plt.plot(psth_bins, m_playback, c='black', label='playback')
        plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='red', alpha=0.2)
        plt.fill_between(psth_bins, m_playback - sem_pb, m_playback + sem_pb, color='black', alpha=0.2)
        plt.legend()
        title = f'{session_name} — Tracking vs playback (Average over all clusters)'

    elif session_type == 'TrackingOnly':
        c_tracking = np.nanmean(tracking, axis=0)
        m_tracking = np.nanmean(c_tracking, axis=0)
        sem_tr = get_sem(c_tracking)
        plt.plot(psth_bins, m_tracking, c='red', label='tracking')
        plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='red', alpha=0.2)
        plt.legend()
        title = f'{session_name} — Tracking only (Average over all clusters)'

    elif session_type == 'PlaybackOnly':
        c_playback = np.nanmean(playback, axis=0)
        m_playback = np.nanmean(c_playback, axis=0)
        sem_pb = get_sem(c_playback)
        plt.plot(psth_bins, m_playback, c='black', label='playback')
        plt.fill_between(psth_bins, m_playback - sem_pb, m_playback + sem_pb, color='grey', alpha=0.2)
        plt.legend()
        title = f'{session_name} — Playback only (Average over all clusters)'

    elif session_type == 'MappingChange':
        c_tracking = np.nanmean(tracking, axis=0)
        c_mc = np.nanmean(mc, axis=0)
        m_tracking = np.nanmean(c_tracking, axis=0)
        m_mc = np.nanmean(c_mc, axis=0)
        sem_tr = get_sem(c_tracking)
        sem_mc = get_sem(c_mc)
        plt.plot(psth_bins, m_tracking, c='red', label='tracking')
        plt.plot(psth_bins, m_mc, c='black', label='mapping change')
        plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='red', alpha=0.2)
        plt.fill_between(psth_bins, m_mc - sem_mc, m_mc + sem_mc, color='purple', alpha=0.2)
        plt.legend()
        title = f'{session_name} — Tracking vs mapping change (Average over all clusters)'

    elif session_type == 'Tonotopy':
        c_tracking = np.nanmean(tracking, axis=0)
        m_tracking = np.nanmean(c_tracking, axis=0)
        sem_tr = get_sem(c_tracking)
        plt.plot(psth_bins, m_tracking, c='red', label='tonotopy')
        plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='red', alpha=0.2)
        plt.legend()
        title = f'{session_name} — Tonotopy (Average over all clusters)'

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (spikes/s)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.path.join(save_path, f'{session_name}_psth_average.png'))
    plt.close()

    print(f'{session_name}: DONE ({session_type})')

print('All sessions processed.')
