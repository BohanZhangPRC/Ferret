import matplotlib.pyplot as plt
import os
import numpy as np
from utils import *
import math

t_pre = 0.2
t_post = 0.30
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post, bin_width)

base_path = '//129.199.81.18/data5/eTheremin/SKIEUR'
save_root = r'C:\Users\PenPen\Desktop\Ferret\Results&PLots'
save_path = os.path.join(save_root, 'playback_plots')
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

    n_block = int(np.max([elt['Block'] for elt in features]))
    if n_block < 1:
        print(f'{session_name}: SKIP (n_block={n_block}, no block structure)')
        continue

    cols = 3
    rows = math.ceil(n_block / cols)

    # END tr vs END pb
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f'{session_name} — End of tracking vs end of playback', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axes = axes.flatten()

    for bloc in range(1, n_block):
        ax = axes[bloc - 1]
        tr = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking'))
        pb = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback'))

        n = int(len(tr[0]) / 5)
        end_tr = tr[:, 4 * n:]
        end_pb = pb[:, 4 * n:]

        mc_tr = np.nanmean(end_tr, axis=1)
        mc_pb = np.nanmean(end_pb, axis=1)
        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)
        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)
        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{session_name}_end_end.png'), bbox_inches='tight')
    plt.close()

    # BEGINNING vs BEGINNING
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f'{session_name} — Beginning of tracking vs beginning of playback', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axes = axes.flatten()

    for bloc in range(1, n_block):
        ax = axes[bloc - 1]
        tr = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking'))
        pb = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback'))

        n = int(len(tr[0]) / 5)
        beg_tr = tr[:, :n]
        beg_pb = pb[:, :n]

        mc_tr = np.nanmean(beg_tr, axis=1)
        mc_pb = np.nanmean(beg_pb, axis=1)
        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)
        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)
        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block + 1:]:
        ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{session_name}_beg_beg.png'), bbox_inches='tight')
    plt.close()

    # END tracking vs BEGINNING playback
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f'{session_name} — End of tracking vs beginning of playback', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axes = axes.flatten()

    for bloc in range(1, n_block):
        ax = axes[bloc - 1]
        tr = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking'))
        pb = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback'))

        n = int(len(tr[0]) / 5)
        end_tr = tr[:, 4 * n:]
        beg_pb = pb[:, :n]

        mc_tr = np.nanmean(end_tr, axis=1)
        mc_pb = np.nanmean(beg_pb, axis=1)
        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)
        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)
        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{session_name}_end_beg.png'), bbox_inches='tight')
    plt.close()

    # Tracking evolution (first half vs second half)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f'{session_name} — Evolution of tracking in a block', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axes = axes.flatten()

    for bloc in range(1, n_block):
        ax = axes[bloc - 1]
        tr = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking'))

        n = int(len(tr[0]) / 2)
        beg_tr = tr[:, :n]
        end_tr = tr[:, n:]

        mc_beg_tr = np.nanmean(beg_tr, axis=1)
        mc_end_tr = np.nanmean(end_tr, axis=1)
        m_beg_tr = np.nanmean(mc_beg_tr, axis=0)
        m_end_tr = np.nanmean(mc_end_tr, axis=0)
        sem_beg_tr = get_sem(mc_beg_tr)
        sem_end_tr = get_sem(mc_end_tr)

        ax.plot(psth_bins, m_beg_tr, c='orange', label='First half tracking')
        ax.plot(psth_bins, m_end_tr, c='red', label='Second half tracking')
        ax.fill_between(psth_bins, m_beg_tr - sem_beg_tr, m_beg_tr + sem_beg_tr, color='orange', alpha=0.2)
        ax.fill_between(psth_bins, m_end_tr - sem_end_tr, m_end_tr + sem_end_tr, color='red', alpha=0.2)
        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{session_name}_tracking_divided.png'), bbox_inches='tight')
    plt.close()

    # Playback evolution (first half vs second half)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(f'{session_name} — Evolution of playback in a block', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axes = axes.flatten()

    for bloc in range(1, n_block):
        ax = axes[bloc - 1]
        pb = np.array(get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback'))

        n = int(len(pb[0]) / 2)
        beg_pb = pb[:, :n]
        end_pb = pb[:, n:]

        mc_beg_pb = np.nanmean(beg_pb, axis=1)
        mc_end_pb = np.nanmean(end_pb, axis=1)
        m_beg_pb = np.nanmean(mc_beg_pb, axis=0)
        m_end_pb = np.nanmean(mc_end_pb, axis=0)
        sem_beg_pb = get_sem(mc_beg_pb)
        sem_end_pb = get_sem(mc_end_pb)

        ax.plot(psth_bins, m_beg_pb, c='grey', label='First half playback')
        ax.plot(psth_bins, m_end_pb, c='black', label='Second half playback')
        ax.fill_between(psth_bins, m_beg_pb - sem_beg_pb, m_beg_pb + sem_beg_pb, color='grey', alpha=0.2)
        ax.fill_between(psth_bins, m_end_pb - sem_end_pb, m_end_pb + sem_end_pb, color='black', alpha=0.2)
        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(os.path.join(save_path, f'{session_name}_playback_divided.png'), bbox_inches='tight')
    plt.close()

    # Shihab figure — evolution across blocks
    m_tracking, m_playback = [], []
    warmup = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, 0, 'tail')
    warmdown = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, n_block, 'tail')
    c_warmup = np.nanmean(warmup, axis=1)
    m_warmup = np.nanmean(c_warmup, axis=0)
    c_warmdown = np.nanmean(warmdown, axis=1)
    m_warmdown = np.nanmean(c_warmdown, axis=0)

    for i in range(1, n_block):
        tracking = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'tracking')
        playback = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'playback')
        c_tracking = np.nanmean(tracking, axis=1)
        c_playback = np.nanmean(playback, axis=1)
        m_tracking.append(np.nanmean(c_tracking, axis=0))
        m_playback.append(np.nanmean(c_playback, axis=0))

    x_tr = np.arange(0, 0 + (n_block + 1), 1)
    x_pb = np.arange(0.5, 0.5 + n_block, 1)
    plt.plot(psth_bins + np.full_like(psth_bins, -1), m_warmup, c='red', label='Warmup')
    plt.plot(psth_bins + np.full_like(psth_bins, x_tr[-1]), m_warmdown, c='red', label='Warmdown')
    for i, elt in enumerate(m_tracking):
        plt.plot(psth_bins + np.full_like(psth_bins, x_tr[i]), elt, c='red')
        plt.plot(psth_bins + np.full_like(psth_bins, x_pb[i]), m_playback[i], c='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (spikes/s)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(f'{session_name} — Evolution of PSTH block by block')
    plt.savefig(os.path.join(save_path, f'{session_name}_plot_shihab.png'), bbox_inches='tight')
    plt.close()

    print(f'{session_name}: DONE')

print('All sessions processed.')
