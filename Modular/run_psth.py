import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DT, N_BINS_POST, N_BINS_PRE, OUTPUT_DIR, SESSION_CONFIGS
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt
from src.processing import extract_segments


def _condition_label(value):
    if value == 0:
        return "Tracking"
    if value == 1:
        return "Playback"
    return f"Condition_{value}"


def _safe_block_text(block_value):
    if pd.isna(block_value):
        return "nan"
    if float(block_value).is_integer():
        return str(int(block_value))
    return str(block_value).replace('.', 'p')


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"PSTH_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING PSTH ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)

    if not n_data_s or not f_data_s:
        print("CRITICAL ERROR: No data loaded.")
        return

    expected_len = N_BINS_PRE + N_BINS_POST
    time_axis = (np.arange(expected_len) - N_BINS_PRE) * DT

    session_level_rows = []

    for session_idx, (n_data, f_data) in enumerate(zip(n_data_s, f_data_s)):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)

        if not is_change.any():
            continue

        # Pair by each available block and condition independently.
        block_condition_groups = f_data[is_change].groupby(['Block', 'Condition'])
        for (block, condition_val), sub in block_condition_groups:
            if pd.isna(block) or pd.isna(condition_val):
                continue

            event_indices = sub.index.to_list()
            if not event_indices:
                continue

            mean_seg = extract_segments(n_data, event_indices)
            if mean_seg is None:
                continue

            # Collapse neuron dimension to produce one PSTH trace per session-block-condition.
            psth_trace = np.nanmean(mean_seg, axis=0)
            if psth_trace.shape[0] != expected_len:
                continue

            condition_label = _condition_label(condition_val)
            for time_s, value in zip(time_axis, psth_trace):
                session_level_rows.append(
                    {
                        'Session': session_idx,
                        'Block': float(block),
                        'Condition': condition_label,
                        'Time_s': float(time_s),
                        'PSTH_Value': float(value),
                        'n_events': int(len(event_indices)),
                    }
                )

    if not session_level_rows:
        print("CRITICAL ERROR: No valid PSTH traces were computed.")
        return

    session_df = pd.DataFrame(session_level_rows)
    session_csv = os.path.join(run_dir, 'psth_session_level.csv')
    session_df.to_csv(session_csv, index=False)

    summary_df = (
        session_df
        .groupby(['Block', 'Condition', 'Time_s'], as_index=False)
        .agg(
            PSTH_Mean=('PSTH_Value', 'mean'),
            PSTH_STD=('PSTH_Value', 'std'),
            N_Sessions=('Session', 'nunique')
        )
    )
    summary_df['PSTH_SEM'] = summary_df['PSTH_STD'] / np.sqrt(summary_df['N_Sessions'].clip(lower=1))

    summary_csv = os.path.join(run_dir, 'psth_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    global_summary_df = (
        session_df
        .groupby(['Condition', 'Time_s'], as_index=False)
        .agg(
            PSTH_Mean=('PSTH_Value', 'mean'),
            PSTH_STD=('PSTH_Value', 'std'),
            N_Sessions=('Session', 'nunique')
        )
    )
    global_summary_df['PSTH_SEM'] = global_summary_df['PSTH_STD'] / np.sqrt(global_summary_df['N_Sessions'].clip(lower=1))
    global_event_counts = (
        session_df
        .groupby('Condition', as_index=False)['n_events']
        .sum()
        .rename(columns={'n_events': 'Total_Events'})
    )
    global_summary_csv = os.path.join(run_dir, 'psth_global_summary.csv')
    global_summary_df.to_csv(global_summary_csv, index=False)

    plot_count = 0
    color_map = {'Tracking': '#d62728', 'Playback': '#4d4d4d'}

    blocks = sorted(summary_df['Block'].dropna().unique())
    if blocks:
        fig_width = max(8.0, 4.8 * len(blocks))
        fig, axes = plt.subplots(1, len(blocks), figsize=(fig_width, 4.8), sharey=True)
        if len(blocks) == 1:
            axes = [axes]

        for ax, block in zip(axes, blocks):
            block_df = summary_df[summary_df['Block'] == block]
            if block_df.empty:
                continue

            for condition in ['Tracking', 'Playback']:
                cond_df = block_df[block_df['Condition'] == condition].sort_values('Time_s')
                if cond_df.empty:
                    continue

                x = cond_df['Time_s'].to_numpy()
                y = cond_df['PSTH_Mean'].to_numpy()
                sem = cond_df['PSTH_SEM'].fillna(0.0).to_numpy()

                n_sessions = int(cond_df['N_Sessions'].max())
                block_events = int(
                    session_df[
                        (session_df['Block'] == block) &
                        (session_df['Condition'] == condition)
                    ]['n_events'].sum()
                )
                label = f"{condition} (n_sessions={n_sessions}, n_events={block_events})"
                ax.plot(x, y, label=label, color=color_map.get(condition, '#1f77b4'), linewidth=2)
                ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition, '#1f77b4'), alpha=0.2)

            ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"Block {_safe_block_text(block)}")
            ax.set_xlabel('Time from trigger (s)')
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        axes[0].set_ylabel('Mean activity (a.u.)')
        fig.suptitle('PSTH by Block', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        block_grid_path = os.path.join(run_dir, 'psth_blocks_grid.png')
        plt.savefig(block_grid_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_count = len(blocks)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for condition in ['Tracking', 'Playback']:
        cond_df = global_summary_df[global_summary_df['Condition'] == condition].sort_values('Time_s')
        if cond_df.empty:
            continue

        x = cond_df['Time_s'].to_numpy()
        y = cond_df['PSTH_Mean'].to_numpy()
        sem = cond_df['PSTH_SEM'].fillna(0.0).to_numpy()

        n_sessions = int(cond_df['N_Sessions'].max())
        event_match = global_event_counts[global_event_counts['Condition'] == condition]
        total_events = int(event_match['Total_Events'].iloc[0]) if not event_match.empty else 0
        label = f"{condition} (n_sessions={n_sessions}, n_events={total_events})"
        ax.plot(x, y, label=label, color=color_map.get(condition, '#1f77b4'), linewidth=2.2)
        ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition, '#1f77b4'), alpha=0.2)

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Global PSTH (All Blocks)')
    ax.set_xlabel('Time from trigger (s)')
    ax.set_ylabel('Mean activity (a.u.)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    global_plot_path = os.path.join(run_dir, 'psth_global.png')
    plt.savefig(global_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved session-level PSTH CSV: {session_csv}")
    print(f"Saved summary PSTH CSV: {summary_csv}")
    print(f"Saved global PSTH summary CSV: {global_summary_csv}")
    print(f"Saved global PSTH plot: {global_plot_path}")
    if blocks:
        print(f"Saved block grid PSTH plot: {block_grid_path}")
    print(f"Saved {plot_count} block subplot(s) in the grid.")
    print(f"--- PSTH analysis finished. Results saved in {run_dir} ---")


if __name__ == '__main__':
    main()
