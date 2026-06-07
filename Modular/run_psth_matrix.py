"""
PSTH Matrix Analysis (Block × N Matrix)

Generates a matrix of PSTH traces visualizing block and n-value dependency.
For each block, shows overlaid first-N and last-N PSTH for multiple n-values from config.

Output:
- psth_first_n_session.csv: First-N PSTH session-level data
- psth_last_n_session.csv: Last-N PSTH session-level data
- Grid of PSTH matrix plots per condition (one plot per condition, showing blocks × n-values)
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DT, N_BINS_PRE, N_BINS_POST, N_VALUES, OUTPUT_DIR, SESSION_CONFIGS
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt
from src.psth import extract_first_n_psth, extract_last_n_psth, format_psth_summary


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
    run_dir = os.path.join(OUTPUT_DIR, f"PSTH_Matrix_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING PSTH MATRIX ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")
    print(f"N-values: {N_VALUES}")

    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    if not n_data_s or not f_data_s:
        print("CRITICAL ERROR: No data loaded.")
        return

    # Extract first-N and last-N PSTH for all n-values
    first_n_df = extract_first_n_psth(n_data_s, f_data_s, n_values=N_VALUES, exclude_block_0=True)
    last_n_df = extract_last_n_psth(n_data_s, f_data_s, n_values=N_VALUES, exclude_block_0=True)

    if first_n_df.empty or last_n_df.empty:
        print("CRITICAL ERROR: No PSTH data extracted for any n-values.")
        return

    # Save session-level data
    first_n_csv = os.path.join(run_dir, 'psth_first_n_session.csv')
    last_n_csv = os.path.join(run_dir, 'psth_last_n_session.csv')
    first_n_df.to_csv(first_n_csv, index=False)
    last_n_df.to_csv(last_n_csv, index=False)

    # Compute summary for each n-value
    first_n_summary = format_psth_summary(
        first_n_df,
        groupby_cols=['Block', 'Condition', 'n_value', 'Time_s']
    )

    last_n_summary = format_psth_summary(
        last_n_df,
        groupby_cols=['Block', 'Condition', 'n_value', 'Time_s']
    )

    # Create matrix plots: one per condition, showing blocks × n-values
    color_map_first = plt.cm.Blues(np.linspace(0.4, 0.9, len(N_VALUES)))
    color_map_last = plt.cm.Reds(np.linspace(0.4, 0.9, len(N_VALUES)))

    for condition in ['Tracking', 'Playback']:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(
            len(first_n_summary[first_n_summary['Condition'] == condition]['Block'].unique()),
            len(N_VALUES),
            hspace=0.35,
            wspace=0.35
        )

        blocks = sorted(first_n_summary[first_n_summary['Condition'] == condition]['Block'].unique())

        for block_idx, block in enumerate(blocks):
            for n_idx, n_val in enumerate(N_VALUES):
                ax = fig.add_subplot(gs[block_idx, n_idx])

                # First-N for this block and n-value
                first_data = first_n_summary[
                    (first_n_summary['Block'] == block) &
                    (first_n_summary['Condition'] == condition) &
                    (first_n_summary['n_value'] == n_val)
                ].sort_values('Time_s')

                # Last-N for this block and n-value
                last_data = last_n_summary[
                    (last_n_summary['Block'] == block) &
                    (last_n_summary['Condition'] == condition) &
                    (last_n_summary['n_value'] == n_val)
                ].sort_values('Time_s')

                if not first_data.empty:
                    x = first_data['Time_s'].to_numpy()
                    y = first_data['PSTH_Mean'].to_numpy()
                    sem = first_data['PSTH_SEM'].fillna(0.0).to_numpy()
                    ax.plot(x, y, linewidth=2, color=color_map_first[n_idx], label=f'First-N (n={n_val})', alpha=0.8)
                    ax.fill_between(x, y - sem, y + sem, color=color_map_first[n_idx], alpha=0.2)

                if not last_data.empty:
                    x = last_data['Time_s'].to_numpy()
                    y = last_data['PSTH_Mean'].to_numpy()
                    sem = last_data['PSTH_SEM'].fillna(0.0).to_numpy()
                    ax.plot(x, y, linewidth=2, color=color_map_last[n_idx], label=f'Last-N (n={n_val})', alpha=0.8, linestyle='--')
                    ax.fill_between(x, y - sem, y + sem, color=color_map_last[n_idx], alpha=0.2)

                first_sessions = int(first_data['N_Sessions'].max()) if not first_data.empty else 0
                last_sessions = int(last_data['N_Sessions'].max()) if not last_data.empty else 0
                first_events = int(
                    first_n_df[
                        (first_n_df['Block'] == block) &
                        (first_n_df['Condition'] == condition) &
                        (first_n_df['n_value'] == n_val)
                    ]['n_events'].sum()
                )
                last_events = int(
                    last_n_df[
                        (last_n_df['Block'] == block) &
                        (last_n_df['Condition'] == condition) &
                        (last_n_df['n_value'] == n_val)
                    ]['n_events'].sum()
                )
                ax.text(
                    0.02,
                    0.98,
                    f"nS(F/L)={first_sessions}/{last_sessions}\nnE(F/L)={first_events}/{last_events}",
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none')
                )

                ax.axvline(0.0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.grid(True, axis='y', alpha=0.2)

                # Titles and labels
                if block_idx == 0:
                    ax.set_title(f'N={n_val}', fontsize=10, fontweight='bold')
                if n_idx == 0:
                    ax.set_ylabel(f'Block {_safe_block_text(block)}\nActivity (a.u.)', fontsize=9)
                if block_idx == len(blocks) - 1:
                    ax.set_xlabel('Time (s)', fontsize=9)
                else:
                    ax.set_xticklabels([])

                ax.tick_params(labelsize=8)

        # Add legend to the figure
        handles = [
            plt.Line2D([0], [0], color=color_map_first[-1], linewidth=2, label='First-N (n = requested N)'),
            plt.Line2D([0], [0], color=color_map_last[-1], linewidth=2, linestyle='--', label='Last-N (n = requested N)')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)

        fig.suptitle(f'PSTH Matrix (Block × N) | {condition}', fontsize=16, fontweight='bold', y=0.995)
        out_path = os.path.join(run_dir, f"psth_matrix_{_safe_name(condition)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved first-N PSTH CSV: {first_n_csv}")
    print(f"Saved last-N PSTH CSV: {last_n_csv}")
    print(f"--- PSTH Matrix analysis finished. Results saved in {run_dir} ---")


if __name__ == '__main__':
    main()
