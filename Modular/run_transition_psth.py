"""
Transition-Anchored PSTH Analysis

Identifies chronological transitions between Tracking and Playback blocks.
For each contiguous TR→PB block pair, extracts PSTH at transition boundary.
Visualizes: (1) TR endpoint and (2) PB onset overlaid for each transition pair.

Output:
- transition_psth_session.csv: Session-level PSTH data for all transitions
- Per-transition pair PSTH plots with overlaid Tracking (endpoint) vs Playback (onset)
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DT, N_BINS_PRE, N_BINS_POST, OUTPUT_DIR, SESSION_CONFIGS
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt
from src.psth import extract_transition_psth, format_psth_summary


def _safe_pair_text(tr_block, pb_block):
    """Safe filename text for transition pair."""
    tr_safe = (str(int(tr_block)) if float(tr_block).is_integer() else str(tr_block).replace('.', 'p'))
    pb_safe = (str(int(pb_block)) if float(pb_block).is_integer() else str(pb_block).replace('.', 'p'))
    return f"TR{tr_safe}_PB{pb_safe}"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"Transition_PSTH_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING TRANSITION-ANCHORED PSTH ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    if not n_data_s or not f_data_s:
        print("CRITICAL ERROR: No data loaded.")
        return

    # Extract transition PSTH (identifies TR->PB pairs automatically)
    transition_df = extract_transition_psth(n_data_s, f_data_s, exclude_block_0=True)

    if transition_df.empty:
        print("WARNING: No transition pairs found. (Requires at least 2 contiguous blocks with TR->PB structure)")
        return

    # Save session-level transition data
    transition_csv = os.path.join(run_dir, 'transition_psth_session.csv')
    transition_df.to_csv(transition_csv, index=False)

    # Aggregate by transition pair
    transition_pairs = transition_df.groupby(['TR_Block', 'PB_Block']).size().reset_index().drop(columns=0)

    print(f"Found {len(transition_pairs)} transition pair(s)")

    # Create summary for each transition pair
    for idx, row in transition_pairs.iterrows():
        tr_block = row['TR_Block']
        pb_block = row['PB_Block']
        
        pair_df = transition_df[
            (transition_df['TR_Block'] == tr_block) & 
            (transition_df['PB_Block'] == pb_block)
        ]

        # Aggregate statistics
        pair_summary = format_psth_summary(
            pair_df,
            groupby_cols=['Condition', 'Time_s']
        )

        # Create plot for this transition
        fig, ax = plt.subplots(figsize=(11, 6))

        color_map = {'Tracking': '#d62728', 'Playback': '#4d4d4d'}

        for condition in ['Tracking', 'Playback']:
            cond_data = pair_summary[pair_summary['Condition'] == condition].sort_values('Time_s')

            if cond_data.empty:
                continue

            x = cond_data['Time_s'].to_numpy()
            y = cond_data['PSTH_Mean'].to_numpy()
            sem = cond_data['PSTH_SEM'].fillna(0.0).to_numpy()

            line_style = '-' if condition == 'Tracking' else '--'
            n_sessions = int(cond_data['N_Sessions'].max()) if not cond_data.empty else 0
            n_events = int(pair_df[pair_df['Condition'] == condition]['n_events'].sum())
            label = (
                f"{condition} (Block {_format_block(tr_block)}, n_sessions={n_sessions}, n_events={n_events})"
                if condition == 'Tracking'
                else f"{condition} (Block {_format_block(pb_block)}, n_sessions={n_sessions}, n_events={n_events})"
            )

            ax.plot(x, y, label=label, color=color_map.get(condition), linewidth=2.5, linestyle=line_style)
            ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition), alpha=0.25)

        ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(
            f"Transition PSTH | Tracking Block {_format_block(tr_block)} → Playback Block {_format_block(pb_block)} | n_sessions={pair_df['Session'].nunique()}",
            fontsize=13, fontweight='bold'
        )
        ax.set_xlabel('Time from trigger (s)', fontsize=11)
        ax.set_ylabel('Mean activity (a.u.)', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        pair_text = _safe_pair_text(tr_block, pb_block)
        out_path = os.path.join(run_dir, f"transition_psth_{pair_text}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved transition PSTH CSV: {transition_csv}")
    print(f"Saved {len(transition_pairs)} transition PSTH plot(s).")
    print(f"--- Transition PSTH analysis finished. Results saved in {run_dir} ---")


def _format_block(block_value):
    """Format block value for display."""
    if pd.isna(block_value):
        return "nan"
    if float(block_value).is_integer():
        return str(int(block_value))
    return f"{block_value:.2f}"


if __name__ == '__main__':
    main()
