"""
First vs Last Block PSTH Comparison Analysis

Compares population PSTH between first and last blocks for each condition.
Computes Last-First delta trace with error propagation.

Output:
- first_block_psth.csv: First block session-level PSTH
- last_block_psth.csv: Last block session-level PSTH
- delta_psth.csv: Last-First delta with propagated uncertainty
- Overlaid PSTH plot (first vs last with SEM bands)
- Delta trace plot with error bounds
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
from src.psth import extract_global_psth, format_psth_summary


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


def extract_first_last_blocks(n_data_list, f_data_list):
    """
    For each session, identify first and last valid blocks after excluding Block 0.
    
    Returns:
        Dict mapping session_idx → {'first_block': float, 'last_block': float}
    """
    block_mapping = {}
    
    for session_idx, f_data in enumerate(f_data_list):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        valid_blocks = sorted([b for b in f_data[is_change]['Block'].unique() if not pd.isna(b)])
        
        # Remove Block 0
        valid_blocks = [b for b in valid_blocks if float(b) != 0.0]
        
        if len(valid_blocks) >= 2:
            block_mapping[session_idx] = {
                'first_block': float(valid_blocks[0]),
                'last_block': float(valid_blocks[-1])
            }
        elif len(valid_blocks) == 1:
            # If only one block, use it for both
            block_mapping[session_idx] = {
                'first_block': float(valid_blocks[0]),
                'last_block': float(valid_blocks[0])
            }
    
    return block_mapping


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"FirstLastBlock_PSTH_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING FIRST vs LAST BLOCK PSTH ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    if not n_data_s or not f_data_s:
        print("CRITICAL ERROR: No data loaded.")
        return

    # Get global PSTH (all events aggregated)
    global_psth_df = extract_global_psth(n_data_s, f_data_s, exclude_block_0=True)
    
    if global_psth_df.empty:
        print("CRITICAL ERROR: No PSTH data extracted.")
        return

    # Identify first and last blocks per session
    block_mapping = extract_first_last_blocks(n_data_s, f_data_s)
    
    if not block_mapping:
        print("WARNING: No valid block pairs found (need at least 2 blocks per session after excluding Block 0).")
        return

    # Filter PSTH data for first and last blocks only
    first_block_psth = []
    last_block_psth = []
    
    for session_idx, blocks in block_mapping.items():
        first_b = blocks['first_block']
        last_b = blocks['last_block']
        
        session_data = global_psth_df[global_psth_df['Session'] == session_idx]
        
        first_data = session_data[session_data['Block'] == first_b].copy()
        first_data['Block_Type'] = 'First'
        first_block_psth.append(first_data)
        
        last_data = session_data[session_data['Block'] == last_b].copy()
        last_data['Block_Type'] = 'Last'
        last_block_psth.append(last_data)
    
    first_block_df = pd.concat(first_block_psth, ignore_index=True)
    last_block_df = pd.concat(last_block_psth, ignore_index=True)
    
    # Save session-level data
    first_block_csv = os.path.join(run_dir, 'first_block_psth.csv')
    last_block_csv = os.path.join(run_dir, 'last_block_psth.csv')
    first_block_df.to_csv(first_block_csv, index=False)
    last_block_df.to_csv(last_block_csv, index=False)
    
    # Compute summary statistics
    first_summary = format_psth_summary(
        first_block_df,
        groupby_cols=['Condition', 'Time_s']
    )
    first_summary['Block_Type'] = 'First'
    
    last_summary = format_psth_summary(
        last_block_df,
        groupby_cols=['Condition', 'Time_s']
    )
    last_summary['Block_Type'] = 'Last'
    
    # Compute delta (Last - First) with error propagation
    delta_rows = []
    for condition in ['Tracking', 'Playback']:
        first_cond = first_summary[first_summary['Condition'] == condition].sort_values('Time_s')
        last_cond = last_summary[last_summary['Condition'] == condition].sort_values('Time_s')
        
        if first_cond.empty or last_cond.empty:
            continue
        
        # Align on Time_s (should be same)
        merged = pd.merge(
            first_cond[['Time_s', 'PSTH_Mean', 'PSTH_SEM']].rename(
                columns={'PSTH_Mean': 'First_Mean', 'PSTH_SEM': 'First_SEM'}
            ),
            last_cond[['Time_s', 'PSTH_Mean', 'PSTH_SEM']].rename(
                columns={'PSTH_Mean': 'Last_Mean', 'PSTH_SEM': 'Last_SEM'}
            ),
            on='Time_s'
        )
        
        # Compute delta with propagated uncertainty
        merged['Delta'] = merged['Last_Mean'] - merged['First_Mean']
        merged['Delta_SEM'] = np.sqrt(merged['First_SEM']**2 + merged['Last_SEM']**2)
        merged['Condition'] = condition
        
        delta_rows.append(merged[['Condition', 'Time_s', 'Delta', 'Delta_SEM', 
                                   'First_Mean', 'First_SEM', 'Last_Mean', 'Last_SEM']])
    
    if delta_rows:
        delta_df = pd.concat(delta_rows, ignore_index=True)
        delta_csv = os.path.join(run_dir, 'delta_psth.csv')
        delta_df.to_csv(delta_csv, index=False)
    else:
        print("WARNING: No delta data computed.")
        delta_df = pd.DataFrame()
    
    # Plot 1: Overlaid first vs last PSTH per condition
    color_map = {'Tracking': '#d62728', 'Playback': '#4d4d4d'}
    
    for condition in ['Tracking', 'Playback']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        first_cond = first_summary[first_summary['Condition'] == condition].sort_values('Time_s')
        last_cond = last_summary[last_summary['Condition'] == condition].sort_values('Time_s')
        
        if not first_cond.empty:
            x = first_cond['Time_s'].to_numpy()
            y = first_cond['PSTH_Mean'].to_numpy()
            sem = first_cond['PSTH_SEM'].fillna(0.0).to_numpy()
            n_first_sessions = int(first_cond['N_Sessions'].max())
            first_events = int(first_block_df[first_block_df['Condition'] == condition]['n_events'].sum())
            first_label = f'First Block (n_sessions={n_first_sessions}, n_events={first_events})'
            ax.plot(x, y, label=first_label, color=color_map.get(condition), linewidth=2.5, linestyle='-')
            ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition), alpha=0.3)
        
        if not last_cond.empty:
            x = last_cond['Time_s'].to_numpy()
            y = last_cond['PSTH_Mean'].to_numpy()
            sem = last_cond['PSTH_SEM'].fillna(0.0).to_numpy()
            n_last_sessions = int(last_cond['N_Sessions'].max())
            last_events = int(last_block_df[last_block_df['Condition'] == condition]['n_events'].sum())
            last_label = f'Last Block (n_sessions={n_last_sessions}, n_events={last_events})'
            ax.plot(x, y, label=last_label, color=color_map.get(condition), linewidth=2.5, linestyle='--')
            ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition), alpha=0.15)
        
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"First vs Last Block PSTH | {condition}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Time from trigger (s)', fontsize=11)
        ax.set_ylabel('Mean activity (a.u.)', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        out_path = os.path.join(run_dir, f"psth_overlay_{_safe_name(condition)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 2: Delta trace with error bounds
    if not delta_df.empty:
        for condition in ['Tracking', 'Playback']:
            cond_delta = delta_df[delta_df['Condition'] == condition].sort_values('Time_s')
            
            if cond_delta.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = cond_delta['Time_s'].to_numpy()
            y = cond_delta['Delta'].to_numpy()
            sem = cond_delta['Delta_SEM'].fillna(0.0).to_numpy()

            first_events = int(first_block_df[first_block_df['Condition'] == condition]['n_events'].sum())
            last_events = int(last_block_df[last_block_df['Condition'] == condition]['n_events'].sum())
            first_cond_summary = first_summary[first_summary['Condition'] == condition]
            last_cond_summary = last_summary[last_summary['Condition'] == condition]
            first_sessions = int(first_cond_summary['N_Sessions'].max()) if not first_cond_summary.empty else 0
            last_sessions = int(last_cond_summary['N_Sessions'].max()) if not last_cond_summary.empty else 0
            n_sessions = max(first_sessions, last_sessions)
            
            delta_label = f'Last - First (n_sessions={n_sessions}, first_events={first_events}, last_events={last_events})'
            ax.plot(x, y, label=delta_label, color=color_map.get(condition), linewidth=2.5)
            ax.fill_between(x, y - sem, y + sem, color=color_map.get(condition), alpha=0.3)
            ax.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
            
            ax.set_title(f"PSTH Delta (Last - First) | {condition}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Time from trigger (s)', fontsize=11)
            ax.set_ylabel('Δ Activity (a.u.)', fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            out_path = os.path.join(run_dir, f"delta_psth_{_safe_name(condition)}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved first block PSTH CSV: {first_block_csv}")
    print(f"Saved last block PSTH CSV: {last_block_csv}")
    if not delta_df.empty:
        print(f"Saved delta PSTH CSV: {delta_csv}")
    print(f"--- First vs Last Block analysis finished. Results saved in {run_dir} ---")


if __name__ == '__main__':
    main()
