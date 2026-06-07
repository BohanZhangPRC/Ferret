"""
Session Diagnostics Analysis

Provides data quality assessment and session mismatch reporting.
Checks for:
- Tracking/Playback event count mismatches
- Invalid conditions or blocks
- Sessions with insufficient events
- Block structure issues
- Frequency change filtering coverage

Output:
- session_diagnostics.csv: Per-session diagnostic metrics
- diagnostics_summary.txt: Human-readable summary report
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, SESSION_CONFIGS
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"Diagnostics_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING SESSION DIAGNOSTICS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    
    # If data loading failed, report why
    if not n_data_s or not f_data_s:
        print("CRITICAL ERROR: Data loading failed. Check configuration.")
        return

    diagnostic_rows = []
    summary_lines = []

    summary_lines.append("=" * 80)
    summary_lines.append("SESSION DIAGNOSTICS REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    total_sessions = len(f_data_s)
    valid_sessions = 0
    sessions_with_warnings = 0
    block_condition_counts = []

    for session_idx, f_data in enumerate(f_data_s):
        session_id = f"Session_{session_idx:03d}"
        diagnostic_row = {'Session_ID': session_id}

        # Total events
        total_events = len(f_data)
        diagnostic_row['Total_Events'] = total_events

        # Frequency change events (events of interest)
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        n_freq_change = is_change.sum()
        diagnostic_row['Frequency_Change_Events'] = n_freq_change

        # Valid blocks (non-NaN)
        valid_blocks_mask = f_data['Block'].notna()
        n_valid_blocks = valid_blocks_mask.sum()
        diagnostic_row['Valid_Block_Events'] = n_valid_blocks

        # Valid conditions (0 or 1 only, exclude condition -1)
        valid_conditions_mask = f_data['Condition'].isin([0, 1])
        n_valid_conditions = valid_conditions_mask.sum()
        diagnostic_row['Valid_Condition_Events'] = n_valid_conditions

        # Events with both frequency change AND valid block AND valid condition
        combined_mask = is_change & valid_blocks_mask & valid_conditions_mask
        n_usable = combined_mask.sum()
        diagnostic_row['Usable_Events'] = n_usable

        # Block 0 presence
        block_0_count = ((f_data['Block'] == 0) & is_change & valid_conditions_mask).sum()
        diagnostic_row['Block_0_Events'] = block_0_count

        # Tracking vs Playback event counts (on usable events)
        usable_df = f_data[combined_mask]
        n_tracking = (usable_df['Condition'] == 0).sum()
        n_playback = (usable_df['Condition'] == 1).sum()
        diagnostic_row['Tracking_Events'] = n_tracking
        diagnostic_row['Playback_Events'] = n_playback

        # Store per-(Block, Condition) event counts so we can report global min/max.
        if not usable_df.empty:
            grouped_counts = usable_df.groupby(['Block', 'Condition']).size()
            for (block_val, condition_val), count_val in grouped_counts.items():
                block_condition_counts.append({
                    'Session_ID': session_id,
                    'Block': block_val,
                    'Condition': int(condition_val),
                    'Count': int(count_val),
                })

        # Mismatch check
        mismatch = abs(n_tracking - n_playback)
        diagnostic_row['TR_PB_Mismatch'] = mismatch
        diagnostic_row['Mismatch_Pct'] = (mismatch / max(n_tracking, n_playback, 1)) * 100

        # Blocks
        if is_change.any():
            blocks = sorted([b for b in f_data[combined_mask]['Block'].unique() if not pd.isna(b)])
            diagnostic_row['Num_Blocks'] = len(blocks)
            diagnostic_row['Block_List'] = str(blocks)
        else:
            diagnostic_row['Num_Blocks'] = 0
            diagnostic_row['Block_List'] = "NONE"

        diagnostic_rows.append(diagnostic_row)

        # Summary reporting
        summary_lines.append(f"Session {session_idx}: {len(f_data)} total events")
        summary_lines.append(f"  - Frequency change events: {n_freq_change}")
        summary_lines.append(f"  - Usable events (freq change + valid block + valid condition): {n_usable}")
        summary_lines.append(f"  - Tracking: {n_tracking}, Playback: {n_playback}")

        is_valid = n_usable > 0
        if is_valid:
            valid_sessions += 1

        issues = []
        if n_freq_change == 0:
            issues.append("NO frequency change events")
        if n_usable == 0:
            issues.append("NO usable events")
        if mismatch > 0:
            issues.append(f"TR/PB mismatch: {mismatch} events ({diagnostic_row['Mismatch_Pct']:.1f}%)")
        if block_0_count > 0:
            issues.append(f"Contains Block 0: {block_0_count} events")

        if issues:
            sessions_with_warnings += 1
            summary_lines.append(f"  [WARNING] {'; '.join(issues)}")

        summary_lines.append("")

    # Summary statistics
    summary_lines.append("=" * 80)
    summary_lines.append("SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total sessions: {total_sessions}")
    summary_lines.append(f"Sessions with usable events: {valid_sessions}")
    summary_lines.append(f"Sessions with warnings: {sessions_with_warnings}")
    summary_lines.append(f"Sessions ready for analysis: {valid_sessions}")

    if block_condition_counts:
        min_entry = min(block_condition_counts, key=lambda x: x['Count'])
        max_entry = max(block_condition_counts, key=lambda x: x['Count'])

        summary_lines.append("")
        summary_lines.append("GLOBAL BLOCK+CONDITION EVENT EXTREMES")
        summary_lines.append("-" * 80)
        summary_lines.append(
            f"Global minimum: {min_entry['Count']} events "
            f"(Session={min_entry['Session_ID']}, Block={int(min_entry['Block'])}, "
            f"Condition={min_entry['Condition']})"
        )
        summary_lines.append(
            f"Global maximum: {max_entry['Count']} events "
            f"(Session={max_entry['Session_ID']}, Block={int(max_entry['Block'])}, "
            f"Condition={max_entry['Condition']})"
        )
    summary_lines.append("")

    if sessions_with_warnings > 0:
        summary_lines.append("ACTION REQUIRED:")
        summary_lines.append("-" * 80)
        for idx, row in enumerate(diagnostic_rows):
            if row['Usable_Events'] == 0:
                summary_lines.append(f"  - {row['Session_ID']}: SKIP (0 usable events)")
            elif row['Mismatch_Pct'] > 20:
                mismatch = row['TR_PB_Mismatch']
                summary_lines.append(
                    f"  - {row['Session_ID']}: REVIEW (TR/PB mismatch: {int(mismatch)} events, "
                    f"{row['Mismatch_Pct']:.1f}%)"
                )

    # Save diagnostics
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    diagnostic_csv = os.path.join(run_dir, 'session_diagnostics.csv')
    diagnostic_df.to_csv(diagnostic_csv, index=False)

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(run_dir, 'diagnostics_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(summary_text)

    print(f"\nSaved diagnostic CSV: {diagnostic_csv}")
    print(f"Saved summary report: {summary_path}")
    print(f"--- Session diagnostics finished. Results saved in {run_dir} ---")


if __name__ == '__main__':
    main()
