import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DT, OUTPUT_DIR, SESSION_CONFIGS
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt

LENGTH_TOLERANCE_SECONDS = 60.0


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


def _assign_length_groups(length_df, tolerance_seconds=LENGTH_TOLERANCE_SECONDS):
    """Assign sessions into block-length groups where lengths agree within +/- tolerance seconds."""
    if length_df.empty:
        return length_df

    ordered = length_df.sort_values('Block_Length_s').copy()
    group_ids = []
    current_group = 1
    current_ref = float(ordered.iloc[0]['Block_Length_s'])
    running_sum = current_ref
    running_count = 1

    for idx, row in enumerate(ordered.itertuples(index=False)):
        if idx == 0:
            group_ids.append(current_group)
            continue

        length_val = float(row.Block_Length_s)
        if abs(length_val - current_ref) <= tolerance_seconds:
            group_ids.append(current_group)
            running_sum += length_val
            running_count += 1
            current_ref = running_sum / running_count
        else:
            current_group += 1
            group_ids.append(current_group)
            current_ref = length_val
            running_sum = length_val
            running_count = 1

    ordered['Length_Group'] = group_ids
    return ordered


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"Trigger_Stats_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)

    print("--- STARTING TRIGGER STATS ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    _, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)

    trigger_rows = []
    for session_idx, f_data in enumerate(f_data_s):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        if not is_change.any():
            continue

        subset = f_data.loc[is_change, ['Block', 'Condition']].copy()
        subset['Session'] = session_idx
        subset['Trigger_Index'] = subset.index.astype(int)

        # Convert trigger order inside each session/block/condition into block-relative time.
        subset['Block_Time_s'] = (
            subset.groupby(['Session', 'Block', 'Condition'])['Trigger_Index']
            .transform(lambda s: (s - s.min()) * DT)
            .astype(float)
        )
        trigger_rows.append(subset)

    if not trigger_rows:
        print("CRITICAL ERROR: No trigger events found across sessions.")
        return

    trigger_df = pd.concat(trigger_rows, ignore_index=True)
    trigger_df = trigger_df.dropna(subset=['Block', 'Condition'])

    if trigger_df.empty:
        print("CRITICAL ERROR: Trigger table is empty after removing missing Block/Condition.")
        return

    trigger_df['Condition_Label'] = trigger_df['Condition'].map(_condition_label)

    summary_df = (
        trigger_df
        .groupby(['Block', 'Condition_Label'], as_index=False)
        .size()
        .rename(columns={'size': 'Total_Triggers'})
        .sort_values(['Block', 'Condition_Label'])
    )

    counts_path = os.path.join(run_dir, 'trigger_counts_by_block_condition.csv')
    summary_df.to_csv(counts_path, index=False)
    print(f"Saved trigger count summary: {counts_path}")

    detailed_counts = (
        trigger_df
        .groupby(['Session', 'Block', 'Condition_Label'], as_index=False)
        .size()
        .rename(columns={'size': 'Session_Trigger_Count'})
        .sort_values(['Session', 'Block', 'Condition_Label'])
    )
    detailed_path = os.path.join(run_dir, 'trigger_counts_by_session_block_condition.csv')
    detailed_counts.to_csv(detailed_path, index=False)

    global_counts_df = (
        trigger_df
        .groupby('Condition_Label', as_index=False)
        .size()
        .rename(columns={'size': 'Total_Triggers'})
        .sort_values('Condition_Label')
    )
    global_counts_path = os.path.join(run_dir, 'trigger_counts_global_by_condition.csv')
    global_counts_df.to_csv(global_counts_path, index=False)

    global_time_df = (
        trigger_df[['Condition_Label', 'Block_Time_s']]
        .copy()
    )
    global_time_df['Block_Time_s'] = pd.to_numeric(global_time_df['Block_Time_s'], errors='coerce')
    global_time_df = global_time_df.dropna(subset=['Block_Time_s'])

    global_hist_path = os.path.join(run_dir, 'trigger_density_global.png')
    if not global_time_df.empty:
        condition_counts = global_time_df['Condition_Label'].value_counts().to_dict()
        hue_order = sorted(condition_counts.keys())
        count_palette = {
            label: f"{label} (n={condition_counts[label]})"
            for label in hue_order
        }
        plot_df = global_time_df.copy()
        plot_df['Condition_With_N'] = plot_df['Condition_Label'].map(count_palette)
        hue_with_n_order = [count_palette[label] for label in hue_order]

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        sns.histplot(
            data=plot_df,
            x='Block_Time_s',
            hue='Condition_With_N',
            hue_order=hue_with_n_order,
            bins='auto',
            stat='density',
            common_norm=False,
            element='step',
            fill=True,
            alpha=0.25,
            kde=True,
            ax=ax,
        )
        ax.set_title('Global Trigger Density vs Block Time')
        ax.set_xlabel('Block Time (s)')
        ax.set_ylabel('Density')
        ax.grid(True, axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(global_hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    hist_count = 0
    grouped = trigger_df.groupby(['Block', 'Condition_Label'])
    for (block, condition_label), group in grouped:
        block_text = _safe_block_text(block)

        session_lengths = (
            group.groupby('Session', as_index=False)['Trigger_Index']
            .agg(min_index='min', max_index='max')
        )
        session_lengths['Block_Length_Index'] = session_lengths['max_index'] - session_lengths['min_index']
        session_lengths['Block_Length_s'] = session_lengths['Block_Length_Index'] * DT
        session_lengths = _assign_length_groups(session_lengths, tolerance_seconds=LENGTH_TOLERANCE_SECONDS)

        if session_lengths['Length_Group'].nunique() > 1:
            print(
                f"Block {block_text} | {condition_label}: split into "
                f"{session_lengths['Length_Group'].nunique()} length groups "
                f"(tolerance +/-{LENGTH_TOLERANCE_SECONDS:.1f}s)."
            )

        group_with_len = group.merge(
            session_lengths[['Session', 'Length_Group', 'Block_Length_Index', 'Block_Length_s']],
            on='Session',
            how='left',
        )

        for len_group, subgroup in group_with_len.groupby('Length_Group'):
            block_time = pd.to_numeric(subgroup['Block_Time_s'], errors='coerce').dropna()
            if block_time.empty:
                continue

            length_vals = session_lengths.loc[
                session_lengths['Length_Group'] == len_group,
                'Block_Length_s'
            ]
            len_min = float(length_vals.min())
            len_max = float(length_vals.max())
            n_sessions = int(subgroup['Session'].nunique())

            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.histplot(
                block_time,
                bins='auto',
                stat='density',
                kde=(len(block_time) >= 3),
                color='#4C72B0',
                edgecolor='white',
                linewidth=0.4,
                ax=ax,
            )

            ax.set_title(
                f"Density vs Block Time | Block {block_text} | {condition_label} | "
                f"LenGroup {int(len_group)} [{len_min:.1f}s-{len_max:.1f}s] | sessions={n_sessions} | n={len(block_time)}"
            )
            ax.set_xlabel('Block Time (s)')
            ax.set_ylabel('Density')
            ax.grid(True, axis='y', alpha=0.25)

            fname = (
                f"density_hist_block_{_safe_name(block_text)}"
                f"_condition_{_safe_name(condition_label)}"
                f"_len_group_{int(len_group)}"
                f"_{int(round(len_min))}s_{int(round(len_max))}s.png"
            )
            fig_path = os.path.join(run_dir, fname)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            hist_count += 1

    print(f"Saved {hist_count} density histogram(s).")
    print(f"Saved global trigger count summary: {global_counts_path}")
    if not global_time_df.empty:
        print(f"Saved global trigger density plot: {global_hist_path}")
    print(f"--- Trigger stats analysis finished. Results saved in {run_dir} ---")


if __name__ == '__main__':
    main()
