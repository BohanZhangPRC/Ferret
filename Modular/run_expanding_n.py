import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import SESSION_CONFIGS, OUTPUT_DIR, N_VALUES
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt
from src.processing import extract_segments, compute_z_scored_metrics
from src.stats import run_paired_stats
from src.plotting import plot_annotated_boxplot

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"Expanding_N_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)
    
    print(f"--- STARTING EXPANDING 'N' ANALYSIS ---")
    print(f"Saved used file list: {used_files_path}")
    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    
    all_raw_results = []

    for n_val in N_VALUES:
        found_data_for_n = False
        print(f"Processing N={n_val}...")
        
        for i, (n_data, f_data) in enumerate(zip(n_data_s, f_data_s)):
            # 1. Identify valid triggers using Boolean masks
            is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
            is_tr = (f_data['Condition'] == 0)
            is_pb = (f_data['Condition'] == 1)

            tr_idx_all = f_data.index[is_change & is_tr].tolist()
            pb_idx_all = f_data.index[is_change & is_pb].tolist()
            
            if not tr_idx_all or not pb_idx_all:
                continue

            # 2. Find blocks present in both Tracking and Playback
            blocks = f_data['Block'].values
            valid_blocks = sorted(set(blocks[tr_idx_all]).intersection(set(blocks[pb_idx_all])))
            valid_blocks = [b for b in valid_blocks if b != 0 and not np.isnan(b)]
            
            for b in valid_blocks:
                b_tr_idx = [idx for idx in tr_idx_all if blocks[idx] == b]
                b_pb_idx = [idx for idx in pb_idx_all if blocks[idx] == b]

                # --- THE FIX: Take subset if count >= n_val ---
                if len(b_tr_idx) >= n_val and len(b_pb_idx) >= n_val:
                    # Grab exactly n_val trials
                    tr_subset = b_tr_idx[-n_val:] # Last N tracking
                    pb_subset = b_pb_idx[:n_val]  # First N playback
                    
                    tr_mean = extract_segments(n_data, tr_subset)
                    pb_mean = extract_segments(n_data, pb_subset)
                    
                    if tr_mean is not None and pb_mean is not None:
                        # compute_z_scored_metrics expects (Neurons x Time)
                        df_tr, df_pb = compute_z_scored_metrics(tr_mean, pb_mean, b, n_val)
                        if df_tr is not None:
                            all_raw_results.append(df_tr)
                            all_raw_results.append(df_pb)
                            found_data_for_n = True

    # 3. Save and Plot
    if not all_raw_results:
        print("CRITICAL ERROR: No data blocks matched the N requirements.")
        return

    df_raw = pd.concat(all_raw_results, ignore_index=True)
    metrics = ['Peak_Z', 'Base_Z', 'Evoked_Z']
    
    for n_val in N_VALUES:
        df_n = df_raw[df_raw['n_triggers'] == n_val]
        if df_n.empty: continue
            
        for metric in metrics:
            stats_df = run_paired_stats(df_n, metric)
            stats_df.to_csv(os.path.join(run_dir, f'stats_{metric}_n_{n_val}.csv'), index=False)
            
            plot_path = os.path.join(run_dir, f'plot_{metric}_n_{n_val}.png')
            plot_annotated_boxplot(df_n, stats_df, metric, f'{metric} (N={n_val})', output_path=plot_path)

    print(f"Success! Analysis saved to {run_dir}")

if __name__ == "__main__":
    main()