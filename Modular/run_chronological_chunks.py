import os
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import datetime
from config import SESSION_CONFIGS, OUTPUT_DIR, N_PER_GROUP, N_GROUPS, RUN_CHUNK_FDR
from src.data_loader import get_cached_or_processed_data, write_used_npy_files_txt
from src.processing import extract_chronological_chunks, compute_chunk_z_scores
from src.stats import run_global_anova, run_pairwise_matrix
from src.plotting import plot_chunk_progression, plot_significance_heatmap

def generate_labels_and_palette():
    labels_tr = [f"TR: Last {i*N_PER_GROUP}-{(i-1)*N_PER_GROUP+1}" for i in range(N_GROUPS, 0, -1)]
    labels_pb = [f"PB: First {i*N_PER_GROUP+1}-{(i+1)*N_PER_GROUP}" for i in range(N_GROUPS)]
    
    tr_colors = sns.color_palette("Reds", n_colors=N_GROUPS + 2)[2:]
    pb_colors = sns.color_palette("Greys", n_colors=N_GROUPS + 2)[2:] 
    
    palette = {}
    for i, lbl in enumerate(labels_tr): palette[lbl] = mcolors.to_hex(tr_colors[i])
    for i, lbl in enumerate(labels_pb): palette[lbl] = mcolors.to_hex(pb_colors[i])
        
    return labels_tr, labels_pb, palette

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"Chronological_Chunks_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    used_files_path = write_used_npy_files_txt(run_dir, SESSION_CONFIGS)
    
    print(f"--- STARTING CHRONOLOGICAL CHUNKS ANALYSIS ---")
    print(f"Results will be saved in: {run_dir}")
    print(f"Saved used file list: {used_files_path}")

    # 1. Load Data
    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_CONFIGS)
    
    # Define labels and palette with explicit N_PER_GROUP ranges for axis display.
    tr_labels, pb_labels, palette = generate_labels_and_palette()
    ordered_chunks = tr_labels + pb_labels

    all_raw_data = []

    # 2. Extraction and Processing
    for i, (n_data, f_data) in enumerate(zip(n_data_s, f_data_s)):
        # Identify "Change" events (Boolean True or Numeric 1)
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        
        # Identify Tracking (0) and Playback (1)
        events_tr = f_data.index[is_change & (f_data['Condition'] == 0)].tolist()
        events_pb = f_data.index[is_change & (f_data['Condition'] == 1)].tolist()
        
        # Extract the mean traces for each chunk
        tr_chunk_means, pb_chunk_means = extract_chronological_chunks(
            n_data, events_tr, events_pb, N_GROUPS, N_PER_GROUP
        )
        
        # Compute metrics (Base, Peak, Evoked) for these chunks
        if tr_chunk_means and pb_chunk_means:
            tr_metrics = compute_chunk_z_scores(tr_chunk_means, tr_labels, 'Tracking', 0)
            pb_metrics = compute_chunk_z_scores(pb_chunk_means, pb_labels, 'Playback', N_GROUPS)
            all_raw_data.extend(tr_metrics + pb_metrics)

    if not all_raw_data:
        print("CRITICAL ERROR: No sessions had enough trials to form the requested chunks.")
        return

    # 3. Final Consolidation and Stats
    df_raw = pd.concat(all_raw_data, ignore_index=True)
    metrics = ['Peak_Z', 'Base_Z', 'Evoked_Z']
    
    # Always generate the Global ANOVA and Boxplots
    for metric in metrics:
        print(f"Analyzing {metric}...")
        anova_table = run_global_anova(df_raw, metric)
        if anova_table is not None:
            anova_table.to_csv(os.path.join(run_dir, f'anova_{metric}.csv'))
        
        plot_path = os.path.join(run_dir, f'boxplot_progression_{metric}.png')
        plot_chunk_progression(
            df_raw, metric, f'Chronological Progression: {metric}', 
            palette, N_GROUPS, output_path=plot_path
        )

    # 4. Conditional Pairwise FDR Testing
    # This checks the "tick" you set in the GUI
    if RUN_CHUNK_FDR:
        print("Running FDR-corrected pairwise tests (this may take a moment)...")
        for metric in metrics:
            p_matrix, sig_matrix = run_pairwise_matrix(df_raw, metric, ordered_chunks)
            
            # Save the raw P-values
            p_matrix.to_csv(os.path.join(run_dir, f'pairwise_p_{metric}.csv'))
            
            # Generate the significance heatmap
            heatmap_path = os.path.join(run_dir, f'heatmap_{metric}.png')
            plot_significance_heatmap(
                p_matrix, sig_matrix, f'Pairwise Significance: {metric}', 
                output_path=heatmap_path,
                corrected_p=RUN_CHUNK_FDR,
            )
    else:
        print("Skipping pairwise FDR tests (Checkbox was unchecked).")

    print(f"--- Analysis Finished. Results saved in {run_dir} ---")

if __name__ == "__main__":
    main()