import os
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
from config import BASE_PATH, OUTPUT_DIR, N_VALUES, N_PER_GROUP, N_GROUPS
from src.data_loader import get_cached_or_processed_data
from src.processing import extract_chronological_chunks, compute_chunk_z_scores
from src.stats import run_global_anova, run_pairwise_matrix
from src.plotting import plot_chunk_progression, plot_significance_heatmap

def generate_labels_and_palette():
    """Generates labels and color palettes dynamically based on N_GROUPS."""
    labels_tr = [f"TR: Last {i*N_PER_GROUP}-{(i-1)*N_PER_GROUP+1}" for i in range(N_GROUPS, 0, -1)]
    labels_pb = [f"PB: First {i*N_PER_GROUP+1}-{(i+1)*N_PER_GROUP}" for i in range(N_GROUPS)]
    
    tr_colors = sns.color_palette("Reds", n_colors=N_GROUPS + 2)[2:]
    pb_colors = sns.color_palette("Greys", n_colors=N_GROUPS + 2)[2:] 
    
    palette = {}
    for i, lbl in enumerate(labels_tr): palette[lbl] = mcolors.to_hex(tr_colors[i])
    for i, lbl in enumerate(labels_pb): palette[lbl] = mcolors.to_hex(pb_colors[i])
        
    return labels_tr, labels_pb, palette

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Loading Data...")
    n_data_s, f_data_s = get_cached_or_processed_data(SESSION_PATHS)
    
    print("2. Extracting Events and Calculating Metrics...")
    labels_tr, labels_pb, palette = generate_labels_and_palette()
    
    tr_chunks_master = [[] for _ in range(N_GROUPS)]
    pb_chunks_master = [[] for _ in range(N_GROUPS)]
    
    for n_data, f_data in zip(n_data_s, f_data_s):
        events_pb = f_data.index[(f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1)].tolist()
        events_tr = f_data.index[(f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0)].tolist()
        
        tr_c, pb_c = extract_chronological_chunks(n_data, events_tr, events_pb, N_GROUPS, N_PER_GROUP)
        
        if tr_c and pb_c:
            for i in range(N_GROUPS):
                if tr_c[i] is not None: tr_chunks_master[i].append(tr_c[i])
                if pb_c[i] is not None: pb_chunks_master[i].append(pb_c[i])

    print("3. Computing Z-Scores and Formatting Data...")
    master_data = []
    master_data.extend(compute_chunk_z_scores(tr_chunks_master, labels_tr, 'Tracking', 0))
    master_data.extend(compute_chunk_z_scores(pb_chunks_master, labels_pb, 'Playback', N_GROUPS))
    
    if not master_data:
        print("No valid data extracted. Exiting.")
        return
        
    df_plot = pd.concat(master_data, ignore_index=True)
    ordered_chunks = df_plot.sort_values('Order')['Chunk'].unique()
    
    print("4. Running Statistics and Plotting...")
    metrics = [('Base_Z', 'Baseline Z-Score'), ('Peak_Z', 'Peak Z-Score'), ('Evoked_Z', 'Evoked Z-Score')]
    
    for metric, title in metrics:
        # ANOVA
        print(f"\n--- {metric} Global ANOVA ---")
        anova = run_global_anova(df_plot, metric)
        if anova is not None:
            print(anova)
            
        # Boxplots
        plot_path = os.path.join(OUTPUT_DIR, f'chunk_progression_{metric}.png')
        plot_chunk_progression(df_plot, metric, title, palette, N_GROUPS, output_path=plot_path)
        
        # Heatmap
        matrix_p, matrix_sig = run_pairwise_matrix(df_plot, metric, ordered_chunks)
        heatmap_path = os.path.join(OUTPUT_DIR, f'heatmap_{metric}.png')
        plot_significance_heatmap(matrix_p, matrix_sig, f"{title} Pairwise Matrix", output_path=heatmap_path)

    print(f"\nPipeline complete! All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()