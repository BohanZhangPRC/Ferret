# --- CRITICAL FIX: Force non-interactive backend for background processing ---
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. EXPANDING N PLOTS ---

def plot_annotated_boxplot(df_plot, stats_df, metric_col, title, output_path=None, show_plot=False):
    """Generates a boxplot annotated with N counts and statistical significance brackets."""
    unique_blocks = sorted(df_plot['Block'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df_plot, x='Block', y=metric_col, hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
    )
    
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.20) 
    
    for x_idx, block in enumerate(unique_blocks):
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')][metric_col]
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')][metric_col]
        
        y_pos_tr = tr_data.max() + (y_range * 0.02) if not tr_data.empty else y_min
        y_pos_pb = pb_data.max() + (y_range * 0.02) if not pb_data.empty else y_min
        
        tr_n = len(tr_data)
        pb_n = len(pb_data)
        ax.text(x_idx, min(y_pos_tr, y_pos_pb) - (y_range * 0.08), f'nTR={tr_n} | nPB={pb_n}', 
                ha='center', va='top', fontsize=9, color='gray')

        # Add Brackets
        stat_row = stats_df[stats_df['Block'] == block]
        sig = stat_row['Significance'].values[0] if not stat_row.empty else 'ns'
        
        if sig != 'ns' and not tr_data.empty and not pb_data.empty:
            bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.08)
            bracket_tip = y_range * 0.02
            x1, x2 = x_idx - 0.2, x_idx + 0.2 
            ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    ax.set_ylabel(metric_col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig) # Closes background figure to free memory


# --- 2. CHRONOLOGICAL CHUNKS PLOTS ---

def plot_chunk_progression(df_plot, metric_col, title, palette, n_groups, output_path=None, show_plot=False):
    """Plots boxplots of chunk progression with N counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df_plot.sort_values('Order'), x='Chunk', y=metric_col, hue='Chunk',
        palette=palette, ax=ax, width=0.6, fliersize=3, dodge=False
    )

    ax.axvline(n_groups - 0.5, color='black', linestyle='--', alpha=0.5, zorder=0)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.25)

    # Add N counts
    unique_chunks = df_plot.sort_values('Order')['Chunk'].unique()
    for idx, chunk_label in enumerate(unique_chunks):
        n_count = len(df_plot[df_plot['Chunk'] == chunk_label])
        if n_count > 0:
            ax.text(idx, y_max + y_range * 0.05, f'n={n_count}', 
                    ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_significance_heatmap(matrix_p, matrix_sig, title, output_path=None, show_plot=False, corrected_p=False):
    """Plots a heatmap of pairwise p-values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ensure data is explicitly float type for the heatmap
    p_data = matrix_p.astype(float)
    
    cbar_label = 'Corrected P-Value' if corrected_p else 'Uncorrected P-Value'

    sns.heatmap(
        p_data, annot=matrix_sig, fmt="", cmap='coolwarm_r', 
        vmin=0, vmax=0.05, cbar_kws={'label': cbar_label}, 
        ax=ax, linewidths=1, linecolor='white'
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)