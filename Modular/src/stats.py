import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests

# --- FOR "EXPANDING N" ANALYSIS ---

def run_paired_stats(df, metric_col):
    """Runs paired t-tests across blocks, returning FDR-corrected stats."""
    unique_blocks = sorted(df['Block'].unique())
    stats_list, p_values = [], []

    for block in unique_blocks:
        tr_data = df[(df['Block'] == block) & (df['Condition'] == 'Tracking')].sort_values('Neuron_UID')[metric_col]
        pb_data = df[(df['Block'] == block) & (df['Condition'] == 'Playback')].sort_values('Neuron_UID')[metric_col]
        
        paired_n = len(tr_data)
        if paired_n > 1 and len(pb_data) > 1:
            stat, p_val = ttest_rel(tr_data, pb_data, nan_policy='omit')
        else:
            stat, p_val = np.nan, np.nan
            
        p_values.append(p_val)
        stats_list.append({'Block': block, 'N_Pairs': paired_n, 't_stat': stat, 'raw_p_value': p_val})
        
    # FDR Correction
    valid_p = [p for p in p_values if not np.isnan(p)]
    p_adj_valid = multipletests(valid_p, alpha=0.05, method='fdr_bh')[1] if valid_p else []
    
    adj_idx = 0
    for res in stats_list:
        if np.isnan(res['raw_p_value']):
            res['adj_p_value'], res['Significance'] = np.nan, 'ns'
        else:
            adj_p = p_adj_valid[adj_idx]
            res['adj_p_value'] = adj_p
            adj_idx += 1
            if adj_p < 0.001: res['Significance'] = '***'
            elif adj_p < 0.01: res['Significance'] = '**'
            elif adj_p < 0.05: res['Significance'] = '*'
            else: res['Significance'] = 'ns'
            
    return pd.DataFrame(stats_list)


# --- FOR "CHRONOLOGICAL CHUNKS" ANALYSIS ---

def run_global_anova(df, metric_col):
    """Runs a global ANOVA across all chunks to test for general significance."""
    try:
        model = ols(f'{metric_col} ~ C(Chunk)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table
    except Exception as e:
        print(f"Could not perform ANOVA for {metric_col}: {e}")
        return None

def run_pairwise_matrix(df, metric_col, ordered_chunks):
    """Calculates full pairwise t-tests for heatmap generation."""
    matrix_p = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=float)
    matrix_sig = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=str)

    for c1 in ordered_chunks:
        for c2 in ordered_chunks:
            if c1 == c2:
                matrix_p.loc[c1, c2] = 1.0
                matrix_sig.loc[c1, c2] = '-'
            else:
                d1 = df[df['Chunk'] == c1][metric_col]
                d2 = df[df['Chunk'] == c2][metric_col]
                
                # Welch's t-test
                if len(d1) > 1 and len(d2) > 1:
                    p = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')[1]
                else:
                    p = 1.0
                
                matrix_p.loc[c1, c2] = p
                
                # Significance markers
                if p < 0.001: matrix_sig.loc[c1, c2] = '***'
                elif p < 0.01: matrix_sig.loc[c1, c2] = '**'
                elif p < 0.05: matrix_sig.loc[c1, c2] = '*'
                else: matrix_sig.loc[c1, c2] = ''
                
    return matrix_p, matrix_sig