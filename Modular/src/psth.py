"""
Shared PSTH extraction module for multi-analysis support.

Provides reusable functions for:
- Global PSTH (all blocks aggregated)
- First-N / Last-N within-block comparisons
- Transition-anchored PSTH (contiguous TR→PB pairs)
- Standardized session-level and summary aggregation
"""

import numpy as np
import pandas as pd
from config import DT, N_BINS_PRE, N_BINS_POST
from src.processing import extract_segments


def _condition_label(value):
    """Convert numeric condition to human-readable label."""
    if value == 0:
        return "Tracking"
    if value == 1:
        return "Playback"
    return f"Condition_{value}"


def _safe_block_text(block_value):
    """Convert block value to safe filename/label text."""
    if pd.isna(block_value):
        return "nan"
    if float(block_value).is_integer():
        return str(int(block_value))
    return str(block_value).replace('.', 'p')


def extract_global_psth(n_data_list, f_data_list, exclude_block_0=True):
    """
    Extracts global population PSTH per block and condition.
    
    Args:
        n_data_list: List of neural data arrays (session × neurons × time)
        f_data_list: List of feature DataFrames with Frequency_changes, Block, Condition
        exclude_block_0: If True, exclude Block 0 from analysis
    
    Returns:
        DataFrame with columns: [Session, Block, Condition, Time_s, PSTH_Value, n_events]
    """
    expected_len = N_BINS_PRE + N_BINS_POST
    time_axis = (np.arange(expected_len) - N_BINS_PRE) * DT
    rows = []
    
    for session_idx, (n_data, f_data) in enumerate(zip(n_data_list, f_data_list)):
        # Filter for frequency changes (events of interest)
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        
        if not is_change.any():
            continue
        
        # Group by block and condition
        block_condition_groups = f_data[is_change].groupby(['Block', 'Condition'])
        
        for (block, condition_val), sub in block_condition_groups:
            # Skip invalid blocks/conditions
            if pd.isna(block) or pd.isna(condition_val):
                continue
            if exclude_block_0 and float(block) == 0:
                continue
            
            event_indices = sub.index.to_list()
            if not event_indices:
                continue
            
            # Extract mean PSTH across events
            mean_seg = extract_segments(n_data, event_indices)
            if mean_seg is None:
                continue
            
            # Average across neurons to get population PSTH
            psth_trace = np.nanmean(mean_seg, axis=0)
            if psth_trace.shape[0] != expected_len:
                continue
            
            condition_label = _condition_label(condition_val)
            
            # Record per-timepoint
            for time_s, value in zip(time_axis, psth_trace):
                rows.append({
                    'Session': session_idx,
                    'Block': float(block),
                    'Condition': condition_label,
                    'Time_s': float(time_s),
                    'PSTH_Value': float(value),
                    'n_events': int(len(event_indices))
                })
    
    if not rows:
        print("WARNING: No valid PSTH traces extracted (extract_global_psth)")
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def extract_first_n_psth(n_data_list, f_data_list, n_values, exclude_block_0=True):
    """
    Extracts first-N PSTH for within-block comparisons.
    
    Args:
        n_data_list: List of neural data arrays
        f_data_list: List of feature DataFrames
        n_values: List of N values to extract (e.g., [10, 50, 100])
        exclude_block_0: If True, exclude Block 0
    
    Returns:
        DataFrame with columns: [Session, Block, Condition, n_value, Time_s, PSTH_Value, n_events]
    """
    expected_len = N_BINS_PRE + N_BINS_POST
    time_axis = (np.arange(expected_len) - N_BINS_PRE) * DT
    rows = []
    
    for session_idx, (n_data, f_data) in enumerate(zip(n_data_list, f_data_list)):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        
        if not is_change.any():
            continue
        
        block_condition_groups = f_data[is_change].groupby(['Block', 'Condition'])
        
        for (block, condition_val), sub in block_condition_groups:
            if pd.isna(block) or pd.isna(condition_val):
                continue
            if exclude_block_0 and float(block) == 0:
                continue
            
            condition_label = _condition_label(condition_val)
            event_indices = sub.index.to_list()
            
            # For each requested N value
            for n_val in n_values:
                # Take first n_val events
                first_n_indices = event_indices[:min(n_val, len(event_indices))]
                
                if not first_n_indices:
                    continue
                
                mean_seg = extract_segments(n_data, first_n_indices)
                if mean_seg is None:
                    continue
                
                psth_trace = np.nanmean(mean_seg, axis=0)
                if psth_trace.shape[0] != expected_len:
                    continue
                
                for time_s, value in zip(time_axis, psth_trace):
                    rows.append({
                        'Session': session_idx,
                        'Block': float(block),
                        'Condition': condition_label,
                        'n_value': int(n_val),
                        'Time_s': float(time_s),
                        'PSTH_Value': float(value),
                        'n_events': int(len(first_n_indices))
                    })
    
    if not rows:
        print("WARNING: No valid PSTH traces extracted (extract_first_n_psth)")
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def extract_last_n_psth(n_data_list, f_data_list, n_values, exclude_block_0=True):
    """
    Extracts last-N PSTH for within-block comparisons.
    
    Args:
        n_data_list: List of neural data arrays
        f_data_list: List of feature DataFrames
        n_values: List of N values to extract
        exclude_block_0: If True, exclude Block 0
    
    Returns:
        DataFrame with columns: [Session, Block, Condition, n_value, Time_s, PSTH_Value, n_events]
    """
    expected_len = N_BINS_PRE + N_BINS_POST
    time_axis = (np.arange(expected_len) - N_BINS_PRE) * DT
    rows = []
    
    for session_idx, (n_data, f_data) in enumerate(zip(n_data_list, f_data_list)):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        
        if not is_change.any():
            continue
        
        block_condition_groups = f_data[is_change].groupby(['Block', 'Condition'])
        
        for (block, condition_val), sub in block_condition_groups:
            if pd.isna(block) or pd.isna(condition_val):
                continue
            if exclude_block_0 and float(block) == 0:
                continue
            
            condition_label = _condition_label(condition_val)
            event_indices = sub.index.to_list()
            
            # For each requested N value
            for n_val in n_values:
                # Take last n_val events
                last_n_indices = event_indices[-min(n_val, len(event_indices)):]
                
                if not last_n_indices:
                    continue
                
                mean_seg = extract_segments(n_data, last_n_indices)
                if mean_seg is None:
                    continue
                
                psth_trace = np.nanmean(mean_seg, axis=0)
                if psth_trace.shape[0] != expected_len:
                    continue
                
                for time_s, value in zip(time_axis, psth_trace):
                    rows.append({
                        'Session': session_idx,
                        'Block': float(block),
                        'Condition': condition_label,
                        'n_value': int(n_val),
                        'Time_s': float(time_s),
                        'PSTH_Value': float(value),
                        'n_events': int(len(last_n_indices))
                    })
    
    if not rows:
        print("WARNING: No valid PSTH traces extracted (extract_last_n_psth)")
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def extract_transition_psth(n_data_list, f_data_list, exclude_block_0=True):
    """
    Extracts transition-anchored PSTH for contiguous Tracking→Playback block pairs.
    
    Identifies block boundaries where Tracking events precede Playback events
    chronologically within the same session. Groups as (TR_block, PB_block) pairs.
    
    Args:
        n_data_list: List of neural data arrays
        f_data_list: List of feature DataFrames
        exclude_block_0: If True, exclude Block 0
    
    Returns:
        DataFrame with columns: [Session, TR_Block, PB_Block, Condition, Time_s, PSTH_Value, n_events, PairType]
        PairType indicates whether it's 'Tracking' or 'Playback' within the pair
    """
    expected_len = N_BINS_PRE + N_BINS_POST
    time_axis = (np.arange(expected_len) - N_BINS_PRE) * DT
    rows = []
    
    for session_idx, (n_data, f_data) in enumerate(zip(n_data_list, f_data_list)):
        is_change = (f_data['Frequency_changes'] == True) | (f_data['Frequency_changes'] == 1)
        
        if not is_change.any():
            continue
        
        # Get valid blocks and conditions
        valid_blocks = sorted(f_data[is_change]['Block'].unique())
        if exclude_block_0 and 0.0 in valid_blocks:
            valid_blocks = [b for b in valid_blocks if float(b) != 0.0]
        
        # For each pair of consecutive blocks, check for TR→PB transition
        for i in range(len(valid_blocks) - 1):
            block_tr = valid_blocks[i]
            block_pb = valid_blocks[i + 1]
            
            # Get chronologically last TR event in block i and first PB event in block i+1
            tr_events = f_data[
                is_change & 
                (f_data['Block'] == block_tr) & 
                (f_data['Condition'] == 0)
            ].index.to_list()
            
            pb_events = f_data[
                is_change & 
                (f_data['Block'] == block_pb) & 
                (f_data['Condition'] == 1)
            ].index.to_list()
            
            if not tr_events or not pb_events:
                continue
            
            # Extract PSTH for Tracking (last event from this block)
            tr_seg = extract_segments(n_data, [tr_events[-1]])
            if tr_seg is not None:
                psth_tr = np.nanmean(tr_seg, axis=0)
                if psth_tr.shape[0] != expected_len:
                    continue
                for time_s, value in zip(time_axis, psth_tr):
                    rows.append({
                        'Session': session_idx,
                        'TR_Block': float(block_tr),
                        'PB_Block': float(block_pb),
                        'Condition': 'Tracking',
                        'Time_s': float(time_s),
                        'PSTH_Value': float(value),
                        'n_events': 1,
                        'PairType': 'Transition'
                    })
            
            # Extract PSTH for Playback (first event from next block)
            pb_seg = extract_segments(n_data, [pb_events[0]])
            if pb_seg is not None:
                psth_pb = np.nanmean(pb_seg, axis=0)
                if psth_pb.shape[0] != expected_len:
                    continue
                for time_s, value in zip(time_axis, psth_pb):
                    rows.append({
                        'Session': session_idx,
                        'TR_Block': float(block_tr),
                        'PB_Block': float(block_pb),
                        'Condition': 'Playback',
                        'Time_s': float(time_s),
                        'PSTH_Value': float(value),
                        'n_events': 1,
                        'PairType': 'Transition'
                    })
    
    if not rows:
        print("WARNING: No valid transition pairs extracted (extract_transition_psth)")
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def format_psth_summary(session_df, groupby_cols, corrected_p=False):
    """
    Aggregates session-level PSTH DataFrame to summary statistics with SEM.
    
    Args:
        session_df: Session-level DataFrame from extract_*_psth functions
        groupby_cols: Columns to group by for aggregation (e.g., ['Block', 'Condition', 'Time_s'])
        corrected_p: If True, label SEM column for corrected p-values (for documentation)
    
    Returns:
        DataFrame with columns: groupby_cols + [PSTH_Mean, PSTH_STD, PSTH_SEM, N_Sessions]
    """
    if session_df.empty:
        print("WARNING: Empty session_df passed to format_psth_summary")
        return pd.DataFrame()
    
    summary_df = (
        session_df
        .groupby(groupby_cols, as_index=False)
        .agg(
            PSTH_Mean=('PSTH_Value', 'mean'),
            PSTH_STD=('PSTH_Value', 'std'),
            N_Sessions=('Session', 'nunique')
        )
    )
    
    summary_df['PSTH_SEM'] = summary_df['PSTH_STD'] / np.sqrt(summary_df['N_Sessions'].clip(lower=1))
    
    return summary_df
