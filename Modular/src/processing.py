import numpy as np
import pandas as pd
from config import N_BINS_PRE, N_BINS_POST, EXPECTED_LENGTH, IDX_BASE_START, IDX_BASE_END, IDX_PEAK_START, IDX_PEAK_END

def extract_segments(n_data, event_indices):
    """Extracts and averages neural segments around given event indices."""
    segments = []
    for idx in event_indices:
        start, stop = int(idx - N_BINS_PRE), int(idx + N_BINS_POST)
        if start >= 0 and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == EXPECTED_LENGTH:
                segments.append(seg)
        # Handle zero-padding for events near the end of the recording
        elif start >= 0 and start < n_data.shape[-1]:
            raw_seg = n_data[:, start:]
            pad_width = EXPECTED_LENGTH - raw_seg.shape[1]
            seg = np.pad(raw_seg, ((0, 0), (0, pad_width)), mode='constant')
            segments.append(seg)
            
    if segments:
        return np.mean(np.stack(segments, axis=0), axis=0)
    return None

# --- FOR run_expanding_n.py ---

def compute_z_scored_metrics(tracking_arr, playback_arr, block_id, n_val):
    """Computes Z-scores and metrics for a 2D array (Neurons x Time)."""
    if tracking_arr is None or playback_arr is None: return None, None
    if tracking_arr.size == 0 or playback_arr.size == 0: return None, None
    
    n_neurons = tracking_arr.shape[0]
    neuron_uids = np.arange(n_neurons)
    
    results = []
    for cond_name, arr in [('Tracking', tracking_arr), ('Playback', playback_arr)]:
        # Z-score across the time axis (axis 1)
        mean_t = np.nanmean(arr, axis=1, keepdims=True)
        std_t = np.nanstd(arr, axis=1, keepdims=True) + 1e-8
        z_arr = (arr - mean_t) / std_t
        
        # Calculate Base and Peak
        base = np.nanmean(z_arr[:, IDX_BASE_START:IDX_BASE_END], axis=1)
        peak = np.nanmax(z_arr[:, IDX_PEAK_START:IDX_PEAK_END], axis=1)
        evoked = peak - base
        
        results.append(pd.DataFrame({
            'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val,
            'Condition': cond_name, 'Base_Z': base, 'Peak_Z': peak, 'Evoked_Z': evoked
        }))
        
    return results[0], results[1]

# --- FOR run_chronological_chunks.py ---

def extract_chronological_chunks(n_data, events_tr, events_pb, n_groups, n_per_group):
    """Slices events into chronological chunks (Last N for TR, First N for PB)."""
    tr_chunks, pb_chunks = [], []
    total_needed = n_groups * n_per_group
    
    if len(events_tr) >= total_needed and len(events_pb) >= total_needed:
        # Tracking Chunks (Chronological from the end)
        for i in range(n_groups, 0, -1):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            mean_trace = extract_segments(n_data, events_tr[start:end])
            tr_chunks.append(mean_trace)

        # Playback Chunks (Chronological from the start)
        for i in range(n_groups):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            mean_trace = extract_segments(n_data, events_pb[start:end])
            pb_chunks.append(mean_trace)
            
    return tr_chunks, pb_chunks

def compute_chunk_z_scores(chunk_list, labels, condition, start_order_idx):
    """Computes Z-scores and metrics for a list of data chunks."""
    master_data = []
    
    for chunk_idx, chunk in enumerate(chunk_list):
        if chunk is not None and len(chunk) > 0:
            # Handle list of session means or a single stacked array
            final_dat = np.concatenate(chunk, axis=0) if isinstance(chunk, list) else chunk
            
            # Z-Score
            mean_time = np.nanmean(final_dat, axis=1, keepdims=True)
            std_time = np.nanstd(final_dat, axis=1, keepdims=True) + 1e-8
            z_dat = (final_dat - mean_time) / std_time
            
            # Extract Metrics
            base = np.nanmean(z_dat[:, IDX_BASE_START:IDX_BASE_END], axis=1)
            peak = np.nanmax(z_dat[:, IDX_PEAK_START:IDX_PEAK_END], axis=1)
            evoked = peak - base
            
            master_data.append(pd.DataFrame({
                'Chunk': labels[chunk_idx], 
                'Condition': condition, 
                'Order': start_order_idx + chunk_idx,
                'Base_Z': base, 
                'Peak_Z': peak, 
                'Evoked_Z': evoked
            }))
            
    return master_data