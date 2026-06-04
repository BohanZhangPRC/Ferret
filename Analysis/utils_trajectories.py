import numpy as np
import matplotlib as mpl
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import pickle
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter

mpl.rcdefaults()


### --- Get events vicinity (PSTH) ---

def get_event_vicinity(bool_array, t_pre, t_post, dt, overlap_thresh=0.2):
    """
    Return a 2D array of indices around True values in a boolean array,
    skipping events that overlap more than allowed.

    Parameters:
        bool_array (np.ndarray): 1D boolean array.
        t_pre (float): Time before event in seconds.
        t_post (float): Time after event in seconds.
        dt (float): Duration of each bin in seconds.
        overlap_thresh (float or int): 
            If float in [0,1], fraction of window allowed to overlap previous event.
            If int >=1, number of bins allowed to overlap. 0 = no overlap.

    Returns:
        np.ndarray: 2D array of index windows, one per event.
    """
    
    # Convert times to number of bins
    n_pre = int(np.round(t_pre / dt))
    n_post = int(np.round(t_post / dt))
    full_window_size = n_pre + n_post + 1  # total window size

    event_indices = np.where(bool_array)[0]
    total_len = len(bool_array)
    last_window_end = -np.inf
    result = []

    # Compute allowed overlap in bins
    if isinstance(overlap_thresh, float):
        max_allowed_overlap = int(full_window_size * overlap_thresh)
    else:
        max_allowed_overlap = int(overlap_thresh)

    for idx in event_indices:
        start = max(0, idx - n_pre)
        end = min(total_len, idx + n_post + 1)
        indices = np.arange(start, end)
        
        # Pad at edges if needed
        if len(indices) < full_window_size:
            pad_len = full_window_size - len(indices)
            indices = np.pad(indices, (0, pad_len), mode='constant', constant_values=-1)
            

        # Check overlap with previous window
        if start < last_window_end and (last_window_end - start) > max_allowed_overlap:
            continue  # skip this event if overlap too large

        result.append(indices)
        last_window_end = end  # update last window end

    return np.array(result)


def compute_mean_trajectories(trajectories, vicinity_indices, event_values, n_pre=None):
    """
    Computes the mean trajectory in the vicinity of events grouped by event label,
    with optional baseline removal (subtracted per event before averaging).

    Parameters:
        trajectories (np.ndarray): [T, D] array of trajectory over time.
        vicinity_indices (np.ndarray): [N_events, window_size] of indices for each event.
        event_values (np.ndarray): [N_events] of numerical event labels (can be float or int).
        n_pre (int, optional): Number of pre-trigger bins used as baseline window.
            If None, no baseline removal is applied.

    Returns:
        dict: {label: mean_trajectory} where mean_trajectory is [window_size, D]
    """
    unique_labels = np.unique(event_values)
    result = {}

    for label in unique_labels:
        label_mask = event_values == label
        label_indices = vicinity_indices[label_mask]

        valid_segments = []
        for inds in label_indices:
            valid_mask = inds >= 0
            if np.any(valid_mask):
                valid_inds = inds[valid_mask]
                segment = trajectories[valid_inds]

                # Pad with NaNs if needed (for edge events)
                if segment.shape[0] < inds.shape[0]:
                    padded = np.full((inds.shape[0], trajectories.shape[1]), np.nan)
                    padded[valid_mask] = segment
                    segment = padded

                # Baseline removal: subtract mean of pre-trigger window per dimension
                if n_pre is not None:
                    baseline = np.nanmean(segment[:n_pre], axis=0)  # [D]
                    segment = segment - baseline  # broadcasts over window_size

                valid_segments.append(segment)

        if valid_segments:
            mean_traj = np.nanmean(np.stack(valid_segments), axis=0)
            result[label] = mean_traj

    return result


### --- Get average traj 

def process_traj(data, event_idx, n_pre=None):
    """
    Compute the mean trajectory across all triggers (no grouping).

    Parameters:
        data (np.ndarray): [D, T] array of trajectory over time.e
        event_idx (np.ndarray): [N_events, window_size] of indices for each event.
        n_pre (int, optional): Number of pre-trigger bins used as baseline window.
            If None, no baseline removal is applied.

    Returns:
        np.ndarray: [window_size, D] mean trajectory, or None if no valid segments.
    """
    segments = []
    
    for inds in event_idx:
        valid_mask = inds >= 0
        if np.any(valid_mask):
            valid_inds = inds[valid_mask]
            segment = data.T[valid_inds]

            if segment.shape[0] < inds.shape[0]:
                padded = np.full((inds.shape[0], data.T.shape[1]), np.nan)
                padded[valid_mask] = segment
                segment = padded

            if n_pre is not None:
                baseline = np.nanmean(segment[:n_pre], axis=0)  # [D]
                segment = segment - baseline

            segments.append(segment)

    if len(segments) == 0:
        return None

    return np.nanmean(np.stack(segments), axis=0)


def extract_traj(n_data_s, f_data_s, t_pre, t_post, dt, overlap_thresh=0.2, n_pre = None, full=True):
    """"
    extract the average trajectory per neuron. 
    output : session x time neuron
    """

    all_traj_track = []
    all_traj_track_p, all_traj_track_m = [], []
    all_traj_pb = []
    all_traj_pb_p, all_traj_pb_m = [], []
    all_traj_mock = []
    all_traj_mock_p, all_traj_mock_m = [], []

    for n_data, f_data in tqdm(zip(n_data_s, f_data_s), total=len(n_data_s), desc="extract_traj"):

        direc = f_data['Change_direction'].to_numpy()
        mock_direc = f_data['Mock_direction'].to_numpy()

        triggers = f_data['Frequency_changes'].to_numpy()
        triggers_mock = f_data['Mock_change'].to_numpy()
        condition = f_data['Condition'].to_numpy()

        trigger_map = {
            'track':  triggers * (condition == 0),
            'track_p': triggers * (condition == 0) * (direc == 1),
            'track_m': triggers * (condition == 0) * (direc == -1),
            'pb': triggers * (condition == 1),
            'pb_p': triggers * (condition == 1) * (direc == 1),
            'pb_m': triggers * (condition == 1) * (direc == -1),
            'mock': triggers_mock * (condition == 1),
            'mock_p': triggers_mock * (condition == 1) * (mock_direc == 1),
            'mock_m': triggers_mock * (condition == 1) * (mock_direc == -1)
        }

        event_idx_map = {
            k: get_event_vicinity(v, t_pre, t_post, dt, overlap_thresh)
            for k, v in trigger_map.items()
        }

        all_traj_track.append(process_traj(n_data, event_idx_map['track'], n_pre=n_pre))
        all_traj_track_p.append(process_traj(n_data, event_idx_map['track_p'], n_pre=n_pre))
        all_traj_track_m.append(process_traj(n_data, event_idx_map['track_m'], n_pre=n_pre))

        all_traj_pb.append(process_traj(n_data, event_idx_map['pb'], n_pre=n_pre))
        all_traj_pb_p.append(process_traj(n_data, event_idx_map['pb_p'], n_pre=n_pre))
        all_traj_pb_m.append(process_traj(n_data, event_idx_map['pb_m'], n_pre=n_pre))

        all_traj_mock.append(process_traj(n_data, event_idx_map['mock'], n_pre=n_pre))
        all_traj_mock_p.append(process_traj(n_data, event_idx_map['mock_p'], n_pre=n_pre))
        all_traj_mock_m.append(process_traj(n_data, event_idx_map['mock_m'], n_pre=n_pre))

    if full:
        return (all_traj_track, all_traj_pb, all_traj_mock,
                all_traj_track_p, all_traj_track_m,
                all_traj_pb_p, all_traj_pb_m,
                all_traj_mock_p, all_traj_mock_m)

    else:
        return all_traj_track, all_traj_pb, all_traj_mock

### --- Extract average trajectory per neuron

def select_trials(bool_arr, trial_start, trial_end):
    indices = np.where(bool_arr)[0]
    i_start = int(len(indices) * trial_start)
    i_end   = int(len(indices) * trial_end)
    
    #print(f"  n_events={len(indices)}, i_start={i_start}, i_end={i_end}")  # debug
    
    mask = np.zeros_like(bool_arr)
    if i_start < i_end:
        mask[indices[i_start:i_end]] = 1
    return mask

def extract_traj_subset(n_data_s, f_data_s, t_pre, t_post, dt, overlap_thresh=0.2, n_pre=None, full=True, 
                 trial_start=0.0, trial_end=1.0):
    """
    Extract the average trajectory per neuron.
    
    Parameters:
        ...
        trial_start (float): Fraction of trials to start from (0.0 = beginning).
        trial_end (float): Fraction of trials to end at (1.0 = all, 0.25 = first quarter).
    
    Output: session x time x neuron
    """

    all_traj_track = []
    all_traj_track_p, all_traj_track_m = [], []
    all_traj_pb = []
    all_traj_pb_p, all_traj_pb_m = [], []
    all_traj_mock = []
    all_traj_mock_p, all_traj_mock_m = [], []

    for n_data, f_data in tqdm(zip(n_data_s, f_data_s), total=len(n_data_s), desc="extract_traj_subset"):

        direc = f_data['Change_direction'].to_numpy()
        mock_direc = f_data['Mock_direction'].to_numpy()

        triggers = f_data['Frequency_changes'].to_numpy()
        triggers_mock = f_data['Mock_change'].to_numpy()
        condition = f_data['Condition'].to_numpy()

        trigger_map = {
            'track':   select_trials(triggers * (condition == 0), trial_start, trial_end ),
            'track_p': select_trials(triggers * (condition == 0) * (direc == 1), trial_start, trial_end),
            'track_m': select_trials(triggers * (condition == 0) * (direc == -1), trial_start, trial_end),
            'pb':      select_trials(triggers * (condition == 1), trial_start, trial_end),
            'pb_p':    select_trials(triggers * (condition == 1) * (direc == 1), trial_start, trial_end),
            'pb_m':    select_trials(triggers * (condition == 1) * (direc == -1), trial_start, trial_end),
            'mock':    select_trials(triggers_mock * (condition == 1), trial_start, trial_end),
            'mock_p':  select_trials(triggers_mock * (condition == 1) * (mock_direc == 1), trial_start, trial_end),
            'mock_m':  select_trials(triggers_mock * (condition == 1) * (mock_direc == -1), trial_start, trial_end)
        }

        event_idx_map = {
            k: get_event_vicinity(v, t_pre, t_post, dt, overlap_thresh)
            for k, v in trigger_map.items()
        }

        all_traj_track.append(process_traj(n_data, event_idx_map['track'], n_pre=n_pre))
        all_traj_track_p.append(process_traj(n_data, event_idx_map['track_p'], n_pre=n_pre))
        all_traj_track_m.append(process_traj(n_data, event_idx_map['track_m'], n_pre=n_pre))

        all_traj_pb.append(process_traj(n_data, event_idx_map['pb'], n_pre=n_pre))
        all_traj_pb_p.append(process_traj(n_data, event_idx_map['pb_p'], n_pre=n_pre))
        all_traj_pb_m.append(process_traj(n_data, event_idx_map['pb_m'], n_pre=n_pre))

        all_traj_mock.append(process_traj(n_data, event_idx_map['mock'], n_pre=n_pre))
        all_traj_mock_p.append(process_traj(n_data, event_idx_map['mock_p'], n_pre=n_pre))
        all_traj_mock_m.append(process_traj(n_data, event_idx_map['mock_m'], n_pre=n_pre))

    if full:
        return (all_traj_track, all_traj_pb, all_traj_mock,
                all_traj_track_p, all_traj_track_m,
                all_traj_pb_p, all_traj_pb_m,
                all_traj_mock_p, all_traj_mock_m)
    else:
        return all_traj_track, all_traj_pb, all_traj_mock   


### --- Get average traj per frequency ---

def process_traj_per_freq(data, event_idx, event_freq, f_min, f_max):
    """Compute and filter mean trajectories by frequency range."""
    m_traj = compute_mean_trajectories(data.T, event_idx, event_freq)
    return {k: v for k, v in m_traj.items() if f_min <= k <= f_max}



def extract_traj_per_frequency(n_data_s,f_data_s, t_pre, t_post, dt, f_min = 1000, f_max = 4000, overlap_thresh = 0.2, full=True):
    """"
    compute the average trajectory per channels and per frequency. 

    eg: all_traj_track, all_traj_pb, all_traj_mock, all_traj_track_p,all_traj_track_m, all_traj_pb_p,all_traj_pb_m, all_traj_mock_p,all_traj_mock_m = extract_traj_per_frequency(n_data_reorganised,f_data_reorganised, t_pre, t_post, dt, f_min, f_max, full=True)
    
    """
    all_traj_track = []
    all_traj_track_p, all_traj_track_m = [], []
    all_traj_pb = []
    all_traj_pb_p, all_traj_pb_m = [], []
    all_traj_mock = []
    all_traj_mock_p,all_traj_mock_m  = [], []
    
    for n_data, f_data in zip(n_data_s, f_data_s):

        min_len = min(len(f_data), n_data.shape[1])
        f_data = f_data.iloc[:min_len]
        n_data = n_data[:, :min_len]
            
        freq = f_data['Played_frequency'].to_numpy()
        direc = f_data['Change_direction'].to_numpy()
        mock_direc = f_data['Mock_direction'].to_numpy()
        mock = f_data['Mock_frequency'].to_numpy() 
        triggers = f_data['Frequency_changes'].to_numpy()
        triggers_mock = f_data['Mock_change'].to_numpy()
        condition = f_data['Condition'].to_numpy()
    
        trigger_map = {
            'track':  triggers * (condition == 0),
            'track_p': triggers * (condition == 0) * (direc == 1),
            'track_m': triggers * (condition == 0) * (direc == -1),
            'pb': triggers*(condition == 1), 
            'pb_p':    triggers * (condition == 1) * (direc == 1),
            'pb_m':    triggers * (condition == 1) * (direc == -1),
            'mock': triggers_mock * (condition == 1),
            'mock_p': triggers_mock * (condition == 1) * (mock_direc == 1),
            'mock_m': triggers_mock * (condition == 1) * (mock_direc == -1)
        }

        n_pre = int(np.round(t_pre/dt))   
        n_post = int(np.round(t_post/dt)) 
        w_size = n_pre + n_post + 1 

    
        event_idx_map = {k: get_event_vicinity(v, t_pre, t_post, dt, overlap_thresh) for k, v in trigger_map.items()}
        freq_map = {
            'track': freq[event_idx_map['track'][:, n_pre+1]],
            'track_p': freq[event_idx_map['track_p'][:, n_pre+1]],
            'track_m': freq[event_idx_map['track_m'][:, n_pre+1]],
            'pb': freq[event_idx_map['pb'][:, n_pre+1]],
            'pb_p':    freq[event_idx_map['pb_p'][:,n_pre+1]],
            'pb_m':    freq[event_idx_map['pb_m'][:, n_pre+1]],
            'mock':  mock[event_idx_map['mock'][:, n_pre+1]],
            'mock_p': mock[event_idx_map['mock_p'][:, n_pre+1]],
            'mock_m': mock[event_idx_map['mock_m'][:, n_pre+1]]
        }
    
        all_traj_track.append(process_traj_per_freq(n_data, event_idx_map['track'], freq_map['track'], f_min, f_max))
        all_traj_track_p.append(process_traj_per_freq(n_data, event_idx_map['track_p'], freq_map['track_p'],f_min, f_max))
        all_traj_track_m.append(process_traj_per_freq(n_data, event_idx_map['track_m'], freq_map['track_m'], f_min, f_max))
        
        all_traj_pb.append(process_traj_per_freq(n_data, event_idx_map['pb'], freq_map['pb'], f_min, f_max))
        all_traj_pb_p.append(process_traj_per_freq(n_data, event_idx_map['pb_p'], freq_map['pb_p'], f_min, f_max))
        all_traj_pb_m.append(process_traj_per_freq(n_data, event_idx_map['pb_m'], freq_map['pb_m'], f_min, f_max))
        
        all_traj_mock.append(process_traj_per_freq(n_data, event_idx_map['mock'], freq_map['mock'], f_min, f_max))
        all_traj_mock_p.append(process_traj_per_freq(n_data, event_idx_map['mock_p'], freq_map['mock_p'], f_min, f_max))
        all_traj_mock_m.append(process_traj_per_freq(n_data, event_idx_map['mock_m'], freq_map['mock_m'], f_min, f_max))

    if full :
        return all_traj_track, all_traj_pb, all_traj_mock,\
        all_traj_track_p, all_traj_track_m, all_traj_pb_p, all_traj_pb_m, all_traj_mock_p, all_traj_mock_m
        
    else :
        return all_traj_track, all_traj_pb, all_traj_mock
    
def concatenate_dicts(dicts,axis=2):  
    return np.stack(list(dicts.values()))

def pseudo_trajectories(mean_traj_dicts):
    """
    Concatenate mean trajectories across sessions by label along the feature axis.

    Parameters:
        mean_traj_dicts (list of dict): Each dict maps label to [T, D] mean trajectory arrays.

    Returns:
        pd.DataFrame: with columns ['trajectory', 'label'].
                     'trajectory' is a [T, D_total] array for each label.
    """
    grouped_traj = {key:[] for key in mean_traj_dicts[0].keys()}

    # Group all trajectories by label
    for session_dict in mean_traj_dicts:
        for label, traj in session_dict.items():
            grouped_traj[label].append(traj)

    # Concatenate along columns (axis=1)
    data = []
    for label, traj_list in grouped_traj.items():
        #print(label, [traj.shape[1] for traj in traj_list])
        concatenated = np.concatenate(traj_list, axis=1)  # shape: [T, D_total]
        data.append((concatenated, label))
    

    # Build the DataFrame
    df = pd.DataFrame(data, columns=["trajectory", "label"])
    return df



def process_traj_per_speed(data, event_idx, event_freq, speed_min, speed_max):
    """Compute and filter mean trajectories by frequency range."""
    m_traj = compute_mean_trajectories(data.T, event_idx, event_freq)
    return {k: v for k, v in m_traj.items() if speed_min <= k <= speed_max}


def extract_traj_per_speed_old(n_data_s,f_data_s, t_pre, t_post, dt,speed_bins, speed_min = 0, speed_max = 4000,overlap_thresh = 0.2, full=True):
    """"
    compute the average trajectory per channels and per frequency. 

    eg: all_traj_track, all_traj_pb, all_traj_mock, all_traj_track_p,all_traj_track_m, all_traj_pb_p,all_traj_pb_m, all_traj_mock_p,all_traj_mock_m = extract_traj_per_frequency(n_data_reorganised,f_data_reorganised, t_pre, t_post, dt, f_min, f_max, full=True)
    
    """
    all_traj_track = []
    all_traj_track_p, all_traj_track_m = [], []
    all_traj_pb = []
    all_traj_pb_p, all_traj_pb_m = [], []
    all_traj_mock = []
    all_traj_mock_p,all_traj_mock_m  = [], []
    
    for n_data, f_data in zip(n_data_s, f_data_s):
        freq = f_data['Played_frequency'].to_numpy()
        direc = f_data['Change_direction'].to_numpy()
        mock_direc = f_data['Mock_direction'].to_numpy()
        mock = f_data['Mock_frequency'].to_numpy() 
        triggers = f_data['Frequency_changes'].to_numpy()
        triggers_mock = f_data['Mock_change'].to_numpy()
        condition = f_data['Condition'].to_numpy()
        speed = f_data['Speed_x'].to_numpy()
        speed_in_bins = pd.cut(speed,bins=speed_bins,labels=False,include_lowest=True)

    
        trigger_map = {
            'track':  triggers * (condition == 0),
            'track_p': triggers * (condition == 0) * (direc == 1),
            'track_m': triggers * (condition == 0) * (direc == -1),
            'pb': triggers*(condition == 1), 
            'pb_p':    triggers * (condition == 1) * (direc == 1),
            'pb_m':    triggers * (condition == 1) * (direc == -1),
            'mock': triggers_mock * (condition == 1),
            'mock_p': triggers_mock * (condition == 1) * (mock_direc == 1),
            'mock_m': triggers_mock * (condition == 1) * (mock_direc == -1)
        }

        n_pre = int(np.round(t_pre/dt))   
        n_post = int(np.round(t_post/dt)) 
        w_size = n_pre + n_post + 1 

    
        event_idx_map = {k: get_event_vicinity(v, t_pre, t_post, dt, overlap_thresh) for k, v in trigger_map.items()}
        speed_map = {
            'track': speed_in_bins[event_idx_map['track'][:, n_pre]],
            'track_p': speed_in_bins[event_idx_map['track_p'][:, n_pre]],
            'track_m': speed_in_bins[event_idx_map['track_m'][:, n_pre]],
            'pb': speed_in_bins[event_idx_map['pb'][:, n_pre]],
            'pb_p':    speed_in_bins[event_idx_map['pb_p'][:,n_pre]],
            'pb_m':    speed_in_bins[event_idx_map['pb_m'][:, n_pre]],
            'mock':  speed_in_bins[event_idx_map['mock'][:, n_pre]],
            'mock_p': speed_in_bins[event_idx_map['mock_p'][:, n_pre]],
            'mock_m': speed_in_bins[event_idx_map['mock_m'][:, n_pre]]
        }
    
        all_traj_track.append(process_traj_per_speed(n_data, event_idx_map['track'], speed_map['track'], speed_min, speed_max))
        all_traj_track_p.append(process_traj_per_speed(n_data, event_idx_map['track_p'], speed_map['track_p'],speed_min, speed_max))
        all_traj_track_m.append(process_traj_per_speed(n_data, event_idx_map['track_m'], speed_map['track_m'], speed_min, speed_max))
        all_traj_pb.append(process_traj_per_speed(n_data, event_idx_map['pb'], speed_map['pb'], speed_min, speed_max))
        all_traj_pb_p.append(process_traj_per_speed(n_data, event_idx_map['pb_p'], speed_map['pb_p'], speed_min, speed_max))
        all_traj_pb_m.append(process_traj_per_speed(n_data, event_idx_map['pb_m'], speed_map['pb_m'], speed_min, speed_max))
        all_traj_mock.append(process_traj_per_speed(n_data, event_idx_map['mock'], speed_map['mock'], speed_min, speed_max))
        all_traj_mock_p.append(process_traj_per_speed(n_data, event_idx_map['mock_p'], speed_map['mock_p'], speed_min, speed_max))
        all_traj_mock_m.append(process_traj_per_speed(n_data, event_idx_map['mock_m'], speed_map['mock_m'], speed_min, speed_max))

    if full :
        return all_traj_track, all_traj_pb, all_traj_mock,\
        all_traj_track_p, all_traj_track_m, all_traj_pb_p, all_traj_pb_m, all_traj_mock_p, all_traj_mock_m
        
    else :
        return all_traj_track, all_traj_pb, all_traj_mock
    
def concatenate_dicts(dicts,axis=2):  
    return np.stack(list(dicts.values()))



def apply_refractory(trigger_vec, n_refractory):
    """Keep only triggers with no preceding trigger within n_refractory samples."""
    indices = np.where(trigger_vec)[0]
    if len(indices) == 0:
        return trigger_vec.copy()
    keep = np.array([
        not trigger_vec[max(0, idx - n_refractory):idx].any()
        for idx in indices
    ])
    clean = np.zeros_like(trigger_vec)
    clean[indices[keep]] = trigger_vec[indices[keep]]
    
    return clean


## --- Get average traj per speed ---

def process_traj_per_speed(data, event_idx, event_freq, speed_min, speed_max):
    """Compute and filter mean trajectories by frequency range."""
    m_traj = compute_mean_trajectories(data.T, event_idx, event_freq)
    return {k: v for k, v in m_traj.items() if speed_min <= k <= speed_max}

def extract_traj_per_speed(n_data_s, f_data_s, t_pre, t_post, dt, speed_bins,
                            speed_min=0, speed_max=4000,
                            refractory=0.2, full=True):
    """
    Compute the average trajectory per channels and per speed bin.
    - Refractory applied on parent conditions, propagated to +/- sub-conditions.
    - Speed evaluated at trigger time (column n_pre).
    - Missing speed bins for a session are skipped gracefully.
    """

    all_traj_track,   all_traj_track_p,  all_traj_track_m  = [], [], []
    all_traj_pb,      all_traj_pb_p,     all_traj_pb_m     = [], [], []
    all_traj_mock,    all_traj_mock_p,   all_traj_mock_m   = [], [], []

    for n_data, f_data in zip(n_data_s, f_data_s):

        direc         = f_data['Change_direction'].to_numpy()
        mock_direc    = f_data['Mock_direction'].to_numpy()
        triggers      = f_data['Frequency_changes'].to_numpy()
        triggers_mock = f_data['Mock_change'].to_numpy()
        condition     = f_data['Condition'].to_numpy()
        speed         = f_data['Speed_x'].to_numpy()

        speed_in_bins = np.array(pd.cut(speed, bins=speed_bins,
                                        labels=False, include_lowest=True))

        n_pre        = int(np.round(t_pre  / dt))
        n_post       = int(np.round(t_post / dt))
        w_size       = n_pre + n_post + 1
        n_refractory = int(np.round(refractory / dt))

        # ── Réfractaire sur conditions parentes, propagé aux sous-conditions ──
        clean_track = apply_refractory(triggers      * (condition == 0), n_refractory)
        clean_pb    = apply_refractory(triggers      * (condition == 1), n_refractory)
        clean_mock  = apply_refractory(triggers_mock * (condition == 1), n_refractory)

        trigger_map = {
            'track':   clean_track,
            'track_p': clean_track * (direc ==  1),
            'track_m': clean_track * (direc == -1),
            'pb':      clean_pb,
            'pb_p':    clean_pb    * (direc ==  1),
            'pb_m':    clean_pb    * (direc == -1),
            'mock':    clean_mock,
            'mock_p':  clean_mock  * (mock_direc ==  1),
            'mock_m':  clean_mock  * (mock_direc == -1),
        }

        # ── overlap_thresh=0 : get_event_vicinity ne filtre rien ─────────────
        event_idx_map = {
            k: get_event_vicinity(v, t_pre, t_post, dt, overlap_thresh=0)
            for k, v in trigger_map.items()
        }

        # ── Speed au moment du trigger (colonne n_pre) ────────────────────────
        speed_map = {
            k: speed_in_bins[event_idx_map[k][:, n_pre-2]]
            for k in event_idx_map
        }

        all_traj_track.append(  process_traj_per_speed(n_data, event_idx_map['track'],   speed_map['track'],   speed_min, speed_max))
        all_traj_track_p.append(process_traj_per_speed(n_data, event_idx_map['track_p'], speed_map['track_p'], speed_min, speed_max))
        all_traj_track_m.append(process_traj_per_speed(n_data, event_idx_map['track_m'], speed_map['track_m'], speed_min, speed_max))
        all_traj_pb.append(     process_traj_per_speed(n_data, event_idx_map['pb'],      speed_map['pb'],      speed_min, speed_max))
        all_traj_pb_p.append(   process_traj_per_speed(n_data, event_idx_map['pb_p'],    speed_map['pb_p'],    speed_min, speed_max))
        all_traj_pb_m.append(   process_traj_per_speed(n_data, event_idx_map['pb_m'],    speed_map['pb_m'],    speed_min, speed_max))
        all_traj_mock.append(   process_traj_per_speed(n_data, event_idx_map['mock'],    speed_map['mock'],    speed_min, speed_max))
        all_traj_mock_p.append( process_traj_per_speed(n_data, event_idx_map['mock_p'],  speed_map['mock_p'],  speed_min, speed_max))
        all_traj_mock_m.append( process_traj_per_speed(n_data, event_idx_map['mock_m'],  speed_map['mock_m'],  speed_min, speed_max))

    if full:
        return (all_traj_track,   all_traj_pb,   all_traj_mock,
                all_traj_track_p, all_traj_track_m,
                all_traj_pb_p,    all_traj_pb_m,
                all_traj_mock_p,  all_traj_mock_m)
    else:
        return all_traj_track, all_traj_pb, all_traj_mock