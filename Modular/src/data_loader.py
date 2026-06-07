import os
import pickle
import re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from config import DT, UNIQUE_TONES_PATH, CACHE_PATH


def list_used_npy_files(session_configs, include_optional=True):
    """Builds a sorted list of .npy file paths referenced by the current session config."""
    used_paths = set()

    if UNIQUE_TONES_PATH and UNIQUE_TONES_PATH.lower().endswith('.npy'):
        used_paths.add(os.path.normpath(UNIQUE_TONES_PATH))

    for session in session_configs:
        s_type = session.get('type')
        if s_type == 'scanned':
            folder = session.get('path', '')
            if not folder:
                continue

            data_path = os.path.join(folder, f'data_{DT}.npy')
            features_path = os.path.join(folder, f'features_{DT}.npy')
            used_paths.add(os.path.normpath(data_path))
            used_paths.add(os.path.normpath(features_path))

            if include_optional:
                gc_path = os.path.join(folder, 'good_clusters.npy')
                if os.path.exists(gc_path):
                    used_paths.add(os.path.normpath(gc_path))

        elif s_type == 'manual':
            data_path = session.get('data_file', '')
            if data_path:
                used_paths.add(os.path.normpath(data_path))
                folder = os.path.dirname(data_path)
                custom_dt = session.get('dt')
                features_path = os.path.join(folder, f'features_{custom_dt}.npy')
                used_paths.add(os.path.normpath(features_path))

                if include_optional:
                    gc_path = os.path.join(folder, 'good_clusters.npy')
                    if os.path.exists(gc_path):
                        used_paths.add(os.path.normpath(gc_path))

    return sorted(used_paths)


def write_used_npy_files_txt(output_dir, session_configs, filename='used_npy_files.txt'):
    """Writes the list of referenced .npy files for this run into a text file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    npy_files = list_used_npy_files(session_configs)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('Used NPY files\n')
        f.write('=' * 80 + '\n')
        for file_path in npy_files:
            f.write(file_path + '\n')

    return out_path


def _resolve_features_path(folder, dt_val, manual_data_path=None):
    """Resolves feature file path across common naming variants."""
    candidates = [
        os.path.join(folder, f'features_{dt_val}.npy'),
        os.path.join(folder, f'feature_{dt_val}.npy'),
    ]

    if manual_data_path:
        base_name = os.path.basename(manual_data_path)
        if re.search(r'features?', base_name, flags=re.IGNORECASE):
            candidates.append(manual_data_path)
        else:
            candidates.append(os.path.join(folder, re.sub(r'data', 'features', base_name, count=1, flags=re.IGNORECASE)))
            candidates.append(os.path.join(folder, re.sub(r'data', 'feature', base_name, count=1, flags=re.IGNORECASE)))

    for path in candidates:
        if os.path.exists(path):
            return path

    # Keep legacy default behavior if none of the variants exists yet.
    return os.path.join(folder, f'features_{dt_val}.npy')


def _resolve_manual_data_path(data_path, dt_val):
    """If user picks a feature file manually, try to infer the matching data file."""
    if not os.path.exists(data_path):
        return data_path

    base_name = os.path.basename(data_path)
    if not re.search(r'features?', base_name, flags=re.IGNORECASE):
        return data_path

    folder = os.path.dirname(data_path)
    candidates = [os.path.join(folder, f'data_{dt_val}.npy')]

    replaced_features = re.sub(r'features', 'data', base_name, count=1, flags=re.IGNORECASE)
    if replaced_features != base_name:
        candidates.append(os.path.join(folder, replaced_features))

    replaced_feature = re.sub(r'feature', 'data', base_name, count=1, flags=re.IGNORECASE)
    if replaced_feature != base_name:
        candidates.append(os.path.join(folder, replaced_feature))

    for path in candidates:
        if os.path.exists(path):
            print(f"Manual file looked like feature data; using inferred data file: {path}")
            return path

    return data_path

def load_and_preprocess_data(session_configs, gc_or_not=True):
    """Loads neural data and feature data from an explicit list of configuration dictionaries."""
    n_data_s, f_data_s = [] , []

    def _to_feature_df(raw_feature):
        if isinstance(raw_feature, pd.DataFrame):
            return raw_feature.copy()

        if isinstance(raw_feature, dict):
            return pd.DataFrame([raw_feature])

        if isinstance(raw_feature, (list, tuple)):
            if len(raw_feature) == 0:
                return pd.DataFrame()
            if isinstance(raw_feature[0], pd.DataFrame):
                return pd.concat(raw_feature, ignore_index=True)
            if isinstance(raw_feature[0], dict):
                feature_dict = {
                    k: [item.get(k) for item in raw_feature]
                    for k in ['Played_frequency', 'Condition', 'Block', 'Frequency_changes', 'Mock_frequency', 'Mock_change']
                }
                return pd.DataFrame(feature_dict)

        if isinstance(raw_feature, np.ndarray) and raw_feature.dtype == object:
            obj_list = list(raw_feature)
            return _to_feature_df(obj_list)

        raise ValueError(f"Unsupported feature structure: {type(raw_feature)}")

    def _normalize_n_data(raw_data):
        if isinstance(raw_data, np.ndarray):
            arr = raw_data
        else:
            arr = np.asarray(raw_data)

        if isinstance(arr, np.ndarray) and arr.dtype == object:
            raise ValueError("Neural data is object-typed; expected a 2D numeric array per recording.")

        if arr.ndim != 2:
            raise ValueError(f"Neural data must be 2D (neurons x time), got shape {getattr(arr, 'shape', None)}")

        return arr.astype(float)

    def _process_pair(raw_n_data, raw_feature, folder):
        n_data = _normalize_n_data(raw_n_data)
        f_data = _to_feature_df(raw_feature)

        # Use good clusters if applicable
        gc_path = os.path.join(folder, 'good_clusters.npy')
        if gc_or_not and os.path.exists(gc_path):
            gc = np.load(gc_path)
        else:
            gc = np.arange(len(n_data))

        # Guard against malformed cluster files.
        gc = np.asarray(gc, dtype=int)
        gc = gc[(gc >= 0) & (gc < n_data.shape[0])]
        if gc.size == 0:
            gc = np.arange(len(n_data))

        n_data = n_data[gc, :]

        # Sanitize Data
        if 'Condition' in f_data.columns and 'Frequency_changes' in f_data.columns:
            f_data.loc[f_data['Condition'] == -1.0, 'Frequency_changes'] = False

        # Smoothing
        n_data_o = n_data - n_data.mean(axis=1, keepdims=True)
        n_data_smooth = gaussian_filter(n_data_o, sigma=1, axes=1)

        n_data_s.append(n_data_smooth)
        f_data_s.append(f_data)
    
    unique_tones_sorted = np.sort(np.load(UNIQUE_TONES_PATH))
    # pixels_sorted = np.linspace(0, 28, len(unique_tones_sorted)) # Optional, used in notebook

    # --- FIX IS HERE: Change session_paths to session_configs ---
    for session in tqdm(session_configs, desc="Loading Sessions", ascii=" #"):
        try:
            # Check if SCANNED or MANUAL
            if session['type'] == 'scanned':
                folder = session['path']
                data_path = os.path.join(folder, f'data_{DT}.npy')
                features_path = _resolve_features_path(folder, DT)
            elif session['type'] == 'manual':
                data_path = _resolve_manual_data_path(session['data_file'], session['dt'])
                folder = os.path.dirname(data_path)
                custom_dt = session['dt']
                features_path = _resolve_features_path(folder, custom_dt, session['data_file'])
            
            # Load Arrays
            try:
                n_data_raw = np.load(data_path)
            except ValueError as e:
                if 'allow_pickle' in str(e):
                    n_data_raw = np.load(data_path, allow_pickle=True)
                else:
                    raise

            f_data_raw = np.load(features_path, allow_pickle=True)

            # Bundled format: each file contains a list of recordings.
            if isinstance(n_data_raw, list) and isinstance(f_data_raw, list):
                if len(n_data_raw) != len(f_data_raw):
                    raise ValueError(
                        f"Bundled data/features length mismatch: {len(n_data_raw)} vs {len(f_data_raw)}"
                    )
                for raw_n, raw_f in zip(n_data_raw, f_data_raw):
                    _process_pair(raw_n, raw_f, folder)
            elif isinstance(n_data_raw, np.ndarray) and n_data_raw.dtype == object and isinstance(f_data_raw, list):
                n_list = list(n_data_raw)
                if len(n_list) != len(f_data_raw):
                    raise ValueError(
                        f"Bundled object-array data/features length mismatch: {len(n_list)} vs {len(f_data_raw)}"
                    )
                for raw_n, raw_f in zip(n_list, f_data_raw):
                    _process_pair(raw_n, raw_f, folder)
            else:
                _process_pair(n_data_raw, f_data_raw, folder)

        except Exception as e:
            name = session.get('data_file', session.get('path', 'Unknown'))
            print(f"Error loading {name}: {e}")
            
    return n_data_s, f_data_s

def get_cached_or_processed_data(session_configs, gc_or_not=True):
    """Checks for cache; processes the configured sessions if missing."""
    cache_dir = os.path.dirname(CACHE_PATH)
    os.makedirs(cache_dir, exist_ok=True)

    def _is_valid_cache(payload):
        if not isinstance(payload, tuple) or len(payload) != 2:
            return False
        n_data_s, f_data_s = payload
        if not isinstance(n_data_s, list) or not isinstance(f_data_s, list):
            return False
        if len(n_data_s) == 0 or len(f_data_s) == 0:
            return False
        if len(n_data_s) != len(f_data_s):
            return False
        return True

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}...")
        try:
            with open(CACHE_PATH, 'rb') as f:
                payload = pickle.load(f)
        except Exception as e:
            print(f"Cache read failed ({e}). Rebuilding cache from raw data...")
            payload = None

        if payload is not None and _is_valid_cache(payload):
            n_data_s, f_data_s = payload
            print("Cache loaded successfully!")
            return n_data_s, f_data_s

        print("Cache is empty or invalid. Reprocessing raw data...")

    else:
        print("No cache found. Processing raw data...")

    n_data_s, f_data_s = load_and_preprocess_data(session_configs, gc_or_not)

    if not n_data_s or not f_data_s or len(n_data_s) != len(f_data_s):
        print("WARNING: Processed data is empty or inconsistent; cache will not be overwritten.")
        return n_data_s, f_data_s

    print(f"Saving processed data to cache at {CACHE_PATH}...")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((n_data_s, f_data_s), f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data cached successfully!")

    return n_data_s, f_data_s