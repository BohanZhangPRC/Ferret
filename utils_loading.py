import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle

### --- Get sessions name and directories --- ###

def get_sessions(sheet_name, sheet_id, headstage_of_interest = 0, session_filter=None, mounted = 'Volumes'):

    """Retourne les chemins de toutes les sessions valides d'une feuille Google Sheet.
         sheet_name : name of the sheet on the google sheet (name of the animal in capital)    
         sheet_id 
         headstage_of_interest : number if the headstage (0 or 1)
         session_filter: type of session (silent, playback, playback_block, mapping_change)
         mounted : where the NAS is mounted on your device
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)

    # filtrage basique : uniquement celles marquées "yes"
    filtered = df[df['use'] == 'yes']

    # appliquer un filtre supplémentaire si demandé
    if session_filter is not None:
        filtered = filtered[filtered['type'].isin(session_filter)]

    sessions = filtered['session'].tolist()

    root_directory = f'/{mounted}/data6/eTheremin/{sheet_name}/'
    headstages = [headstage_of_interest]

    # construire les chemins
    paths = [f"{root_directory}{s}/headstage_{hs}/" for s in sessions for hs in headstages]
    return paths




def load_sessions(sheet_ids, headstage_of_interest, session_type, mounted):
    """"
    Return directories of all the sessions 
    """
    sessions = []

    for sheet_name, sheet_id in sheet_ids.items():
        sessions.extend(get_sessions(
            sheet_name,
            sheet_id,
            headstage_of_interest = headstage_of_interest,
            session_filter=[str(session_type)], 
            mounted = mounted
        ))
        
    return sessions


### --- Load data from directories ---###

def compare_diff(x):
    """
    Given a 1D array x, return an array y of same length with:
        y[i] =  1 if x[i] > x[i-1]
        y[i] = -1 if x[i] < x[i-1]
        y[i] =  0 if x[i] == x[i-1]
    y[0] is set to 0 by convention.
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=int)
    diff = np.diff(x)
    y[1:] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return y


def get_data_spike_sorted(all_sessions_path, dt, remove_baseline = False) :

    """
    Load the spike sorted data from all_sessions_path
    """
    # gc = True si A1, si PMC alors False 
    # Initialisation
    n_data_s = []
    f_data_s = []
    neuron_ids_s = []
    global_id = 0

    for file in tqdm(all_sessions_path):
        print(file)
        try:
            # Chargement
            n_data = np.load(os.path.join(file, f'spike_sorting/data_{dt}.npy'))
            f_data_raw = np.load(os.path.join(file, f'spike_sorting/features_{dt}.npy'), allow_pickle=True)
            unique_tones = np.load(os.path.join(file, 'unique_tones.npy'))


            # Construction DataFrame
            f_data_dict = {'Played_frequency':[], 'Condition':[], 'Block':[],
                        'Frequency_changes':[], 'Mock_frequency':[], 'Mock_change':[]}
            for item in f_data_raw:
                for key, value in item.items():
                    f_data_dict[key].append(value)
            f_data = pd.DataFrame(f_data_dict)

            # IDs des neurones
            #N_neurons = len(gc)
            N_neurons = len(n_data)

            # Nom de la session
            session_name = file # juste le nom du dossier/fichier

            # IDs uniques : session + neuron number
            neuron_ids = [f"{session_name}_neuron{n}" for n in range(N_neurons)]

            # Ajouter à la liste globale
            neuron_ids_s.append(neuron_ids)

            # Direction
            f_data['Change_direction'] = compare_diff(f_data['Played_frequency'].to_numpy())
            f_data['Mock_direction'] = compare_diff(f_data['Mock_frequency'].to_numpy())

    

            # --- mapping des fréquences vers pixels ---
            unique_tones_sorted = np.sort(unique_tones)
            pixels_sorted = np.linspace(0, 28, len(unique_tones_sorted))  # 28 cm

            # --- choisir Played ou Mock selon la condition ---
            positional_freq = np.where(f_data['Condition'].isin([0, -1]),
                                    f_data['Played_frequency'],
                                    f_data['Mock_frequency'])

            # --- interpolation pour toutes les fréquences inconnues ---
            positions = np.interp(positional_freq, unique_tones_sorted, pixels_sorted)

            # --- lissage ---
            pos_smooth = gaussian_filter(positions, sigma=10)

            # --- calcul de Speed_x ---
            speed_x = np.diff(pos_smooth)
            speed_x = np.append(0, speed_x)
            speed_x = np.abs(speed_x) * 100
            speed_x[~np.isfinite(speed_x)] = 0   # supprime NaN / inf
            f_data['Speed_x'] = speed_x

            # --- calcul de Sound_speed ---
            freq = np.array(f_data['Played_frequency'])
            freq_pos = np.interp(freq, unique_tones_sorted, pixels_sorted)
            freq_pos_smooth = gaussian_filter(freq_pos, sigma=10)
            sound_speed = np.diff(freq_pos_smooth)
            sound_speed = np.append(0, sound_speed)
            sound_speed = np.abs(sound_speed) * 100
            sound_speed[~np.isfinite(sound_speed)] = 0
            f_data['Sound_speed'] = sound_speed
            f_data['Position'] = pos_smooth
            f_data['Freq_position'] = freq_pos_smooth   # les positions correspondantes aux fréquences jouées


            speed_bins = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 6]
            f_data['Speed_bin'] = pd.cut(f_data['Speed_x'],
                                    bins=speed_bins,
                                    labels=False,
                                    include_lowest=True)
            acc_x = np.diff(speed_x, prepend=speed_x[0])
            f_data["Acc_x"] = acc_x
            

            # smooth n_data
            if remove_baseline:
                n_data_o = n_data - n_data.mean(axis=1, keepdims=True)
            else: 
                n_data_o = n_data
            # Sauvegarde
            n_data_s.append(n_data_o)
            f_data_s.append(f_data)

        except Exception as e:
            print(f"Error for file {file}: {e}")
    return n_data_s, f_data_s, neuron_ids_s


def get_data(all_sessions_path, dt, gc_or_not = True) :

    """
    Load the non spike sorted data from all_sessions_path
    """
    # gc = True si A1, si PMC alors False 
    # Initialisation
    n_data_s = []
    f_data_s = []
    neuron_ids_s = []
    global_id = 0

    for file in tqdm(all_sessions_path):
        print(file)
        try:
            # Chargement
            n_data = np.load(os.path.join(file, f'data_{dt}.npy'))
            f_data_raw = np.load(os.path.join(file, f'features_{dt}.npy'), allow_pickle=True)
            if gc_or_not:
                gc = np.load(os.path.join(file, 'good_clusters.npy'))
            else : 
                gc = np.arange(32)
            unique_tones = np.load(os.path.join(file, 'unique_tones.npy'))

            # Neurones valides
            n_data = n_data[gc, :].astype(float)

            # Construction DataFrame
            f_data_dict = {'Played_frequency':[], 'Condition':[], 'Block':[],
                        'Frequency_changes':[], 'Mock_frequency':[], 'Mock_change':[]}
            for item in f_data_raw:
                for key, value in item.items():
                    f_data_dict[key].append(value)
            f_data = pd.DataFrame(f_data_dict)

            # IDs des neurones
            #N_neurons = len(gc)
            N_neurons = len(n_data)
            neuron_ids = list(range(global_id, global_id + N_neurons))
            global_id += N_neurons
            neuron_ids_s.append(neuron_ids)

            # Direction
            f_data['Change_direction'] = compare_diff(f_data['Played_frequency'].to_numpy())
            f_data['Mock_direction'] = compare_diff(f_data['Mock_frequency'].to_numpy())

            # --- mapping des fréquences vers pixels ---
            unique_tones_sorted = np.sort(unique_tones)
            pixels_sorted = np.linspace(0, 28, len(unique_tones_sorted))  # 28 cm

            # --- choisir Played ou Mock selon la condition ---
            positional_freq = np.where(f_data['Condition'].isin([0, -1]),
                                    f_data['Played_frequency'],
                                    f_data['Mock_frequency'])

            # --- interpolation pour toutes les fréquences inconnues ---
            positions = np.interp(positional_freq, unique_tones_sorted, pixels_sorted)

            # --- lissage ---
            pos_smooth = gaussian_filter(positions, sigma=10)

            # --- calcul de Speed_x ---
            speed_x = np.diff(pos_smooth)
            speed_x = np.append(0, speed_x)
            speed_x = np.abs(speed_x) * 100
            speed_x[~np.isfinite(speed_x)] = 0   # supprime NaN / inf
            f_data['Speed_x'] = speed_x

            # --- calcul de Sound_speed ---
            freq = np.array(f_data['Played_frequency'])
            freq_pos = np.interp(freq, unique_tones_sorted, pixels_sorted)
            freq_pos_smooth = gaussian_filter(freq_pos, sigma=10)
            sound_speed = np.diff(freq_pos_smooth)
            sound_speed = np.append(0, sound_speed)
            sound_speed = np.abs(sound_speed) * 100
            sound_speed[~np.isfinite(sound_speed)] = 0
            f_data['Sound_speed'] = sound_speed
            f_data['Position'] = pos_smooth
            f_data['Freq_position'] = freq_pos_smooth   # les positions correspondantes aux fréquences jouées


            speed_bins = [0, 0.01, 1, 2,3,  4, 5, 6, np.inf] 
            f_data['Speed_bin'] = pd.cut(f_data['Speed_x'],
                                    bins=speed_bins,
                                    labels=False,
                                    include_lowest=True)
            acc_x = np.diff(speed_x, prepend=speed_x[0])
            f_data["Acc_x"] = acc_x
            
            # Sauvegarde
            n_data_s.append(n_data)
            f_data_s.append(f_data)

        except Exception as e:
            print(f"Error for file {file}: {e}")
    return n_data_s, f_data_s


### create the dataset ###

def create_data_set(all_sheets, headstage_of_interest, session_type, dt, root_directory_base, spike_sorted = False, remove_baseline = False, gc_or_not = False):

    """
    create 2 .npy file per animal and per headstage : 
     - one with the neural_data binned (data)
      - one with the features binned (features)

      headtsage_of_interest : 0 or 1
      session_type: 'playback', 'silent', 'mapping_change'
    """

    for sheet in all_sheets :
        animal = list(sheet.keys())[0]
        print(animal)

        sessions_a1 = load_sessions(sheet, headstage_of_interest, session_type)
        
        if spike_sorted:
        
            n_data_s, f_data_s, idx = get_data_spike_sorted(sessions_a1, dt,  gc_or_not = False, remove_baseline = False)

            with open(root_directory_base + animal + f"_hs_{headstage_of_interest}_{session_type}_{dt}_data_ss", "wb") as fp:
                pickle.dump(n_data_s, fp)
            
            with open(root_directory_base+ animal + f"_hs_{headstage_of_interest}_{session_type}_{dt}_feature_ss", "wb") as fp:
                pickle.dump(f_data_s, fp)

        else:
            n_data_s, f_data_s = get_data(sessions_a1, dt, gc_or_not)

            with open(root_directory_base + animal + f"_hs_{headstage_of_interest}_{session_type}_{dt}_data", "wb") as fp:
                pickle.dump(n_data_s, fp)
        
            with open(root_directory_base+ animal + f"_hs_{headstage_of_interest}_{session_type}_{dt}_feature", "wb") as fp:
                pickle.dump(f_data_s, fp)



### --- tools to smooth and remove baseline ---

def smooth_data(n_data_s, sigma =1):
    """
    Smooth neural data
    e.g : 
    for n_data_s in n_data_all :
        n_data_s = smooth_data(n_data_s, sigma = 1)
    """
    for i in tqdm(range(len(n_data_s))) :
        n_data_smooth = gaussian_filter(n_data_s[i],sigma=sigma,axes=1)
        n_data_s[i] = n_data_smooth

    return n_data_s

def remove_average(n_data_s):
    """
    Remove baseline from each channel
    e.g : 
    for n_data_s in n_data_all :
        n_data_s = remove_average(n_data_s)
    """
    for i in tqdm(range(len(n_data_s))):
        row_mean = np.mean(n_data_s[i], axis=1, keepdims=True)
        n_data_s[i] = n_data_s[i] - row_mean

    return n_data_s


### --- Orangize dataset in chronological order of the sessions (from naive to expert)---

def re_organise_data(n_data_all, f_data_all):

    n_data_reorganised, f_data_reorganised = [], []

    sessions = [len(elt) for elt in n_data_all]
    n_sessions = np.min(sessions)

    for i in range(n_sessions):#pour chaque numéro de session
        for j in range(len(n_data_all)): # parcourir les animaux et prendre la sessions i
            n_data_reorganised.append(n_data_all[j][i])
            f_data_reorganised.append(f_data_all[j][i])

    return n_data_reorganised, f_data_reorganised