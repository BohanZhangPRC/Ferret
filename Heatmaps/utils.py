import numpy as np
from scipy.signal import find_peaks
import os
import json

from scipy.signal import butter, filtfilt

def match_triggers(features):
    """"
    Fonction pour faire matcher les triggers entre les blocks de tracking et de playback
    input : features
    ouptut : un dictionnaire qui, pour chaque block, renvoie les indices de triggers en tracking et les indices de triggers en playabck correspondant
    """
    matching_triggers = {} 

    blocks = np.array([elt['Block'] for elt in features])
    conditions = np.array([elt['Condition'] for elt in features])
    played_tones = np.array([elt['Played_frequency'] for elt in features])
    frequency_changes = np.array([elt['Frequency_changes'] for elt in features])

    for block in np.sort(np.unique(blocks)):

        indices_tracking_fc = np.where((frequency_changes == True) & (conditions == 0) & (blocks == block))[0] # récupérer les indices où on a un changement de fréquence dans le block de tracking
        indices_playback_fc = np.where((frequency_changes == True) & (conditions == 1) & (blocks == block))[0] # récupérer les indices où on a un changement de fréquence dans le block de playback
        # ici j'ai un ton de playabck en plus que dans le trackinhc chelou
        if len(indices_playback_fc)>len(indices_tracking_fc):
            # je retire le premier ton de playback
            indices_playback_fc = indices_playback_fc[1:]
        if len(indices_playback_fc)<len(indices_tracking_fc):
            #je retirer le premier ton de tracking
            indices_tracking_fc = indices_tracking_fc[1:]

        # pour etre sur que ca colle
        if features[indices_tracking_fc[0]]['Played_frequency'] != features[indices_playback_fc[0]]['Played_frequency']:
            indices_playback_fc = indices_playback_fc[1:]
            indices_tracking_fc = indices_tracking_fc[:-1]

        matching_triggers[int(block)] = {
            'idx_tracking': indices_tracking_fc.tolist(),
            'idx_playback': indices_playback_fc.tolist()
        }
  
    return matching_triggers


def est_premier(nombre):
    if nombre <= 1:
        return False
    elif nombre <= 3:
        return True
    elif nombre % 2 == 0 or nombre % 3 == 0:
        return False
    i = 5
    while i * i <= nombre:
        if nombre % i == 0 or nombre % (i + 2) == 0:
            return False
        i += 6
    return True

def get_plot_coords(channel_number):
    """
    Fonction qui calcule la position en 2D d'un canal sur une Microprobe.
    Retourne la ligne et la colonne.
    """
    if channel_number in list(range(8)):
        row = 3
        col = channel_number % 8

    elif channel_number in list(range(8, 16)):
        row = 1
        col = 7 - channel_number % 8

    elif channel_number in list(range(16, 24)):
        row = 0
        col = 7 - channel_number % 8

    else:
        row = 2
        col = channel_number % 8

    return row, col



def get_plot_geometry(good_clusters):
    n_clus = len(good_clusters)
    if est_premier(n_clus):
        n_clus=n_clus-1

    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
        
        #print(num_columns)
    num_rows = -(-n_clus // num_columns)
    return num_rows, num_columns

def get_better_plot_geometry(good_clusters):
    # Calculate number of rows and columns for subplots
    num_plots = len(good_clusters)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    return num_plots, num_rows, num_cols






def get_all_psth(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """

    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0  :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth

def get_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth





def get_mock_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Mock_change']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth


def get_psth_spaced(data, features, t_spaced, t_pre, t_post, bin_width, good_clusters):
    """
    Pour voir, pour chaque neurone, les psth, espacés d'au moins 200 ms (0.2s) : t_spaced en s
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    
    
    psth=[] 
    for cluster in good_clusters:
        x_speed, x_freq, x_condition, x_position = [] ,[] ,[] ,[] 
        psth_clus = []
        t_last_psth = 0
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features) and bin-t_last_psth >t_spaced/bin_width :
                if features[bin]['Frequency_changes']>0 :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    t_last_psth = bin
                    x_freq.append(features[bin]['Played_frequency'])
                    x_condition.append(features[bin]['Condition'])
                    x_position.append(features[bin]['Position'])
                    x_speed.append(features[bin]['Speed'])
        psth.append(psth_clus)
        #x_speed.append(features[bin]['Speed'])
    return psth, x_speed, x_freq, x_condition, x_position


 





def get_psth_and_speed(data_speed, features, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    

    psth=[] 
    for cluster in good_clusters:
        speed  = []
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data_speed[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    speed.append(features[bin]['Speed'])
        psth.append(psth_clus)
    return psth, speed





def get_psth_at_bf(data, features, t_pre, t_post, bin_width, good_clusters,bf, condition):
    """
    Pour voir, pour chaque neurone, les psth uniquement pris à la best freqeuncy (bf) du neurone
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for i, cluster in enumerate(good_clusters):
        best_freq = bf[i]
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c and features[bin]['Played_frequency']==best_freq:
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth


def get_psth_in_block(data, features, t_pre, t_post, bin_width, good_clusters, block, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Block']==block:
                    if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth


def get_psth_in_index(data, features, t_pre, t_post, bin_width, good_clusters, indexes):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clusters, indexes (un tableau qui contient les indices auxquels chercher les psth)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in indexes:
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth


def get_played_frequency(features, t_pre, t_post, bin_width, condition):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    if condition=="tracking":
        c = 0
    elif condition=="playback":
        c=1
    elif condition=="tail":
        c = -1
    elif condition == "mappingchange":
        c = 2
    frequency = []
    for bin in range(len(features)):
        if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                frequency.append(features[bin]['Played_frequency'])
    return frequency
        


def get_mock_frequency(features):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    c=1
    frequency = []
    for bin in range(len(features)):
        if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
            frequency.append(features[bin]['Mock_frequency'])
    return frequency
        


def moving_average(data, window_size):
    # Calculer la moyenne mobile avec remplissage
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def mean_psth(group):
    # Empiler toutes les listes de psth en une matrice de 100 colonnes
    psth_matrix = np.array(group['psth'].tolist())
    # Calculer la moyenne sur les lignes (axis=0), pour chaque position de la liste
    return np.mean(psth_matrix, axis=0)


def replace_zeros(features):
    found_first_zero = False

    for i in range(len(features)):
        if features[i]['Condition'] == 0 and not found_first_zero:
            found_first_zero = True  # Début du remplacement
        elif features[i]['Condition'] == -1 and found_first_zero:
            break  # Stop dès qu'on rencontre -1 après un 0
        elif features[i]['Condition'] == 0 and found_first_zero:
            features[i]['Condition'] = -1  # Remplacement des 0 par -1

    return features




def get_sustained_activity(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    return (np.nanmean(psth[0: int(t_pre/bin_width)-3]))




def get_sustained_activity_nan(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    --> dans la cas où on aurait des nan gênants
    
    
    """
    if psth is not np.nan and psth is not None:
    
        return (np.nanmean(psth[0: int(t_pre/bin_width)-5]))
    else:
        return np.nan




def mean_maxima(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau dont les indices sont compris
    entre t0 et t1
    """
    # Find peaks in the array
    pics, _ = find_peaks(arr[t0:t1], distance=thresh)

    # Check if there are at least two peaks
    if len(pics) >= 2:
        # Get the indices of the two maximum values
        max_indices = np.argsort(arr[pics])[-2:]
        # Calculate the mean of the two maximum values
        #mean = np.mean(arr[pics][max_indices])
        mean = np.max(arr[pics][max_indices])

        # Get the actual maximum values
        max_values = arr[pics][max_indices]
    else:
        mean = np.nan
        max_values = np.nan

    return mean, pics, max_values


def mean_maxima_nan(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau cont les indices sont compris
    entre t0 et t1
    
    --> cas où on aurait des nan gênants
    """
    # Find peaks in the array
    if arr is not np.nan:
        pics, _ = find_peaks(arr, distance=thresh)

        # Check if there are at least two peaks
        if len(pics) >= 2:
            # Get the indices of the two maximum values
            max_indices = np.argsort(arr[pics])[-2:]

            # Calculate the mean of the two maximum values
            #mean = np.mean(arr[pics][max_indices])
            mean = np.max(arr[pics][max_indices])
            # Get the actual maximum values
            max_values = arr[pics][max_indices]
        else:
            mean = np.nan
            max_values = np.nan
    else:
        mean = np.nan
        max_values = np.nan
        pics=np.nan
        

    return mean, pics, max_values

def get_total_evoked_response_unique(psth, t_pre, t_post, bin_width, thresh, t0, t1):
    """
    Calcule la total evoked response pour un seul vecteur PSTH.
    
    Inputs :
    - psth : vecteur 1D (PSTH d'un neurone ou d'une session)
    - t_pre, t_post : temps avant/après l'événement (non utilisés ici si déjà inclus dans t0/t1)
    - bin_width : largeur des bins temporels (optionnel selon ta fonction mean_maxima)
    - thresh : seuil pour mean_maxima
    - t0, t1 : intervalle de temps sur lequel calculer la réponse
    
    Output :
    - total_evoked_response : scalaire (réponse totale)
    """
    total_evoked_response = mean_maxima(psth, thresh, t0, t1)[0]
    return total_evoked_response


def get_total_evoked_response(psth, t_pre, t_post, bin_width, thresh, t0, t1):
    """"
    Function qui renvoie la total evoked reponse pour un tableau contenant des psth
    input : un tableau psth contenant des psth
    output : un tableau contenant la total evoked response pour chaque psth
    
    """
    total_evoked_response = []
    for elt in psth:
        total_evoked_response.append(mean_maxima(elt, thresh, t0,t1)[0])
        #total_evoked_response.append(np.max(elt))
    return total_evoked_response


def get_indexes(tableau, a):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est égale à a

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i] == a:
            indices_a.append(i)

    return indices_a

def get_indexes_in(tableau, a, b):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est comprise entre a et b

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i]>=a and tableau[i]<=b:
            indices_a.append(i)

    return indices_a


def get_psth_for_indexes(data, features, indexes, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), indexes (les indices des bin qui nous intéressent)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in indexes:
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth



def get_mean_psth_in_bandwidth(data, features, bandwidth, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, renvoie la moyenne des psth pour toutes les fréquences comprises dans la badnwidth du cluster
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), bandwidth
    output : 
     - une liste contenant le psth moyen par cluster [cluster x [t_pre, t_post] ] in la bandwidth
      et une autre out la bandwidth
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    else : 
        c=1
        
    
    in_psth, out_psth=[] , []
    for idx, cluster in enumerate(good_clusters):
        psth_clus, out_clus = [], []
        low_f, high_f = bandwidth[idx][0],  bandwidth[idx][1]
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c:
                    if low_f<=features[bin]['Played_frequency']<=high_f:
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    else:
                        out_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        if len(psth_clus)==0:
            psth_clus = [[np.nan]*(len(psth_bins)-1)]*2
        if len(out_clus)==0:
            out_clus = [[np.nan]*(len(psth_bins)-1)]*2
        in_psth.append(np.nanmean(psth_clus, axis=0))
        out_psth.append(np.nanmean(out_clus, axis=0))
       
    return in_psth, out_psth    


def get_sem(neurones):
    """""
    Fonction qui renvoie la sem pour un tableau de format (neurones x bin)
    
    input : un tableau [neurones, bins]
    output: liste [bins] contenant la SEM
    """
    sem = []
    for bin in range(len(neurones[0])):
        sem.append(np.nanstd(np.array(neurones)[:,bin])/np.sqrt(len(neurones)))
    return sem  










def get_sustained_activity_OLD(psth, t_pre, t_post, bin_width):
    """""
    PAS UTILE POUR L'INSTANT !!!
    Fonction qui renvoie l'activité moyenne d'un tableau de PSTH
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    sustained = []
    for elt in psth:
        sustained.append(np.nanmean(elt[0: int(t_pre/bin_width)-2]))
    return sustained 


def indices_valeurs_egales(tableau, valeur_cible):
    """
    

    Args:
        tableau (_type_): un tableau
        valeur_cible (_type_): la valeur qu'on recherche dans le tableau

    Returns:
        indices: les indices des éléments dans le tableau dont la valeur est égale à la valeur cible
    """
    indices = []
    for i in range(len(tableau)):
        if tableau[i] == valeur_cible:
            indices.append(i)
    return indices


def indices_valeurs_comprises(tableau, valeur_min, valeur_max):
    
    """"
       Args:
        tableau (_type_): un tableau
        valeur_min, valeur_max (_type_): valeurs qui définissent l'intervalle dans lequel on cherche des valeurs dans le tableau

        Returns:
            indices: les indices des éléments dans le tableau dont la valeur est comprise dans l'intervalle.
    """
    indices = []
    for i in range(len(tableau)):
        if valeur_min<=tableau[i]<valeur_max:
            indices.append(i)
    return indices




def get_mean_psth_in_bandwidth(data, features, bandwidth, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, renvoie la moyenne des psth pour toutes les fréquences comprises dans la badnwidth du cluster
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), bandwidth
    output : 
     - une liste contenant le psth moyen par cluster [cluster x [t_pre, t_post] ] in la bandwidth
      et une autre out la bandwidth
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
        
    
    in_psth, out_psth=[] , []
    for idx, cluster in enumerate(good_clusters):
        psth_clus, out_clus = [], []
        low_f, high_f = bandwidth[idx][0],  bandwidth[idx][1]
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c:
                    if low_f<=features[bin]['Played_frequency']<=high_f:
                        psth_clus.append(data[idx][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    else:
                        out_clus.append(data[idx][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        print(len(psth_clus))
        print(len(out_clus))
        if len(psth_clus)<1500:
            psth_clus = [[np.nan]*(len(psth_bins)-1)]*2
        if len(out_clus)<1500:
            out_clus = [[np.nan]*(len(psth_bins)-1)]*2
        in_psth.append(np.nanmean(psth_clus, axis=0))
        out_psth.append(np.nanmean(out_clus, axis=0))
       
    return in_psth, out_psth    
 
    

def get_session_type_final(path):
    """
    Fonction qui renvoie le type de la session parmi TrackingOnly, PlaybackOnly etc
    elle va chercher dans le fichier json le type de session
    """
    # List all files in the folder
    files = os.listdir(path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]
    # Check if only one JSON file is found
    if len(json_files) == 1:
        json_file_path = os.path.join(path, json_files[0])
        print("Found JSON file:", json_file_path)
        # Load the JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        # Extract the "Type" field
        try : 
            type_value = data['Block_000']['Type']

            if type_value=="Pause":
                type_value = data['Block_001']['Type']
                
            print("Type:", type_value)
        except :
            type_value = [data[key]["Type"] for key in data if key.startswith("Experiment_")][1]
            print("Type:", type_value)       
            
            
    else:
        print("Error: No JSON files found.")
    return type_value



def get_mean_neurone_spaced_frequency(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par neurone
    Attention ici je ne prends que les changements de fréquence qui sont 
    séparés de plus de 200ms (pour vérifier que les oscillations sont bien
    dûes aux changements de fréquence précédents le stim d'intéret)
    --> si tu veux l'utiliser : change l'appel à la fonction dans get_mean_psth
    input: fichier data.npy d'une session, features.npy, t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        previousbin=0
        for bin in range(len(features)):
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0 and bin-previousbin>0.2/bin_width:
                    mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    previousbin=bin
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and bin-previousbin>0.2/bin_width:
                    mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    previousbin=bin
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback



import statistics
def get_psth_speed(data, features, t_pre, t_post, bin_width, good_clusters, condition, threshold):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
       - threshold : seuil pour les vitesses (0 si on fait mouvement/pas mouvement, sinon médian par exemple)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    speed = [x['Speed'] for x in features]
    if threshold== 0 : 
        thresh = 0
    elif threshold == 'median':
        thresh = np.median(np.abs(speed))

    psth_still, psth_moving=[], []
    for cluster in good_clusters:
        psth_clus_still, psth_clus_moving = [], []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c and np.abs(features[bin]['Speed'])<= thresh  :
                    psth_clus_still.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])

                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c and np.abs(features[bin]['Speed'])> thresh  :
                    psth_clus_moving.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth_still.append(psth_clus_still)
        psth_moving.append(psth_clus_moving)
    return psth_still, psth_moving


def moving_average(data, window_size):
    # Calculer la moyenne mobile avec remplissage
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def get_speed(path, bin_width, window_size):
    """"
    Fonction pour créer un features dans lequl il y a également la vitesse par bin
    window_size = c'est le nombre de bin sur lequel on fait la moyenne de vitesse 
    """
    features = np.load(path+'/features_0.005.npy', allow_pickle=True)
    f_tr = [elt['Played_frequency'] for elt in features if elt['Condition']==0]
    f_pb = [elt['Mock_frequency'] for elt in features if elt['Condition'] == 1]
    # attention ca ne marche que pour le tracking ! en playback il faut prendre les mock_frequency!
    f_total = np.append(f_tr, f_pb)
    speed_x = np.diff(moving_average(f_total, window_size))

    speed_x = np.append(0, speed_x[0:])  # a affiner ici !!

    for i, feature in enumerate(features):
        feature['Speed'] = speed_x[i]
    np.save(path+f'/features_speed_{bin_width}.npy', features)


def get_psth_tracking_speed(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Pour voir, pour chaque neurone, les psth en fonction de la vitesse en tracking : 
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    c = 0
    speed = [x['Speed'] for x in features]
    median = np.nanmedian(speed)
    psth_slow, psth_fast=[], []
    for cluster in good_clusters:
        psth_clus_slow, psth_clus_fast = [], []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c and np.abs(features[bin]['Speed'])<= median  :
                    psth_clus_slow.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])

                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c and np.abs(features[bin]['Speed'])>median  :
                    psth_clus_fast.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth_slow.append(psth_clus_slow)
        psth_fast.append(psth_clus_fast)
    return psth_slow, psth_fast



def get_total_evoked_response_individual_old(psth, t_pre, t_post, bin_width, thresh, t0, t1):
    """
    Fonction qui renvoie la total evoked response pour un psth individuel
    Prend la valeur du psth à t_pre/bin_width et les deux bins voisins autour.
    
    input : un tableau psth contenant des psth
    output : un tableau contenant la total evoked response pour chaque psth
    """
    if psth is not np.nan:
        # Calculer l'indice correspondant à t_pre/bin_width
        index_t_pre = int(t_pre / bin_width)
        
        # Vérifier que l'index est dans les limites de l'array
        if index_t_pre - 2 >= 0 and index_t_pre + 2 < len(psth):
            # Extraire les 5 valeurs autour de index_t_pre (2 bins avant et 2 bins après)
            values_around_t_pre = psth[index_t_pre - 2 : index_t_pre + 3]
            
            # Vous pouvez ici soit renvoyer la moyenne de ces valeurs ou les valeurs elles-mêmes
            return np.mean(values_around_t_pre)
        else:
            return np.nan
    else: 
        return np.nan







def get_total_evoked_response_individual(psth, t_pre, t_post, bin_width, thresh, t0, t1):
    """"
    Function qui renvoie la total evoked reponse pour un psth individuel
    input : un tableau psth contenant des psth
    output : un tableau contenant la total evoked response pour chaque psth
    
    """
    if psth is not np.nan:
        return mean_maxima(psth, thresh, t0,t1)[0]
    else : 
        return np.nan
    
def average_psth(psth_list):
    return np.mean(np.vstack(psth_list), axis=0)


def convert_freq_to_cm(freq):
    """"
    Fonction qui prend un tableau de fréquences et renvoie un tableau contenant les positions associées à ces fréquences
    
    """

    unique_tones = [    0.,   190.,   220.,   255.,   296.,   343.,   397.,   460.,
         533.,   617.,   715.,   828.,   959.,  1111.,  1287.,  1490.,
        1727.,  2000.,  2317.,  2684.,  3109.,  3601.,  4172.,  4832.,
        5598.,  6484.,  7511.,  8701., 10079., 11676., 13525., 15667.,
       18149., 21024.]
    
    pixels = np.linspace(0, 28, len(unique_tones))  
    freq_to_pixel = {tone: pixel for tone, pixel in zip(unique_tones, pixels)} # tableau de correspondance des fréquences en pixels
    f_to_pixels = np.array([freq_to_pixel.get(f, np.nan) for f in freq])

    return f_to_pixels



def find_movement(all_pos, integWin = 15,pauseBeforeMovement = 100,  threshold = 2, sweepRefractoryPeriod = 300,preSweepPeriod = 150 ):

    # Calcul des différences
    datadiff = np.diff(all_pos)
    changeIdx = np.where(datadiff != 0)[0]

    # Intégration sur les changements précédents
    integPos = np.zeros(len(all_pos))
    for changeNum in range(integWin, len(changeIdx)):
        idx = changeIdx[changeNum]
        window = changeIdx[changeNum - integWin : changeNum + 1]
        integPos[idx] = np.sum(datadiff[window - 1])  # -1 car datadiff est plus court que data

    # Détection des pics dépassant le seuil
    threshIdx = []
    for signC in [-1, 1]:
        integPos_T = signC * integPos
        threshIdx_T = np.where(integPos_T > threshold)[0].astype(float)  # permet les NaN

        # Vérification de pic local
        for i in range(len(threshIdx_T)):
            idx = int(threshIdx_T[i])
            start = int(max(0, idx - sweepRefractoryPeriod))
            end = int(min(len(integPos_T), idx + sweepRefractoryPeriod + 1))
            if np.any(integPos_T[start:end] > integPos_T[idx]):
                threshIdx_T[i] = np.nan

        threshIdx.extend(threshIdx_T)

    # Nettoyage des indices
    threshIdx = np.array(threshIdx)
    threshIdx = threshIdx[~np.isnan(threshIdx)].astype(int)
    threshIdx = np.sort(threshIdx)

    # Détection des débuts de mouvement
    onsetIdx = []
    for threshNum in range(1, len(threshIdx)):
        peakIdx = threshIdx[threshNum]
        cursorBackInTime = peakIdx
        while cursorBackInTime > threshIdx[threshNum - 1]:
            start = max(0, cursorBackInTime - preSweepPeriod)
            if np.all(all_pos[start:cursorBackInTime + 1] == all_pos[cursorBackInTime]):
                onsetIdx.append((cursorBackInTime, peakIdx))
                break
            cursorBackInTime -= 1
    return integPos, onsetIdx

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