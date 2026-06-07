import numpy as np
import scipy.io
import pandas as pd

# fichier avec fonctions utiles 

def convert_mat_to_npy(mat_file, npy_file):
    # Charger le fichier .mat#create_data_features(path, bin_width, sr)
    mat_data = scipy.io.loadmat(mat_file)
    
    # pour sauvegarder tout le contenu du fichier .mat dans un fichier .npy
    #np.save(npy_file, mat_data)
    
    np.save(npy_file, mat_data['cluster_class'])

#Convertir .mat en .npy


def create_spikes_clusters(save_path, num_channel,nbr_spikes_min):
  #Créer les fichers spike_times et spike_clusters à partir des données spike sortées. 

    spk_clus_f = []
    spk_times_f = []
    # Parcourir chaque canal
    for channel in num_channel:
        try : 
            mat_file = save_path + 'times_C' + str(channel) + '.mat'
            npy_file = save_path + 'times_C' + str(channel) + '.npy'
            convert_mat_to_npy(mat_file, npy_file)  # Convertit le fichier .mat en .npy
            # Charger le fichier .npy
            ss = np.load(npy_file, allow_pickle=True)
            valeurs, occurences = np.unique(ss[:, 0], return_counts=True)

            print("nbr_spikes_min", nbr_spikes_min)
            masque = np.isin(ss[:, 0], valeurs[occurences >= nbr_spikes_min])

            ss_filtre = ss[masque]


            # Diviser le fichier en temps de spikes et clusters associés
            spk_clus = ss_filtre[:, 0]
        
            spk_clus = [[channel,x] for x in spk_clus]  # Ajoute le décalage du canal
            #spk_clus = [int(elt) for elt in spk_clus]
            spk_times = ss_filtre[:, 1]
            
            # Ajouter les valeurs au tableau final
            spk_clus_f.extend(spk_clus)
            spk_times_f.extend(spk_times)
        except:
            pass

    # Combiner spk_times_f et spk_clus_f dans une liste de tuples
    combined = list(zip(spk_times_f, spk_clus_f))

    # Trier en fonction de spk_times_f (le premier élément de chaque tuple)
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Séparer les listes triées
    spk_times_f_sorted, spk_clus_f_sorted = zip(*combined_sorted)

    # Convertir en listes (si nécessaire)
    spk_times_f_sorted = list(spk_times_f_sorted)
    spk_clus_f_sorted = list(spk_clus_f_sorted)

    # Sauvegarder les résultats triés
    np.save(save_path + '/ss_spike_clusters.npy', spk_clus_f_sorted)
    np.save(save_path + '/ss_spike_times.npy', spk_times_f_sorted)


def get_sessions(sheet_name, sheet_id, session_filter=None):
    """Retourne les chemins de toutes les sessions valides d'une feuille Google Sheet."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)

    # filtrage basique : uniquement celles marquées "yes"
    filtered = df[df['use'] == 'yes']

    # appliquer un filtre supplémentaire si demandé
    if session_filter is not None:
        filtered = filtered[filtered['type'].isin(session_filter)]

    sessions = filtered['session'].tolist()

    # règles pour headstages
    if sheet_name == "HERCULE":
        root_directory = f'/auto/data6/eTheremin/{sheet_name}/'
        headstages = [0]
    if sheet_name == "FETA":
        root_directory = f'/auto/data6/eTheremin/{sheet_name}/'
        headstages = [1] # 1 c'est le PMC et 0 c'est A1
    elif sheet_name == "ALTAI":
        root_directory = f'/auto/data2/eTheremin/{sheet_name}/'
        headstages = [0, 1]
    else:
        root_directory = f'/auto/data6/eTheremin/{sheet_name}/'
        headstages = [0, 1]

    # construire les chemins
    paths = [f"{root_directory}{s}/headstage_{hs}/" for s in sessions for hs in headstages]
    return paths
