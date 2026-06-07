# main_pipeline.py
import numpy as np
from preprocess_save import preprocess_and_save
from waveclus_pipeline import run_waveclus
from utils import * 
import os
from create_npy import *


#

sessions = [ 
                   'SKIEUR/SKIEUR_20260331_SESSION_01/headstage_0', 
                   
                   ]

base_data_path = r"\\129.199.81.18\data6\eTheremin\SKIEUR" 
save_base_path = r"C:\Users\PenPen\Desktop\Ferret\Data" + sessions[0]

# Paramètres pour la création des spikes-==================================
fs = 30e3
freq_min = 3

for session in sessions:
    session_path = os.path.join(base_data_path, session)
    exist = False
    
    if exist == False: 
        # Étape 1 : prétraitement et sauvegarde des fichiers .mat
        preprocess_and_save(session_path)
        
        # Étape 2 : clustering Wave_Clus
        run_waveclus(session_path, os.path.join(session_path, 'spike_sorting'))

    else:
        None
    # Étape 3 : création des fichiers spike_times et spike_clusters
    create_spike_data(
    sessions= [session_path],
    base_data_path=base_data_path,
    save_base_path=save_base_path,
    fs=fs,
    freq_min=freq_min
    )

