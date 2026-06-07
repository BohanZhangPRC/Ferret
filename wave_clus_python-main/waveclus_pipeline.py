# waveclus_pipeline.py
import os
import time
import subprocess
import numpy as np

# run_waveclus.py
import os
import time
import subprocess
import numpy as np

def run_waveclus(
    session_path,
    raw_data_path,
    nbr_channel=32,
    interval_verification=0.5,
    timeout_creation=300
):
    """
    Run Wave_Clus spike detection and clustering for all good channels.

    Parameters
    ----------
    session_path : str
        Path to the session folder (headstage_0).
    raw_data_path : str
        Path to the folder containing C{k}.mat files.
    nbr_channel : int, optional
        Number of channels, by default 32.
    interval_verification : float, optional
        Time to wait between checking for output files, by default 0.5.
    timeout_creation : int, optional
        Maximum time to wait for the clustering files, by default 300.
    """

    session_name = os.path.basename(session_path)
    waveclus_path = '/home/felicie/Downloads/wave_clus-master'  # Chemin vers Wave_Clus

    # Load good channels if available
    good_clusters_file = os.path.join(session_path, 'good_clusters.npy')
    if os.path.exists(good_clusters_file):
        good_channels = np.load(good_clusters_file, allow_pickle=True)
    else:
        good_channels = np.arange(nbr_channel)

    print(f"Good channels for {session_name}: {good_channels}")

    for channel in good_channels:
        try : 
            fichier_mat = os.path.join(raw_data_path, f'C{channel}.mat')
            fichier_spikes = os.path.join(raw_data_path, f'C{channel}_spikes.mat')
            fichier_times = os.path.join(raw_data_path, f'times_C{channel}.mat')

            # Commandes MATLAB Wave_Clus avec tous les sous-dossiers
            #get_spikes_cmd = (
            #    f"matlab -nodesktop -nosplash -batch "
            #    f"\"addpath(genpath('{waveclus_path}')); cd('{raw_data_path}'); "
            #    f"clear all; close all; Get_spikes('{fichier_mat}'); exit;\""
            #)

            #do_clustering_cmd = (
            #   f"matlab -nodesktop -nosplash -batch "
            #  f"\"addpath(genpath('{waveclus_path}')); cd('{raw_data_path}'); Do_clustering('{fichier_spikes}');\""
            #)

            get_spikes_cmd = (
                f"matlab -nodisplay -nosplash -batch "
                f"\"addpath(genpath('{waveclus_path}')); "
                f"cd('{raw_data_path}'); "
                f"Get_spikes('C{channel}.mat');\""
            )

            do_clustering_cmd = (
                f"matlab -nodisplay -nosplash -batch "
                f"\"addpath(genpath('{waveclus_path}')); "
                f"cd('{raw_data_path}'); "
                f"Do_clustering('C{channel}_spikes.mat');\""
            )


            # Étape 1 : Get_spikes
            try:
                print(f"Executing Get_spikes for channel {channel}...")
                with open('log_get_spikes.txt', 'a') as log_file:
                    subprocess.run(get_spikes_cmd, shell=True, check=True, stdout=log_file, stderr=log_file)
            except subprocess.CalledProcessError as e:
                print(f"Error during Get_spikes for channel {channel}: {e}")
                print(f"See log_get_spikes.txt for details.")
                exit(1)

            # Attente du fichier spikes
            time_start = time.time()
            while not os.path.exists(fichier_spikes):
                if time.time() - time_start > timeout_creation:
                    print(f"Timeout: {fichier_spikes} not created.")
                    exit(1)
                print(f"Waiting for {fichier_spikes}...")
                time.sleep(interval_verification)

            # Étape 2 : Do_clustering
            try:
                print(f"Executing Do_clustering for channel {channel}...")
                with open('log_do_clustering.txt', 'a') as log_file:
                    subprocess.run(do_clustering_cmd, shell=True, check=True, stdout=log_file, stderr=log_file)
            except subprocess.CalledProcessError as e:
                print(f"Error during Do_clustering for channel {channel}: {e}")
                print(f"See log_do_clustering.txt for details.")
                exit(1)

            # Attente du fichier times
            time_start = time.time()
            while not os.path.exists(fichier_times):
                if time.time() - time_start > timeout_creation:
                    print(f"Timeout: {fichier_times} not created.")
                    exit(1)
                time.sleep(interval_verification)

            print(f"Channel {channel} processing completed.")
        except:
            pass



