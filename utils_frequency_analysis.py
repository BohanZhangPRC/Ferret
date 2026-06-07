import numpy as np


"""
Récupérer toutes les réponses de tous les neurones
"""

def get_psth_triggers_all_neurons(
    n_data_s,
    f_data_s,
    t_pre,
    t_post,
    dt, 
    condition
):

    pre_bins = int(t_pre / dt)
    post_bins = int(t_post / dt)
    n_bins = pre_bins + post_bins

    if condition == 'tracking':
        c = 0
    elif condition =='playback':
        c = 1
    else:
        print('error')
    # --- Récupération des fréquences globales ---
    all_freqs = set()
    for f_data in f_data_s:
        mask = (
            (f_data['Frequency_changes'] == 1) &
            (f_data['Condition'] == 0) &
            (f_data['Played_frequency'] >= 500) &
            (f_data['Played_frequency'] <= 15000)
        )
        all_freqs.update(np.unique(f_data['Played_frequency'][mask]))

    freqs = np.sort(np.array(list(all_freqs)))
    freq_to_idx = {f: i for i, f in enumerate(freqs)}

    psth_list = []

    # --- Boucle sessions / neurones ---
    for session_idx in range(len(n_data_s)):
        n_data = n_data_s[session_idx]
        f_data = f_data_s[session_idx]

        trigger_mask = (
            (f_data['Frequency_changes'] == 1) &
            (f_data['Condition'] == c) &
            (f_data['Played_frequency'] >= 500) &
            (f_data['Played_frequency'] <= 15000)
        )

        trigger_indices = np.where(trigger_mask)[0]
        trigger_frequencies = f_data['Played_frequency'][trigger_indices]

        for neurons in n_data:
            psth_neuron = np.full((len(freqs), n_bins), np.nan)

            for trig_idx, freq in zip(trigger_indices, trigger_frequencies):
                if trig_idx - pre_bins < 0 or trig_idx + post_bins > len(neurons):
                    continue

                f_idx = freq_to_idx[freq]
                psth = neurons[trig_idx - pre_bins : trig_idx + post_bins]

                if np.isnan(psth_neuron[f_idx]).all():
                    psth_neuron[f_idx] = psth
                else:
                    psth_neuron[f_idx] += psth

            # moyenne sur les répétitions
            for f_idx in range(len(freqs)):
                valid_trigs = np.sum(~np.isnan(psth_neuron[f_idx]))
                if valid_trigs > 0:
                    psth_neuron[f_idx] /= valid_trigs

            psth_list.append(psth_neuron)

    psth_all = np.stack(psth_list, axis=0)

    return psth_all, freqs


#récupérer les fréquences qui activent les neurones
def get_selectivity(psth_all, freqs, t_pre, t_post, dt, t_start, t_end):

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    time = np.arange(-n_bins_pre, n_bins_post) * dt
    idx_start = np.searchsorted(time, t_start)
    idx_end = np.searchsorted(time, t_end)

    psth_window = psth_all[:, :, idx_start:idx_end] # récupérer uniquement entre 0 et 0.1s (le pic)

    # intégrale par neurone et fréquence
    auc = np.trapz(psth_window, dx=dt, axis=2)   # shape : neurones x fréquences

    # --- calcul de l'indice de sélectivité --- #
    selectivity = auc / np.nanmean(auc, axis=1, keepdims=True)  # shape : neurones x fréquences

    # --- selectivity +/- 2std --- #
    # On considère qu'une fréquence active un neurone si selectivity > 2*std
    # si pas de selectivity>2std alors on prend la freq qui a la selectivity la plus grande

    mean_sel = selectivity.mean(axis=1, keepdims=True)
    std_sel  = selectivity.std(axis=1, keepdims=True)

    threshold = mean_sel + 1 * std_sel

    # masque actif initialisé
    mask_active = np.zeros_like(selectivity, dtype=bool)

    for n in range(selectivity.shape[0]):
        # sélection par seuil
        mask = selectivity[n, :] > threshold[n]
        
        # si aucune fréquence ne dépasse le seuil, on prend la fréquence maximale
        if not np.any(mask):
            max_idx = np.argmax(selectivity[n, :])
            mask[max_idx] = True
        
        mask_active[n, :] = mask

    # masque inactif
    mask_inactive = ~mask_active

    # -----------------------
    # Variables similaires à ton code précédent
    # -----------------------
    neurons_active_idx, freqs_active_idx = np.where(mask_active)
    neurons_inactive_idx, freqs_inactive_idx = np.where(mask_inactive)

    freqs_active = freqs[freqs_active_idx]
    neurons_active = neurons_active_idx

    freqs_inactive = freqs[freqs_inactive_idx]
    neurons_inactive = neurons_inactive_idx

    return neurons_active, freqs_active, neurons_inactive, freqs_inactive





def get_psth_triggers_all_neurons_spaced(
    n_data_s,
    f_data_s,
    t_pre,
    t_post,
    dt, 
    dt_trigs,   # nouveau paramètre : temps minimum entre triggers (en secondes)
    condition
):
    pre_bins = int(t_pre / dt)
    post_bins = int(t_post / dt)
    n_bins = pre_bins + post_bins

    if condition == 'tracking':
        c = 0
    elif condition =='playback':
        c = 1
    else:
        raise ValueError('Condition must be "tracking" or "playback"')

    # --- Récupération des fréquences globales ---
    all_freqs = set()
    for f_data in f_data_s:
        mask = (
            (f_data['Frequency_changes'] == 1) &
            (f_data['Condition'] == 0) &
            (f_data['Played_frequency'] >= 500) &
            (f_data['Played_frequency'] <= 15000)
        )
        all_freqs.update(np.unique(f_data['Played_frequency'][mask]))

    freqs = np.sort(np.array(list(all_freqs)))
    freq_to_idx = {f: i for i, f in enumerate(freqs)}

    psth_list = []

    # --- Boucle sessions / neurones ---
    for session_idx in range(len(n_data_s)):
        n_data = n_data_s[session_idx]
        f_data = f_data_s[session_idx]

        trigger_mask = (
            (f_data['Frequency_changes'] == 1) &
            (f_data['Condition'] == c) &
            (f_data['Played_frequency'] >= 500) &
            (f_data['Played_frequency'] <= 15000)
        )

        trigger_indices = np.where(trigger_mask)[0]
        trigger_frequencies = f_data['Played_frequency'][trigger_indices]

        # --- filtrer triggers trop proches ---
        if dt_trigs is not None and len(trigger_indices) > 1:
            min_bins = int(dt_trigs / dt)
            keep_trigs = [trigger_indices[0]]
            for t in trigger_indices[1:]:
                if t - keep_trigs[-1] >= min_bins:
                    keep_trigs.append(t)
            trigger_indices = np.array(keep_trigs)
            trigger_frequencies = f_data['Played_frequency'][trigger_indices]

        for neurons in n_data:
            psth_neuron = np.full((len(freqs), n_bins), np.nan)

            for trig_idx, freq in zip(trigger_indices, trigger_frequencies):
                if trig_idx - pre_bins < 0 or trig_idx + post_bins > len(neurons):
                    continue

                f_idx = freq_to_idx[freq]
                psth = neurons[trig_idx - pre_bins : trig_idx + post_bins]

                if np.isnan(psth_neuron[f_idx]).all():
                    psth_neuron[f_idx] = psth
                else:
                    psth_neuron[f_idx] += psth

            # moyenne sur les répétitions
            for f_idx in range(len(freqs)):
                valid_trigs = np.sum(~np.isnan(psth_neuron[f_idx]))
                if valid_trigs > 0:
                    psth_neuron[f_idx] /= valid_trigs

            psth_list.append(psth_neuron)

    psth_all = np.stack(psth_list, axis=0)

    return psth_all, freqs



def get_psth_mock_triggers_all_neurons_spaced(
    n_data_s,
    f_data_s,
    t_pre,
    t_post,
    dt, 
    dt_trigs,   # nouveau paramètre : temps minimum entre triggers (en secondes)
    condition = 'playback'
):
    
    """"
    fonction pour les mocks
    """
    pre_bins = int(t_pre / dt)
    post_bins = int(t_post / dt)
    n_bins = pre_bins + post_bins

    if condition == 'playback':
        c = 1
    else:
        raise ValueError('Condition must be "playback"')

    # --- Récupération des fréquences globales ---
    all_freqs = set()
    for f_data in f_data_s:
        mask = (
            (f_data['Mock_change'] == 1) &
            (f_data['Condition'] == 1) &
            (f_data['Mock_frequency'] >= 500) &
            (f_data['Mock_frequency'] <= 15000)
        )
        all_freqs.update(np.unique(f_data['Mock_frequency'][mask]))

    freqs = np.sort(np.array(list(all_freqs)))
    freq_to_idx = {f: i for i, f in enumerate(freqs)}

    psth_list = []

    # --- Boucle sessions / neurones ---
    for session_idx in range(len(n_data_s)):
        n_data = n_data_s[session_idx]
        f_data = f_data_s[session_idx]

        trigger_mask = (
            (f_data['Mock_change'] == 1) &
            (f_data['Condition'] == c) &
            (f_data['Mock_frequency'] >= 500) &
            (f_data['Mock_frequency'] <= 15000)
        )

        trigger_indices = np.where(trigger_mask)[0]
        trigger_frequencies = f_data['Mock_frequency'][trigger_indices]

        # --- filtrer triggers trop proches ---
        if dt_trigs is not None and len(trigger_indices) > 1:
            min_bins = int(dt_trigs / dt)
            keep_trigs = [trigger_indices[0]]
            for t in trigger_indices[1:]:
                if t - keep_trigs[-1] >= min_bins:
                    keep_trigs.append(t)
            trigger_indices = np.array(keep_trigs)
            trigger_frequencies = f_data['Mock_frequency'][trigger_indices]

        for neurons in n_data:
            psth_neuron = np.full((len(freqs), n_bins), np.nan)

            for trig_idx, freq in zip(trigger_indices, trigger_frequencies):
                if trig_idx - pre_bins < 0 or trig_idx + post_bins > len(neurons):
                    continue

                f_idx = freq_to_idx[freq]
                psth = neurons[trig_idx - pre_bins : trig_idx + post_bins]

                if np.isnan(psth_neuron[f_idx]).all():
                    psth_neuron[f_idx] = psth
                else:
                    psth_neuron[f_idx] += psth

            # moyenne sur les répétitions
            for f_idx in range(len(freqs)):
                valid_trigs = np.sum(~np.isnan(psth_neuron[f_idx]))
                if valid_trigs > 0:
                    psth_neuron[f_idx] /= valid_trigs

            psth_list.append(psth_neuron)

    psth_all = np.stack(psth_list, axis=0)

    return psth_all, freqs