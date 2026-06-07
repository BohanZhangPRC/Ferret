#!/usr/bin/env python
# coding: utf-8

# ## Import Data

# In[1]:


# pip install pandas


# In[2]:


import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
def get_data(all_sessions_path, dt, gc_or_not=True):
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
            else:
                gc = np.arange(len(n_data))
            unique_tones_path = r"C:\Users\PenPen\Desktop\Ferret\Data\Bohan\unique_tones\unique_tones.npy"
            try:
                unique_tones = np.load(unique_tones_path)
            except FileNotFoundError:
                print(f"Unique tones file not found at path: {unique_tones_path}")
                raise
            # Neurones valides
            n_data = n_data[gc, :].astype(float)

            # Construction DataFrame
            f_data_dict = {
                'Played_frequency': [],
                'Condition': [],
                'Block': [],
                'Frequency_changes': [],
                'Mock_frequency': [],
                'Mock_change': []
            }
            for item in f_data_raw:
                for key, value in item.items():
                    f_data_dict[key].append(value)
            f_data = pd.DataFrame(f_data_dict)

            # IDs des neurones
            N_neurons = len(n_data)
            neuron_ids = list(range(global_id, global_id + N_neurons))
            global_id += N_neurons
            neuron_ids_s.append(neuron_ids)

            # Direction
            #f_data['Change_direction'] = compare_diff(f_data['Played_frequency'].to_numpy())
            #f_data['Mock_direction'] = compare_diff(f_data['Mock_frequency'].to_numpy())

            # --- mapping des fréquences vers pixels ---
            unique_tones_sorted = np.sort(unique_tones)
            pixels_sorted = np.linspace(0, 28, len(unique_tones_sorted))  # 28 cm

            # --- choisir Played ou Mock selon la condition ---
            positional_freq = np.where(
                f_data['Condition'].isin([0, -1]),
                f_data['Played_frequency'],
                f_data['Mock_frequency']
            )

            # --- interpolation pour toutes les fréquences inconnues ---
            positions = np.interp(positional_freq, unique_tones_sorted, pixels_sorted)

            # --- lissage ---
            pos_smooth = gaussian_filter(positions, sigma=10)

            # --- calcul de Speed_x ---
            speed_x = np.diff(pos_smooth)
            speed_x = np.append(0, speed_x)
            speed_x = np.abs(speed_x) * 100
            speed_x[~np.isfinite(speed_x)] = 0  # supprime NaN / inf
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
            f_data['Freq_position'] = freq_pos_smooth  # les positions correspondantes aux fréquences jouées

            speed_bins = [0, 0.01, 1, 2, 3, 4, 5, 6, np.inf]
            f_data['Speed_bin'] = pd.cut(
                f_data['Speed_x'],
                bins=speed_bins,
                labels=False,
                include_lowest=True
            )
            acc_x = np.diff(speed_x, prepend=speed_x[0])
            f_data["Acc_x"] = acc_x

            # smooth n_data
            n_data_o = n_data - n_data.mean(axis=1, keepdims=True)
            n_data_smooth = gaussian_filter(n_data_o, sigma=1, axes=1)
            # Sauvegarde
            n_data_s.append(n_data_smooth)
            f_data_s.append(f_data)

        except Exception as e:
            print(f"Error for file {file}: {e}")
    return n_data_s, f_data_s


# In[3]:


import os

base_path = r"C:\Users\PenPen\Desktop\Ferret\Data\Bohan"

all_sessions_path = [
    os.path.join(base_path, name) 
    for name in os.listdir(base_path) 
    if os.path.isdir(os.path.join(base_path, name))
]

all_sessions_path[:2]


# ## Set Up Essential Variables & Examining
# 

# In[4]:


dt = 0.005
#gc_or_not = True (if you only want the good clusters... keep it True)
n_data_s, f_data_s = get_data(all_sessions_path, dt, gc_or_not= True)


# In[5]:


# 1. Setup Parameters
t_window_pre = 0.32
t_window_post = 0.22

n_bins_pre = int(t_window_pre / dt)
n_bins_post = int(t_window_post / dt)
expected_length = n_bins_pre + n_bins_post


# In[6]:


for n_data, f_data in zip(n_data_s, f_data_s):
    print(f_data.groupby(['Block', 'Condition', 'Frequency_changes']).size())


# In[7]:


for n_data, f_data in zip(n_data_s, f_data_s):
    print(f_data[(f_data['Block'] == 5) & (f_data['Condition'] == 1) & (f_data['Frequency_changes'] == 1)].shape[0])


# ## Overall Average

# In[8]:


# List to collect individual NEURON means (PSTHs) from all sessions
all_neuron_means = []

# 2. Loop through all sessions 
for n_data, f_data in zip(n_data_s, f_data_s):

    event_playback = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))

    # Store events specifically for THIS session
    session_events = [] 

    for i in event_playback:
        start = int(i - n_bins_pre)
        stop = int(i + n_bins_post)

        if start >= 0 and stop <= n_data.shape[-1]:
            # segment shape: (n_channels, expected_length)
            segment = n_data[:, start:stop]
            if segment.shape[1] == expected_length:
                session_events.append(segment)

    # If we found valid events in this session
    if len(session_events) > 0:
        # Stack to shape: (n_events, n_channels, expected_length)
        session_stack = np.stack(session_events, axis=0)

        # STEP 1: Average across multiple events (axis=0)
        # Resulting shape: (n_channels, expected_length)
        # This represents the average trace for each individual channel in this session
        session_channel_means = np.mean(session_stack, axis=0)

        # Append these channel means to our global list
        all_neuron_means.append(session_channel_means)

# 3. Combine all 16 sessions * ~32 channels and Calculate Grand Average
if len(all_neuron_means) > 0:
    # Concatenate along axis=0 (the channel dimension)
    # Using concatenate because one session might have 32 channels, another 31, etc.
    # Resulting shape: (Total_Channels_Across_All_Sessions, expected_length)
    final_pool = np.concatenate(all_neuron_means, axis=0)

    # STEP 2: Calculate the population mean across all 16*32 channels
    # Resulting shape: (expected_length,)
    mean_playback = np.mean(final_pool, axis=0)

    # Calculate Standard Error (SEM) based on the number of CHANNELS, not events
    sem_playback = np.std(final_pool, axis=0) / np.sqrt(final_pool.shape[0])

    print(f"Successfully collected averaged traces from {final_pool.shape[0]} individual channels/neurons.")
    print(f"Mean trajectory calculated with shape: {mean_playback.shape}")
else:
    print("No valid events found.")


# In[9]:


mean_playback


# In[10]:


# 1. Create the time axis
# Ensure the length matches mean_playback (n_bins_pre + n_bins_post)
time = np.arange(-n_bins_pre, n_bins_post) * dt

# 2. Compatibility check: 
# Option 1 output 'mean_playback' is already a 1D array (the Global Average).
# We can wrap it in a Series or DataFrame for easy handling.
df_plot = pd.DataFrame({'Global_Avg': mean_playback}, index=time)

# 3. Plotting
plt.figure(figsize=(8, 5))

# Plot the 1D mean directly
plt.plot(df_plot.index, df_plot['Global_Avg'], 
         color='black', linewidth=2.5, label='Block 5 Playback (Global Avg)')

# 4. Formatting
plt.axvline(0, color='black', linestyle='--', label='Event')
plt.title(f'Playback Average Neural Response (Pooled Across {final_pool.shape[0]} neuron samples)', fontsize=14)
plt.xlabel('Time relative to event (s)', fontsize=12)
plt.ylabel('Mean Activity (Firing Rate)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save or display
# plt.savefig('global_average_response.png')
plt.show()


# ## Average of Tracking vs. Playback 

# In[11]:


# Lists to collect means for each neuron/channel from all sessions
all_neuron_means_playback = []
all_neuron_means_tracking = []

# 2. Loop through all sessions 
for n_data, f_data in zip(n_data_s, f_data_s):

    # Get event indices for both conditions
    event_playback = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    event_tracking = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # --- Helper function: extract average response for current session ---
    def get_session_channel_means(event_indices):
        session_events = []
        for i in event_indices:
            start = int(i - n_bins_pre)
            stop = int(i + n_bins_post)
            if start >= 0 and stop <= n_data.shape[-1]:
                segment = n_data[:, start:stop]
                if segment.shape[1] == expected_length:
                    session_events.append(segment)
        if len(session_events) > 0:
            session_stack = np.stack(session_events, axis=0)
            return np.mean(session_stack, axis=0) # shape: (n_channels, expected_length)
        return None
    # -------------------------------------------------------------

    # Extract Playback and append to list
    pb_means = get_session_channel_means(event_playback)
    if pb_means is not None:
        all_neuron_means_playback.append(pb_means)

    # Extract Tracking and append to list
    tr_means = get_session_channel_means(event_tracking)
    if tr_means is not None:
        all_neuron_means_tracking.append(tr_means)

# 3. Combine and Calculate Grand Averages for BOTH conditions
# Playback
if len(all_neuron_means_playback) > 0:
    final_pool_pb = np.concatenate(all_neuron_means_playback, axis=0)
    mean_playback = np.mean(final_pool_pb, axis=0)
    sem_playback = np.std(final_pool_pb, axis=0) / np.sqrt(final_pool_pb.shape[0])
    print(f"Playback: {final_pool_pb.shape[0]} neurons pooled.")

# Tracking
if len(all_neuron_means_tracking) > 0:
    final_pool_tr = np.concatenate(all_neuron_means_tracking, axis=0)
    mean_tracking = np.mean(final_pool_tr, axis=0)
    sem_tracking = np.std(final_pool_tr, axis=0) / np.sqrt(final_pool_tr.shape[0])
    print(f"Tracking: {final_pool_tr.shape[0]} neurons pooled.")


# In[12]:


mean_playback


# In[13]:


# Create time axis
time = np.arange(-n_bins_pre, n_bins_post) * dt

plt.figure(figsize=(9, 6))

# Plot Playback curve and shaded error
plt.plot(time, mean_playback, color='black', linewidth=2.5, label='Playback')
plt.fill_between(time, 
                 mean_playback - sem_playback, 
                 mean_playback + sem_playback, 
                 color='black', alpha=0.2, label='SEM-Playback')

# Plot Tracking curve and shaded error
plt.plot(time, mean_tracking, color='red', linewidth=2.5, label='Tracking')
plt.fill_between(time, 
                 mean_tracking - sem_tracking, 
                 mean_tracking + sem_tracking, 
                 color='red', alpha=0.2, label='SEM-Tracking')

# Figure decorations
plt.axvline(0, color='black', linestyle='--', label='Event Trigger')
plt.title(f'Population Neural Response: Playback ({final_pool_pb.shape[0]} neurons) vs Tracking ({final_pool_tr.shape[0]} neurons) (bin = {dt}s)', fontsize=14)
plt.xlabel('Time relative to event (s)', fontsize=12)
plt.ylabel('Mean Activity (Firing Rate)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# In[14]:


# Iterate through the list of DataFrames (f_data_s)
for i, f_data in enumerate(f_data_s):
    # Get the session folder name from the path list if available
    session_name = os.path.basename(all_sessions_path[i])

    print(f"--- Session {i}: {session_name} ---")

    # Check if 'Block' column exists
    if 'Block' in f_data.columns:
        # Sort by block index (e.g., Block 1, 2, 3...)
        block_counts = f_data['Block'].value_counts().sort_index()
        print(block_counts)
    else:
        print("Column 'Block' not found in this session.")
    print("\n")


# ## PSTH Tracking vs. Playback First/Last Block

# In[15]:


import numpy as np
import matplotlib.pyplot as plt


# Containers for the 4 groups
# Format: (First/Last) Block x (Playback/Tracking) Condition
data_groups = {
    'first_playback': [], 'first_tracking': [],
    'last_playback': [], 'last_tracking': []
}

# 2. Helper function to get channel means for specific criteria
def get_channel_means(n_data, f_data, block_id, condition_id):
    events = np.flatnonzero((f_data['Block'] == block_id) & 
                           (f_data['Condition'] == condition_id) & 
                           (f_data['Frequency_changes'] == 1))

    segments = []
    for i in events:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)

    if len(segments) > 0:
        return np.mean(np.stack(segments), axis=0) # Mean per channel
    return None

# 3. Main Processing Loop
for n_data, f_data in zip(n_data_s, f_data_s):
    # Find blocks that have BOTH condition 0 and 1
    blocks_with_both = []
    for b in sorted(f_data['Block'].unique()):
        has_cond0 = not f_data[(f_data['Block'] == b) & (f_data['Condition'] == 0)].empty
        has_cond1 = not f_data[(f_data['Block'] == b) & (f_data['Condition'] == 1)].empty
        if has_cond0 and has_cond1:
            blocks_with_both.append(b)

    if len(blocks_with_both) < 1:
        continue # Skip session if no block meets criteria

    first_b = blocks_with_both[0]
    last_b = blocks_with_both[-1]

    # Extract for First Block
    fp = get_channel_means(n_data, f_data, first_b, 1)
    ft = get_channel_means(n_data, f_data, first_b, 0)
    if fp is not None: data_groups['first_playback'].append(fp)
    if ft is not None: data_groups['first_tracking'].append(ft)

    # Extract for Last Block (only if it's different, or include anyway)
    lp = get_channel_means(n_data, f_data, last_b, 1)
    lt = get_channel_means(n_data, f_data, last_b, 0)
    if lp is not None: data_groups['last_playback'].append(lp)
    if lt is not None: data_groups['last_tracking'].append(lt)

# 4. Calculate Final Means and SEMs
results = {}
for key, list_of_means in data_groups.items():
    if list_of_means:
        combined = np.concatenate(list_of_means, axis=0)
        results[key] = {
            'mean': np.mean(combined, axis=0),
            'sem': np.std(combined, axis=0) / np.sqrt(combined.shape[0])
        }



# In[16]:


results = {}
for key, list_of_means in data_groups.items():
    if list_of_means:
        combined = np.concatenate(list_of_means, axis=0)
        results[key] = {
            'mean': np.mean(combined, axis=0),
            'sem': np.std(combined, axis=0) / np.sqrt(combined.shape[0]),
            'count': combined.shape[0]  # <--- 关键：把神经元数量存下来
        }

# 创建两个并排的子图
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=0)

# --- 左图：First Block ---
ax1 = axes[0]
n_pb_first = results['first_playback']['count']
n_tr_first = results['first_tracking']['count']

ax1.plot(time, results['first_playback']['mean'], color='black', linewidth=2.5, 
         label=f'Playback (n={n_pb_first})') # 在图例显示数量
ax1.plot(time, results['first_tracking']['mean'], color='red', linewidth=2.5, 
         label=f'Tracking (n={n_tr_first})')
ax1.fill_between(time, 
                 results['first_tracking']['mean'] - results['first_tracking']['sem'], 
                 results['first_tracking']['mean'] + results['first_tracking']['sem'], 
                 color='red', alpha=0.2, label='SEM-Tracking')
ax1.fill_between(time, 
                 results['first_playback']['mean'] - results['first_playback']['sem'], 
                 results['first_playback']['mean'] + results['first_playback']['sem'], 
                 color='black', alpha=0.2, label='SEM-Playback')                 
ax1.set_title(f'First Block\n(n={n_tr_first} neurons)', fontsize=14)


# --- 右图：Last Block ---
ax2 = axes[1]
n_pb_last = results['last_playback']['count']
n_tr_last = results['last_tracking']['count']

ax2.plot(time, results['last_playback']['mean'], color='black', linewidth=2.5, 
         label=f'Playback (n={n_pb_last})')
ax2.plot(time, results['last_tracking']['mean'], color='red', linewidth=2.5, 
         label=f'Tracking (n={n_tr_last})')
ax2.fill_between(time, 
                 results['last_tracking']['mean'] - results['last_tracking']['sem'], 
                 results['last_tracking']['mean'] + results['last_tracking']['sem'], 
                 color='red', alpha=0.2)
ax2.fill_between(time, 
                 results['last_playback']['mean'] - results['last_playback']['sem'], 
                 results['last_playback']['mean'] + results['last_playback']['sem'], 
                 color='black', alpha=0.2)

ax2.set_title(f'Last Block\n(n={n_tr_last} neurons)', fontsize=14)


# --- 总标题修改 ---
# 如果你想在总标题显示总数，可以直接取一个代表性的数字
plt.suptitle(f'Neural Response Comparison: Tracking (Red) vs Playback (Black) (bin = {dt}s)', 
             fontsize=16, y=1.05)

# --- 通用格式设置 ---
for ax in axes:
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Event')
    ax.set_xlabel('Time relative to event (s)', fontsize=12)
    ax.set_ylabel('Firing Rate', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[17]:


# 1. 计算差值 (Last - First) 和 误差传递 (Error Propagation)
# Playback Difference
diff_pb_mean = results['last_playback']['mean'] - results['first_playback']['mean']
diff_pb_sem = np.sqrt(results['first_playback']['sem']**2 + results['last_playback']['sem']**2)

# Tracking Difference
diff_tr_mean = results['last_tracking']['mean'] - results['first_tracking']['mean']
diff_tr_sem = np.sqrt(results['first_tracking']['sem']**2 + results['last_tracking']['sem']**2)

# 2. 设置画布：3行1列
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# --- Top Plot: First Block ---
ax1 = axes[0]
ax1.plot(time, results['first_playback']['mean'], color='black', linewidth=2, label='Playback')
ax1.fill_between(time, results['first_playback']['mean'] - results['first_playback']['sem'], 
                 results['first_playback']['mean'] + results['first_playback']['sem'], color='black', alpha=0.1)
ax1.plot(time, results['first_tracking']['mean'], color='red', linewidth=2, label='Tracking')
ax1.fill_between(time, results['first_tracking']['mean'] - results['first_tracking']['sem'], 
                 results['first_tracking']['mean'] + results['first_tracking']['sem'], color='red', alpha=0.1)
ax1.set_title(f"First Common Block (n={results['first_tracking']['count']} neurons)", fontsize=14)

# --- Middle Plot: Last Block ---
ax2 = axes[1]
ax2.plot(time, results['last_playback']['mean'], color='black', linewidth=2, label='Playback')
ax2.fill_between(time, results['last_playback']['mean'] - results['last_playback']['sem'], 
                 results['last_playback']['mean'] + results['last_playback']['sem'], color='black', alpha=0.1)
ax2.plot(time, results['last_tracking']['mean'], color='red', linewidth=2, label='Tracking')
ax2.fill_between(time, results['last_tracking']['mean'] - results['last_tracking']['sem'], 
                 results['last_tracking']['mean'] + results['last_tracking']['sem'], color='red', alpha=0.1)
ax2.set_title(f"Last Common Block (n={results['last_tracking']['count']} neurons)", fontsize=14)

# --- Bottom Plot: Difference (Last - First) ---
ax3 = axes[2]
# 绘制 Playback 差值
ax3.plot(time, diff_pb_mean, color='black', linewidth=2.5, label='Playback Difference')
# ax3.fill_between(time, diff_pb_mean - diff_pb_sem, diff_pb_mean + diff_pb_sem, color='black', alpha=0.1)

# 绘制 Tracking 差值
ax3.plot(time, diff_tr_mean, color='red', linewidth=2.5, label='Tracking Difference')
# ax3.fill_between(time, diff_tr_mean - diff_tr_sem, diff_tr_mean + diff_tr_sem, color='red', alpha=0.1)

# 在差值图增加 y=0 的水平基准线，方便看增减
ax3.axhline(0, color='blue', linestyle=':', alpha=0.5)
ax3.set_title("Difference (Last Block - First Block)", fontsize=14)
ax3.set_ylabel('$\Delta$ Firing Rate', fontsize=12)

# --- 通用格式设置 ---
for ax in axes:
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time relative to event (s)', fontsize=12)
    ax.set_ylabel('Firing Rate', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=10)

plt.suptitle(f'Neural Dynamics: First vs Last Block with Difference (bin = {dt}s)', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# ## Last n Tracking vs. First n Playback

# In[18]:


# 1. Setup Parameters (re-using your existing variables if already defined)
time = np.arange(-n_bins_pre, n_bins_post) * dt
n_triggers = 1  # Number of events to take from each condition for the subset analysis

# Lists to collect means for each channel across sessions
means_first_n_pb = []
means_last_n_tr = []

# 2. Helper function to extract and average segments for specific event indices
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)

    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) # Mean per channel
    return None

# 3. Process each session
for n_data, f_data in zip(n_data_s, f_data_s):
    # Get all valid events for Playback (Condition 1) and Tracking (Condition 0)
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # Subset: First 100 Playback, Last 100 Tracking
    events_pb_subset = events_pb[:n_triggers] if len(events_pb) >= n_triggers else events_pb
    events_tr_subset = events_tr[-n_triggers:] if len(events_tr) >= n_triggers else events_tr

    # Extract and store
    pb_mean = get_subset_channel_means(n_data, events_pb_subset)
    tr_mean = get_subset_channel_means(n_data, events_tr_subset)

    if pb_mean is not None: means_first_n_pb.append(pb_mean)
    if tr_mean is not None: means_last_n_tr.append(tr_mean)

# 4. Calculate Grand Averages
final_pb = np.concatenate(means_first_n_pb, axis=0)
final_tr = np.concatenate(means_last_n_tr, axis=0)

res_pb = {
    'mean': np.mean(final_pb, axis=0),
    'sem': np.std(final_pb, axis=0) / np.sqrt(final_pb.shape[0]),
    'count': final_pb.shape[0]
}

res_tr = {
    'mean': np.mean(final_tr, axis=0),
    'sem': np.std(final_tr, axis=0) / np.sqrt(final_tr.shape[0]),
    'count': final_tr.shape[0]
}

# Calculate Difference (Tracking - Playback)
diff_mean = res_tr['mean'] - res_pb['mean']
# Standard error of the difference
diff_sem = np.sqrt(res_tr['sem']**2 + res_pb['sem']**2)

# 5. Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# --- Top Plot: Overlaid Means ---
ax1 = axes[0]
ax1.plot(time, res_pb['mean'], color='black', linewidth=2.5, 
         label=f"First {n_triggers} Playback (n={res_pb['count']})")
ax1.fill_between(time, res_pb['mean'] - res_pb['sem'], res_pb['mean'] + res_pb['sem'], 
                 color='black', alpha=0.15)

ax1.plot(time, res_tr['mean'], color='red', linewidth=2.5, 
         label=f"Last {n_triggers} Tracking (n={res_tr['count']})")
ax1.fill_between(time, res_tr['mean'] - res_tr['sem'], res_tr['mean'] + res_tr['sem'], 
                 color='red', alpha=0.15)

ax1.set_title(f"Neural Response: Last {n_triggers} Tracking vs First {n_triggers} Playback", fontsize=14)
ax1.set_ylabel('Firing Rate', fontsize=12)

# --- Bottom Plot: Difference ---
ax2 = axes[1]
ax2.plot(time, diff_mean, color='purple', linewidth=2.5, label='Difference (Tracking - Playback)')
ax2.fill_between(time, diff_mean - diff_sem, diff_mean + diff_sem, color='purple', alpha=0.15)
ax2.axhline(0, color='gray', linestyle=':', linewidth=2, alpha=0.8)

ax2.set_title("Response Difference", fontsize=14)
ax2.set_xlabel('Time relative to event (s)', fontsize=12)
ax2.set_ylabel('$\Delta$ Firing Rate', fontsize=12)

# Format both axes
for ax in axes:
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Event Trigger' if ax == ax1 else "")
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[19]:


import matplotlib.ticker as ticker
# Define indices for calculating metrics
# Baseline: 0.2s before frequency change
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Frequency change (time 0)

# Peak: Max response after frequency change
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Define the different "n_triggers" (number of events) you want to evaluate
n_triggers_list = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300, 500]

# Dictionary to store the results
results_metrics = {
    'n_events': [],
    'pb_baseline': [], 'pb_peak': [], 'pb_evoked': [],
    'tr_baseline': [], 'tr_peak': [], 'tr_evoked': []
}

# Helper function to extract and average segments
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) 
    return None

# 2. Main Processing Loop
for n_b in n_triggers_list:
    means_first_n_pb = []
    means_last_n_tr = []

    for n_data, f_data in zip(n_data_s, f_data_s):
        events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
        events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

        # Subset: First n_b Playback, Last n_b Tracking
        events_pb_subset = events_pb[:n_b] if len(events_pb) >= n_b else events_pb
        events_tr_subset = events_tr[-n_b:] if len(events_tr) >= n_b else events_tr

        pb_mean = get_subset_channel_means(n_data, events_pb_subset)
        tr_mean = get_subset_channel_means(n_data, events_tr_subset)

        if pb_mean is not None: means_first_n_pb.append(pb_mean)
        if tr_mean is not None: means_last_n_tr.append(tr_mean)

    # Calculate Grand Averages for this n_b
    final_pb = np.concatenate(means_first_n_pb, axis=0)
    final_tr = np.concatenate(means_last_n_tr, axis=0)

    mean_pb_trace = np.mean(final_pb, axis=0)
    mean_tr_trace = np.mean(final_tr, axis=0)

    # --- Calculate Metrics for Playback ---
    pb_base = np.mean(mean_pb_trace[idx_base_start:idx_base_end])
    pb_pk = np.max(mean_pb_trace[idx_peak_start:idx_peak_end])
    pb_evk = pb_pk - pb_base

    # --- Calculate Metrics for Tracking ---
    tr_base = np.mean(mean_tr_trace[idx_base_start:idx_base_end])
    tr_pk = np.max(mean_tr_trace[idx_peak_start:idx_peak_end])
    tr_evk = tr_pk - tr_base

    # Store results
    results_metrics['n_events'].append(n_b)
    results_metrics['pb_baseline'].append(pb_base)
    results_metrics['pb_peak'].append(pb_pk)
    results_metrics['pb_evoked'].append(pb_evk)
    results_metrics['tr_baseline'].append(tr_base)
    results_metrics['tr_peak'].append(tr_pk)
    results_metrics['tr_evoked'].append(tr_evk)

# 3. Create a DataFrame for easy viewing
df_metrics = pd.DataFrame(results_metrics)
display(df_metrics)

# 4. Plotting the metrics across different subset sizes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Baseline Plot
axes[0].plot(df_metrics['n_events'], df_metrics['pb_baseline'], marker='o', color='black', label='Playback')
axes[0].plot(df_metrics['n_events'], df_metrics['tr_baseline'], marker='o', color='red', label='Tracking')
axes[0].set_title('Baseline (-0.2s to 0s)')
axes[0].set_ylabel('Mean Firing Rate')

# Peak Plot
axes[1].plot(df_metrics['n_events'], df_metrics['pb_peak'], marker='o', color='black', label='Playback')
axes[1].plot(df_metrics['n_events'], df_metrics['tr_peak'], marker='o', color='red', label='Tracking')
axes[1].set_title('Peak Response')

# Total Evoked Response Plot
axes[2].plot(df_metrics['n_events'], df_metrics['pb_evoked'], marker='o', color='black', label='Playback')
axes[2].plot(df_metrics['n_events'], df_metrics['tr_evoked'], marker='o', color='red', label='Tracking')
axes[2].set_title('Total Evoked Response (Peak - Baseline)')

for ax in axes:
    # Set x-axis to logarithmic scale
    ax.set_xscale('log')

    # Force the x-ticks to be exactly your n_triggers_list values
    ax.set_xticks(n_triggers_list)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Number of Events (n_triggers) [Log Scale]')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()


# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# Define the range from 1 to 500
start = 1
stop = 512

# Generate powers of 2: 2^0, 2^1, 2^2, ... up to 500
# We use log2 to determine the maximum exponent
n_triggers_list = [int(round(2**i)) for i in range(int(math.log2(stop)) + 1)]

# Define indices for calculating metrics
# Baseline: 0.2s before sound onset
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Sound onset (time 0)

# Peak: Max response after sound onset
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Define the different "n_triggers" (number of events/trials) you want to evaluate
# n_triggers_list = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300, 500]

# Dictionary to store the results (Means and SEMs)
results_metrics = {
    'n_triggers': [],
    'pb_base_mean': [], 'pb_base_sem': [],
    'pb_peak_mean': [], 'pb_peak_sem': [],
    'pb_evoked_mean': [], 'pb_evoked_sem': [],
    'tr_base_mean': [], 'tr_base_sem': [],
    'tr_peak_mean': [], 'tr_peak_sem': [],
    'tr_evoked_mean': [], 'tr_evoked_sem': []
}

# Helper function to extract and average segments across selected triggers for each channel
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) 
    return None

# 2. Main Processing Loop
for n_trig in n_triggers_list:
    means_first_n_pb = []
    means_last_n_tr = []

    for n_data, f_data in zip(n_data_s, f_data_s):
        events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
        events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

        # Subset: First n_trig Playback, Last n_trig Tracking
        events_pb_subset = events_pb[:n_trig] if len(events_pb) >= n_trig else events_pb
        events_tr_subset = events_tr[-n_trig:] if len(events_tr) >= n_trig else events_tr

        pb_mean = get_subset_channel_means(n_data, events_pb_subset)
        tr_mean = get_subset_channel_means(n_data, events_tr_subset)

        if pb_mean is not None: means_first_n_pb.append(pb_mean)
        if tr_mean is not None: means_last_n_tr.append(tr_mean)

    # Concatenate across all sessions. Shape: (n_total_neurons, expected_length)
    final_pb = np.concatenate(means_first_n_pb, axis=0) if means_first_n_pb else np.array([])
    final_tr = np.concatenate(means_last_n_tr, axis=0) if means_last_n_tr else np.array([])

    if final_pb.size == 0 or final_tr.size == 0:
        continue

    n_pb_neurons = final_pb.shape[0]
    n_tr_neurons = final_tr.shape[0]

    # --- Calculate Metrics for EACH neuron individually for Playback ---
    pb_base_per_neuron = np.mean(final_pb[:, idx_base_start:idx_base_end], axis=1)
    pb_pk_per_neuron = np.max(final_pb[:, idx_peak_start:idx_peak_end], axis=1)
    pb_evk_per_neuron = pb_pk_per_neuron - pb_base_per_neuron

    # --- Calculate Metrics for EACH neuron individually for Tracking ---
    tr_base_per_neuron = np.mean(final_tr[:, idx_base_start:idx_base_end], axis=1)
    tr_pk_per_neuron = np.max(final_tr[:, idx_peak_start:idx_peak_end], axis=1)
    tr_evk_per_neuron = tr_pk_per_neuron - tr_base_per_neuron

    # Calculate Means and SEMs and store them
    results_metrics['n_triggers'].append(n_trig)

    results_metrics['pb_base_mean'].append(np.mean(pb_base_per_neuron))
    results_metrics['pb_base_sem'].append(np.std(pb_base_per_neuron) / np.sqrt(n_pb_neurons))
    results_metrics['pb_peak_mean'].append(np.mean(pb_pk_per_neuron))
    results_metrics['pb_peak_sem'].append(np.std(pb_pk_per_neuron) / np.sqrt(n_pb_neurons))
    results_metrics['pb_evoked_mean'].append(np.mean(pb_evk_per_neuron))
    results_metrics['pb_evoked_sem'].append(np.std(pb_evk_per_neuron) / np.sqrt(n_pb_neurons))

    results_metrics['tr_base_mean'].append(np.mean(tr_base_per_neuron))
    results_metrics['tr_base_sem'].append(np.std(tr_base_per_neuron) / np.sqrt(n_tr_neurons))
    results_metrics['tr_peak_mean'].append(np.mean(tr_pk_per_neuron))
    results_metrics['tr_peak_sem'].append(np.std(tr_pk_per_neuron) / np.sqrt(n_tr_neurons))
    results_metrics['tr_evoked_mean'].append(np.mean(tr_evk_per_neuron))
    results_metrics['tr_evoked_sem'].append(np.std(tr_evk_per_neuron) / np.sqrt(n_tr_neurons))




# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'results_metrics' is defined earlier in your script
df_metrics = pd.DataFrame(results_metrics)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_vals = df_metrics['n_triggers'].values
n_points = len(x_vals)
center_x = n_points - 0.5  # The exact middle point of the plot

# Create sequential x-coordinates
x_tr = np.arange(n_points)
x_pb = np.arange(n_points) + n_points

# Reverse Tracking data so it plots chronologically toward the center (512 down to 1)
tr_base_mean_rev = df_metrics['tr_base_mean'].values[::-1]
tr_base_sem_rev  = df_metrics['tr_base_sem'].values[::-1]
tr_peak_mean_rev = df_metrics['tr_peak_mean'].values[::-1]
tr_peak_sem_rev  = df_metrics['tr_peak_sem'].values[::-1]
tr_evk_mean_rev  = df_metrics['tr_evoked_mean'].values[::-1]
tr_evk_sem_rev   = df_metrics['tr_evoked_sem'].values[::-1]

# Playback data (normal order)
pb_base_mean = df_metrics['pb_base_mean'].values
pb_base_sem  = df_metrics['pb_base_sem'].values
pb_peak_mean = df_metrics['pb_peak_mean'].values
pb_peak_sem  = df_metrics['pb_peak_sem'].values
pb_evk_mean  = df_metrics['pb_evoked_mean'].values
pb_evk_sem   = df_metrics['pb_evoked_sem'].values

# Create X-tick labels: Reversed for Tracking, Normal for Playback
labels_tr = list(x_vals)[::-1]
labels_pb = list(x_vals)
all_labels = labels_tr + labels_pb

# Formatting helper for the axis
def setup_sequential_axis(ax):
    ax.set_xticks(list(x_tr) + list(x_pb))
    ax.set_xticklabels(all_labels, rotation=45)
    ax.set_xlabel('Number of Triggers')
    # The center dividing line
    ax.axvline(x=center_x, color='gray', linestyle='--', alpha=0.8, zorder=3)
    ax.grid(True, alpha=0.3)
    ax.legend()

# Helper function to plot means, fill SEMs, and draw horizontal dashed lines
def plot_metric_with_fills(ax, tr_m, tr_s, pb_m, pb_s, title, ylabel=None):
    # --- Tracking (Red) ---
    ax.plot(x_tr, tr_m, marker='o', color='red', label='Tracking Mean')
    # ADD LABEL HERE
    ax.fill_between(x_tr, tr_m - tr_s, tr_m + tr_s, color='red', alpha=0.2, label='Tracking SEM')
    ax.hlines(y=tr_m, xmin=x_tr, xmax=center_x, color='red', linestyle='--', alpha=0.4, zorder=1)

    # --- Playback (Black) ---
    ax.plot(x_pb, pb_m, marker='o', color='black', label='Playback Mean')
    # ADD LABEL HERE
    ax.fill_between(x_pb, pb_m - pb_s, pb_m + pb_s, color='black', alpha=0.2, label='Playback SEM')
    ax.hlines(y=pb_m, xmin=center_x, xmax=x_pb, color='black', linestyle='--', alpha=0.4, zorder=1)

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    setup_sequential_axis(ax)

# 3A. Baseline Plot
plot_metric_with_fills(axes[0], tr_base_mean_rev, tr_base_sem_rev, 
                       pb_base_mean, pb_base_sem, 
                       title='Baseline (-0.2s to 0s)', ylabel='Mean Firing Rate')

# 3B. Peak Plot
plot_metric_with_fills(axes[1], tr_peak_mean_rev, tr_peak_sem_rev, 
                       pb_peak_mean, pb_peak_sem, 
                       title='Peak Response')

# 3C. Total Evoked Response Plot
plot_metric_with_fills(axes[2], tr_evk_mean_rev, tr_evk_sem_rev, 
                       pb_evk_mean, pb_evk_sem, 
                       title='Total Evoked Response (Peak - Baseline)')

plt.suptitle(f'Last n Triggers of Tracking vs First n Triggers of Playback', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()


# In[22]:


# Define the range from 1 to 500
start = 1
stop = 512

# Generate powers of 2: 2^0, 2^1, 2^2, ... up to 500
n_triggers_list = [int(round(2**i)) for i in range(int(math.log2(stop)) + 1)]

# Define indices for calculating metrics
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Sound onset (time 0)

idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Dictionary to store the results (Means and SEMs)
results_metrics = {
    'n_triggers': [],
    'pb_base_mean': [], 'pb_base_sem': [],
    'pb_peak_mean': [], 'pb_peak_sem': [],
    'pb_evoked_mean': [], 'pb_evoked_sem': [],
    'tr_base_mean': [], 'tr_base_sem': [],
    'tr_peak_mean': [], 'tr_peak_sem': [],
    'tr_evoked_mean': [], 'tr_evoked_sem': []
}

# Helper function
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) 
    return None

# 2. Main Processing Loop
for n_trig in n_triggers_list:
    means_first_n_pb = []
    means_last_n_tr = []

    for n_data, f_data in zip(n_data_s, f_data_s):
        events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
        events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

        events_pb_subset = events_pb[:n_trig] if len(events_pb) >= n_trig else events_pb
        events_tr_subset = events_tr[-n_trig:] if len(events_tr) >= n_trig else events_tr

        pb_mean = get_subset_channel_means(n_data, events_pb_subset)
        tr_mean = get_subset_channel_means(n_data, events_tr_subset)

        if pb_mean is not None: means_first_n_pb.append(pb_mean)
        if tr_mean is not None: means_last_n_tr.append(tr_mean)

    final_pb = np.concatenate(means_first_n_pb, axis=0) if means_first_n_pb else np.array([])
    final_tr = np.concatenate(means_last_n_tr, axis=0) if means_last_n_tr else np.array([])

    if final_pb.size == 0 or final_tr.size == 0:
        continue

    n_pb_neurons = final_pb.shape[0]
    n_tr_neurons = final_tr.shape[0]

    # =========================================================
    # APPLY Z-SCORE (按每个神经元独立计算 Z-score)
    # =========================================================
    # 为了避免除以 0，给标准差加上一个极小值 (1e-8)

    # Playback Z-score (基于整个时段计算均值和标准差)
    pb_mean_time = np.mean(final_pb, axis=1, keepdims=True)
    pb_std_time = np.std(final_pb, axis=1, keepdims=True) + 1e-8
    final_pb_z = (final_pb - pb_mean_time) / pb_std_time

    # Tracking Z-score
    tr_mean_time = np.mean(final_tr, axis=1, keepdims=True)
    tr_std_time = np.std(final_tr, axis=1, keepdims=True) + 1e-8
    final_tr_z = (final_tr - tr_mean_time) / tr_std_time

    # --- 基于 Z-scored 数据计算 Playback 的 Metrics ---
    pb_base_per_neuron = np.mean(final_pb_z[:, idx_base_start:idx_base_end], axis=1)
    pb_pk_per_neuron = np.max(final_pb_z[:, idx_peak_start:idx_peak_end], axis=1)
    pb_evk_per_neuron = pb_pk_per_neuron - pb_base_per_neuron

    # --- 基于 Z-scored 数据计算 Tracking 的 Metrics ---
    tr_base_per_neuron = np.mean(final_tr_z[:, idx_base_start:idx_base_end], axis=1)
    tr_pk_per_neuron = np.max(final_tr_z[:, idx_peak_start:idx_peak_end], axis=1)
    tr_evk_per_neuron = tr_pk_per_neuron - tr_base_per_neuron

    # Calculate Means and SEMs and store them
    results_metrics['n_triggers'].append(n_trig)

    results_metrics['pb_base_mean'].append(np.mean(pb_base_per_neuron))
    results_metrics['pb_base_sem'].append(np.std(pb_base_per_neuron) / np.sqrt(n_pb_neurons))
    results_metrics['pb_peak_mean'].append(np.mean(pb_pk_per_neuron))
    results_metrics['pb_peak_sem'].append(np.std(pb_pk_per_neuron) / np.sqrt(n_pb_neurons))
    results_metrics['pb_evoked_mean'].append(np.mean(pb_evk_per_neuron))
    results_metrics['pb_evoked_sem'].append(np.std(pb_evk_per_neuron) / np.sqrt(n_pb_neurons))

    results_metrics['tr_base_mean'].append(np.mean(tr_base_per_neuron))
    results_metrics['tr_base_sem'].append(np.std(tr_base_per_neuron) / np.sqrt(n_tr_neurons))
    results_metrics['tr_peak_mean'].append(np.mean(tr_pk_per_neuron))
    results_metrics['tr_peak_sem'].append(np.std(tr_pk_per_neuron) / np.sqrt(n_tr_neurons))
    results_metrics['tr_evoked_mean'].append(np.mean(tr_evk_per_neuron))
    results_metrics['tr_evoked_sem'].append(np.std(tr_evk_per_neuron) / np.sqrt(n_tr_neurons))


# In[23]:


# =========================================================
# PLOTTING
# =========================================================
df_metrics = pd.DataFrame(results_metrics)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_vals = df_metrics['n_triggers'].values
n_points = len(x_vals)
center_x = n_points - 0.5

x_tr = np.arange(n_points)
x_pb = np.arange(n_points) + n_points

# Reverse Tracking data
tr_base_mean_rev = df_metrics['tr_base_mean'].values[::-1]
tr_base_sem_rev  = df_metrics['tr_base_sem'].values[::-1]
tr_peak_mean_rev = df_metrics['tr_peak_mean'].values[::-1]
tr_peak_sem_rev  = df_metrics['tr_peak_sem'].values[::-1]
tr_evk_mean_rev  = df_metrics['tr_evoked_mean'].values[::-1]
tr_evk_sem_rev   = df_metrics['tr_evoked_sem'].values[::-1]

# Playback data
pb_base_mean = df_metrics['pb_base_mean'].values
pb_base_sem  = df_metrics['pb_base_sem'].values
pb_peak_mean = df_metrics['pb_peak_mean'].values
pb_peak_sem  = df_metrics['pb_peak_sem'].values
pb_evk_mean  = df_metrics['pb_evoked_mean'].values
pb_evk_sem   = df_metrics['pb_evoked_sem'].values

labels_tr = list(x_vals)[::-1]
labels_pb = list(x_vals)
all_labels = labels_tr + labels_pb

def setup_sequential_axis(ax):
    ax.set_xticks(list(x_tr) + list(x_pb))
    ax.set_xticklabels(all_labels, rotation=45)
    ax.set_xlabel('Number of Triggers')
    ax.axvline(x=center_x, color='gray', linestyle='--', alpha=0.8, zorder=3)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_metric_with_fills(ax, tr_m, tr_s, pb_m, pb_s, title, ylabel=None):
    # Tracking (Red)
    ax.plot(x_tr, tr_m, marker='o', color='red', label='Tracking Mean')
    ax.fill_between(x_tr, tr_m - tr_s, tr_m + tr_s, color='red', alpha=0.2, label='Tracking SEM')
    ax.hlines(y=tr_m, xmin=x_tr, xmax=center_x, color='red', linestyle='--', alpha=0.4, zorder=1)

    # Playback (Black)
    ax.plot(x_pb, pb_m, marker='o', color='black', label='Playback Mean')
    ax.fill_between(x_pb, pb_m - pb_s, pb_m + pb_s, color='black', alpha=0.2, label='Playback SEM')
    ax.hlines(y=pb_m, xmin=center_x, xmax=x_pb, color='black', linestyle='--', alpha=0.4, zorder=1)

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    setup_sequential_axis(ax)

# 3A. Baseline Plot
plot_metric_with_fills(axes[0], tr_base_mean_rev, tr_base_sem_rev, 
                       pb_base_mean, pb_base_sem, 
                       title='Baseline Z-Score (-0.2s to 0s)', ylabel='Z-Score')

# 3B. Peak Plot
plot_metric_with_fills(axes[1], tr_peak_mean_rev, tr_peak_sem_rev, 
                       pb_peak_mean, pb_peak_sem, 
                       title='Peak Response Z-Score', ylabel='Z-Score')

# 3C. Total Evoked Response Plot
plot_metric_with_fills(axes[2], tr_evk_mean_rev, tr_evk_sem_rev, 
                       pb_evk_mean, pb_evk_sem, 
                       title='Total Evoked Response (Peak - Baseline Z-Score)', ylabel='Z-Score')

plt.suptitle(f'Last n Triggers of Tracking vs First n Triggers of Playback (Z-Scored)', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CUSTOMIZABLE VARIABLES ---
n_per_group = 1    # Number of triggers per group
n_groups = 3        # Number of chronological groups per condition
# ==========================================

# Generate keys dynamically based on variables
tr_keys = []
for i in range(n_groups, 0, -1):
    start_idx = i * n_per_group
    end_idx = (i - 1) * n_per_group + 1
    tr_keys.append(f'Tracking: Last {start_idx}-{end_idx}')

pb_keys = []
for i in range(n_groups):
    start_idx = i * n_per_group + 1
    end_idx = (i + 1) * n_per_group
    pb_keys.append(f'Playback: First {start_idx}-{end_idx}')

all_keys = tr_keys + pb_keys
all_data = {key: [] for key in all_keys}

# Helper function to extract and average segments
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0)
    return None

# 2. Main Processing Loop
for n_data, f_data in zip(n_data_s, f_data_s):
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # --- Tracking: Slice chronological chunks from the end ---
    total_tr_needed = n_groups * n_per_group
    tr_chunks = []

    # Only process if the session has enough tracking events
    if len(events_tr) >= total_tr_needed:
        for i in range(n_groups, 0, -1):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            tr_chunks.append(events_tr[start:end])
    else:
        tr_chunks = [[] for _ in range(n_groups)] 

    for key, evs in zip(tr_keys, tr_chunks):
        if len(evs) > 0:
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                all_data[key].append(mean_trace)

    # --- Playback: Slice chronological chunks from the beginning ---
    total_pb_needed = n_groups * n_per_group
    pb_chunks = []

    # Only process if the session has enough playback events
    if len(events_pb) >= total_pb_needed:
        for i in range(n_groups):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            pb_chunks.append(events_pb[start:end])
    else:
        pb_chunks = [[] for _ in range(n_groups)]

    for key, evs in zip(pb_keys, pb_chunks):
        if len(evs) > 0:
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                all_data[key].append(mean_trace)

# 3. Plotting in a Dynamic Grid
# Create a 2 x n_groups grid (Top row: Tracking, Bottom row: Playback)
fig, axes = plt.subplots(1, n_groups * 2 , figsize=(20 * n_groups, 8), sharey=True, sharex=True)

# Ensure axes is a 2D array even if n_groups=1
axes = np.array(axes).reshape(2, n_groups)

# Generate dynamic colors (from light to dark)
tr_colors = [plt.cm.Reds(x) for x in np.linspace(0.4, 0.9, n_groups)]
pb_colors = [plt.cm.Greys(x) for x in np.linspace(0.4, 0.9, n_groups)]

# --- Plot Tracking (Top Row) ---
for i, (key, color) in enumerate(zip(tr_keys, tr_colors)):
    ax = axes[0, i]
    if len(all_data[key]) > 0:
        final_dat = np.concatenate(all_data[key], axis=0)
        mean_dat = np.mean(final_dat, axis=0)
        sem_dat = np.std(final_dat, axis=0) / np.sqrt(final_dat.shape[0])

        ax.plot(time, mean_dat, color=color, linewidth=2.5, label=f'Neurons: n={final_dat.shape[0]}')
        ax.fill_between(time, mean_dat - sem_dat, mean_dat + sem_dat, color=color, alpha=0.15)
        ax.legend(loc='upper right')

    ax.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Sound Onset' if i==0 else "")
    ax.set_title(key, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0: ax.set_ylabel('Mean Firing Rate\n[Tracking]', fontsize=12)

# --- Plot Playback (Bottom Row) ---
for i, (key, color) in enumerate(zip(pb_keys, pb_colors)):
    ax = axes[1, i]
    if len(all_data[key]) > 0:
        final_dat = np.concatenate(all_data[key], axis=0)
        mean_dat = np.mean(final_dat, axis=0)
        sem_dat = np.std(final_dat, axis=0) / np.sqrt(final_dat.shape[0])

        ax.plot(time, mean_dat, color=color, linewidth=2.5, label=f'Neurons: n={final_dat.shape[0]}')
        ax.fill_between(time, mean_dat - sem_dat, mean_dat + sem_dat, color=color, alpha=0.15)
        ax.legend(loc='upper right')

    ax.axvline(0, color='blue', linestyle='--', alpha=0.5)
    ax.set_title(key, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time relative to event (s)', fontsize=12)
    if i == 0: ax.set_ylabel('Mean Firing Rate\n[Playback]', fontsize=12)

plt.suptitle(f'Chronological PSTH sEvolution ({n_per_group} triggers per group)', fontsize=16, y=1.02)
plt.tight_layout()  
plt.show()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CUSTOMIZABLE VARIABLES ---
n_per_group = 5    # Number of triggers per group
n_groups = 3        # Number of chronological groups per condition
# ==========================================

# Generate keys dynamically based on variables
tr_keys = []
for i in range(n_groups, 0, -1):
    start_idx = i * n_per_group
    end_idx = (i - 1) * n_per_group + 1
    tr_keys.append(f'Tracking: Last {start_idx}-{end_idx}')

pb_keys = []
for i in range(n_groups):
    start_idx = i * n_per_group + 1
    end_idx = (i + 1) * n_per_group
    pb_keys.append(f'Playback: First {start_idx}-{end_idx}')

all_keys = tr_keys + pb_keys
all_data = {key: [] for key in all_keys}

# Helper function to extract and average segments
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0)
    return None

# 2. Main Processing Loop
for n_data, f_data in zip(n_data_s, f_data_s):
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # --- Tracking: Slice chronological chunks from the end ---
    total_tr_needed = n_groups * n_per_group
    tr_chunks = []

    # Only process if the session has enough tracking events
    if len(events_tr) >= total_tr_needed:
        for i in range(n_groups, 0, -1):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            tr_chunks.append(events_tr[start:end])
    else:
        tr_chunks = [[] for _ in range(n_groups)] 

    for key, evs in zip(tr_keys, tr_chunks):
        if len(evs) > 0:
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                all_data[key].append(mean_trace)

    # --- Playback: Slice chronological chunks from the beginning ---
    total_pb_needed = n_groups * n_per_group
    pb_chunks = []

    # Only process if the session has enough playback events
    if len(events_pb) >= total_pb_needed:
        for i in range(n_groups):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            pb_chunks.append(events_pb[start:end])
    else:
        pb_chunks = [[] for _ in range(n_groups)]

    for key, evs in zip(pb_keys, pb_chunks):
        if len(evs) > 0:
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                all_data[key].append(mean_trace)

# 3. Plotting in a Dynamic Grid
# Create a 2 x n_groups grid (Top row: Tracking, Bottom row: Playback)
fig, axes = plt.subplots(1, n_groups * 2 , figsize=(20 * n_groups, 8), sharey=True, sharex=True)

# Ensure axes is a 2D array even if n_groups=1
axes = np.array(axes).reshape(2, n_groups)

# Generate dynamic colors (from light to dark)
tr_colors = [plt.cm.Reds(x) for x in np.linspace(0.4, 0.9, n_groups)]
pb_colors = [plt.cm.Greys(x) for x in np.linspace(0.4, 0.9, n_groups)]

# --- Plot Tracking (Top Row) ---
for i, (key, color) in enumerate(zip(tr_keys, tr_colors)):
    ax = axes[0, i]
    n_neurons = 0 # Default to 0 if no data
    if len(all_data[key]) > 0:
        final_dat = np.concatenate(all_data[key], axis=0)
        mean_dat = np.mean(final_dat, axis=0)
        sem_dat = np.std(final_dat, axis=0) / np.sqrt(final_dat.shape[0])
        n_neurons = final_dat.shape[0] # <--- Extract number of neurons here

        ax.plot(time, mean_dat, color=color, linewidth=2.5, label='Mean FR')
        ax.fill_between(time, mean_dat - sem_dat, mean_dat + sem_dat, color=color, alpha=0.15)

        # Add a text box directly inside the plot
        ax.text(0.05, 0.95, f'n = {n_neurons} neurons', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))

    ax.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Sound Onset' if i==0 else "")

    # Update the title to include the neuron count
    ax.set_title(f"{key}\n(n={n_neurons})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0: ax.set_ylabel('Mean Firing Rate\n[Tracking]', fontsize=12)
    ax.legend(loc='lower right') # Moved legend so it doesn't overlap the text box

# --- Plot Playback (Bottom Row) ---
for i, (key, color) in enumerate(zip(pb_keys, pb_colors)):
    ax = axes[1, i]
    n_neurons = 0
    if len(all_data[key]) > 0:
        final_dat = np.concatenate(all_data[key], axis=0)
        mean_dat = np.mean(final_dat, axis=0)
        sem_dat = np.std(final_dat, axis=0) / np.sqrt(final_dat.shape[0])
        n_neurons = final_dat.shape[0] # <--- Extract number of neurons here

        ax.plot(time, mean_dat, color=color, linewidth=2.5, label='Mean FR')
        ax.fill_between(time, mean_dat - sem_dat, mean_dat + sem_dat, color=color, alpha=0.15)

        # Add a text box directly inside the plot
        ax.text(0.05, 0.95, f'n = {n_neurons} neurons', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))

    ax.axvline(0, color='blue', linestyle='--', alpha=0.5)

    # Update the title to include the neuron count
    ax.set_title(f"{key}\n(n={n_neurons})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time relative to event (s)', fontsize=12)
    if i == 0: ax.set_ylabel('Mean Firing Rate\n[Playback]', fontsize=12)
    ax.legend(loc='lower right')

plt.suptitle(f'Chronological PSTH Evolution ({n_per_group} triggers per group)', fontsize=18, y=1.05)
plt.tight_layout()  
plt.show()


# In[26]:


# Set your thresholds (matching your last cell)
n_per_group = 5    
n_groups = 3        
total_needed = n_groups * n_per_group  # 15 events needed

print(f"--- Diagnostic: Checking for Session Mismatches (Threshold: {total_needed} events) ---\n")

# Keep track of totals
total_pb_neurons = 0
total_tr_neurons = 0
mismatched_sessions = []

for i, (n_data, f_data) in enumerate(zip(n_data_s, f_data_s)):
    # 1. Count the events
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # 2. Check against your threshold
    has_enough_pb = len(events_pb) >= total_needed
    has_enough_tr = len(events_tr) >= total_needed

    # Get session name safely (using your all_sessions_path variable from Cell 3)
    try:
        session_name = os.path.basename(all_sessions_path[i])
    except NameError:
        session_name = f"Session Index {i}"

    num_neurons = n_data.shape[0]

    # 3. Flag the mismatches
    if has_enough_pb and not has_enough_tr:
        print(f"⚠️ MISMATCH | {session_name}:")
        print(f"   -> Kept for Playback ({len(events_pb)} events)")
        print(f"   -> EXCLUDED for Tracking ({len(events_tr)} events)")
        print(f"   -> Result: +{num_neurons} neurons added to Playback ONLY.\n")
        mismatched_sessions.append(session_name)
        total_pb_neurons += num_neurons

    elif has_enough_tr and not has_enough_pb:
        print(f"⚠️ MISMATCH | {session_name}:")
        print(f"   -> Kept for Tracking ({len(events_tr)} events)")
        print(f"   -> EXCLUDED for Playback ({len(events_pb)} events)")
        print(f"   -> Result: +{num_neurons} neurons added to Tracking ONLY.\n")
        mismatched_sessions.append(session_name)
        total_tr_neurons += num_neurons

    # 4. Tally matches (for your own sanity check)
    elif has_enough_pb and has_enough_tr:
        total_pb_neurons += num_neurons
        total_tr_neurons += num_neurons

print("--- Summary ---")
print(f"Total mismatched sessions: {len(mismatched_sessions)}")
print(f"Final Expected Playback Neurons: {total_pb_neurons}")
print(f"Final Expected Tracking Neurons: {total_tr_neurons}")


# ## Z-scored Metrics of Groups of Triggers 

# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CUSTOMIZABLE VARIABLES ---
n_per_group = 100    # Number of triggers per group
n_groups = 3        # Number of chronological groups per condition
# ==========================================

total_needed = n_groups * n_per_group

# Define indices for calculating metrics
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Sound onset (time 0)
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Generate x-axis labels for the chunks
# Tracking: Last chunks (15-11, 10-6, 5-1) chronological
# Playback: First chunks (1-5, 6-10, 11-15) chronological
x_labels_tr = [f"TR: Last {i*n_per_group}-{(i-1)*n_per_group+1}" for i in range(n_groups, 0, -1)]
x_labels_pb = [f"PB: First {i*n_per_group+1}-{(i+1)*n_per_group}" for i in range(n_groups)]
all_x_labels = x_labels_tr + x_labels_pb

# Containers for the chunks
tr_data_chunks = [[] for _ in range(n_groups)]
pb_data_chunks = [[] for _ in range(n_groups)]

# Helper function
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0)
    return None

# 1. Main Processing Loop
for n_data, f_data in zip(n_data_s, f_data_s):
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # STRICT INCLUSION: Only process if the session has enough events for BOTH conditions
    if len(events_tr) >= total_needed and len(events_pb) >= total_needed:

        # --- Tracking Chunks (Chronological from the end) ---
        for chunk_idx, i in enumerate(range(n_groups, 0, -1)):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            evs = events_tr[start:end]
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                tr_data_chunks[chunk_idx].append(mean_trace)

        # --- Playback Chunks (Chronological from the start) ---
        for chunk_idx, i in enumerate(range(n_groups)):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            evs = events_pb[start:end]
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                pb_data_chunks[chunk_idx].append(mean_trace)

# 2. Compute Z-Scores and Metrics
def process_metrics(data_chunks):
    means = {'base': [], 'peak': [], 'evoked': []}
    sems = {'base': [], 'peak': [], 'evoked': []}
    n_neurons = 0

    for chunk in data_chunks:
        if len(chunk) > 0:
            final_dat = np.concatenate(chunk, axis=0)
            n_neurons = final_dat.shape[0]

            # Z-Score per neuron
            mean_time = np.mean(final_dat, axis=1, keepdims=True)
            std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
            z_dat = (final_dat - mean_time) / std_time

            # Extract Metrics
            base = np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
            peak = np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1)
            evoked = peak - base

            # Store Means and SEMs
            means['base'].append(np.mean(base))
            sems['base'].append(np.std(base) / np.sqrt(n_neurons))

            means['peak'].append(np.mean(peak))
            sems['peak'].append(np.std(peak) / np.sqrt(n_neurons))

            means['evoked'].append(np.mean(evoked))
            sems['evoked'].append(np.std(evoked) / np.sqrt(n_neurons))
        else:
            for k in means:
                means[k].append(np.nan)
                sems[k].append(np.nan)

    return means, sems, n_neurons

tr_means, tr_sems, n_tr_neurons = process_metrics(tr_data_chunks)
pb_means, pb_sems, n_pb_neurons = process_metrics(pb_data_chunks)

# 3. Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define sequential x-coordinates for Tracking (left) and Playback (right)
x_tr = np.arange(n_groups)
x_pb = np.arange(n_groups) + n_groups
all_x_vals = list(x_tr) + list(x_pb)
divider_x = n_groups - 0.5  # The midpoint between Tracking and Playback

metrics = ['base', 'peak', 'evoked']
titles = ['Baseline Z-Score (-0.2s to 0s)', 'Peak Response Z-Score', 'Total Evoked Response (Peak - Baseline)']

for i, metric in enumerate(metrics):
    ax = axes[i]

    # Plot Tracking (Red, Left side)
    tr_m = np.array(tr_means[metric])
    tr_s = np.array(tr_sems[metric])
    ax.errorbar(x_tr, tr_m, yerr=tr_s, color='red', marker='o', markersize=8, 
                linewidth=2.5, label=f'Tracking (n={n_tr_neurons})', capsize=5)

    # Plot Playback (Black, Right side)
    pb_m = np.array(pb_means[metric])
    pb_s = np.array(pb_sems[metric])
    ax.errorbar(x_pb, pb_m, yerr=pb_s, color='black', marker='o', markersize=8, 
                linewidth=2.5, label=f'Playback (n={n_pb_neurons})', capsize=5)

    # Formatting
    ax.axvline(x=divider_x, color='gray', linestyle='--', alpha=0.8, zorder=1) # Visual separator
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xticks(all_x_vals)
    ax.set_xticklabels(all_x_labels, rotation=45, ha='right') # Rotated for readability
    ax.grid(True, alpha=0.3)

    if i == 0:
        ax.set_ylabel('Mean Z-Score', fontsize=12)
    ax.legend(loc='best')

plt.suptitle(f'Chronological Z-Score Metrics ({n_per_group} triggers/chunk)', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CUSTOMIZABLE VARIABLES ---
n_per_group = 100    # Number of triggers per group
n_groups = 3       # Number of chronological groups per condition
# ==========================================

total_needed = n_groups * n_per_group

# Define indices for calculating metrics
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Sound onset (time 0)
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Generate x-axis labels
x_labels_tr = [f"TR: Last {i*n_per_group}-{(i-1)*n_per_group+1}" for i in range(n_groups, 0, -1)]
x_labels_pb = [f"PB: First {i*n_per_group+1}-{(i+1)*n_per_group}" for i in range(n_groups)]
all_x_labels = x_labels_tr + x_labels_pb

# Containers for the chunks
tr_data_chunks = [[] for _ in range(n_groups)]
pb_data_chunks = [[] for _ in range(n_groups)]

# Helper function
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) # Average over the 5 events, keep channels
    return None

# 1. Main Processing Loop (Transition-Based)
for n_data, f_data in zip(n_data_s, f_data_s):
    # Get all valid frequency change events and their conditions
    events_all = np.flatnonzero(f_data['Frequency_changes'] == 1)
    if len(events_all) == 0:
        continue

    conds_all = f_data['Condition'].values[events_all]

    # Find exact indices where condition changes from 0 (Tracking) to 1 (Playback)
    transition_indices = np.where((conds_all[:-1] == 0) & (conds_all[1:] == 1))[0]

    for idx in transition_indices:
        # Extract contiguous tracking events BEFORE the transition
        tr_evs = []
        for j in range(idx, -1, -1):
            if conds_all[j] == 0:
                tr_evs.append(events_all[j])
            else:
                break
        tr_evs = tr_evs[::-1] # Reverse to chronological order

        # Extract contiguous playback events AFTER the transition
        pb_evs = []
        for j in range(idx + 1, len(conds_all)):
            if conds_all[j] == 1:
                pb_evs.append(events_all[j])
            else:
                break

        # Only process THIS transition if it has enough events on both sides
        if len(tr_evs) >= total_needed and len(pb_evs) >= total_needed:

            # --- Tracking Chunks (Chronological from the end of the tracking block) ---
            for chunk_idx, i in enumerate(range(n_groups, 0, -1)):
                start = -i * n_per_group
                end = -(i - 1) * n_per_group if i > 1 else None
                evs = tr_evs[start:end]
                mean_trace = get_subset_channel_means(n_data, evs)
                if mean_trace is not None:
                    tr_data_chunks[chunk_idx].append(mean_trace)

            # --- Playback Chunks (Chronological from the start of the playback block) ---
            for chunk_idx, i in enumerate(range(n_groups)):
                start = i * n_per_group
                end = (i + 1) * n_per_group
                evs = pb_evs[start:end]
                mean_trace = get_subset_channel_means(n_data, evs)
                if mean_trace is not None:
                    pb_data_chunks[chunk_idx].append(mean_trace)

# 2. Compute Z-Scores and Metrics
def process_metrics(data_chunks):
    means = {'base': [], 'peak': [], 'evoked': []}
    sems = {'base': [], 'peak': [], 'evoked': []}
    n_samples = 0

    for chunk in data_chunks:
        if len(chunk) > 0:
            # Concatenate along axis=0. Shape = (n_channels * valid_transitions, expected_length)
            final_dat = np.concatenate(chunk, axis=0)
            n_samples = final_dat.shape[0] # N = Transitions * Channels

            # Z-Score independently per sample
            mean_time = np.mean(final_dat, axis=1, keepdims=True)
            std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
            z_dat = (final_dat - mean_time) / std_time

            # Extract Metrics
            base = np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
            peak = np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1)
            evoked = peak - base

            # Store Means and SEMs
            means['base'].append(np.mean(base))
            sems['base'].append(np.std(base) / np.sqrt(n_samples))

            means['peak'].append(np.mean(peak))
            sems['peak'].append(np.std(peak) / np.sqrt(n_samples))

            means['evoked'].append(np.mean(evoked))
            sems['evoked'].append(np.std(evoked) / np.sqrt(n_samples))
        else:
            for k in means:
                means[k].append(np.nan)
                sems[k].append(np.nan)

    return means, sems, n_samples

tr_means, tr_sems, n_tr_samples = process_metrics(tr_data_chunks)
pb_means, pb_sems, n_pb_samples = process_metrics(pb_data_chunks)

# 3. Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

x_tr = np.arange(n_groups)
x_pb = np.arange(n_groups) + n_groups
all_x_vals = list(x_tr) + list(x_pb)
divider_x = n_groups - 0.5 

metrics = ['base', 'peak', 'evoked']
titles = ['Baseline Z-Score (-0.2s to 0s)', 'Peak Response Z-Score', 'Total Evoked Response (Peak - Baseline)']

for i, metric in enumerate(metrics):
    ax = axes[i]

    # Plot Tracking (Red, Left side)
    tr_m = np.array(tr_means[metric])
    tr_s = np.array(tr_sems[metric])
    ax.errorbar(x_tr, tr_m, yerr=tr_s, color='red', marker='o', markersize=8, 
                linewidth=2.5, label=f'Tracking', capsize=5)

    # Plot Playback (Black, Right side)
    pb_m = np.array(pb_means[metric])
    pb_s = np.array(pb_sems[metric])
    ax.errorbar(x_pb, pb_m, yerr=pb_s, color='black', marker='o', markersize=8, 
                linewidth=2.5, label=f'Playback', capsize=5)

    # Formatting
    ax.axvline(x=divider_x, color='gray', linestyle='--', alpha=0.8, zorder=1) 
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xticks(all_x_vals)
    ax.set_xticklabels(all_x_labels, rotation=45, ha='right') 
    ax.grid(True, alpha=0.3)

    # Add N annotation dynamically inside each plot
    ax.text(0.05, 0.95, f'N = {n_pb_samples}\n(Channels × Transitions)', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    if i == 0:
        ax.set_ylabel('Mean Z-Score', fontsize=12)
    ax.legend(loc='lower left')

plt.suptitle(f'Chronological Z-Score Metrics Anchored to Local Block Transitions', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()


# ## PSTH of triggers in Last n Tracking and First n Playback

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Paramètres ##
t_pre = 0.3   # temps avant trigger (s)
t_post = 0.3  # temps après trigger (s)
n_first = 25  # Nombre de triggers à utiliser (first/last)

# Remarque : assurez-vous que 'dt' est bien défini dans votre environnement (ex: dt = 0.005)

def get_psth_last_n_triggers_tracking(data, f_data, t_pre, t_post, dt, n_last=10):
    """
    Calcule le PSTH moyen autour des n derniers triggers pour CHAQUE block,
    uniquement pour les trials Tracking (condition == 0).

    Returns:
        block_psths : dict {block_id: array(neurones x bins)}
        time_axis : temps relatif au trigger
    """
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    # --- Indices des triggers de Tracking ---
    trigger_idx = np.where((frequency_changes == 1) & (condition == 0))[0]

    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            last_triggers = b_triggers[-n_last:]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in last_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            # --- Moyenne par neurone POUR CE BLOCK ---
            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


def get_psth_first_n_triggers_playback(data, f_data, t_pre, t_post, dt, n_first=10):
    """
    Calcule le PSTH moyen autour des n premiers triggers pour CHAQUE block,
    uniquement pour les trials Playback (condition == 1).

    Returns:
        block_psths : dict {block_id: array(neurones x bins)}
        time_axis : temps relatif au trigger
    """
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    # --- Indices des triggers de Playback ---
    trigger_idx = np.where((frequency_changes == 1) & (condition == 1))[0]

    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            first_triggers = b_triggers[:n_first]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in first_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            # --- Moyenne par neurone POUR CE BLOCK ---
            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


# ==========================================================
# Exécution Principale
# ==========================================================

all_playback, all_tracking = [], []

# --- Collecte des PSTHs par transition Tracking -> Playback ---
for n_data, f_data in zip(n_data_s, f_data_s):
    # Dictionnaires contenant les PSTHs calculés par block
    dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_first)
    dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_first)

    # Identifier les transitions en regardant l'ordre chronologique des blocks
    blocks_in_order = f_data['Block'].unique()

    for i in range(len(blocks_in_order) - 1):
        b_curr = blocks_in_order[i]
        b_next = blocks_in_order[i+1]

        # Si le block courant est dans notre dictionnaire Tracking 
        # ET que le block suivant est dans notre dictionnaire Playback : c'est une transition !
        if b_curr in dict_tr and b_next in dict_pb:
            all_tracking.append(dict_tr[b_curr])
            all_playback.append(dict_pb[b_next])

if len(all_tracking) > 0 and len(all_playback) > 0:
    # --- Convertir en array et empiler ---
    # np.vstack empile les tableaux de taille (n_canaux, bins). 
    # Le résultat est de taille (n_canaux * transitions, bins)
    tracking_arr = np.vstack(all_tracking)  
    playback_arr = np.vstack(all_playback)

    # N = Nombre total de canaux multiplié par le nombre de transitions identifiées
    n_total_samples = tracking_arr.shape[0]

    # --- Moyenne et SEM (en calculant sur l'axe 0 = échantillons totaux) ---
    tracking_mean = np.nanmean(tracking_arr, axis=0)
    tracking_sem  = np.nanstd(tracking_arr, axis=0) / np.sqrt(n_total_samples)

    playback_mean = np.nanmean(playback_arr, axis=0)
    playback_sem  = np.nanstd(playback_arr, axis=0) / np.sqrt(n_total_samples)

    # --- Plot ---
    plt.figure(figsize=(8, 6))

    # Tracking
    plt.plot(psth_bins, tracking_mean, c='red', linewidth=2, label=f'Tracking (last {n_first} triggers)')
    plt.fill_between(psth_bins, tracking_mean - tracking_sem, tracking_mean + tracking_sem, color='red', alpha=0.2)

    # Playback
    plt.plot(psth_bins, playback_mean, c='black', linewidth=2, label=f'Playback (first {n_first} triggers)')
    plt.fill_between(psth_bins, playback_mean - playback_sem, playback_mean + playback_sem, color='black', alpha=0.2)

    # Esthétique
    plt.axvline(0, color='gray', linestyle='--', alpha=0.8)
    plt.xlabel('Time relative to event (s)', fontsize=12)
    plt.ylabel('Firing Rate (spk/s)', fontsize=12)
    plt.legend(frameon=False)
    plt.title('PSTH Tracking vs Playback\nAnchored to Block Transitions', fontsize=14)

    # Retirer les axes du haut et de droite
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ajouter explicitement le 'N' calculé dans un encadré
    ax.text(0.05, 0.95, f'N = {n_total_samples}\n(Channels × Transitions)', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.show()
else:
    print("Aucune transition Tracking -> Playback trouvée avec ces critères.")


# In[30]:


import numpy as np
import matplotlib.pyplot as plt

## Paramètres ##
t_pre = 0.3   # temps avant trigger (s)
t_post = 0.3  # temps après trigger (s)

# Liste des puissances de 2 : [1, 2, 4, 8, 16, 32, 64, 128, 256]
n_values = [2**i for i in range(10)]

def get_psth_last_n_triggers_tracking(data, f_data, t_pre, t_post, dt, n_last=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 0))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            last_triggers = b_triggers[-n_last:]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in last_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


def get_psth_first_n_triggers_playback(data, f_data, t_pre, t_post, dt, n_first=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 1))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            first_triggers = b_triggers[:n_first]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in first_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


# ==========================================================
# Exécution Principale - Grille de Subplots
# ==========================================================

# Création de la figure avec une rangée et autant de colonnes que de valeurs de n
fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 5), sharey=True)

# S'assurer que axes est itérable si on n'a qu'une seule valeur (au cas où)
if len(n_values) == 1:
    axes = [axes]

for ax_idx, n_first in enumerate(n_values):
    all_playback, all_tracking = [], []
    ax = axes[ax_idx]

    # Collecte des PSTHs par transition pour le n_first courant
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_first)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_first)

        blocks_in_order = f_data['Block'].unique()

        for i in range(len(blocks_in_order) - 1):
            b_curr = blocks_in_order[i]
            b_next = blocks_in_order[i+1]

            if b_curr in dict_tr and b_next in dict_pb:
                all_tracking.append(dict_tr[b_curr])
                all_playback.append(dict_pb[b_next])

    if len(all_tracking) > 0 and len(all_playback) > 0:
        # Conversion et moyenne
        tracking_arr = np.vstack(all_tracking)  
        playback_arr = np.vstack(all_playback)
        n_total_samples = tracking_arr.shape[0]

        tracking_mean = np.nanmean(tracking_arr, axis=0)
        tracking_sem  = np.nanstd(tracking_arr, axis=0) / np.sqrt(n_total_samples)

        playback_mean = np.nanmean(playback_arr, axis=0)
        playback_sem  = np.nanstd(playback_arr, axis=0) / np.sqrt(n_total_samples)

        # Plot Tracking (Rouge)
        ax.plot(psth_bins, tracking_mean, c='red', linewidth=2, label='Tracking')
        ax.fill_between(psth_bins, tracking_mean - tracking_sem, tracking_mean + tracking_sem, color='red', alpha=0.2)

        # Plot Playback (Noir)
        ax.plot(psth_bins, playback_mean, c='black', linewidth=2, label='Playback')
        ax.fill_between(psth_bins, playback_mean - playback_sem, playback_mean + playback_sem, color='black', alpha=0.2)

        # Esthétique locale du subplot
        ax.axvline(0, color='gray', linestyle='--', alpha=0.8)
        ax.set_title(f'n = {n_first}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)

        # Le label 'Y' uniquement sur le graphique le plus à gauche
        if ax_idx == 0:
            ax.set_ylabel('Firing Rate (spk/s)', fontsize=12)

        # Ajouter N dans le subplot
        ax.text(0.05, 0.95, f'N = {n_total_samples}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    else:
        ax.set_title(f'n = {n_first}\n(No data)', fontsize=12)

    # Retirer les cadres superflus
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# On place la légende sur le dernier graphique pour ne pas surcharger
axes[-1].legend(frameon=False, loc='upper right')

plt.suptitle('PSTH Last Trackings vs First Playbacks across n Triggers', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

## Paramètres ##
t_pre = 0.3   # temps avant trigger (s)
t_post = 0.3  # temps après trigger (s)

# Liste des puissances de 2 : [1, 2, 4, 8, 16, 32, 64, 128, 256]
n_values = [2**i for i in range(10)]

# ==========================================================
# 0. DATA SANITIZATION (Safe Removal of Condition -1.0)
# ==========================================================
# We neutralize Condition -1.0 instead of dropping rows to preserve 
# the 1-to-1 index alignment with the n_data timebins.
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False  # Changed from 0 to False
        # Remove the condition label entirely
# ==========================================================

def get_psth_last_n_triggers_tracking(data, f_data, t_pre, t_post, dt, n_last=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    # La condition -1.0 étant neutralisée, ceci ne trouvera que les vrais 0
    trigger_idx = np.where((frequency_changes == 1) & (condition == 0))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            last_triggers = b_triggers[-n_last:]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in last_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


def get_psth_first_n_triggers_playback(data, f_data, t_pre, t_post, dt, n_first=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 1))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            first_triggers = b_triggers[:n_first]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in first_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


# ==========================================================
# Exécution Principale - Grille de Subplots
# ==========================================================

# 1. Structure pour stocker les données
data_collection = {}

# 2. Collecte des données
for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Chercher tous les blocs réels de cette session qui ont À LA FOIS Tracking et Playback
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

# 3. Création de la figure
unique_blocks = sorted(data_collection.keys())
n_rows = len(unique_blocks)
n_cols = len(n_values)

if n_rows == 0:
    print("Aucun bloc mixte (contenant Tracking ET Playback) n'a été trouvé.")
else:
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(3.5 * n_cols, 3 * n_rows), 
                             sharey='row', sharex=True)

    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    for row_idx, actual_b in enumerate(unique_blocks):
        for col_idx, n_val in enumerate(n_values):
            ax = axes[row_idx, col_idx]

            tr_list = data_collection[actual_b][n_val]['tr']
            pb_list = data_collection[actual_b][n_val]['pb']

            if len(tr_list) > 0 and len(pb_list) > 0:
                tracking_arr = np.vstack(tr_list)
                playback_arr = np.vstack(pb_list)
                n_total_samples = tracking_arr.shape[0]

                tracking_mean = np.nanmean(tracking_arr, axis=0)
                tracking_sem  = np.nanstd(tracking_arr, axis=0) / np.sqrt(n_total_samples)

                playback_mean = np.nanmean(playback_arr, axis=0)
                playback_sem  = np.nanstd(playback_arr, axis=0) / np.sqrt(n_total_samples)

                # Trace les courbes
                ax.plot(psth_bins, tracking_mean, c='red', linewidth=2)
                ax.fill_between(psth_bins, tracking_mean - tracking_sem, tracking_mean + tracking_sem, color='red', alpha=0.2)

                ax.plot(psth_bins, playback_mean, c='black', linewidth=2)
                ax.fill_between(psth_bins, playback_mean - playback_sem, playback_mean + playback_sem, color='black', alpha=0.2)

                ax.text(0.05, 0.95, f'N = {n_total_samples}', 
                        transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')

            # Esthétique de base
            ax.axvline(0, color='gray', linestyle='--', alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # --- Formatting de la grille ---
            if row_idx == 0:
                ax.set_title(f'n = {n_val}', fontsize=14, fontweight='bold')

            if row_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=12)

            if col_idx == 0:
                ax.set_ylabel(f'Block {actual_b}\nFR (spk/s)', fontsize=12, fontweight='bold')

    # Ajout d'une légende globale
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='black', lw=2)]
    fig.legend(custom_lines, ['Tracking (Last n)', 'Playback (First n)'], 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=14)

    plt.suptitle('PSTH Evolution Across Blocks (Rows) and Trigger Quantity (Columns)', fontsize=18, y=1.08)
    plt.tight_layout()
    plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt

## Paramètres ##
t_pre = 0.3   # temps avant trigger (s)
t_post = 0.3  # temps après trigger (s)

# Only n = 32
n_values = [32]

# ==========================================================
# 0. DATA SANITIZATION (Safe Removal of Condition -1.0)
# ==========================================================
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False
# ==========================================================

def get_psth_last_n_triggers_tracking(data, f_data, t_pre, t_post, dt, n_last=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 0))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            last_triggers = b_triggers[-n_last:]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in last_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


def get_psth_first_n_triggers_playback(data, f_data, t_pre, t_post, dt, n_first=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 1))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            first_triggers = b_triggers[:n_first]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in first_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


# ==========================================================
# Exécution Principale - Grille Horizontale (sans bloc 0)
# ==========================================================

data_collection = {}

# Collecte des données (uniquement pour n = 32)
for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n_val: {'tr': [], 'pb': []}}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

# === EXCLUSION DU BLOC 0 ===
unique_blocks = sorted(data_collection.keys())
unique_blocks = [b for b in unique_blocks if b != 0]   # on retire le bloc 0

n_blocks = len(unique_blocks)

if n_blocks == 0:
    print("Aucun bloc (autre que 0) contenant Tracking ET Playback n'a été trouvé.")
else:
    # Création de la figure horizontale : 1 ligne, n_blocks colonnes
    fig, axes = plt.subplots(1, n_blocks, 
                             figsize=(4 * n_blocks, 4), 
                             sharey='row', sharex=True,
                             squeeze=False)

    for col_idx, actual_b in enumerate(unique_blocks):
        ax = axes[0, col_idx]

        tr_list = data_collection[actual_b][32]['tr']
        pb_list = data_collection[actual_b][32]['pb']

        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)
            n_total_samples = tracking_arr.shape[0]

            tracking_mean = np.nanmean(tracking_arr, axis=0)
            tracking_sem  = np.nanstd(tracking_arr, axis=0) / np.sqrt(n_total_samples)

            playback_mean = np.nanmean(playback_arr, axis=0)
            playback_sem  = np.nanstd(playback_arr, axis=0) / np.sqrt(n_total_samples)

            ax.plot(psth_bins, tracking_mean, c='red', linewidth=2)
            ax.fill_between(psth_bins, tracking_mean - tracking_sem, tracking_mean + tracking_sem, color='red', alpha=0.2)

            ax.plot(psth_bins, playback_mean, c='black', linewidth=2)
            ax.fill_between(psth_bins, playback_mean - playback_sem, playback_mean + playback_sem, color='black', alpha=0.2)

            ax.text(0.05, 0.95, f'N = {n_total_samples}', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')

        # Esthétique de base
        ax.axvline(0, color='gray', linestyle='--', alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Étiquetage ---
        ax.set_title(f'Block {actual_b}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)

        if col_idx == 0:
            ax.set_ylabel('FR (spk/s)', fontsize=12, fontweight='bold')

    # Légende globale
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='black', lw=2)]
    fig.legend(custom_lines, ['Tracking (last 32)', 'Playback (first 32)'], 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=14)

    plt.suptitle('PSTH for n = 32 triggers (excluding block 0)', fontsize=18, y=1.08)
    plt.tight_layout()
    plt.show()


# ### Z-scored (n_triggers defined here)

# In[33]:


import numpy as np
import matplotlib.pyplot as plt

## Paramètres ##
t_pre = 0.3   # temps avant trigger (s)
t_post = 0.3  # temps après trigger (s)

# Seulement n = 32
n_values = [16]

# ==========================================================
# 0. DATA SANITIZATION (Safe Removal of Condition -1.0)
# ==========================================================
# We neutralize Condition -1.0 instead of dropping rows to preserve 
# the 1-to-1 index alignment with the n_data timebins.
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False  # Changed from 0 to False
        # Remove the condition label entirely
# ==========================================================

def get_psth_last_n_triggers_tracking(data, f_data, t_pre, t_post, dt, n_last=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    # La condition -1.0 étant neutralisée, ceci ne trouvera que les vrais 0
    trigger_idx = np.where((frequency_changes == 1) & (condition == 0))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            last_triggers = b_triggers[-n_last:]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in last_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


def get_psth_first_n_triggers_playback(data, f_data, t_pre, t_post, dt, n_first=10):
    frequency_changes = f_data['Frequency_changes'].values
    condition = f_data['Condition'].values
    block = f_data['Block'].values
    n_neurons, n_bins = data.shape

    n_bins_pre = int(t_pre / dt)
    n_bins_post = int(t_post / dt)

    trigger_idx = np.where((frequency_changes == 1) & (condition == 1))[0]
    unique_blocks = np.unique(block[trigger_idx])
    block_psths = {}

    for b in unique_blocks:
        b_triggers = trigger_idx[block[trigger_idx] == b]
        if len(b_triggers) > 0:
            first_triggers = b_triggers[:n_first]

            psth_all = np.zeros((n_neurons, n_bins_pre + n_bins_post))
            n_segments = 0

            for idx in first_triggers:
                start = int(idx - n_bins_pre)
                end = int(idx + n_bins_post)
                if start >= 0 and end <= n_bins:
                    psth_all += data[:, start:end]
                    n_segments += 1

            if n_segments > 0:
                block_psths[b] = psth_all / n_segments

    time_axis = np.arange(-n_bins_pre, n_bins_post) * dt
    return block_psths, time_axis


# ==========================================================
# Exécution Principale - Grille de Subplots
# ==========================================================

# 1. Structure pour stocker les données
data_collection = {}

# 2. Collecte des données
for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Chercher tous les blocs réels (SAUF LE BLOC 0) qui ont À LA FOIS Tracking et Playback
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb and b != 0])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

# 3. Création de la figure
unique_blocks = sorted(data_collection.keys())

# --- Changement d'orientation de la grille ---
n_rows = len(n_values)  # Sera de 1 car n_values = [32]
n_cols = len(unique_blocks) # Les blocs sont maintenant en colonnes

if n_cols == 0:
    print("Aucun bloc mixte (contenant Tracking ET Playback et différent de 0) n'a été trouvé.")
else:
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(3.5 * n_cols, 4), # Hauteur fixée, largeur adaptative
                             sharey=True, sharex=True)

    # Sécuriser le formatage du tableau d'axes (2D array)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Inversion de la boucle pour s'adapter à la nouvelle orientation
    for col_idx, actual_b in enumerate(unique_blocks):
        for row_idx, n_val in enumerate(n_values):
            ax = axes[row_idx, col_idx]

            tr_list = data_collection[actual_b][n_val]['tr']
            pb_list = data_collection[actual_b][n_val]['pb']

            if len(tr_list) > 0 and len(pb_list) > 0:
                tracking_arr = np.vstack(tr_list)
                playback_arr = np.vstack(pb_list)
                n_total_samples = tracking_arr.shape[0]

                tracking_mean = np.nanmean(tracking_arr, axis=0)
                tracking_sem  = np.nanstd(tracking_arr, axis=0) / np.sqrt(n_total_samples)

                playback_mean = np.nanmean(playback_arr, axis=0)
                playback_sem  = np.nanstd(playback_arr, axis=0) / np.sqrt(n_total_samples)

                # Trace les courbes
                ax.plot(psth_bins, tracking_mean, c='red', linewidth=2)
                ax.fill_between(psth_bins, tracking_mean - tracking_sem, tracking_mean + tracking_sem, color='red', alpha=0.2)

                ax.plot(psth_bins, playback_mean, c='black', linewidth=2)
                ax.fill_between(psth_bins, playback_mean - playback_sem, playback_mean + playback_sem, color='black', alpha=0.2)

                ax.text(0.05, 0.95, f'N = {n_total_samples}', 
                        transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')

            # Esthétique de base
            ax.axvline(0, color='gray', linestyle='--', alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # --- Formatting de la grille ---
            # Le titre devient le numéro de bloc
            if row_idx == 0:
                ax.set_title(f'Block {actual_b}', fontsize=14, fontweight='bold')

            # L'axe X s'applique à tous (car une seule ligne)
            if row_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=12)

            # L'axe Y s'applique uniquement à la première colonne
            if col_idx == 0:
                ax.set_ylabel('FR (spk/s)', fontsize=12, fontweight='bold')

    # Ajout d'une légende globale
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='black', lw=2)]
    fig.legend(custom_lines, ['Tracking (Last 32)', 'Playback (First 32)'], 
               loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=14)

    plt.suptitle('PSTH Comparison: Tracking vs Playback Across Blocks (n=32)', fontsize=16, y=1.18)
    plt.tight_layout()
    plt.show()


# ### Quantifying Block groupped Tracking-Playback

# #### Firing Rate

# In[34]:


import pandas as pd
import numpy as np

# Define time windows based on your dt
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Event trigger (time 0)
idx_peak_start = n_bins_pre
idx_peak_end = n_bins_pre + int(t_post / dt)

quantified_results = []

# Loop through the data_collection
for block_id, nt_dict in sorted(data_collection.items()):

    # SAFER ITERATION: Loop directly through the keys inside the dictionary
    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        # Only process if both conditions have data for this 'n_val'
        if len(tr_list) > 0 and len(pb_list) > 0:
            # Stack the trial averages
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)

            n_samples = tracking_arr.shape[0]

            # --- Calculate Tracking Metrics (Raw Firing Rate) ---
            # If your arrays are in "spikes per bin", you can divide by dt here to get spk/s 
            # e.g., tr_base = np.mean(tracking_arr[:, idx_base_start:idx_base_end], axis=1) / dt
            tr_base = np.mean(tracking_arr[:, idx_base_start:idx_base_end], axis=1)
            tr_peak = np.max(tracking_arr[:, idx_peak_start:idx_peak_end], axis=1)
            tr_evoked = tr_peak - tr_base

            # --- Calculate Playback Metrics (Raw Firing Rate) ---
            pb_base = np.mean(playback_arr[:, idx_base_start:idx_base_end], axis=1)
            pb_peak = np.max(playback_arr[:, idx_peak_start:idx_peak_end], axis=1)
            pb_evoked = pb_peak - pb_base

            # Append to our list
            quantified_results.append({
                'Block': block_id,
                'n_triggers': n_val,
                'N_samples': n_samples,

                'TR_Base_Mean': np.mean(tr_base),
                'TR_Base_SEM': np.std(tr_base) / np.sqrt(n_samples),
                'TR_Peak_Mean': np.mean(tr_peak),
                'TR_Peak_SEM': np.std(tr_peak) / np.sqrt(n_samples),
                'TR_Evoked_Mean': np.mean(tr_evoked),
                'TR_Evoked_SEM': np.std(tr_evoked) / np.sqrt(n_samples),

                'PB_Base_Mean': np.mean(pb_base),
                'PB_Base_SEM': np.std(pb_base) / np.sqrt(n_samples),
                'PB_Peak_Mean': np.mean(pb_peak),
                'PB_Peak_SEM': np.std(pb_peak) / np.sqrt(n_samples),
                'PB_Evoked_Mean': np.mean(pb_evoked),
                'PB_Evoked_SEM': np.std(pb_evoked) / np.sqrt(n_samples)
            })

# Convert to DataFrame
df_quantified = pd.DataFrame(quantified_results)

# Display the top rows
display(df_quantified.head(10))


# In[35]:


import matplotlib.pyplot as plt
import numpy as np

# Get unique blocks and sort them to ensure chronological order
unique_blocks = sorted(df_quantified['Block'].unique())
n_blocks = len(unique_blocks)

# Updated Titles for Z-Scored Metrics
metrics_to_plot = [
    ('Base', 'Baseline FR\n(-0.2s to 0s)'),
    ('Peak', 'Peak Response\nFR'),
    ('Evoked', 'Evoked FR\n(Peak - Base)')
]

# Create a figure: rows = 3 metrics, cols = n_blocks
fig, axes = plt.subplots(3, n_blocks, figsize=(4 * n_blocks, 10), sharex=0, sharey='row')

# Safety check: ensure axes is a 2D array even if there is only 1 block
if n_blocks == 1:
    axes = np.expand_dims(axes, axis=1)

for col_idx, block in enumerate(unique_blocks):
    # Filter data for the current block and sort by trigger quantity
    df_block = df_quantified[df_quantified['Block'] == block].sort_values('n_triggers')
    x_vals = df_block['n_triggers'].values

    for row_idx, (metric_prefix, metric_title) in enumerate(metrics_to_plot):
        ax = axes[row_idx, col_idx]

        # Extract Tracking data
        tr_mean = df_block[f'TR_{metric_prefix}_Mean'].values
        tr_sem = df_block[f'TR_{metric_prefix}_SEM'].values

        # Extract Playback data
        pb_mean = df_block[f'PB_{metric_prefix}_Mean'].values
        pb_sem = df_block[f'PB_{metric_prefix}_SEM'].values

        # Plot Tracking (Red)
        ax.errorbar(x_vals, tr_mean, yerr=tr_sem, color='red', marker='o', 
                    linewidth=2.5, markersize=6, label='Tracking', capsize=4)

        # Plot Playback (Black)
        ax.errorbar(x_vals, pb_mean, yerr=pb_sem, color='black', marker='o', 
                    linewidth=2.5, markersize=6, label='Playback', capsize=4)

        # Formatting
        ax.set_xscale('log', base=2)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals, rotation=45)
        ax.grid(True, alpha=0.3)

        # Setup titles and labels dynamically based on grid position
        if row_idx == 0:
            # Top row gets the Block titles
            ax.set_title(f'Block {block}', fontsize=14, fontweight='bold')

        if col_idx == 0:
            # First column gets the Metric Y-axis labels
            ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')

        if row_idx == 2:
            # Bottom row gets the X-axis labels
            ax.set_xlabel('Number of Triggers (Log Scale)', fontsize=12)

        # Add legend only to the top right plot to avoid clutter
        if row_idx == 0 and col_idx == n_blocks - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, fontsize=12)

# Updated Main Title
plt.suptitle('#Metrics Evolution: Blocks (Columns) vs Metrics (Rows)', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()


# #### Z-Scored

# In[36]:


import pandas as pd
import numpy as np

# Define time windows based on your dt
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Event trigger (time 0)
idx_peak_start = n_bins_pre
idx_peak_end = n_bins_pre + int(t_post / dt)

raw_results = []

# Loop through the data_collection
for block_id, nt_dict in sorted(data_collection.items()):

    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        # Only process if both conditions have data for this 'n_val'
        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)

            # ==========================================
            # --- Z-SCORE TRANSFORMATION PER SAMPLE ---
            # ==========================================
            tr_mean_time = np.mean(tracking_arr, axis=1, keepdims=True)
            tr_std_time = np.std(tracking_arr, axis=1, keepdims=True) + 1e-8
            tracking_z = (tracking_arr - tr_mean_time) / tr_std_time

            pb_mean_time = np.mean(playback_arr, axis=1, keepdims=True)
            pb_std_time = np.std(playback_arr, axis=1, keepdims=True) + 1e-8
            playback_z = (playback_arr - pb_mean_time) / pb_std_time

            # ==========================================
            # --- CALCULATE METRICS PER SAMPLE ---
            # ==========================================
            # Notice we are NO LONGER taking np.mean() over axis=0.
            # These are now arrays of shape (n_samples,)
            tr_base = np.mean(tracking_z[:, idx_base_start:idx_base_end], axis=1)
            tr_peak = np.max(tracking_z[:, idx_peak_start:idx_peak_end], axis=1)
            tr_evoked = tr_peak - tr_base

            pb_base = np.mean(playback_z[:, idx_base_start:idx_base_end], axis=1)
            pb_peak = np.max(playback_z[:, idx_peak_start:idx_peak_end], axis=1)
            pb_evoked = pb_peak - pb_base

            # --- Store individual samples for Tracking ---
            df_tr = pd.DataFrame({
                'Block': block_id,
                'n_triggers': n_val,
                'Condition': 'Tracking',
                'Base_Z': tr_base,
                'Peak_Z': tr_peak,
                'Evoked_Z': tr_evoked
            })

            # --- Store individual samples for Playback ---
            df_pb = pd.DataFrame({
                'Block': block_id,
                'n_triggers': n_val,
                'Condition': 'Playback',
                'Base_Z': pb_base,
                'Peak_Z': pb_peak,
                'Evoked_Z': pb_evoked
            })

            # Add to master list
            raw_results.extend([df_tr, df_pb])

# Combine into a single long-format DataFrame!
df_raw = pd.concat(raw_results, ignore_index=True)

# Display a preview
display(df_raw.head())


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Setup Data for Plotting
# Exclude Block 0 and pick the max_triggers for a clean block-to-block comparison
df_plot = df_raw[df_raw['Block'] != 0.0].copy()
max_trig = df_plot['n_triggers'].max()
df_plot = df_plot[df_plot['n_triggers'] == max_trig]

metrics = [
    ('Base_Z', 'Baseline Z-Score\n(-0.2s to 0s)'),
    ('Peak_Z', 'Peak Response\nZ-Score'),
    ('Evoked_Z', 'Evoked Z-Score\n(Peak - Base)')
]

# 2. Create Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=0)

# Get sorted unique blocks so our x-indices match Seaborn's internal plotting order
unique_blocks = sorted(df_plot['Block'].unique())

for i, (col_name, title) in enumerate(metrics):
    ax = axes[i]

    # A. Draw the Box Plot (IQR, medians, and outliers)
    sns.boxplot(
        data=df_plot, x='Block', y=col_name, hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4,
        boxprops=dict(alpha=0.7)
    )

    # B. Add Sample Size (N) Text above each box
    # Grab current y-limits and expand the top by 10% to fit the text
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.1)

    for x_idx, block in enumerate(unique_blocks):
        # -- Tracking Text --
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]
        if not tr_data.empty:
            n_tr = len(tr_data)
            # Find the max value (outlier or top whisker) to place text right above it
            y_pos_tr = tr_data[col_name].max() + (y_range * 0.02)
            # -0.2 x-offset aligns with Seaborn's left-shifted 'hue' box
            ax.text(x_idx - 0.2, y_pos_tr, f'n={n_tr}', ha='center', va='bottom', 
                    fontsize=9, color='darkred', fontweight='bold')

        # -- Playback Text --
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]
        if not pb_data.empty:
            n_pb = len(pb_data)
            y_pos_pb = pb_data[col_name].max() + (y_range * 0.02)
            # +0.2 x-offset aligns with Seaborn's right-shifted 'hue' box
            ax.text(x_idx + 0.2, y_pos_pb, f'n={n_pb}', ha='center', va='bottom', 
                    fontsize=9, color='black', fontweight='bold')

    # C. Formatting
    ax.set_ylabel(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Manage legends
    if i == 0:
        ax.set_title(f'Tracking-Playback Transition Across Blocks (n_triggers={max_trig})', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', frameon=True, fontsize=11)
    else:
        # Remove legends from subsequent plots to keep it clean
        ax.legend().set_visible(False)

axes[-1].set_xlabel('Block Number', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# #### Statistical Tests

# In[38]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. Prepare the data (Filter for specific trigger count and exclude Block 0)
max_trig = df_raw['n_triggers'].max()
df_stats = df_raw[(df_raw['Block'] != 0.0) & (df_raw['n_triggers'] == max_trig)].copy()

# 2. Define the model formula
# Peak_Z is predicted by Condition, Block, and their interaction (Condition:Block)
# C() tells statsmodels to treat the variable as categorical
formula = 'Peak_Z ~ C(Condition) + C(Block) + C(Condition):C(Block)'

# 3. Fit the model and calculate the ANOVA table
model = ols(formula, data=df_stats).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the results
display(anova_table)


# In[39]:


import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# We will store the results here
posthoc_results = []
p_values = []

# Get the unique blocks from your statistics dataframe
blocks = sorted(df_stats['Block'].unique())

for block in blocks:
    # Isolate data for this specific block
    tr_data = df_stats[(df_stats['Block'] == block) & (df_stats['Condition'] == 'Tracking')]['Peak_Z']
    pb_data = df_stats[(df_stats['Block'] == block) & (df_stats['Condition'] == 'Playback')]['Peak_Z']

    # Run Welch's t-test (equal_var=False is generally safer for neuroscience data)
    stat, p_val = ttest_ind(tr_data, pb_data, equal_var=False, nan_policy='omit')

    p_values.append(p_val)
    posthoc_results.append({
        'Block': block,
        'Tracking_N': len(tr_data),
        'Playback_N': len(pb_data),
        't_stat': stat,
        'raw_p_value': p_val
    })

# Apply FDR (False Discovery Rate) correction to the p-values
_, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Add the adjusted p-values and significance markers to our results
for i, res in enumerate(posthoc_results):
    res['adj_p_value'] = p_adjusted[i]

    # Determine the star rating for the plot later
    if res['adj_p_value'] < 0.001:
        res['Significance'] = '***'
    elif res['adj_p_value'] < 0.01:
        res['Significance'] = '**'
    elif res['adj_p_value'] < 0.05:
        res['Significance'] = '*'
    else:
        res['Significance'] = 'ns'

df_posthoc = pd.DataFrame(posthoc_results)
display(df_posthoc)


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Setup Data
df_plot = df_raw[(df_raw['Block'] != 0.0) & (df_raw['n_triggers'] == df_raw['n_triggers'].max())].copy()
unique_blocks = sorted(df_plot['Block'].unique())

# 2. Create Figure for Peak_Z
fig, ax = plt.subplots(figsize=(10, 6))

# A. Draw the Box Plot
sns.boxplot(
    data=df_plot, x='Block', y='Peak_Z', hue='Condition',
    palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
    ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
)

# B. Add Sample Sizes (N) and Statistical Brackets
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min
ax.set_ylim(y_min, y_max + y_range * 0.15) # Add 15% headroom for brackets and text

for x_idx, block in enumerate(unique_blocks):

    # -- 1. Calculate heights for N labels --
    tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Peak_Z']
    pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Peak_Z']

    y_pos_tr = tr_data.max() + (y_range * 0.02)
    y_pos_pb = pb_data.max() + (y_range * 0.02)

    # Add N text
    ax.text(x_idx - 0.2, y_pos_tr, f'n={len(tr_data)}', ha='center', va='bottom', fontsize=9, color='darkred')
    ax.text(x_idx + 0.2, y_pos_pb, f'n={len(pb_data)}', ha='center', va='bottom', fontsize=9, color='black')

    # -- 2. Draw Statistical Brackets --
    # Get the significance marker from your df_posthoc dataframe
    sig = df_posthoc.loc[df_posthoc['Block'] == block, 'Significance'].values[0]

    if sig != 'ns':
        # Find the highest point between both boxes to place the bracket above them
        bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.06)
        bracket_tip = y_range * 0.02

        x1, x2 = x_idx - 0.2, x_idx + 0.2 # Centers of the two boxes

        # Draw the horizontal line and vertical downward tips
        ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')

        # Add the stars (* or ***) precisely in the middle of the bracket
        ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

# C. Formatting
ax.set_ylabel('Peak Response\nZ-Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
ax.set_title('Peak_Z: Tracking vs Playback within Blocks', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, axis='y', linestyle=':', alpha=0.7)
ax.legend(loc='upper left', frameon=True, fontsize=11)

plt.tight_layout()
plt.show()


# #### Automation Peak

# ##### Automated Data Extraction and Z-Scoring

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# ==========================================================
# 1. PARAMETERS & INITIALIZATION
# ==========================================================
t_pre = 0.3   
t_post = 0.3  

# Generate n values: 2^0 to 2^8 [1, 2, 4, 8, 16, 32, 64, 128, 256]
n_values = [2**i for i in range(9)]

# Clean Condition -1.0
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False

data_collection = {}

# ==========================================================
# 2. DATA EXTRACTION
# ==========================================================
for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        # Using your existing functions (ensure they are defined above in your notebook)
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Blocks with BOTH Tracking and Playback, excluding Block 0
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb and b != 0])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

# ==========================================================
# 3. Z-SCORING & METRICS COMPUTATION
# ==========================================================
baseline_bins = int(0.2 / dt)
n_bins_pre = int(t_pre / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre
idx_peak_start = n_bins_pre
idx_peak_end = n_bins_pre + int(t_post / dt)

raw_results = []

for block_id, nt_dict in sorted(data_collection.items()):
    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)

            # Z-Score Tracking
            tr_mean_time = np.mean(tracking_arr, axis=1, keepdims=True)
            tr_std_time = np.std(tracking_arr, axis=1, keepdims=True) + 1e-8
            tracking_z = (tracking_arr - tr_mean_time) / tr_std_time

            # Z-Score Playback
            pb_mean_time = np.mean(playback_arr, axis=1, keepdims=True)
            pb_std_time = np.std(playback_arr, axis=1, keepdims=True) + 1e-8
            playback_z = (playback_arr - pb_mean_time) / pb_std_time

            # Extract Metrics
            tr_base = np.mean(tracking_z[:, idx_base_start:idx_base_end], axis=1)
            tr_peak = np.max(tracking_z[:, idx_peak_start:idx_peak_end], axis=1)
            tr_evoked = tr_peak - tr_base

            pb_base = np.mean(playback_z[:, idx_base_start:idx_base_end], axis=1)
            pb_peak = np.max(playback_z[:, idx_peak_start:idx_peak_end], axis=1)
            pb_evoked = pb_peak - pb_base

            df_tr = pd.DataFrame({'Block': block_id, 'n_triggers': n_val, 'Condition': 'Tracking', 
                                  'Base_Z': tr_base, 'Peak_Z': tr_peak, 'Evoked_Z': tr_evoked})
            df_pb = pd.DataFrame({'Block': block_id, 'n_triggers': n_val, 'Condition': 'Playback', 
                                  'Base_Z': pb_base, 'Peak_Z': pb_peak, 'Evoked_Z': pb_evoked})

            raw_results.extend([df_tr, df_pb])

df_raw = pd.concat(raw_results, ignore_index=True)
print("Data extraction and Z-scoring complete!")


# ##### Automated Statistics and Boxplot Generation

# In[42]:


# ==========================================================
# AUTOMATED PLOTTING & STATISTICS LOOP
# ==========================================================

for n_val in n_values:
    # 1. Filter data for the current n_val
    df_plot = df_raw[df_raw['n_triggers'] == n_val].copy()

    # Skip if there's no data for this n_val
    if df_plot.empty:
        continue

    unique_blocks = sorted(df_plot['Block'].unique())

    # 2. Run Statistics (T-test per block)
    posthoc_results = []
    p_values = []

    for block in unique_blocks:
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Peak_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Peak_Z']

        # Only run test if there is enough data
        if len(tr_data) > 1 and len(pb_data) > 1:
            stat, p_val = ttest_ind(tr_data, pb_data, equal_var=False, nan_policy='omit')
        else:
            p_val = np.nan

        p_values.append(p_val)
        posthoc_results.append({'Block': block, 'p_val': p_val})

    # 3. Apply FDR Correction
    valid_p = [p for p in p_values if not np.isnan(p)]
    if valid_p:
        _, p_adj_valid, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')
    else:
        p_adj_valid = []

    # Re-map adjusted p-values and determine star ratings
    adj_idx = 0
    sig_markers = {}
    for res in posthoc_results:
        if np.isnan(res['p_val']):
            sig_markers[res['Block']] = 'ns'
        else:
            adj_p = p_adj_valid[adj_idx]
            adj_idx += 1
            if adj_p < 0.001: sig_markers[res['Block']] = '***'
            elif adj_p < 0.01: sig_markers[res['Block']] = '**'
            elif adj_p < 0.05: sig_markers[res['Block']] = '*'
            else: sig_markers[res['Block']] = 'ns'

    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df_plot, x='Block', y='Peak_Z', hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
    )

    # Scale Y-axis to fit annotations
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.20) # 20% headroom

    # 5. Add N-counts and Stat Brackets
    for x_idx, block in enumerate(unique_blocks):
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Peak_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Peak_Z']

        # Heights for 'n' text
        y_pos_tr = tr_data.max() + (y_range * 0.02) if not tr_data.empty else y_min
        y_pos_pb = pb_data.max() + (y_range * 0.02) if not pb_data.empty else y_min

        ax.text(x_idx - 0.2, y_pos_tr, f'n={len(tr_data)}', ha='center', va='bottom', fontsize=9, color='darkred')
        ax.text(x_idx + 0.2, y_pos_pb, f'n={len(pb_data)}', ha='center', va='bottom', fontsize=9, color='black')

        # Bracket placement
        sig = sig_markers.get(block, 'ns')
        if sig != 'ns' and not tr_data.empty and not pb_data.empty:
            bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.08)
            bracket_tip = y_range * 0.02
            x1, x2 = x_idx - 0.2, x_idx + 0.2 

            ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    # 6. Formatting
    ax.set_ylabel('Peak Response Z-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_title(f'Peak_Z: Tracking vs Playback (n_triggers = {n_val})', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Adjust legend
    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    plt.show()


# ##### Improved Stats

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from IPython.display import display

# ==========================================================
# 1. PARAMETERS & DATA SANITIZATION
# ==========================================================
t_pre = 0.3   
t_post = 0.3  

# Generate n values: 1, 2, 4, 8, 16, 32, 64, 128, 256
n_values = [2**i for i in range(9)]

# Safely sanitize Condition -1.0
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False

# ==========================================================
# 2. DATA EXTRACTION & PAIRED Z-SCORING
# ==========================================================
data_collection = {}

for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Blocks with BOTH Tracking and Playback, excluding Block 0
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb and b != 0])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

baseline_bins = int(0.2 / dt)
n_bins_pre = int(t_pre / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre
idx_peak_start = n_bins_pre
idx_peak_end = n_bins_pre + int(t_post / dt)

raw_results = []

for block_id, nt_dict in sorted(data_collection.items()):
    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)
            n_samples = tracking_arr.shape[0]

            # Create a shared unique ID
            neuron_uids = np.arange(n_samples)

            # Z-Score Tracking
            tr_mean_time = np.mean(tracking_arr, axis=1, keepdims=True)
            tr_std_time = np.std(tracking_arr, axis=1, keepdims=True) + 1e-8
            tracking_z = (tracking_arr - tr_mean_time) / tr_std_time

            # Z-Score Playback
            pb_mean_time = np.mean(playback_arr, axis=1, keepdims=True)
            pb_std_time = np.std(playback_arr, axis=1, keepdims=True) + 1e-8
            playback_z = (playback_arr - pb_mean_time) / pb_std_time

            # Extract Peak Metrics
            tr_peak = np.max(tracking_z[:, idx_peak_start:idx_peak_end], axis=1)
            pb_peak = np.max(playback_z[:, idx_peak_start:idx_peak_end], axis=1)

            # Store Tracking
            df_tr = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Tracking', 'Peak_Z': tr_peak})

            # Store Playback
            df_pb = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Playback', 'Peak_Z': pb_peak})

            raw_results.extend([df_tr, df_pb])

df_raw = pd.concat(raw_results, ignore_index=True)
print("Paired Data Extraction Complete!")


# In[44]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from IPython.display import display

# ==========================================================
# 0. SETUP OUTPUT DIRECTORY
# ==========================================================
output_dir = r"C:\Users\PenPen\Desktop\Ferret\Code\Results"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to: {output_dir}")

# ==========================================================
# 1. PARAMETERS & DATA SANITIZATION
# ==========================================================
t_pre = 0.3   
t_post = 0.3  

# Generate n values: 1, 2, 4, 8, 16, 32, 64, 128, 256
n_values = [2**i for i in range(9)]

# Safely sanitize Condition -1.0
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False


# ==========================================================
# 3. PAIRED STATS, PLOTTING & TABLE GENERATION
# ==========================================================
master_stats_list = [] # We will store ALL statistical results here

for n_val in n_values:
    df_plot = df_raw[df_raw['n_triggers'] == n_val].copy()
    if df_plot.empty:
        continue

    unique_blocks = sorted(df_plot['Block'].unique())
    current_n_stats = []
    p_values = []

    # Run Paired Stats per Block for this n_val
    for block in unique_blocks:
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')].sort_values('Neuron_UID')['Peak_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')].sort_values('Neuron_UID')['Peak_Z']

        paired_n = len(tr_data)

        if paired_n > 1 and len(pb_data) > 1:
            stat, p_val = ttest_rel(tr_data, pb_data, nan_policy='omit')
        else:
            stat, p_val = np.nan, np.nan

        p_values.append(p_val)

        current_n_stats.append({
            'n_triggers': n_val,
            'Block': block,
            'N_Pairs': paired_n,
            't_stat': stat,
            'raw_p_value': p_val
        })

    # Apply FDR Correction across the blocks for this specific n_val
    valid_p = [p for p in p_values if not np.isnan(p)]
    if valid_p:
        _, p_adj_valid, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')
    else:
        p_adj_valid = []

    adj_idx = 0
    sig_markers = {}

    # Map back the adjusted p-values and assign stars
    for res in current_n_stats:
        if np.isnan(res['raw_p_value']):
            res['adj_p_value'] = np.nan
            res['Significance'] = 'ns'
            sig_markers[res['Block']] = 'ns'
        else:
            adj_p = p_adj_valid[adj_idx]
            res['adj_p_value'] = adj_p
            adj_idx += 1

            if adj_p < 0.001: res['Significance'] = '***'
            elif adj_p < 0.01: res['Significance'] = '**'
            elif adj_p < 0.05: res['Significance'] = '*'
            else: res['Significance'] = 'ns'

            sig_markers[res['Block']] = res['Significance']

    master_stats_list.extend(current_n_stats)

    # Generate the Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df_plot, x='Block', y='Peak_Z', hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
    )

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.20) 

    # Annotate N and Significance
    for x_idx, block in enumerate(unique_blocks):
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Peak_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Peak_Z']

        y_pos_tr = tr_data.max() + (y_range * 0.02) if not tr_data.empty else y_min
        y_pos_pb = pb_data.max() + (y_range * 0.02) if not pb_data.empty else y_min

        paired_n = len(tr_data)
        ax.text(x_idx, min(y_pos_tr, y_pos_pb) - (y_range * 0.08), f'n={paired_n}', 
                ha='center', va='top', fontsize=9, color='gray')

        sig = sig_markers.get(block, 'ns')
        if sig != 'ns' and not tr_data.empty and not pb_data.empty:
            bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.08)
            bracket_tip = y_range * 0.02
            x1, x2 = x_idx - 0.2, x_idx + 0.2 

            ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    ax.set_ylabel('Peak Response Z-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_title(f'Paired Peak_Z: Tracking vs Playback (n_triggers = {n_val})', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()

    # --- SAVE PLOT ---
    fig_filename = os.path.join(output_dir, f'paired_peak_z_n_{n_val}.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[45]:


# ==========================================================
# 4. SAVE STATS TABLE
# ==========================================================
df_all_stats = pd.DataFrame(master_stats_list)

print("\n--- Summary of Paired Statistical Tests (FDR Corrected per n_triggers) ---")
display(df_all_stats)

# --- SAVE CSV ---
csv_filename = os.path.join(output_dir, 'paired_stats_summary_n_triggers.csv')
df_all_stats.to_csv(csv_filename, index=False)
print(f"\nAll plots and the statistics CSV have been successfully saved to:\n{output_dir}")


# #### Automation Baseline

# In[46]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from IPython.display import display

# ==========================================================
# 0. SETUP OUTPUT DIRECTORY
# ==========================================================
output_dir = r"C:\Users\PenPen\Desktop\Ferret\Code\Results"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to: {output_dir}")

# ==========================================================
# 1. PARAMETERS & DATA SANITIZATION
# ==========================================================
t_pre = 0.3   
t_post = 0.3  

# Generate n values: 1, 2, 4, 8, 16, 32, 64, 128, 256
n_values = [2**i for i in range(9)]

# Safely sanitize Condition -1.0
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False

# ==========================================================
# 2. DATA EXTRACTION & PAIRED Z-SCORING (FOCUS ON BASELINE)
# ==========================================================
data_collection = {}

for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Blocks with BOTH Tracking and Playback, excluding Block 0
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb and b != 0])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

baseline_bins = int(0.2 / dt)
n_bins_pre = int(t_pre / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Time 0

raw_results = []

for block_id, nt_dict in sorted(data_collection.items()):
    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)
            n_samples = tracking_arr.shape[0]

            # Create a shared unique ID
            neuron_uids = np.arange(n_samples)

            # Z-Score Tracking
            tr_mean_time = np.mean(tracking_arr, axis=1, keepdims=True)
            tr_std_time = np.std(tracking_arr, axis=1, keepdims=True) + 1e-8
            tracking_z = (tracking_arr - tr_mean_time) / tr_std_time

            # Z-Score Playback
            pb_mean_time = np.mean(playback_arr, axis=1, keepdims=True)
            pb_std_time = np.std(playback_arr, axis=1, keepdims=True) + 1e-8
            playback_z = (playback_arr - pb_mean_time) / pb_std_time

            # Extract Baseline Metrics (-0.2s to 0s)
            tr_base = np.mean(tracking_z[:, idx_base_start:idx_base_end], axis=1)
            pb_base = np.mean(playback_z[:, idx_base_start:idx_base_end], axis=1)

            # Store Tracking (Base_Z)
            df_tr = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Tracking', 'Base_Z': tr_base})

            # Store Playback (Base_Z)
            df_pb = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Playback', 'Base_Z': pb_base})

            raw_results.extend([df_tr, df_pb])

df_raw = pd.concat(raw_results, ignore_index=True)
print("Paired Data Extraction (Baseline) Complete!")

# ==========================================================
# 3. PAIRED STATS, PLOTTING & TABLE GENERATION
# ==========================================================
master_stats_list = [] # We will store ALL statistical results here

for n_val in n_values:
    df_plot = df_raw[df_raw['n_triggers'] == n_val].copy()
    if df_plot.empty:
        continue

    unique_blocks = sorted(df_plot['Block'].unique())
    current_n_stats = []
    p_values = []

    # Run Paired Stats per Block for this n_val using Base_Z
    for block in unique_blocks:
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')].sort_values('Neuron_UID')['Base_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')].sort_values('Neuron_UID')['Base_Z']

        paired_n = len(tr_data)

        if paired_n > 1 and len(pb_data) > 1:
            stat, p_val = ttest_rel(tr_data, pb_data, nan_policy='omit')
        else:
            stat, p_val = np.nan, np.nan

        p_values.append(p_val)

        current_n_stats.append({
            'n_triggers': n_val,
            'Block': block,
            'N_Pairs': paired_n,
            't_stat': stat,
            'raw_p_value': p_val
        })

    # Apply FDR Correction across the blocks for this specific n_val
    valid_p = [p for p in p_values if not np.isnan(p)]
    if valid_p:
        _, p_adj_valid, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')
    else:
        p_adj_valid = []

    adj_idx = 0
    sig_markers = {}

    # Map back the adjusted p-values and assign stars
    for res in current_n_stats:
        if np.isnan(res['raw_p_value']):
            res['adj_p_value'] = np.nan
            res['Significance'] = 'ns'
            sig_markers[res['Block']] = 'ns'
        else:
            adj_p = p_adj_valid[adj_idx]
            res['adj_p_value'] = adj_p
            adj_idx += 1

            if adj_p < 0.001: res['Significance'] = '***'
            elif adj_p < 0.01: res['Significance'] = '**'
            elif adj_p < 0.05: res['Significance'] = '*'
            else: res['Significance'] = 'ns'

            sig_markers[res['Block']] = res['Significance']

    master_stats_list.extend(current_n_stats)

    # Generate the Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df_plot, x='Block', y='Base_Z', hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
    )

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.20) 

    # Annotate N and Significance
    for x_idx, block in enumerate(unique_blocks):
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Base_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Base_Z']

        y_pos_tr = tr_data.max() + (y_range * 0.02) if not tr_data.empty else y_min
        y_pos_pb = pb_data.max() + (y_range * 0.02) if not pb_data.empty else y_min

        paired_n = len(tr_data)
        ax.text(x_idx, min(y_pos_tr, y_pos_pb) - (y_range * 0.08), f'n={paired_n}', 
                ha='center', va='top', fontsize=9, color='gray')

        sig = sig_markers.get(block, 'ns')
        if sig != 'ns' and not tr_data.empty and not pb_data.empty:
            bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.08)
            bracket_tip = y_range * 0.02
            x1, x2 = x_idx - 0.2, x_idx + 0.2 

            ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    ax.set_ylabel('Baseline Z-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_title(f'Paired Base_Z: Tracking vs Playback (n_triggers = {n_val})', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()

    # --- SAVE PLOT ---
    fig_filename = os.path.join(output_dir, f'paired_base_z_n_{n_val}.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================================
# 4. SAVE STATS TABLE
# ==========================================================
df_all_stats_base = pd.DataFrame(master_stats_list)

print("\n--- Summary of Paired Statistical Tests (BASELINE - FDR Corrected) ---")
display(df_all_stats_base)

# --- SAVE CSV ---
csv_filename = os.path.join(output_dir, 'paired_base_stats_summary_n_triggers.csv')
df_all_stats_base.to_csv(csv_filename, index=False)
print(f"\nAll baseline plots and the statistics CSV have been successfully saved to:\n{output_dir}")


# #### Automation Evoked Response

# In[47]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from IPython.display import display

# ==========================================================
# 0. SETUP OUTPUT DIRECTORY
# ==========================================================
output_dir = r"C:\Users\PenPen\Desktop\Ferret\Code\Results"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to: {output_dir}")

# ==========================================================
# 1. PARAMETERS & DATA SANITIZATION
# ==========================================================
t_pre = 0.3   
t_post = 0.3  

# Generate n values: 1, 2, 4, 8, 16, 32, 64, 128, 256
n_values = [2**i for i in range(9)]

# Safely sanitize Condition -1.0
for f_data in f_data_s:
    mask = (f_data['Condition'] == -1.0)
    f_data.loc[mask, 'Frequency_changes'] = False

# ==========================================================
# 2. DATA EXTRACTION & PAIRED Z-SCORING (EVOKED RESPONSE)
# ==========================================================
data_collection = {}

for n_val in n_values:
    for n_data, f_data in zip(n_data_s, f_data_s):
        dict_tr, psth_bins = get_psth_last_n_triggers_tracking(n_data, f_data, t_pre, t_post, dt, n_last=n_val)
        dict_pb, _ = get_psth_first_n_triggers_playback(n_data, f_data, t_pre, t_post, dt, n_first=n_val)

        # Blocks with BOTH Tracking and Playback, excluding Block 0
        valid_blocks = sorted([b for b in f_data['Block'].unique() if b in dict_tr and b in dict_pb and b != 0])

        for actual_b in valid_blocks:
            if actual_b not in data_collection:
                data_collection[actual_b] = {n: {'tr': [], 'pb': []} for n in n_values}

            data_collection[actual_b][n_val]['tr'].append(dict_tr[actual_b])
            data_collection[actual_b][n_val]['pb'].append(dict_pb[actual_b])

baseline_bins = int(0.2 / dt)
n_bins_pre = int(t_pre / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre  # Time 0
idx_peak_start = n_bins_pre
idx_peak_end = n_bins_pre + int(t_post / dt)

raw_results = []

for block_id, nt_dict in sorted(data_collection.items()):
    for n_val, trials_data in nt_dict.items():
        tr_list = trials_data['tr']
        pb_list = trials_data['pb']

        if len(tr_list) > 0 and len(pb_list) > 0:
            tracking_arr = np.vstack(tr_list)
            playback_arr = np.vstack(pb_list)
            n_samples = tracking_arr.shape[0]

            # Create a shared unique ID
            neuron_uids = np.arange(n_samples)

            # Z-Score Tracking
            tr_mean_time = np.mean(tracking_arr, axis=1, keepdims=True)
            tr_std_time = np.std(tracking_arr, axis=1, keepdims=True) + 1e-8
            tracking_z = (tracking_arr - tr_mean_time) / tr_std_time

            # Z-Score Playback
            pb_mean_time = np.mean(playback_arr, axis=1, keepdims=True)
            pb_std_time = np.std(playback_arr, axis=1, keepdims=True) + 1e-8
            playback_z = (playback_arr - pb_mean_time) / pb_std_time

            # Extract Metrics
            tr_base = np.mean(tracking_z[:, idx_base_start:idx_base_end], axis=1)
            tr_peak = np.max(tracking_z[:, idx_peak_start:idx_peak_end], axis=1)
            tr_evoked = tr_peak - tr_base

            pb_base = np.mean(playback_z[:, idx_base_start:idx_base_end], axis=1)
            pb_peak = np.max(playback_z[:, idx_peak_start:idx_peak_end], axis=1)
            pb_evoked = pb_peak - pb_base

            # Store Tracking (Evoked_Z)
            df_tr = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Tracking', 'Evoked_Z': tr_evoked})

            # Store Playback (Evoked_Z)
            df_pb = pd.DataFrame({'Neuron_UID': neuron_uids, 'Block': block_id, 'n_triggers': n_val, 
                                  'Condition': 'Playback', 'Evoked_Z': pb_evoked})

            raw_results.extend([df_tr, df_pb])

df_raw = pd.concat(raw_results, ignore_index=True)
print("Paired Data Extraction (Evoked Response) Complete!")

# ==========================================================
# 3. PAIRED STATS, PLOTTING & TABLE GENERATION
# ==========================================================
master_stats_list = [] # We will store ALL statistical results here

for n_val in n_values:
    df_plot = df_raw[df_raw['n_triggers'] == n_val].copy()
    if df_plot.empty:
        continue

    unique_blocks = sorted(df_plot['Block'].unique())
    current_n_stats = []
    p_values = []

    # Run Paired Stats per Block for this n_val using Evoked_Z
    for block in unique_blocks:
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')].sort_values('Neuron_UID')['Evoked_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')].sort_values('Neuron_UID')['Evoked_Z']

        paired_n = len(tr_data)

        if paired_n > 1 and len(pb_data) > 1:
            stat, p_val = ttest_rel(tr_data, pb_data, nan_policy='omit')
        else:
            stat, p_val = np.nan, np.nan

        p_values.append(p_val)

        current_n_stats.append({
            'n_triggers': n_val,
            'Block': block,
            'N_Pairs': paired_n,
            't_stat': stat,
            'raw_p_value': p_val
        })

    # Apply FDR Correction across the blocks for this specific n_val
    valid_p = [p for p in p_values if not np.isnan(p)]
    if valid_p:
        _, p_adj_valid, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')
    else:
        p_adj_valid = []

    adj_idx = 0
    sig_markers = {}

    # Map back the adjusted p-values and assign stars
    for res in current_n_stats:
        if np.isnan(res['raw_p_value']):
            res['adj_p_value'] = np.nan
            res['Significance'] = 'ns'
            sig_markers[res['Block']] = 'ns'
        else:
            adj_p = p_adj_valid[adj_idx]
            res['adj_p_value'] = adj_p
            adj_idx += 1

            if adj_p < 0.001: res['Significance'] = '***'
            elif adj_p < 0.01: res['Significance'] = '**'
            elif adj_p < 0.05: res['Significance'] = '*'
            else: res['Significance'] = 'ns'

            sig_markers[res['Block']] = res['Significance']

    master_stats_list.extend(current_n_stats)

    # Generate the Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df_plot, x='Block', y='Evoked_Z', hue='Condition',
        palette={'Tracking': '#ff9999', 'Playback': '#b3b3b3'}, 
        ax=ax, width=0.6, fliersize=4, boxprops=dict(alpha=0.7)
    )

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.20) 

    # Annotate N and Significance
    for x_idx, block in enumerate(unique_blocks):
        tr_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Tracking')]['Evoked_Z']
        pb_data = df_plot[(df_plot['Block'] == block) & (df_plot['Condition'] == 'Playback')]['Evoked_Z']

        y_pos_tr = tr_data.max() + (y_range * 0.02) if not tr_data.empty else y_min
        y_pos_pb = pb_data.max() + (y_range * 0.02) if not pb_data.empty else y_min

        paired_n = len(tr_data)
        ax.text(x_idx, min(y_pos_tr, y_pos_pb) - (y_range * 0.08), f'n={paired_n}', 
                ha='center', va='top', fontsize=9, color='gray')

        sig = sig_markers.get(block, 'ns')
        if sig != 'ns' and not tr_data.empty and not pb_data.empty:
            bracket_y = max(y_pos_tr, y_pos_pb) + (y_range * 0.08)
            bracket_tip = y_range * 0.02
            x1, x2 = x_idx - 0.2, x_idx + 0.2 

            ax.plot([x1, x1, x2, x2], [bracket_y - bracket_tip, bracket_y, bracket_y, bracket_y - bracket_tip], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, bracket_y, sig, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

    ax.set_ylabel('Evoked Z-Score (Peak - Base)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_title(f'Paired Evoked_Z: Tracking vs Playback (n_triggers = {n_val})', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()

    # --- SAVE PLOT ---
    fig_filename = os.path.join(output_dir, f'paired_evoked_z_n_{n_val}.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================================
# 4. SAVE STATS TABLE
# ==========================================================
df_all_stats_evoked = pd.DataFrame(master_stats_list)

print("\n--- Summary of Paired Statistical Tests (EVOKED - FDR Corrected) ---")
display(df_all_stats_evoked)

# --- SAVE CSV ---
csv_filename = os.path.join(output_dir, 'paired_evoked_stats_summary_n_triggers.csv')
df_all_stats_evoked.to_csv(csv_filename, index=False)
print(f"\nAll evoked plots and the statistics CSV have been successfully saved to:\n{output_dir}")


# ## Chunk of Triggers Tracking - Playback

# 

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools
from IPython.display import display

# ==========================================
# 1. ADJUSTABLE PARAMETERS
# ==========================================
n_per_group = 50     # Adjust: Number of triggers per chunk
n_groups = 3         # Adjust: Number of chunks per condition
total_needed = n_groups * n_per_group

# Define time window indices based on dt (assuming dt, n_bins_pre, expected_length are defined)
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Generate chronological labels
# Tracking: Last to First (e.g., Last 150-101, Last 100-51, Last 50-1)
labels_tr = [f"TR: Last {i*n_per_group}-{(i-1)*n_per_group+1}" for i in range(n_groups, 0, -1)]
# Playback: First to Last (e.g., First 1-50, First 51-100, First 101-150)
labels_pb = [f"PB: First {i*n_per_group+1}-{(i+1)*n_per_group}" for i in range(n_groups)]
chunk_labels = labels_tr + labels_pb

# Dynamically generate color palettes based on the number of groups
tr_colors = sns.color_palette("Reds", n_colors=n_groups + 2)[2:]  # Skip the lightest colors
pb_colors = sns.color_palette("Greys", n_colors=n_groups + 2)[2:] 
palette = {}
for i, lbl in enumerate(labels_tr): palette[lbl] = mcolors.to_hex(tr_colors[i])
for i, lbl in enumerate(labels_pb): palette[lbl] = mcolors.to_hex(pb_colors[i])

# Helper function to average trials per channel
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) 
    return None

# ==========================================
# 2. EXTRACT DATA REGARDLESS OF BLOCKS
# ==========================================
tr_data_chunks = [[] for _ in range(n_groups)]
pb_data_chunks = [[] for _ in range(n_groups)]

for n_data, f_data in zip(n_data_s, f_data_s):
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # Check if session has enough events globally
    if len(events_tr) >= total_needed and len(events_pb) >= total_needed:
        # Tracking Chunks (Chronological from the end)
        for chunk_idx, i in enumerate(range(n_groups, 0, -1)):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            mean_trace = get_subset_channel_means(n_data, events_tr[start:end])
            if mean_trace is not None:
                tr_data_chunks[chunk_idx].append(mean_trace)

        # Playback Chunks (Chronological from the start)
        for chunk_idx, i in enumerate(range(n_groups)):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            mean_trace = get_subset_channel_means(n_data, events_pb[start:end])
            if mean_trace is not None:
                pb_data_chunks[chunk_idx].append(mean_trace)

# ==========================================
# 3. COMPUTE Z-SCORES & FORMAT DATAFRAME
# ==========================================
master_data = []

for chunk_idx, chunk in enumerate(tr_data_chunks):
    if len(chunk) > 0:
        final_dat = np.concatenate(chunk, axis=0)
        mean_time = np.mean(final_dat, axis=1, keepdims=True)
        std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
        z_dat = (final_dat - mean_time) / std_time

        master_data.append(pd.DataFrame({
            'Chunk': labels_tr[chunk_idx], 'Condition': 'Tracking', 'Order': chunk_idx,
            'Base_Z': np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1),
            'Peak_Z': np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1),
            'Evoked_Z': np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1) - np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
        }))

for chunk_idx, chunk in enumerate(pb_data_chunks):
    if len(chunk) > 0:
        final_dat = np.concatenate(chunk, axis=0)
        mean_time = np.mean(final_dat, axis=1, keepdims=True)
        std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
        z_dat = (final_dat - mean_time) / std_time

        master_data.append(pd.DataFrame({
            'Chunk': labels_pb[chunk_idx], 'Condition': 'Playback', 'Order': chunk_idx + n_groups,
            'Base_Z': np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1),
            'Peak_Z': np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1),
            'Evoked_Z': np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1) - np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
        }))

df_plot = pd.concat(master_data, ignore_index=True)
ordered_chunks = df_plot.sort_values('Order')['Chunk'].unique()

# ==========================================
# 4. PLOTTING THE OVERVIEW HORIZONTAL GRID
# ==========================================
metrics_to_plot = [
    ('Base_Z', 'Baseline Z-Score\n(-0.2s to 0s)'),
    ('Peak_Z', 'Peak Response\nZ-Score'),
    ('Evoked_Z', 'Total Evoked Z-Score\n(Peak - Base)')
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

for i, (col_name, title) in enumerate(metrics_to_plot):
    ax = axes[i]
    sns.boxplot(
        data=df_plot.sort_values('Order'), x='Chunk', y=col_name, hue='Chunk',
        palette=palette, ax=ax, width=0.6, fliersize=3, dodge=False
    )

    ax.axvline(n_groups - 0.5, color='black', linestyle='--', alpha=0.5, zorder=0) 
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Z-Score', fontsize=12) if i == 0 else ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    for idx, chunk_label in enumerate(chunk_labels):
        n_count = len(df_plot[df_plot['Chunk'] == chunk_label])
        if n_count > 0:
            ax.text(idx, y_max + y_range * 0.05, f'n={n_count}', 
                    ha='center', va='bottom', fontsize=10, color='black')

plt.suptitle(f'Global Chronological Progression: Tracking -> Playback\n({n_per_group} triggers/chunk, block-agnostic)', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# ==========================================
# 5. AUTOMATED STATS, ANNOTATED BOXPLOTS & HEATMAPS
# ==========================================
for metric, title in metrics_to_plot:
    print(f"\n" + "="*80)
    print(f"  STATISTICAL ANALYSIS: {metric}")
    print("="*80)

    # Global ANOVA
    model = ols(f'{metric} ~ C(Chunk)', data=df_plot).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\n[1] Global ANOVA for {metric}:")
    display(anova_table)

    # Pairwise Stats (Adjacent Pairs specifically for the Boxplot)
    adjacent_pairs = [(ordered_chunks[i], ordered_chunks[i+1]) for i in range(len(ordered_chunks)-1)]
    adj_p_vals = []

    for c1, c2 in adjacent_pairs:
        d1 = df_plot[df_plot['Chunk'] == c1][metric]
        d2 = df_plot[df_plot['Chunk'] == c2][metric]
        p = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')[1] if len(d1) > 1 and len(d2) > 1 else 1.0
        adj_p_vals.append(p)

    _, p_adj, _, _ = multipletests(adj_p_vals, alpha=0.05, method='fdr_bh')

    # --- Plot Annotated Boxplot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_plot.sort_values('Order'), x='Chunk', y=metric, hue='Chunk',
        palette=palette, ax=ax, width=0.6, fliersize=3, dodge=False
    )

    ax.axvline(n_groups - 0.5, color='black', linestyle='--', alpha=0.5, zorder=0)
    ax.set_title(f"{title} - Adjacent Transitions", fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.25)

    h_base = y_max + y_range * 0.05
    step = y_range * 0.08

    for i, p_val in enumerate(p_adj):
        if p_val < 0.001: sig = '***'
        elif p_val < 0.01: sig = '**'
        elif p_val < 0.05: sig = '*'
        else: sig = 'ns'

        if sig != 'ns':
            x1, x2 = i, i + 1
            h = h_base + (i % 2) * step
            ax.plot([x1, x1, x2, x2], [h - step*0.2, h, h, h - step*0.2], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, h, sig, ha='center', va='bottom', color='k', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # --- Plot Complete Significance Heatmap ---
    matrix_p = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=float)
    matrix_sig = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=str)

    # Compute all pairwise comparisons
    for c1 in ordered_chunks:
        for c2 in ordered_chunks:
            if c1 == c2:
                matrix_p.loc[c1, c2] = 1.0
                matrix_sig.loc[c1, c2] = '-'
            else:
                d1 = df_plot[df_plot['Chunk'] == c1][metric]
                d2 = df_plot[df_plot['Chunk'] == c2][metric]
                p = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')[1] if len(d1) > 1 and len(d2) > 1 else 1.0
                matrix_p.loc[c1, c2] = p

                if p < 0.001: matrix_sig.loc[c1, c2] = '***'
                elif p < 0.01: matrix_sig.loc[c1, c2] = '**'
                elif p < 0.05: matrix_sig.loc[c1, c2] = '*'
                else: matrix_sig.loc[c1, c2] = ''

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_p.astype(float), annot=matrix_sig, fmt="", cmap='coolwarm_r', 
        vmin=0, vmax=0.05, cbar_kws={'label': 'Uncorrected P-Value'}, ax=ax, linewidths=1, linecolor='white'
    )

    ax.set_title(f"{title} - All Pairwise Comparisons Heatmap", fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CUSTOMIZABLE VARIABLES
# ==========================================
n_per_group = 50     # Number of triggers per chunk
n_groups = 3         # Number of chunks per condition
total_needed = n_groups * n_per_group

# Define time window indices based on dt
baseline_bins = int(0.2 / dt)
idx_base_start = n_bins_pre - baseline_bins
idx_base_end = n_bins_pre
idx_peak_start = n_bins_pre
idx_peak_end = expected_length

# Generate chronological labels
# Tracking: Last 150-101, Last 100-51, Last 50-1
labels_tr = [f"TR: Last {i*n_per_group}-{(i-1)*n_per_group+1}" for i in range(n_groups, 0, -1)]
# Playback: First 1-50, First 51-100, First 101-150
labels_pb = [f"PB: First {i*n_per_group+1}-{(i+1)*n_per_group}" for i in range(n_groups)]
chunk_labels = labels_tr + labels_pb

# Helper function to average trials per channel
def get_subset_channel_means(n_data, event_indices):
    segments = []
    for i in event_indices:
        start, stop = int(i - n_bins_pre), int(i + n_bins_post)
        if 0 <= start and stop <= n_data.shape[-1]:
            seg = n_data[:, start:stop]
            if seg.shape[1] == expected_length:
                segments.append(seg)
    if len(segments) > 0:
        return np.mean(np.stack(segments, axis=0), axis=0) 
    return None

# ==========================================
# 2. EXTRACT DATA REGARDLESS OF BLOCKS
# ==========================================
tr_data_chunks = [[] for _ in range(n_groups)]
pb_data_chunks = [[] for _ in range(n_groups)]

for n_data, f_data in zip(n_data_s, f_data_s):
    # Ignore blocks, just look at the condition
    events_pb = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 1))
    events_tr = np.flatnonzero((f_data['Frequency_changes'] == 1) & (f_data['Condition'] == 0))

    # Check if session has enough events globally
    if len(events_tr) >= total_needed and len(events_pb) >= total_needed:

        # --- Tracking Chunks (Chronological from the end) ---
        for chunk_idx, i in enumerate(range(n_groups, 0, -1)):
            start = -i * n_per_group
            end = -(i - 1) * n_per_group if i > 1 else None
            evs = events_tr[start:end]
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                tr_data_chunks[chunk_idx].append(mean_trace)

        # --- Playback Chunks (Chronological from the start) ---
        for chunk_idx, i in enumerate(range(n_groups)):
            start = i * n_per_group
            end = (i + 1) * n_per_group
            evs = events_pb[start:end]
            mean_trace = get_subset_channel_means(n_data, evs)
            if mean_trace is not None:
                pb_data_chunks[chunk_idx].append(mean_trace)

# ==========================================
# 3. COMPUTE Z-SCORES & FORMAT DATAFRAME
# ==========================================
master_data = []

# Process Tracking
for chunk_idx, chunk in enumerate(tr_data_chunks):
    if len(chunk) > 0:
        final_dat = np.concatenate(chunk, axis=0)

        # Z-Score
        mean_time = np.mean(final_dat, axis=1, keepdims=True)
        std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
        z_dat = (final_dat - mean_time) / std_time

        base = np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
        peak = np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1)
        evoked = peak - base

        df_temp = pd.DataFrame({
            'Chunk': labels_tr[chunk_idx],
            'Condition': 'Tracking',
            'Order': chunk_idx, # To maintain chronological plotting
            'Base_Z': base,
            'Peak_Z': peak,
            'Evoked_Z': evoked
        })
        master_data.append(df_temp)

# Process Playback
for chunk_idx, chunk in enumerate(pb_data_chunks):
    if len(chunk) > 0:
        final_dat = np.concatenate(chunk, axis=0)

        # Z-Score
        mean_time = np.mean(final_dat, axis=1, keepdims=True)
        std_time = np.std(final_dat, axis=1, keepdims=True) + 1e-8
        z_dat = (final_dat - mean_time) / std_time

        base = np.mean(z_dat[:, idx_base_start:idx_base_end], axis=1)
        peak = np.max(z_dat[:, idx_peak_start:idx_peak_end], axis=1)
        evoked = peak - base

        df_temp = pd.DataFrame({
            'Chunk': labels_pb[chunk_idx],
            'Condition': 'Playback',
            'Order': chunk_idx + n_groups, # To maintain chronological plotting
            'Base_Z': base,
            'Peak_Z': peak,
            'Evoked_Z': evoked
        })
        master_data.append(df_temp)

df_plot = pd.concat(master_data, ignore_index=True)

# ==========================================
# 4. PLOTTING THE HORIZONTAL GRID
# ==========================================
metrics = [
    ('Base_Z', 'Baseline Z-Score\n(-0.2s to 0s)'),
    ('Peak_Z', 'Peak Response\nZ-Score'),
    ('Evoked_Z', 'Total Evoked Z-Score\n(Peak - Base)')
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Define custom color palette for the chunks to show progression
palette = {
    labels_tr[0]: '#ffb3b3', labels_tr[1]: '#ff6666', labels_tr[2]: '#cc0000', # Light to dark red
    labels_pb[0]: '#cccccc', labels_pb[1]: '#8c8c8c', labels_pb[2]: '#404040'  # Light to dark gray
}

for i, (col_name, title) in enumerate(metrics):
    ax = axes[i]

    sns.boxplot(
        data=df_plot.sort_values('Order'), 
        x='Chunk', y=col_name, hue='Chunk',
        palette=palette, ax=ax, width=0.6, fliersize=3, dodge=False
    )

    # Formatting
    ax.axvline(2.5, color='black', linestyle='--', alpha=0.5, zorder=0) # Divides TR and PB
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Z-Score', fontsize=12) if i == 0 else ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Add N counts at the top of the plot
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    for idx, chunk_label in enumerate(chunk_labels):
        n_count = len(df_plot[df_plot['Chunk'] == chunk_label])
        if n_count > 0:
            ax.text(idx, y_max + y_range * 0.05, f'n={n_count}', 
                    ha='center', va='bottom', fontsize=10, color='black')

plt.suptitle(f'Global Chronological Progression: Tracking -> Playback\n({n_per_group} triggers/chunk, block-agnostic)', 
             fontsize=16, y=1.05)
plt.tight_layout()
plt.show()


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools
from IPython.display import display

# The 3 metrics we computed in df_plot
metrics = ['Base_Z', 'Peak_Z', 'Evoked_Z']

for metric in metrics:
    print(f"\n" + "="*70)
    print(f"  STATISTICAL ANALYSIS: {metric}")
    print("="*70)

    # ==========================================
    # 1. ONE-WAY ANOVA
    # ==========================================
    # Testing if there's a significant main effect of 'Chunk' (time) on the metric
    formula = f'{metric} ~ C(Chunk)'
    model = ols(formula, data=df_plot).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(f"\n[1] Global ANOVA for {metric}:")
    display(anova_table)

    # ==========================================
    # 2. POST-HOC PAIRWISE T-TESTS (FDR Corrected)
    # ==========================================
    # Get the unique chunks in their correct chronological order
    ordered_chunks = df_plot.sort_values('Order')['Chunk'].unique()

    # Generate all possible pairwise combinations between the 6 chunks
    pairs = list(itertools.combinations(ordered_chunks, 2))

    p_values = []
    posthoc_results = []

    for chunk1, chunk2 in pairs:
        data1 = df_plot[df_plot['Chunk'] == chunk1][metric]
        data2 = df_plot[df_plot['Chunk'] == chunk2][metric]

        # Run Welch's t-test (equal_var=False)
        if len(data1) > 1 and len(data2) > 1:
            stat, p_val = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        else:
            stat, p_val = np.nan, np.nan

        p_values.append(p_val)
        posthoc_results.append({
            'Group 1': chunk1,
            'Group 2': chunk2,
            't_stat': stat,
            'raw_p_value': p_val
        })

    # Apply Benjamini-Hochberg FDR correction to the p-values
    valid_p = [p if not np.isnan(p) else 1.0 for p in p_values]
    _, p_adj, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')

    # Assign significance markers
    for i, res in enumerate(posthoc_results):
        res['adj_p_value'] = p_adj[i]

        if p_adj[i] < 0.001: res['Significance'] = '***'
        elif p_adj[i] < 0.01: res['Significance'] = '**'
        elif p_adj[i] < 0.05: res['Significance'] = '*'
        else: res['Significance'] = 'ns'

    df_posthoc = pd.DataFrame(posthoc_results)

    print(f"\n[2] Pairwise Post-Hoc Comparisons for {metric} (FDR Corrected):")
    display(df_posthoc)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Define the metrics and their readable titles
metrics = [
    ('Base_Z', 'Baseline Z-Score\n(-0.2s to 0s)'),
    ('Peak_Z', 'Peak Response\nZ-Score'),
    ('Evoked_Z', 'Total Evoked Z-Score\n(Peak - Base)')
]

# Custom palette
palette = {
    labels_tr[0]: '#ffb3b3', labels_tr[1]: '#ff6666', labels_tr[2]: '#cc0000', 
    labels_pb[0]: '#cccccc', labels_pb[1]: '#8c8c8c', labels_pb[2]: '#404040'
}

ordered_chunks = df_plot.sort_values('Order')['Chunk'].unique()

for metric, title in metrics:
    # ---------------------------------------------------------
    # PART 1: BOXPLOT WITH KEY STATISTICAL BRACKETS
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df_plot.sort_values('Order'), 
        x='Chunk', y=metric, hue='Chunk',
        palette=palette, ax=ax, width=0.6, fliersize=3, dodge=False
    )

    # Formatting the plot
    ax.axvline(2.5, color='black', linestyle='--', alpha=0.5, zorder=0) # Divides TR and PB
    ax.set_title(f"{title} - Annotated Stats", fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Expand Y-axis to make room for brackets
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.25)

    # Define adjacent pairs we actually want to draw brackets for
    # (e.g., TR1 vs TR2, TR2 vs TR3, TR3 vs PB1, PB1 vs PB2, PB2 vs PB3)
    adjacent_pairs = [(ordered_chunks[i], ordered_chunks[i+1]) for i in range(len(ordered_chunks)-1)]

    # Extract the posthoc results for this specific metric
    # (Assuming you saved the posthoc dataframe as `df_posthoc` in the previous step)
    # We will recalculate just the necessary ones here for the bracket drawing:
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    # Recalculate p-values for adjacent pairs quickly to annotate
    adj_p_vals = []
    for c1, c2 in adjacent_pairs:
        d1 = df_plot[df_plot['Chunk'] == c1][metric]
        d2 = df_plot[df_plot['Chunk'] == c2][metric]
        if len(d1) > 1 and len(d2) > 1:
            _, p = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
        else:
            p = 1.0
        adj_p_vals.append(p)

    _, p_adj, _, _ = multipletests(adj_p_vals, alpha=0.05, method='fdr_bh')

    # Draw brackets
    bracket_level = y_max + y_range * 0.05
    step = y_range * 0.08

    for i, (c1, c2) in enumerate(adjacent_pairs):
        p_val = p_adj[i]

        # Determine stars
        if p_val < 0.001: sig = '***'
        elif p_val < 0.01: sig = '**'
        elif p_val < 0.05: sig = '*'
        else: sig = 'ns'

        if sig != 'ns':
            x1, x2 = i, i + 1
            # Alternate bracket heights slightly so they don't overlap perfectly if adjacent
            h = bracket_level + (i % 2) * step

            ax.plot([x1, x1, x2, x2], [h - step*0.2, h, h, h - step*0.2], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, h, sig, ha='center', va='bottom', color='k', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # PART 2: HEATMAP OF ALL 15 PAIRWISE P-VALUES
    # ---------------------------------------------------------
    # Build a matrix of p-values for all combinations
    matrix_p = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=float)
    matrix_sig = pd.DataFrame(index=ordered_chunks, columns=ordered_chunks, dtype=str)

    for c1 in ordered_chunks:
        for c2 in ordered_chunks:
            if c1 == c2:
                matrix_p.loc[c1, c2] = 1.0
                matrix_sig.loc[c1, c2] = '-'
            else:
                d1 = df_plot[df_plot['Chunk'] == c1][metric]
                d2 = df_plot[df_plot['Chunk'] == c2][metric]
                if len(d1) > 1 and len(d2) > 1:
                    _, p = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
                else:
                    p = 1.0
                matrix_p.loc[c1, c2] = p

    # We ideally want to apply FDR correction to the flattened upper triangle here,
    # but for visualization purposes, coloring by raw/uncorrected p-value is common in heatmaps.

    # Fill significance matrix
    for c1 in ordered_chunks:
        for c2 in ordered_chunks:
            p = matrix_p.loc[c1, c2]
            if c1 == c2: continue
            if p < 0.001: matrix_sig.loc[c1, c2] = '***'
            elif p < 0.01: matrix_sig.loc[c1, c2] = '**'
            elif p < 0.05: matrix_sig.loc[c1, c2] = '*'
            else: matrix_sig.loc[c1, c2] = ''

    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_p.astype(float), 
        annot=matrix_sig, 
        fmt="", 
        cmap='coolwarm_r', # Red = significant (low p), Blue = not significant (high p)
        vmin=0, vmax=0.05, # Cap color mapping at significance threshold
        cbar_kws={'label': 'P-Value'},
        ax=ax, linewidths=1, linecolor='white'
    )

    ax.set_title(f"{metric} - Pairwise Significance Heatmap", fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

