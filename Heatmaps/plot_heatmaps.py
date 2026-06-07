from utils import *
#from tonotopy import *
import findpeaks
from skimage import measure
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

def plot_heatmap_bandwidth(heatmaps, smooth_sigma, good_clusters, unique_tones,
                           bin_width, psth_bins, t_pre, save_path, condition,
                           session_name=None, max_cols=8):
    """
    将所有神经元的热图绘制在一张大图上并保存。
    
    参数:
        heatmaps : (n_neurons, n_tones, n_timebins) 数组
        smooth_sigma : 高斯平滑sigma
        good_clusters : 神经元编号列表
        unique_tones : 频率列表
        bin_width, psth_bins, t_pre : 时间参数
        save_path : 图片保存路径（文件夹）
        condition : 条件名（tracking/playback）
        session_name : 自定义 session 名（若不传则取 save_path 的最后一级）
        max_cols : 每行最多子图数，用于自动计算行列
    """
    time_centers = psth_bins[:-1] + bin_width / 2
    n_neurons = len(good_clusters)

    # 自动计算行列数
    n_cols = min(max_cols, n_neurons)
    n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(3 * n_cols, 3 * n_rows))
    # 确保 axes 是二维可迭代
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # 添加总标题
    title = f"Tonotopy - {condition}"
    if session_name is None:
        session_name = os.path.basename(save_path.rstrip('/\\'))
    title += f"\n{session_name}"
    fig.suptitle(title, fontsize=14, y=0.98)

    for idx in range(n_neurons):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        hm = heatmaps[idx]
        if smooth_sigma > 0:
            hm_smooth = gaussian_filter1d(hm, sigma=smooth_sigma, axis=1)
        else:
            hm_smooth = hm

        im = ax.imshow(hm_smooth, aspect='auto', origin='lower',
               extent=[time_centers[0], time_centers[-1], 0, len(unique_tones)-1],
               cmap='RdBu_r',                              # 红-蓝，红为高
               vmin=np.percentile(hm_smooth, 1),           # 下限：第1百分位
               vmax=np.percentile(hm_smooth, 99),
               interpolation='nearest')          # 上限：第99百分位
        ax.axvline(0, color='white', linestyle='--', linewidth=0.8)
        
        # y轴刻度：只标几个频率
        tick_step = max(1, len(unique_tones) // 8)
        y_ticks = np.arange(0, len(unique_tones), tick_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(unique_tones[t])}' for t in y_ticks], fontsize=6)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('Freq (Hz)', fontsize=7)
        ax.set_title(f'Ch {good_clusters[idx]}', fontsize=9)

    # 隐藏多余子图
    for idx in range(n_neurons, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为 suptitle 留空间
    fname = f'heatmap_{session_name}.png'
    fpath = os.path.join(save_path, fname)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {fpath}')

path = r'\\129.199.81.18\data5\eTheremin\SKIEUR\SKIEUR_20260403_SESSION_02'
#session = 'MMELOIK_20241029_SESSION_00'
#path = '/Volumes/data6/eTheremin/MMELOIK/'+ session + '/'

#session = 'MUROLS_20230227/MUROLS_20230227_SESSION_00'
#path = '/Volumes/data2/eTheremin/MUROLS/'+ session + '/'
def get_tonotopy_robust(psth, tones, t_pre, t_post, bin_width):
    """
    从 PSTH 和 tones 直接生成热图，无需预先指定 unique_tones。
    
    参数:
        psth : list of list of 1D arrays   (n_neurons x n_trials 的可变长列表)
        tones : list of int                (与 psth 的 trial 一一对应)
        t_pre, t_post, bin_width : 时间参数
    
    返回:
        heatmaps : (n_neurons, n_freq, n_bins) 的数组
        global_freqs : 公共频率列表（所有神经元出现的频率的并集，排序）
    """
    n_bins = int((t_pre + t_post) / bin_width)  # 每个 trial 的 bin 数
    # 确保 tones 为整数数组
    tones = np.array([int(t) for t in tones])

    # 全局频率：取所有神经元出现的频率的并集
    all_freqs = set()
    neuron_freq_data = []  # 每个神经元一个字典 {freq: list of arrays}

    for neuron_psth in psth:
        # neuron_psth 是 list of trial arrays
        freq_dict = defaultdict(list)
        for trial, tone in zip(neuron_psth, tones):
            freq_dict[tone].append(np.asarray(trial))
        neuron_freq_data.append(freq_dict)
        all_freqs.update(freq_dict.keys())

    global_freqs = sorted(all_freqs)
    n_freq = len(global_freqs)
    heatmaps = []

    for freq_dict in neuron_freq_data:
        average_list = []
        for f in global_freqs:
            if f in freq_dict:
                trials = np.array(freq_dict[f])  # (n_trials, n_bins)
                avg = np.mean(trials, axis=0)
            else:
                avg = np.zeros(n_bins)
            average_list.append(avg)
        hm = np.array(average_list)  # (n_freq, n_bins)

        # 基线校正：减去刺激前时间段的均值
        t0_idx = int(t_pre / bin_width)
        baseline = np.mean(hm[:, :t0_idx], axis=1, keepdims=True)
        hm_corrected = hm - baseline

        heatmaps.append(hm_corrected)

    return np.array(heatmaps), global_freqs

t_pre = 0.3#0.2
t_post = 0.3#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
condition = 'playback' #or playback

data = np.load(path+rf'\headstage_0\data_{bin_width}.npy', allow_pickle=True)
features = np.load(path+rf'\headstage_0\features_{bin_width}.npy', allow_pickle=True)
#data = np.load(path+f'/spike_sorting/data_{bin_width}.npy', allow_pickle=True)
#features = np.load(path+f'/spike_sorting/features_{bin_width}.npy', allow_pickle=True)
#gc = np.load(path+'/good_clusters.npy', allow_pickle=True)
gc = np.arange(32)

tones = get_played_frequency(features, t_pre, t_post, bin_width, condition)
tones = [int(x) for x in tones]
# prendre les valeurs uniques de tones
unique_tones = sorted(np.unique(tones))
unique_tones = [int(x) for x in unique_tones]

# ne calculer les heatmaps uniquement si on ne trouve pas le fichier heatmaps.npy
#if not os.path.exists(path + f'heatmap_plot_{condition}.npy'):
print("calculating heatmaps")
print("Actual tones unique values:", np.unique(tones))
print("Provided unique_tones:", unique_tones)
print("Type of tones element:", type(tones[0]))
print("Type of unique_tones element:", type(unique_tones[0]))
# heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, 0, 0, condition, 'heatmaps')
condition = 'playback'  # 或 'tracking'
psth_all = get_psth(data, features, t_pre, t_post, bin_width, gc, condition)
tones_all = get_played_frequency(features, t_pre, t_post, bin_width, condition)

# 生成热图
heatmaps, freqs = get_tonotopy_robust(psth_all, tones_all, t_pre, t_post, bin_width)

# 绘图
local_dir = r'C:\Users\PenPen\Desktop\Ferret\Results&PLots'
os.makedirs(local_dir, exist_ok=True)
session = os.path.basename(path)   # path 末尾不含分隔符时有效
# 若 path 末尾有反斜杠，可先 strip：os.path.basename(path.rstrip('\\'))

# 绘制所有神经元在一张图上
plot_heatmap_bandwidth(
    heatmaps, 3, gc, freqs,
    bin_width, psth_bins, t_pre,
    local_dir, condition,
    session_name=session
)
#else:
   # heatmaps = np.load(path + f'heatmap_plot_{condition}.npy', allow_pickle = True)
    #print('heatmaps already exist')
#print(heatmaps)
#récupérer les heatmaps
# 本地保存路径
# local_save_dir = r'C:\Users\PenPen\Desktop\Ferret\Results&PLots'
# os.makedirs(local_save_dir, exist_ok=True)   # 自动创建文件夹（如不存在）

# plot_heatmap_bandwidth(heatmaps, 3, gc, unique_tones, 2, 2, bin_width, psth_bins, t_pre,
#                        local_save_dir, '', condition)   # 注意替换 path 为 local_save_dir