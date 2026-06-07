# %% LMM Analysis: Track vs Playback, MP1 vs MP2
# 每个神经元贡献 2 行（Track + PB），condition 作为显式因子
# Requires: traj_by_half_mp1, traj_by_half_mp2, time, save_directory
# Run after the AUC violin plot cell

import sys
sys.path.insert(0, r'C:\Users\PenPen\Ferret\Analysis')
from lmm_analysis import run_lmm_analysis, compare_random_effect_structures

# Run full LMM pipeline (Track vs PB as within-neuron factor)
result_lmm, df_lmm = run_lmm_analysis(
    traj_by_half_mp1, traj_by_half_mp2, time, save_directory
)

# Compare random effect structures
compare_random_effect_structures(df_lmm)
