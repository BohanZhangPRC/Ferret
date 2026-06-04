# %% LMM Diagnostics — 检查数据流和模型问题
# 在 LMM 分析 cell 之后运行这个 cell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapezoid
from scipy.stats import ttest_ind, wilcoxon, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
sys.path.insert(0, r'C:\Users\PenPen\Ferret\Analysis')
from lmm_analysis import build_lmm_dataframe

print("=" * 65)
print("  DIAGNOSTIC CHECKLIST")
print("=" * 65)

# ── 1. 检查 traj_by_half 结构 ──────────────────
print("\n1. 数据结构检查")
for name, traj in [('MP1', traj_by_half_mp1), ('MP2', traj_by_half_mp2)]:
    for half_key in [(0.0, 0.5), (0.5, 1.0)]:
        for exp_key in ['beg', 'exp']:
            track_list = traj[half_key]['track'][exp_key]
            pb_list    = traj[half_key]['pb'][exp_key]
            n_sessions = len(track_list)
            n_valid_track = sum(1 for t in track_list if t is not None)
            n_valid_pb    = sum(1 for p in pb_list if p is not None)
            if n_valid_track > 0:
                shapes = [t.shape for t in track_list if t is not None]
                n_neurons_list = [s[1] for s in shapes]
                print(f"  {name} {half_key} {exp_key}: {n_valid_track}/{n_sessions} sessions valid, "
                      f"neurons min={min(n_neurons_list)}, max={max(n_neurons_list)}")

# ── 2. 检查组装后的 DataFrame ──────────────────
print("\n2. DataFrame 检查")
df_mp1 = build_lmm_dataframe(traj_by_half_mp1, 'MP1 (playback)', time)
df_mp2 = build_lmm_dataframe(traj_by_half_mp2, 'MP2 (mapping_change_only)', time)
df_check = pd.concat([df_mp1, df_mp2], ignore_index=True)

print(f"  总行数: {len(df_check)}")
print(f"  总 session 数: {df_check['session_id'].nunique()}")
print(f"  MP1 sessions: {df_check[df_check['group']=='MP1 (playback)']['session_id'].nunique()}")
print(f"  MP2 sessions: {df_check[df_check['group']=='MP2 (mapping_change_only)']['session_id'].nunique()}")

# 检查每个 session 是否有 H1 和 H2
session_half_counts = df_check.groupby('session_id')['half'].nunique()
missing_halves = (session_half_counts < 2).sum()
print(f"  Session 缺少 H1 或 H2: {missing_halves} / {len(session_half_counts)}")

# AUC 值范围
print(f"\n  AUC range: [{df_check['auc'].min():.6f}, {df_check['auc'].max():.6f}]")
print(f"  AUC mean={df_check['auc'].mean():.6f}, std={df_check['auc'].std():.6f}")

# ── 3. Cell-mean 级别检查：session 平均 AUC ──────────────────
print("\n3. Session-mean 级别分析（避免伪重复）")
session_means = df_check.groupby(['session_id', 'group', 'expertise']).agg(
    auc_mean=('auc', 'mean'),
    auc_std=('auc', 'std'),
    n_neurons=('auc', 'count')
).reset_index()

print(f"  MP1 session-mean AUC: {session_means[session_means['group']=='MP1 (playback)']['auc_mean'].mean():.6f} "
      f"+/- {session_means[session_means['group']=='MP1 (playback)']['auc_mean'].std():.6f}")
print(f"  MP2 session-mean AUC: {session_means[session_means['group']=='MP2 (mapping_change_only)']['auc_mean'].mean():.6f} "
      f"+/- {session_means[session_means['group']=='MP2 (mapping_change_only)']['auc_mean'].std():.6f}")

# 用 session-mean 做简单的 t-test（没有随机效应，但避免了伪重复）
mp1_session_means = session_means[session_means['group']=='MP1 (playback)']['auc_mean'].values
mp2_session_means = session_means[session_means['group']=='MP2 (mapping_change_only)']['auc_mean'].values
t_simple, p_simple = ttest_ind(mp1_session_means, mp2_session_means, equal_var=False)
print(f"\n  简单 session-mean t-test: t={t_simple:.3f}, p={p_simple:.6f}")
print(f"  这个 p 值是最保守的检验 — 如果它不显著，说明 session 间变异太大。")

# ── 4. 比较不同模型复杂度 ──────────────────
print("\n4. 模型复杂度比较")

# 最简单的：固定效应 only
formula = "auc ~ group"
model_simple = smf.mixedlm(formula, df_check, groups=df_check["session_id"])
result_simple = model_simple.fit(reml=False)
print(f"  Model 'auc ~ group': LL={result_simple.llf:.2f}, "
      f"group.coef={result_simple.fe_params.get('group[T.MP2 (mapping_change_only)]', 'N/A'):.6f}, "
      f"p={result_simple.pvalues.get('group[T.MP2 (mapping_change_only)]', float('nan')):.6f}")

# 加上 expertise
formula2 = "auc ~ group + expertise"
model2 = smf.mixedlm(formula2, df_check, groups=df_check["session_id"])
result2 = model2.fit(reml=False)
p_group = result2.pvalues.get('group[T.MP2 (mapping_change_only)]', float('nan'))
print(f"  Model 'auc ~ group + expertise': LL={result2.llf:.2f}, "
      f"group p={p_group:.6f}")

# 加上 half
formula3 = "auc ~ group + expertise + half"
model3 = smf.mixedlm(formula3, df_check, groups=df_check["session_id"])
result3 = model3.fit(reml=False)
p_group3 = result3.pvalues.get('group[T.MP2 (mapping_change_only)]', float('nan'))
print(f"  Model 'auc ~ group + expertise + half': LL={result3.llf:.2f}, "
      f"group p={p_group3:.6f}")

# ── 5. 检查随机效应方差 ──────────────────
print("\n5. 随机效应方差分解")
# 组内（session内）vs 组间（session间）方差
within_session_var = df_check.groupby('session_id')['auc'].var().mean()
between_session_var = df_check.groupby('session_id')['auc'].mean().var()
icc = between_session_var / (between_session_var + within_session_var)
print(f"  Session 内方差 (mean): {within_session_var:.8f}")
print(f"  Session 间方差:        {between_session_var:.8f}")
print(f"  ICC (组内相关系数):    {icc:.4f}")
print(f"  ICC > 0.05 说明必须用 LMM，ICC 接近 0 说明 LMM 退化为 OLS")

# ── 6. 可视化诊断 ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(14/2.54, 10/2.54))

# Panel A: Session-mean AUC by group
ax = axes[0, 0]
for group_name, color in [('MP1 (playback)', '#D6604D'), ('MP2 (mapping_change_only)', '#4393C3')]:
    subset = session_means[session_means['group'] == group_name]
    ax.scatter(np.random.normal(0, 0.05, len(subset)), subset['auc_mean'],
               alpha=0.7, s=15, color=color, label=group_name)
    ax.errorbar(0 if 'MP1' in group_name else 1,
                subset['auc_mean'].mean(),
                yerr=subset['auc_mean'].std()/np.sqrt(len(subset)),
                fmt='o', ms=8, color=color, ecolor=color, capsize=3, zorder=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['MP1\n(playback)', 'MP2\n(mapping_change)'], fontsize=6)
ax.set_ylabel('Session-mean AUC', fontsize=7)
ax.set_title(f'Session-mean AUC by Group\n(t={t_simple:.2f}, p={p_simple:.4f})', fontsize=7)
ax.legend(frameon=False, fontsize=5)
ax.axhline(0, color='k', ls='--', lw=0.5)
sns.despine(ax=ax, offset=3, trim=True)

# Panel B: Each neuron's AUC, colored by session
ax = axes[0, 1]
# For clarity, show only first 16 sessions
sessions_to_show = df_check['session_id'].unique()[:16]
subset_plot = df_check[df_check['session_id'].isin(sessions_to_show)]
sns.boxplot(data=subset_plot, x='session_id', y='auc', hue='group',
            palette={'MP1 (playback)': '#D6604D', 'MP2 (mapping_change_only)': '#4393C3'},
            linewidth=0.5, fliersize=0.5, ax=ax)
ax.axhline(0, color='k', ls='--', lw=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
ax.set_title('AUC by Session (first 16)', fontsize=7)
ax.set_ylabel('AUC', fontsize=7)
ax.legend(frameon=False, fontsize=5)
sns.despine(ax=ax, offset=3, trim=True)

# Panel C: Variance components
ax = axes[1, 0]
ax.bar(['Within-Session\n(neuron)', 'Between-Session\n(mean)'],
       [within_session_var, between_session_var],
       color=['#FDB863', '#B2ABD2'], edgecolor='#333333', lw=0.5)
ax.set_ylabel('Variance', fontsize=7)
ax.set_title(f'Variance Components (ICC={icc:.3f})', fontsize=7)
sns.despine(ax=ax, offset=3, trim=True)

# Panel D: Model comparison
ax = axes[1, 1]
models = ['auc ~ group', '~ group +\nexpertise', '~ group +\nexpertise + half']
lls = [result_simple.llf, result2.llf, result3.llf]
p_values = [result_simple.pvalues.get('group[T.MP2 (mapping_change_only)]', np.nan),
            result2.pvalues.get('group[T.MP2 (mapping_change_only)]', np.nan),
            result3.pvalues.get('group[T.MP2 (mapping_change_only)]', np.nan)]
ax.bar(models, [-np.log10(max(p, 1e-16)) for p in p_values],
       color=['#D6604D' if p < 0.05 else '#D1E5F0' for p in p_values],
       edgecolor='#333333', lw=0.5)
ax.axhline(-np.log10(0.05), color='#B2182B', ls='--', lw=0.8, label='p=0.05')
ax.set_ylabel('-log10(p) for Group effect', fontsize=7)
ax.set_title('Group Effect Across Models', fontsize=7)
ax.legend(frameon=False, fontsize=5)
for i, (ll, p) in enumerate(zip(lls, p_values)):
    ax.text(i, 0.1, f'p={p:.4f}', ha='center', fontsize=6)
sns.despine(ax=ax, offset=3, trim=True)

plt.tight_layout()
plt.show()

# ── 7. 最终诊断建议 ──────────────────
print("\n" + "=" * 65)
print("  DIAGNOSTIC SUMMARY")
print("=" * 65)
if p_simple > 0.05:
    print("  [!] Session-mean t-test 也不显著 — session 间变异太大或样本太小。")
    print("      建议：增加 session 数量，或使用更强的 within-session 设计。")
if icc < 0.01:
    print("  [!] ICC 接近 0 — session 间差异很小，LMM 退化为 OLS。")
    print("      可以简化模型，去掉随机效应。")
if len(df_check['session_id'].unique()) < 8:
    print("  [!] Session 太少 (< 8) — 随机效应模型可能不可靠。")
elif len(df_check['session_id'].unique()) < 15:
    print("  [i] Session 数偏少 (8-15) — 随机效应估计可能不稳定。")
else:
    print("  [OK] Session 数量充足。")
if p_simple < 0.05 and icc > 0.05:
    print("  [OK] Session-mean 差异显著 + ICC 非零 = LMM 应该能检测到效应。")
    print("       如果 LMM 不显著，检查模型是否过拟合（固定效应太多）。")
