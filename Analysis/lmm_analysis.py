r"""
LMM Analysis: No vs Yes mapping
==========================================================
Linear Mixed-Effects Model with Random Intercept + Random Slope,
Wald tests, Likelihood Ratio tests, and Post-hoc comparisons.

Key design: Tracking vs Playback is an explicit within-neuron factor (condition),
so the model directly tests Track-vs-PB and whether this difference varies
between MP1 (playback) and MP2 (mapping_change_only).

Response: mean firing rate (baseline-subtracted) in 0-100ms post-trigger.

Data hierarchy:
  - condition (Track / Playback): within-neuron, repeated measure
  - half (H1 / H2): within-session
  - expertise (Beginner / Expert): between-session
  - mapping (No / Yes): between-session
  - session: random effect

Usage: Run from notebook with:
  import sys; sys.path.insert(0, r'C:\Users\PenPen\Ferret\Analysis')
  from lmm_analysis import run_lmm_analysis
  result, df_lmm = run_lmm_analysis(traj_by_half_mp1, traj_by_half_mp2,
                                     time, save_directory)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.integrate import trapezoid
from scipy.stats import chi2, ttest_1samp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


# ===============================================================
# STEP 1: Build long-format DataFrame (Track + PB as separate rows)
# ===============================================================

def build_lmm_dataframe(traj_by_half, mapping_label, time_array,
                        auc_t_start=0.0, auc_t_end=0.1, split_half=True,
                        metric='mean'):
    """
    Build a long-format DataFrame where each neuron contributes TWO rows:
    one for Tracking and one for Playback. This allows condition (Track vs PB)
    to be an explicit fixed effect in the LMM.

    Parameters
    ----------
    traj_by_half : dict
        From build_traj(). Structure:
        {(0.0, 0.5): {'track': {'beg': [...], 'exp': [...]},
                       'pb':    {'beg': [...], 'exp': [...]}},
         (0.5, 1.0): {...}}
        Each [...] is a list of [121, n_neurons] arrays, one per session.
    mapping_label : str
        'MP1' or 'MP2'
    time_array : np.ndarray
        Time vector for the PSTH window.
    auc_t_start, auc_t_end : float
        Time window for mean activity (seconds post-trigger).
    split_half : bool
        If True, split trials into H1 (first half) and H2 (second half).
        If False, use all trials (half='All').
    metric : str
        'mean' -> average over window; 'peak' -> maximum in window.

    Returns
    -------
    pd.DataFrame with columns:
        response, condition, mapping, expertise, half, session_id, neuron_id
    """
    idx_start = np.searchsorted(time_array, auc_t_start)
    idx_end   = np.searchsorted(time_array, auc_t_end)
    if metric == 'peak':
        agg_fn = np.nanmax
    else:
        agg_fn = np.nanmean

    records = []
    neuron_global_id = 0

    halves_map = [
        ((0.0, 0.5), 'H1'),
        ((0.5, 1.0), 'H2'),
    ]
    expertise_map = [
        ('beg', 'Beginner'),
        ('exp', 'Expert'),
    ]

    for exp_key, exp_name in expertise_map:
        # Collect per-session per-half means, then optionally combine
        session_data = {}  # sess_idx -> {'track': [h1_mean, h2_mean], 'pb': [...]}

        for half_key, half_name in halves_map:
            track_sessions = traj_by_half[half_key]['track'][exp_key]
            pb_sessions    = traj_by_half[half_key]['pb'][exp_key]
            n_sessions     = len(track_sessions)

            for sess_idx in range(n_sessions):
                track = track_sessions[sess_idx]
                pb    = pb_sessions[sess_idx]

                if track is None or pb is None:
                    continue

                # Mean activity in post-trigger window, baseline-subtracted
                mean_track = agg_fn(track[idx_start:idx_end, :], axis=0)
                mean_pb    = agg_fn(pb[idx_start:idx_end, :],    axis=0)

                if sess_idx not in session_data:
                    session_data[sess_idx] = {
                        'track': [], 'pb': [],
                        'n_neurons': len(mean_track),
                    }
                session_data[sess_idx]['track'].append(mean_track)
                session_data[sess_idx]['pb'].append(mean_pb)

        for sess_idx, sd in session_data.items():
            n_neurons = sd['n_neurons']
            session_str = f"{mapping_label}_{exp_name}_s{sess_idx:03d}"

            if split_half:
                # Two rows per neuron: H1 and H2
                for hi, half_name in enumerate(['H1', 'H2']):
                    mean_track = sd['track'][hi]
                    mean_pb    = sd['pb'][hi]
                    for n in range(n_neurons):
                        records.append({
                            'response': mean_track[n], 'condition': 'Track',
                            'mapping': mapping_label, 'expertise': exp_name,
                            'half': half_name, 'session_id': session_str,
                            'neuron_id': f"{mapping_label}_{session_str}_n{neuron_global_id}",
                        })
                        records.append({
                            'response': mean_pb[n], 'condition': 'Playback',
                            'mapping': mapping_label, 'expertise': exp_name,
                            'half': half_name, 'session_id': session_str,
                            'neuron_id': f"{mapping_label}_{session_str}_n{neuron_global_id}",
                        })
                        neuron_global_id += 1
            else:
                # Average H1 and H2 for ALL-trial response
                mean_track = np.mean(sd['track'], axis=0)
                mean_pb    = np.mean(sd['pb'], axis=0)
                for n in range(n_neurons):
                    records.append({
                        'response': mean_track[n], 'condition': 'Track',
                        'mapping': mapping_label, 'expertise': exp_name,
                        'half': 'All', 'session_id': session_str,
                        'neuron_id': f"{mapping_label}_{session_str}_n{neuron_global_id}",
                    })
                    records.append({
                        'response': mean_pb[n], 'condition': 'Playback',
                        'mapping': mapping_label, 'expertise': exp_name,
                        'half': 'All', 'session_id': session_str,
                        'neuron_id': f"{mapping_label}_{session_str}_n{neuron_global_id}",
                    })
                    neuron_global_id += 1

    return pd.DataFrame(records)


# ===============================================================
# STEP 2: Run the full LMM analysis
# ===============================================================

def run_lmm_analysis(traj_by_half_mp1, traj_by_half_mp2, time_array,
                     save_directory=None, auc_t_start=0.0, auc_t_end=0.1,
                     metric='mean'):
    """
    Complete LMM pipeline with Track vs Playback as an explicit factor.

    The key model formula includes `condition` (Track / Playback) as a
    fixed effect, testing:
      1. condition main effect: is there a Track-to-PB change overall?
      2. condition:mapping interaction: does this change differ between
         No and Yes mapping sessions?

    Parameters
    ----------
    traj_by_half_mp1, traj_by_half_mp2 : dict
        Trajectory dictionaries from build_traj().
    time_array : np.ndarray
        PSTH time vector.
    save_directory : str or None
        If provided, save figure PDFs.

    Returns
    -------
    result : statsmodels MixedLMResults
        Fitted LMM (random intercept, REML).
    df_lmm : pd.DataFrame
        Long-format DataFrame (2 rows per neuron: Track + PB).
    """
    # -- 2a. Assemble DataFrame ---------------------------------
    print("Building LMM DataFrame (long format: Track + PB as separate rows)...")
    df_mp1 = build_lmm_dataframe(traj_by_half_mp1, 'MP1', time_array, auc_t_start=auc_t_start, auc_t_end=auc_t_end, metric=metric)
    df_mp2 = build_lmm_dataframe(traj_by_half_mp2, 'MP2', time_array, auc_t_start=auc_t_start, auc_t_end=auc_t_end, metric=metric)
    df_lmm = pd.concat([df_mp1, df_mp2], ignore_index=True)

    n_rows = len(df_lmm)
    n_neurons = df_lmm['neuron_id'].nunique()
    n_sessions = df_lmm['session_id'].nunique()
    print(f"  Total rows: {n_rows} ({n_neurons} neurons x 2 conditions)")
    print(f"  Unique sessions: {n_sessions}")
    for g in df_lmm['mapping'].unique():
        mask = df_lmm['mapping'] == g
        print(f"    {g}: {df_lmm[mask]['session_id'].nunique()} sessions, "
              f"{df_lmm[mask]['neuron_id'].nunique()} neurons")

    # -- 2b. Descriptive summary -------------------------------
    print("\n-- Data Summary (mean response +/- SEM) --")
    summary = df_lmm.groupby(['mapping', 'condition', 'expertise', 'half'])['response'].agg(
        ['mean', 'std', 'count', 'sem']
    ).round(5)
    print(summary.to_string())

    # Track-to-PB difference summary
    print("\n-- Track-to-PB Difference (PB - Track) --")
    # Compute per-neuron difference
    df_wide = df_lmm.pivot_table(
        index=['neuron_id', 'session_id', 'mapping', 'expertise', 'half'],
        columns='condition', values='response'
    ).reset_index()
    df_wide['diff'] = df_wide['Playback'] - df_wide['Track']
    diff_summary = df_wide.groupby(['mapping', 'expertise', 'half'])['diff'].agg(
        ['mean', 'std', 'count', 'sem']
    ).round(5)
    print(diff_summary.to_string())

    # -- 2c. Fit models ----------------------------------------
    # Group by NEURON (not session): each neuron has paired Track + PB observations.
    # The random intercept per neuron absorbs baseline firing rate differences,
    # so condition (Track vs PB) tests the within-neuron effect.
    formula_full = "response ~ condition * mapping * expertise * half"
    n_neurons = df_lmm['neuron_id'].nunique()
    n_per_neuron = df_lmm.groupby('neuron_id').size()

    print(f"\n-- Design Summary --")
    print(f"  Total neurons: {n_neurons} (each has {n_per_neuron.iloc[0]:.0f} rows: Track/PB x H1/H2)")
    print(f"  Total sessions: {n_sessions}")
    print(f"  Sessions per mapping: "
          f"{df_lmm.groupby('mapping')['session_id'].nunique().to_dict()}")
    print(f"  Grouping variable: neuron_id (paired Track-PB within each neuron)")
    print(f"  Full formula: {formula_full}")

    # Model 1: Random Intercept per neuron (REML)
    print("\n-- Fitting Model 1: Random Intercept per neuron (REML) --")
    model1 = smf.mixedlm(formula_full, df_lmm, groups=df_lmm["neuron_id"],
                         vc_formula={"session_id": "0 + C(session_id)"})
    result1 = model1.fit(reml=True)
    print(f"  Converged: {result1.converged}, LogLik: {result1.llf:.2f}")

    # Model 2: Random Intercept + Random Slope for condition
    # Each neuron gets its own Track->PB effect size
    print(f"\n-- Fitting Model 2: + Random Slope for condition (REML) --")
    try:
        model2 = smf.mixedlm(
            formula_full, df_lmm, groups=df_lmm["neuron_id"],
            re_formula="~ condition",
            vc_formula={"session_id": "0 + C(session_id)"}
        )
        result2 = model2.fit(reml=True)
        print(f"  Converged: {result2.converged}, LogLik: {result2.llf:.2f}")
        used_random_slope = True
        if not result2.converged:
            print(f"  WARNING: Model 2 did not converge! Falling back to Model 1.")
            result2 = result1
            used_random_slope = False
    except Exception as e:
        print(f"  WARNING: Model 2 failed: {e}")
        print(f"    Falling back to Random Intercept only.")
        result2 = result1
        used_random_slope = False

    print(result2.summary())

    # Refit with ML for Likelihood Ratio Tests
    print("\n-- Refitting with ML for LRT --")
    result1_ml = smf.mixedlm(formula_full, df_lmm, groups=df_lmm["neuron_id"],
                             vc_formula={"session_id": "0 + C(session_id)"}).fit(reml=False)
    print(f"  Model 1 (ML): LL = {result1_ml.llf:.2f}")
    if used_random_slope:
        result2_ml = smf.mixedlm(
            formula_full, df_lmm, groups=df_lmm["neuron_id"],
            re_formula="~ condition",
            vc_formula={"session_id": "0 + C(session_id)"}
        ).fit(reml=False)
        print(f"  Model 2 (ML): LL = {result2_ml.llf:.2f}")
    else:
        result2_ml = result1_ml

    # ==========================================================
    # WALD TESTS
    # ==========================================================
    print("\n" + "=" * 65)
    print("  WALD TESTS (joint test per term)")
    print("=" * 65)
    try:
        wald_result = result2.wald_test_terms()
        print(wald_result)
    except Exception:
        print("(Wald test terms unavailable; see coefficient z-tests in summary above)")

    # Highlight the key tests
    print("\n-- Key Coefficient Tests --")
    coef_names = result2.fe_params.index
    for name in coef_names:
        if 'condition' in name.lower():
            coef = result2.fe_params[name]
            p    = result2.pvalues[name]
            stars = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
            print(f"  {name:<50s}: coef={coef:+.6f}, p={p:.6f}  {stars}")

    # ==========================================================
    # LIKELIHOOD RATIO TESTS
    # ==========================================================
    print("\n" + "=" * 65)
    print("  LIKELIHOOD RATIO TESTS")
    print("=" * 65)

    re_formula_lrt = "~ condition" if used_random_slope else None

    # --- LRT a: Random slope significance ---
    print(f"\n  {'Test':<50} {'LRT stat':>10} {'df':>4} {'p-value':>10}")
    print(f"  {'-'*74}")
    if used_random_slope:
        lr_stat_re = 2 * max(0, result2_ml.llf - result1_ml.llf)
        # random slope for condition (2-level factor) adds var + cov = 2 params
        df_re = 2
        p_re = chi2.sf(lr_stat_re, df_re)
        print(f"  {'Random slope for condition':<50} {lr_stat_re:10.4f} {df_re:4d} {p_re:10.6f}")
        print(f"    H0: var(condition slope) = 0")
    else:
        print(f"  {'Random slope for condition':<50} {'--':>10} {'--':>4} {'--':>10}")
        print(f"    Not fitted.")

    # --- LRT b: condition main effect (Track vs PB) ---
    formula_no_cond = "response ~ mapping * expertise * half"
    try:
        model_no_cond = smf.mixedlm(
            formula_no_cond, df_lmm, groups=df_lmm["neuron_id"],
            re_formula=re_formula_lrt,
            vc_formula={"session_id": "0 + C(session_id)"}
        ).fit(reml=False)
    except np.linalg.LinAlgError:
        print("    (session_id VC singular in reduced model; retrying without it)")
        model_no_cond = smf.mixedlm(
            formula_no_cond, df_lmm, groups=df_lmm["neuron_id"],
            re_formula=re_formula_lrt
        ).fit(reml=False)

    lr_stat_cond = 2 * max(0, result2_ml.llf - model_no_cond.llf)
    df_cond = len(result2_ml.fe_params) - len(model_no_cond.fe_params)
    p_cond = chi2.sf(lr_stat_cond, df_cond)

    print(f"\n  {'condition (Track vs PB)':<50} {lr_stat_cond:10.4f} {df_cond:4d} {p_cond:10.6f}")
    print(f"    H0: no difference between Tracking and Playback")
    print(f"    Full LL={result2_ml.llf:.2f}, No-condition LL={model_no_cond.llf:.2f}")
    if p_cond < 0.05:
        print(f"    ---> Significant Track-vs-PB difference detected!")
    else:
        print(f"    ---> No significant Track-vs-PB difference.")

    # --- LRT c: condition:mapping interaction (key test) ---
    # Build reduced model by stripping condition:mapping AND all higher-order
    # terms that contain it (condition:mapping:expertise, condition:mapping:half,
    # condition:mapping:expertise:half).  All other terms stay.
    formula_no_cond_group = (
        "response ~ condition + mapping + expertise + half + "
        "condition:expertise + condition:half + "
        "mapping:expertise + mapping:half + "
        "expertise:half + "
        "condition:expertise:half + mapping:expertise:half"
    )
    model_no_cg = smf.mixedlm(
        formula_no_cond_group, df_lmm, groups=df_lmm["neuron_id"],
        re_formula=re_formula_lrt,
        vc_formula={"session_id": "0 + C(session_id)"}
    )
    try:
        model_no_cg = model_no_cg.fit(reml=False)
    except np.linalg.LinAlgError:
        print("    (session_id VC singular in reduced model; retrying without it)")
        model_no_cg = smf.mixedlm(
            formula_no_cond_group, df_lmm, groups=df_lmm["neuron_id"],
            re_formula=re_formula_lrt
        ).fit(reml=False)

    lr_stat_cg = 2 * max(0, result2_ml.llf - model_no_cg.llf)
    df_cg = len(result2_ml.fe_params) - len(model_no_cg.fe_params)
    p_cg = chi2.sf(lr_stat_cg, df_cg)

    print(f"\n  {'condition:mapping interaction':<50} {lr_stat_cg:10.4f} {df_cg:4d} {p_cg:10.6f}")
    print(f"    H0: Track-vs-PB difference is the SAME in No and Yes mapping")
    print(f"    Full LL={result2_ml.llf:.2f}, No-interaction LL={model_no_cg.llf:.2f}")
    if p_cg < 0.05:
        print(f"    ---> No and Yes mapping differ in their Track-to-PB modulation!")
    else:
        print(f"    ---> Track-to-PB modulation does NOT differ between No and Yes mapping.")

    # --- LRT summary table ---
    lrt_results = [
        ("condition (Track vs PB)", lr_stat_cond, df_cond, p_cond),
        ("condition:mapping (key test)", lr_stat_cg, df_cg, p_cg),
    ]
    print(f"\n  {'-'*74}")
    print(f"  {'LRT Summary':^74}")
    print(f"  {'-'*74}")
    print(f"  {'Effect':<40} {'chi2':>10} {'df':>4} {'p':>10}  {'Sig.'}")
    print(f"  {'-'*66}")
    for name, stat, d, p in lrt_results:
        sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
        print(f"  {name:<40} {stat:10.4f} {d:4d} {p:10.6f}  {sig}")

    # ==========================================================
    # POST-HOC PAIRWISE COMPARISONS
    # ==========================================================
    print("\n" + "=" * 65)
    print("  POST-HOC: Track vs PB at each Mapping x Expertise x Half")
    print("=" * 65)

    combinations = [
        ('MP1', 'Beginner', 'H1'),
        ('MP1', 'Beginner', 'H2'),
        ('MP1', 'Expert',   'H1'),
        ('MP1', 'Expert',   'H2'),
        ('MP2', 'Beginner', 'H1'),
        ('MP2', 'Beginner', 'H2'),
        ('MP2', 'Expert',   'H1'),
        ('MP2', 'Expert',   'H2'),
    ]

    print(f"\n  {'Mapping x Expertise x Half':<35} {'Track mean':>10} {'PB mean':>10} "
          f"{'Diff':>10} {'t':>8} {'p_raw':>8} {'p_fdr':>8}  {'Sig.'}")
    print(f"  {'-'*97}")

    posthoc_results = []
    for mapping_val, exp, half in combinations:
        mask = ((df_wide["mapping"] == mapping_val) &
                (df_wide["expertise"] == exp) &
                (df_wide["half"] == half))
        subset = df_wide[mask]
        track_vals = subset['Track'].values
        pb_vals    = subset['Playback'].values
        diff_vals  = pb_vals - track_vals

        t_stat, p_val = ttest_1samp(diff_vals, 0)
        posthoc_results.append({
            'contrast': f"{mapping_val[:3]} {exp} {half}",
            'track_mean': np.mean(track_vals),
            'pb_mean': np.mean(pb_vals),
            'diff': np.mean(diff_vals),
            't_stat': t_stat,
            'p_raw': p_val,
            'n': len(track_vals),
        })

    # FDR correction
    p_raw = [r['p_raw'] for r in posthoc_results]
    reject, p_fdr, _, _ = multipletests(p_raw, method='fdr_bh')

    for r, p_corr, rej in zip(posthoc_results, p_fdr, reject):
        sig = '***' if p_corr<0.001 else ('**' if p_corr<0.01 else ('*' if p_corr<0.05 else 'n.s.'))
        print(f"  {r['contrast']:<35} {r['track_mean']:10.5f} {r['pb_mean']:10.5f} "
              f"{r['diff']:10.5f} {r['t_stat']:8.3f} {r['p_raw']:8.4f} {p_corr:8.4f}  {sig}")
        r['p_fdr'] = p_corr
        r['significant'] = rej

    # ==========================================================
    # VISUALIZATION
    # ==========================================================
    print("\n-- Generating LMM diagnostic plots --")

    # Short labels for x-axes (avoid dense unreadable text)
    df_lmm['mapping_short'] = df_lmm['mapping'].map({
        'MP1': 'MP1',
        'MP2': 'MP2',
    })
    df_wide['mapping_short'] = df_wide['mapping'].map({
        'MP1': 'MP1',
        'MP2': 'MP2',
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Panel A: Track vs PB boxplot, split by mapping
    ax = axes[0, 0]
    sns.boxplot(data=df_lmm, x='mapping_short', y='response', hue='condition',
                palette={'Track': '#D6604D', 'Playback': '#2C2C2A'},
                linewidth=0.8, fliersize=1, ax=ax)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_title('Track vs Playback by Mapping', fontsize=10)
    ax.set_ylabel('Mean response (0-100ms)', fontsize=9)
    ax.set_xlabel('')
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax, offset=3, trim=True)

    # Panel B: Track-to-PB difference by mapping x expertise
    ax = axes[0, 1]
    sns.boxplot(data=df_wide, x='expertise', y='diff', hue='mapping_short',
                palette={'MP1': '#FDB863',
                         'MP2': '#B2ABD2'},
                linewidth=0.8, fliersize=1, ax=ax)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_title('PB - Track Difference', fontsize=10)
    ax.set_ylabel('Diff (PB - Track)', fontsize=9)
    ax.set_xlabel('Expertise', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax, offset=3, trim=True)

    # Panel C: 按 group 分面的 cond x half 对比（替代拥挤的 combo 标签）
    ax = axes[1, 0]
    # 只画 MP2 的 group + expert x half 组合，避免标签重叠
    df_wide['label'] = (df_wide['mapping_short'].str.replace('MP1', 'M1')
                                          .str.replace('MP2', 'M2')
                        + ' ' + df_wide['expertise'].str[:3] + ' ' + df_wide['half'])
    label_order = sorted(df_wide['label'].unique())
    sns.boxplot(data=df_wide, x='label', y='diff',
                color='#B2ABD2', linewidth=0.8, fliersize=1, ax=ax)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_title('PB - Track by Mapping / Expertise / Half', fontsize=10)
    ax.set_ylabel('Diff (PB - Track)', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
    ax.set_xlabel('')
    sns.despine(ax=ax, offset=3, trim=True)

    # Panel D: LRT summary
    ax = axes[1, 1]
    lrt_names = [r[0] for r in lrt_results]
    lrt_ps    = [r[3] for r in lrt_results]
    colors    = ['#B2182B' if p < 0.05 else '#D1E5F0' for p in lrt_ps]
    ax.barh(lrt_names, [-np.log10(max(p, 1e-16)) for p in lrt_ps],
            color=colors, edgecolor='#333333', lw=0.5)
    ax.axvline(-np.log10(0.05), color='#B2182B', ls='--', lw=0.8, label='p=0.05')
    ax.axvline(-np.log10(0.01), color='#B2182B', ls=':', lw=0.5, label='p=0.01')
    ax.set_xlabel('-log10(p)', fontsize=9)
    ax.set_title('LRT Summary', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    sns.despine(ax=ax, offset=3, trim=True)

    plt.tight_layout(pad=1.2)
    if save_directory:
        plt.savefig(save_directory + 'lmm_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_directory + 'lmm_analysis.svg', bbox_inches='tight')
    plt.show()
    print("\n  Tip: run '%matplotlib qt' before this cell for a resizable popup window.")

    return result2, df_lmm


# ===============================================================
# Convenience: compare random-effect structures
# ===============================================================

def compare_random_effect_structures(df_lmm, formula="response ~ condition * mapping * expertise * half"):
    """
    Compare different random effect structures using AIC and BIC.
    """
    print("\n" + "=" * 65)
    print("  RANDOM EFFECTS STRUCTURE COMPARISON")
    print("=" * 65)

    structures = [
        ("Fixed effects only", None),
        ("+ Random Intercept", None),  # handled separately
        ("+ Random Intercept + Slope for condition", "~ condition"),
    ]

    results_list = []

    # Model 0: Fixed effects only (true OLS, no random effects)
    print("\n  Fitting: Fixed effects only (OLS) ...")
    m0 = smf.ols(formula, df_lmm)
    r0 = m0.fit()

    # Model 1: Random intercept (default)
    print("  Fitting: Random Intercept ...")
    # Already done, but refit for consistency
    m1 = smf.mixedlm(formula, df_lmm, groups=df_lmm["neuron_id"])
    r1 = m1.fit(reml=False)

    # Model 2: Random intercept + condition slope
    print("  Fitting: Random Intercept + Slope for condition ...")
    try:
        m2 = smf.mixedlm(formula, df_lmm, groups=df_lmm["neuron_id"],
                          re_formula="~ condition")
        r2 = m2.fit(reml=False)
        has_slope = True
    except Exception:
        r2 = r1
        has_slope = False
        print("    (failed, using Random Intercept only)")

    models = [
        ("Fixed effects only", r0),
        ("Random Intercept", r1),
        ("Random Int + Slope (condition)", r2),
    ]

    print(f"\n  {'Model':<35} {'LL':>12} {'AIC':>10} {'BIC':>10}  {'vs prev chi2':>12}  {'p':>8}")
    print(f"  {'-'*93}")

    prev = None
    for name, res in models:
        # Handle both OLS (params) and MixedLM (fe_params + cov_re)
        if hasattr(res, 'fe_params'):
            n_params = len(res.fe_params) + res.cov_re.shape[0] + 1  # MixedLM: fixed + random + residual
        else:
            n_params = len(res.params) + 1  # OLS: params + residual
        aic = -2 * res.llf + 2 * n_params
        bic = -2 * res.llf + np.log(len(df_lmm)) * n_params
        if prev is None:
            print(f"  {name:<35} {res.llf:12.2f} {aic:10.2f} {bic:10.2f}")
        else:
            lr_stat = 2 * max(0, res.llf - prev['llf'])
            df_diff = n_params - prev['n_params']
            p_val = chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
            print(f"  {name:<35} {res.llf:12.2f} {aic:10.2f} {bic:10.2f}  "
                  f"{lr_stat:12.4f}  {p_val:8.6f}" if not np.isnan(p_val) else
                  f"  {name:<35} {res.llf:12.2f} {aic:10.2f} {bic:10.2f}  {'--':>12}  {'--':>8}")
        prev = {'name': name, 'llf': res.llf, 'n_params': n_params, 'aic': aic, 'bic': bic}
        results_list.append(prev)

    best = min(results_list, key=lambda x: x['aic'])
    print(f"\n  -> Best model by AIC: {best['name']} (AIC={best['aic']:.2f})")
    return results_list


# ===============================================================
# MAIN: called from notebook
# ===============================================================

if __name__ == "__main__":
    try:
        result, df_lmm = run_lmm_analysis(
            traj_by_half_mp1, traj_by_half_mp2, time, save_directory
        )
        compare_random_effect_structures(df_lmm)
    except NameError as e:
        print(f"Variable not defined: {e}")
        print("Run this script from the notebook with:  %run -i Analysis/lmm_analysis.py")
        print("Or import and call: from lmm_analysis import run_lmm_analysis")
