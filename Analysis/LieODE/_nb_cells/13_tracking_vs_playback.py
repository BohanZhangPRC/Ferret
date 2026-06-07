# ============================================================
# Phase 4 -- Tracking vs Playback Paired Analysis
# ============================================================
# Note: SR/eigenvalues are from the shared session-level generator J(u)+L,
# so Tracking/PB comparisons are only meaningful for R2_drive (held-out epochs).
# SR/eig are reported as session-level summaries (pooled across conditions).
#
# CAVEAT -- kinematic confound: drive is standardized across pooled TR+PB epochs.
# If the animal moves less during Playback, the drive dynamic range is smaller,
# which may systematically lower R2_drive_rollout for Playback INDEPENDENTLY of
# neural computation.  Observed TR > PB differences cannot be attributed to
# neural mechanisms unless velocity distributions are first shown comparable
# (cf. lie_algebra_method_description.md section 12.9).

# ---- Velocity distribution check (kinematic confound) ----
print("=" * 60)
print("  Velocity Distribution Check (Kinematic Confound)")
print("=" * 60)

vel_stats = []
for idx, (n_data_session, f_df) in enumerate(zip(n_data_all, f_data_all)):
    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        v = f_df[f_df["Condition"] == val]["Velocity_x"].values
        vel_stats.append({
            "Session_Idx": idx, "Condition": label,
            "Mean": np.mean(v), "Std": np.std(v),
            "RMS": np.sqrt(np.mean(v**2)),
            "Range": np.ptp(v),
            "N_tp": len(v),
        })
vel_df = pd.DataFrame(vel_stats)
print(vel_df.groupby("Condition")[["Mean", "Std", "RMS", "Range"]].mean().round(2).to_string())
print()

# Paired t-test on velocity metrics
vel_pivot = vel_df.pivot_table(
    values=["Mean", "Std", "RMS", "Range"],
    index="Session_Idx", columns="Condition").dropna()
if len(vel_pivot) > 1:
    print("  Tracking vs Playback velocity paired t-tests:")
    for metric in ["RMS", "Std", "Range"]:
        t, p = ttest_rel(vel_pivot[metric]["Tracking"],
                         vel_pivot[metric]["Playback"])
        print(f"    {metric:10s}: TR={vel_pivot[metric]['Tracking'].mean():.2f}, "
              f"PB={vel_pivot[metric]['Playback'].mean():.2f}, "
              f"t={t:.3f}, p={p:.4f}")
    # Per-session KS test (formal distribution comparison)
    from scipy.stats import ks_2samp
    ks_results = []
    for idx, f_df in enumerate(f_data_all):
        v_tr = f_df[f_df["Condition"] == 0.0]["Velocity_x"].values
        v_pb = f_df[f_df["Condition"] == 1.0]["Velocity_x"].values
        if len(v_tr) > 10 and len(v_pb) > 10:
            ks_stat, ks_p = ks_2samp(v_tr, v_pb)
            ks_results.append({"Session_Idx": idx, "KS_stat": ks_stat, "KS_p": ks_p})
    if ks_results:
        ks_df = pd.DataFrame(ks_results)
        n_sig = (ks_df["KS_p"] < 0.05).sum()
        print(f"  KS test (Tracking vs Playback per session):")
        print(f"    Sessions with p<0.05: {n_sig}/{len(ks_df)}")
        print(f"    Mean KS stat: {ks_df['KS_stat'].mean():.3f}, "
              f"median p: {ks_df['KS_p'].median():.3f}")
        if n_sig > len(ks_df) / 2:
            print(f"    WARNING: majority of sessions show significantly")
            print(f"      different velocity distributions -- kinematic confound")
            print(f"      is likely.")
    print()

# Velocity histogram
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
for i, (cond, color) in enumerate([("Tracking", "#440154"), ("Playback", "#21918c")]):
    all_v = np.concatenate([
        f_df[f_df["Condition"] == (0.0 if cond == "Tracking" else 1.0)]["Velocity_x"].values
        for f_df in f_data_all])
    axes[i].hist(all_v, bins=50, color=color, alpha=0.7, density=True)
    axes[i].set_title(f"{cond}\n(RMS={np.sqrt(np.mean(all_v**2)):.1f}, "
                      f"N={len(all_v):,} tp)")
    axes[i].set_xlabel("Velocity_x")
axes[0].set_ylabel("Density")
plt.suptitle("Velocity Distribution: Tracking vs Playback (all sessions)",
             y=1.05, fontsize=9, fontweight="bold")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"Velocity_Distribution_TR_PB.{fmt}"),
                dpi=150, bbox_inches="tight")
plt.show()

# ---- Per-condition SR (condition-specific drive distribution) ----
if 'e2e_session_df' in dir() and e2e_session_df is not None:
    if 'SR_Tracking' in e2e_session_df.columns and 'SR_Playback' in e2e_session_df.columns:
        print("  Per-condition SR (drive-distribution-specific):")
        sr_tr = e2e_session_df['SR_Tracking'].dropna()
        sr_pb = e2e_session_df['SR_Playback'].dropna()
        # Align by session
        common = e2e_session_df[['Session_Idx', 'SR_Tracking', 'SR_Playback']].dropna()
        print(f"    SR_Tracking: {common['SR_Tracking'].mean():.4f} +- {common['SR_Tracking'].sem():.4f}")
        print(f"    SR_Playback:  {common['SR_Playback'].mean():.4f} +- {common['SR_Playback'].sem():.4f}")
        if len(common) > 1:
            t_sr, p_sr = ttest_rel(common['SR_Tracking'], common['SR_Playback'])
            print(f"    Paired t-test: t={t_sr:.3f}, p={p_sr:.4f}")
            if p_sr < 0.05:
                print(f"    -> Tracking SR significantly higher: rotation specifically")
                print(f"       enhanced, not just global gain modulation.")
        print()

if e2e_df is not None:
    print("=" * 60)
    print("  Tracking vs Playback -- Paired Comparison")
    print("=" * 60)

    # ---- Session-level SR/eig (pooled across conditions) ----
    if 'e2e_session_df' in dir() and e2e_session_df is not None:
        print("--- Session-level Generator Metrics (shared J(u)+L) ---")
        print(f"  Mean SR:           {e2e_session_df['SR'].mean():.4f} "
              f"(sem={e2e_session_df['SR'].sem():.4f})")
        print(f"  N sessions:        {len(e2e_session_df)}")
        print(f"  (Eigenvalues are per-session diagnostics in arbitrary encoder-scale")
        print(f"   units — not comparable across independently-trained sessions.")
        print(f"   See per-session table in Cell 10 for individual values.)")
        print()

    # ---- Per-condition R2_drive: Tracking vs Playback ----
    # Stratify by Window_Bins: each horizon gets its own paired test
    # Per-horizon paired t-tests
    for wlen in VAL_ROLLOUT_LENS:
        sub = e2e_df[e2e_df["Window_Bins"] == wlen]
        pivot_e2e = sub.pivot_table(
            values=["R2_drive_rollout"],
            index=["Subject", "Session_Idx", "Headstage"],
            columns="Condition").dropna()
        if len(pivot_e2e) > 1:
            print(f"    {wlen} bins ({wlen*dt*1000:.0f}ms): ", end="")
            tr = pivot_e2e["R2_drive_rollout"]["Tracking"].values
            pb = pivot_e2e["R2_drive_rollout"]["Playback"].values
            t, p = ttest_rel(tr, pb)
            print(f"TR={np.mean(tr):.6f}, PB={np.mean(pb):.6f}, "
                  f"t={t:.3f}, p={p:.4f}")
        else:
            print(f"    {wlen} bins: not enough paired sessions")

    # ---- Visualization (primary horizon = shortest window) ----
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    sub_primary = e2e_df[e2e_df["Window_Bins"] == VAL_ROLLOUT_LENS[0]]
    pivot_primary = sub_primary.pivot_table(
        values=["R2_drive_rollout"],
        index=["Subject", "Session_Idx", "Headstage"],
        columns="Condition").dropna()

    # Panel A: Session-level SR histogram
    if 'e2e_session_df' in dir() and e2e_session_df is not None:
        axes[0].hist(e2e_session_df["SR"], bins=min(10, len(e2e_session_df)),
                     color="#440154", alpha=0.7, edgecolor='white')
        axes[0].axvline(e2e_session_df["SR"].mean(), color='#21918c',
                        lw=1.5, ls='--', label=f'Mean={e2e_session_df["SR"].mean():.3f}')
        axes[0].set_xlabel("Skewness Ratio"); axes[0].set_ylabel("Sessions")
        axes[0].set_title("SR Distribution (E2E)")
        axes[0].legend(fontsize=5)

    # Panel B: R2_drive Tracking vs Playback paired (primary horizon)
    if len(pivot_primary) > 1:
        tr_vals = pivot_primary["R2_drive_rollout"]["Tracking"].values
        pb_vals = pivot_primary["R2_drive_rollout"]["Playback"].values
        for i in range(len(tr_vals)):
            axes[1].plot([0, 1], [tr_vals[i], pb_vals[i]], '-',
                         c='gray', lw=0.4, alpha=0.5)
        axes[1].scatter(np.zeros(len(tr_vals)), tr_vals, c="#440154",
                        s=30, zorder=3, label="Tracking")
        axes[1].scatter(np.ones(len(pb_vals)), pb_vals, c="#21918c",
                        s=30, zorder=3, label="Playback")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Tracking", "Playback"], fontsize=7)
        axes[1].set_ylabel("R2_drive (Rollout)")
        axes[1].set_title("R2_drive: TR vs PB")
        axes[1].legend(fontsize=5)

    plt.suptitle("End-to-End Lie-ODE -- Session Metrics",
                 y=1.02, fontsize=9, fontweight="bold")
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(LIE_OUTPUT_DIR,
                                 f"E2E_Tracking_vs_Playback.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("Skipped: no E2E data available.")
