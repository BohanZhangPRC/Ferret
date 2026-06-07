# ============================================================
# Phase 3 -- Results Comparison: Baseline vs End-to-End vs Dummy
# ============================================================

if e2e_df is not None and baseline_df is not None:
    print("=" * 60)
    print("  Baseline vs End-to-End Comparison")
    print("=" * 60)

    # ---- Merge session-level metrics (SR, eig) ----
    # e2e_session_df has one row per session; baseline_df has one per condition.
    # Average baseline SR/eig across conditions per session for fair comparison.
    baseline_session = baseline_df.groupby(
        ["Subject", "Session_Idx", "Headstage"]
    ).agg(SR_baseline=("SR", "mean"),
          Eig_Real_baseline=("Eig_Real_Mean", "mean"),
          Eig_Imag_baseline=("Eig_Imag_Mean", "mean")).reset_index()

    compare_sr = baseline_session.merge(
        e2e_session_df[["Session_Idx", "Headstage", "SR", "Eig_Real_Mean",
                        "Eig_Imag_Mean", "Loss_final"]],
        on=["Session_Idx", "Headstage"],
        suffixes=("", "_e2e"))

    print(f"Session-level merge: {len(compare_sr)} sessions")
    print()
    print("--- Skewness Ratio ---")
    print(f"  Baseline SR:  {compare_sr['SR_baseline'].mean():.4f}")
    print(f"  E2E SR:       {compare_sr['SR'].mean():.4f}")
    if len(compare_sr) > 1:
        t_sr, p_sr = ttest_rel(compare_sr["SR_baseline"], compare_sr["SR"])
        print(f"  Paired t-test: t={t_sr:.3f}, p={p_sr:.4f}")

    print()
    print("--- Eigenvalues ---")
    print(f"  Baseline |Real|: {compare_sr['Eig_Real_baseline'].mean():.4f}, "
          f"E2E |Real|: {compare_sr['Eig_Real_Mean'].mean():.4f}")
    print(f"  Baseline |Imag|: {compare_sr['Eig_Imag_baseline'].mean():.4f}, "
          f"E2E |Imag|: {compare_sr['Eig_Imag_Mean'].mean():.4f}")

    # ---- Merge per-condition R2_drive ----
    # baseline_df has R2_drive per condition; e2e_df has R2_drive_rollout per condition
    compare_r2 = baseline_df[["Subject", "Session_Idx", "Headstage", "Condition",
                               "R2_drive"]].merge(
        e2e_df[["Session_Idx", "Headstage", "Condition", "R2_drive_rollout"]],
        on=["Session_Idx", "Headstage", "Condition"])

    print()
    print("--- R2_drive ---")
    for cond in ["Tracking", "Playback"]:
        sub = compare_r2[compare_r2["Condition"] == cond]
        print(f"  {cond}: R2_drive_baseline={sub['R2_drive'].mean():.6f}, "
              f"R2_drive_e2e={sub['R2_drive_rollout'].mean():.6f}")

    # ---- Dummy-CEBRA gate ----
    if dummy_df is not None:
        compare_r2_dummy = compare_r2.merge(
            dummy_df[["Session_Idx", "Headstage", "Condition", "SR_dummy",
                      "R2_drive_dummy"]],
            on=["Session_Idx", "Headstage", "Condition"])
        gate_e2e = (compare_r2_dummy["R2_drive_rollout"] >
                     compare_r2_dummy["R2_drive_dummy"]).sum()
        gate_base = (compare_r2_dummy["R2_drive"] >
                      compare_r2_dummy["R2_drive_dummy"]).sum()
        print(f"\n  R2_drive gate (E2E > Dummy): {gate_e2e}/{len(compare_r2_dummy)}")
        print(f"  R2_drive gate (Baseline > Dummy): {gate_base}/{len(compare_r2_dummy)}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Row 1, col 1-2: SR scatter per session
    axes[0, 0].scatter(compare_sr["SR_baseline"], compare_sr["SR"],
                       c="#440154", s=25, alpha=0.7)
    lims_sr = [0, 1]
    axes[0, 0].plot(lims_sr, lims_sr, '--', c='gray', lw=0.8)
    axes[0, 0].set_xlim(lims_sr); axes[0, 0].set_ylim(lims_sr)
    axes[0, 0].set_xlabel("SR (Baseline)"); axes[0, 0].set_ylabel("SR (E2E)")
    axes[0, 0].set_title("SR: Baseline vs E2E (per session)")
    if len(compare_sr) > 1:
        t, p = ttest_rel(compare_sr["SR_baseline"], compare_sr["SR"])
        axes[0, 0].text(0.05, 0.95, f"t={t:.2f}, p={p:.3f}",
                        transform=axes[0, 0].transAxes, fontsize=6, va='top')

    # Row 1, col 2: SR bar
    sr_bar = pd.DataFrame({
        "Pipeline": ["Baseline", "E2E"],
        "SR": [compare_sr["SR_baseline"].mean(), compare_sr["SR"].mean()],
        "SEM": [compare_sr["SR_baseline"].sem(), compare_sr["SR"].sem()],
    })
    axes[0, 1].bar(["Baseline", "E2E"], sr_bar["SR"],
                   yerr=sr_bar["SEM"], color=["#440154", "#21918c"],
                   capsize=3, width=0.5)
    axes[0, 1].set_title("Mean SR")
    axes[0, 1].set_ylim(0, 1)

    # Row 1, col 3: loss curve
    axes[0, 2].set_title("Final Training Loss")

    # Row 2, col 1-2: R2_drive per condition scatter
    for i, cond in enumerate(["Tracking", "Playback"]):
        sub = compare_r2[compare_r2["Condition"] == cond]
        axes[1, i].scatter(sub["R2_drive"], sub["R2_drive_rollout"],
                           c="#21918c", s=20, alpha=0.7)
        all_vals = np.concatenate([sub["R2_drive"].values,
                                    sub["R2_drive_rollout"].values])
        vmin = min(0, np.nanmin(all_vals) * 1.1)
        vmax = max(0.01, np.nanmax(all_vals) * 1.1)
        axes[1, i].plot([vmin, vmax], [vmin, vmax], '--', c='gray', lw=0.8)
        axes[1, i].set_xlim(vmin, vmax); axes[1, i].set_ylim(vmin, vmax)
        axes[1, i].set_xlabel("R2_drive (Baseline)")
        axes[1, i].set_ylabel("R2_drive (E2E)")
        axes[1, i].set_title(f"R2_drive: {cond}")

    # Row 2, col 3: R2_drive bar per condition
    r2_bar = compare_r2.groupby("Condition")[
        ["R2_drive", "R2_drive_rollout"]].mean().reset_index()
    r2_bar_melt = pd.melt(r2_bar, id_vars=["Condition"],
                          value_vars=["R2_drive", "R2_drive_rollout"],
                          var_name="Pipeline", value_name="R2_drive")
    sns.barplot(data=r2_bar_melt, x="Condition", y="R2_drive", hue="Pipeline",
                ax=axes[1, 2],
                palette={"R2_drive": "#440154", "R2_drive_rollout": "#21918c"})
    axes[1, 2].set_title("R2_drive: Baseline vs E2E")
    axes[1, 2].legend(fontsize=5)

    plt.suptitle("Baseline vs End-to-End Lie-ODE Comparison",
                 y=1.02, fontsize=10, fontweight="bold")
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"E2E_vs_Baseline.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("Skipped comparison: missing data (e2e or baseline).")
