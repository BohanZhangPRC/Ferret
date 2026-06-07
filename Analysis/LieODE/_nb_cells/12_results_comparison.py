# ============================================================
# Phase 3 -- Results Comparison: Baseline vs End-to-End vs Dummy
# ============================================================
# IMPORTANT: baseline R2_drive (derivative-based OLS) and E2E R2_drive_rollout
# (trajectory-rollout MSE) are DIFFERENT ESTIMATORS of conceptually related
# quantities.  They are NOT directly comparable on a y=x scatter.  Each is
# gated against its OWN null:
#   - Baseline: R2_drive > R2_drive_shuffle (same embedding, shuffled labels)
#   - Baseline: R2_drive > R2_drive_dummy   (dummy-CEBRA embedding)
#   - E2E:     R2_drive_rollout > R2_drive_shuffle (same encoder, shuffled drive)

if e2e_df is not None and baseline_df is not None:
    print("=" * 60)
    print("  Results Comparison")
    print("=" * 60)
    print("  NOTE: baseline R2_drive (derivative OLS) and E2E R2_drive_rollout")
    print("        (trajectory rollout) are different estimators — not directly")
    print("        comparable.  Each is gated against its own null distribution.")
    print()

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
    print("--- Eigenvalues (per-session diagnostics, NOT cross-session averages) ---")
    print(f"  Baseline |Real| (per-session): "
          f"{compare_sr['Eig_Real_baseline'].round(4).tolist()}")
    print(f"  E2E |Real| (per-session):     "
          f"{compare_sr['Eig_Real_Mean'].round(4).tolist()}")
    print(f"  Baseline |Imag| (per-session): "
          f"{compare_sr['Eig_Imag_baseline'].round(4).tolist()}")
    print(f"  E2E |Imag| (per-session):     "
          f"{compare_sr['Eig_Imag_Mean'].round(4).tolist()}")
    print(f"  (E2E eigenvalues are in arbitrary encoder-scale units —")
    print(f"   not comparable across independently-trained sessions.")
    print(f"   Only SR — a scale-invariant ratio — is cross-session comparable.)")

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

    # ---- Gates: each pipeline vs its OWN null ----
    # Baseline gates (derivative-based R2_drive)
    base_shuf = baseline_df[["Session_Idx", "Headstage", "Condition",
                              "R2_drive", "R2_drive_shuffle"]].copy()
    gate_base_shuf = (base_shuf["R2_drive"] >
                       base_shuf["R2_drive_shuffle"]).sum()
    print(f"\n  Gate (Baseline R2_drive > shuffle): {gate_base_shuf}/"
          f"{len(base_shuf)}")

    if dummy_df is not None:
        base_dummy = base_shuf.merge(
            dummy_df[["Session_Idx", "Headstage", "Condition", "R2_drive_dummy"]],
            on=["Session_Idx", "Headstage", "Condition"])
        gate_base_dummy = (base_dummy["R2_drive"] >
                            base_dummy["R2_drive_dummy"]).sum()
        print(f"  Gate (Baseline R2_drive > Dummy-CEBRA): "
              f"{gate_base_dummy}/{len(base_dummy)}")

    # E2E gate: use ONLY shortest window (where shuffle null is computed)
    e2e_gate = e2e_df[e2e_df["Window_Bins"] == VAL_ROLLOUT_LENS[0]][
        ["Session_Idx", "Headstage", "Condition",
         "R2_drive_rollout", "R2_drive_shuffle"]].copy()
    gate_e2e_shuf = (e2e_gate["R2_drive_rollout"].dropna() >
                      e2e_gate["R2_drive_shuffle"].dropna()).sum()
    print(f"  Gate (E2E R2_drive > shuffle, {VAL_ROLLOUT_LENS[0]} bins): "
          f"{gate_e2e_shuf}/{len(e2e_gate.dropna())}")

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

    # Row 2, col 1-2: R2_drive self-null (each pipeline vs its own null)
    # Baseline: R2_drive true vs shuffle
    base_r2_bar = baseline_df.groupby("Condition")[
        ["R2_drive", "R2_drive_shuffle"]].mean().reset_index()
    base_r2_melt = pd.melt(base_r2_bar, id_vars=["Condition"],
                           value_vars=["R2_drive", "R2_drive_shuffle"],
                           var_name="Type", value_name="R2_drive")
    sns.barplot(data=base_r2_melt, x="Condition", y="R2_drive", hue="Type",
                ax=axes[1, 0],
                palette={"R2_drive": "#440154", "R2_drive_shuffle": "#B2B2B2"})
    axes[1, 0].set_title("Baseline R2_drive\n(true vs shuffle)")
    axes[1, 0].legend(fontsize=5)

    # E2E: R2_drive_rollout true vs drive-shuffle
    e2e_short = e2e_df[e2e_df["Window_Bins"] == VAL_ROLLOUT_LENS[0]]
    e2e_r2_bar = e2e_short.groupby("Condition")[
        ["R2_drive_rollout", "R2_drive_shuffle"]].mean().reset_index()
    e2e_r2_melt = pd.melt(e2e_r2_bar, id_vars=["Condition"],
                          value_vars=["R2_drive_rollout", "R2_drive_shuffle"],
                          var_name="Type", value_name="R2_drive")
    sns.barplot(data=e2e_r2_melt, x="Condition", y="R2_drive", hue="Type",
                ax=axes[1, 1],
                palette={"R2_drive_rollout": "#21918c",
                         "R2_drive_shuffle": "#B2B2B2"})
    axes[1, 1].set_title("E2E R2_drive_rollout\n(true vs drive-shuffle)")
    axes[1, 1].legend(fontsize=5)

    # Row 2, col 3: Gate pass/fail summary
    labels = ['Base\nvs Shuf', 'Base\nvs Dummy', 'E2E\nvs Shuf']
    passes = [gate_base_shuf, gate_base_dummy if dummy_df is not None else 0,
              gate_e2e_shuf]
    totals = [len(base_shuf), len(base_dummy) if dummy_df is not None else 0,
              len(e2e_gate)]
    colors_bar = ['#440154', '#440154', '#21918c']
    axes[1, 2].bar(labels, [p/max(t,1) for p, t in zip(passes, totals)],
                   color=colors_bar, alpha=0.7)
    for i, (p, t) in enumerate(zip(passes, totals)):
        if t > 0:
            axes[1, 2].text(i, p/t + 0.02, f'{p}/{t}', ha='center', fontsize=7)
    axes[1, 2].set_ylabel("Pass Fraction"); axes[1, 2].set_ylim(0, 1.1)
    axes[1, 2].set_title("Gate Pass Rates")

    plt.suptitle("Baseline vs End-to-End Lie-ODE Comparison",
                 y=1.02, fontsize=10, fontweight="bold")
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"E2E_vs_Baseline.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("Skipped comparison: missing data (e2e or baseline).")
