# ============================================================
# Phase 4 -- Tracking vs Playback Paired Analysis
# ============================================================
# Note: SR/eigenvalues are from the shared session-level generator J(u)+L,
# so Tracking/PB comparisons are only meaningful for R2_drive (held-out epochs).
# SR/eig are reported as session-level summaries (pooled across conditions).

if e2e_df is not None:
    print("=" * 60)
    print("  Tracking vs Playback -- Paired Comparison")
    print("=" * 60)

    # ---- Session-level SR/eig (pooled across conditions) ----
    if 'e2e_session_df' in dir() and e2e_session_df is not None:
        print("--- Session-level Generator Metrics (shared J(u)+L) ---")
        print(f"  Mean SR:           {e2e_session_df['SR'].mean():.4f} "
              f"(sem={e2e_session_df['SR'].sem():.4f})")
        print(f"  Mean |Real|:       {e2e_session_df['Eig_Real_Mean'].mean():.4f}")
        print(f"  Mean |Imag|:       {e2e_session_df['Eig_Imag_Mean'].mean():.4f}")
        print(f"  N sessions:        {len(e2e_session_df)}")
        print()

    # ---- Per-condition R2_drive: Tracking vs Playback ----
    pivot_e2e = e2e_df.pivot_table(
        values=["R2_drive_rollout"],
        index=["Subject", "Session_Idx", "Headstage"],
        columns="Condition").dropna()

    if len(pivot_e2e) > 1:
        print("--- R2_drive (held-out rollout) ---")
        for metric in ["R2_drive_rollout"]:
            tr = pivot_e2e[metric]["Tracking"].values
            pb = pivot_e2e[metric]["Playback"].values
            t, p = ttest_rel(tr, pb)
            print(f"  Tracking: {np.mean(tr):.6f}, Playback: {np.mean(pb):.6f}, "
                  f"t={t:.3f}, p={p:.4f}")
    else:
        print("  Not enough paired sessions for R2_drive t-test.")

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Panel A: Session-level SR histogram
    if 'e2e_session_df' in dir() and e2e_session_df is not None:
        axes[0].hist(e2e_session_df["SR"], bins=min(10, len(e2e_session_df)),
                     color="#440154", alpha=0.7, edgecolor='white')
        axes[0].axvline(e2e_session_df["SR"].mean(), color='#21918c',
                        lw=1.5, ls='--', label=f'Mean={e2e_session_df["SR"].mean():.3f}')
        axes[0].set_xlabel("Skewness Ratio"); axes[0].set_ylabel("Sessions")
        axes[0].set_title("SR Distribution (E2E)")
        axes[0].legend(fontsize=5)

    # Panel B: R2_drive Tracking vs Playback paired
    if len(pivot_e2e) > 1:
        tr_vals = pivot_e2e["R2_drive_rollout"]["Tracking"].values
        pb_vals = pivot_e2e["R2_drive_rollout"]["Playback"].values
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
