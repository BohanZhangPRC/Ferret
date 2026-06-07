# Quick baseline vs dummy comparison
if baseline_df is not None:
    print("=" * 60)
    print("  Baseline CEBRA-Embedded Lie -- Quick Summary")
    print("=" * 60)
    grp = baseline_df.groupby("Condition")[
        ["SR", "SR_shuffle", "R2", "R2_shuffle", "R2_drive",
         "R2_drive_shuffle"]].mean().round(4)
    print(grp.to_string())
    print()

    # Paired t-test: Tracking vs Playback
    pivot = baseline_df.pivot_table(
        values=["SR", "R2_drive"],
        index=["Subject", "Session_Idx", "Headstage"],
        columns="Condition").dropna()
    if len(pivot) > 1:
        for metric in ["SR", "R2_drive"]:
            t, p = ttest_rel(pivot[metric]["Tracking"],
                             pivot[metric]["Playback"])
            print(f"  {metric} Tracking vs Playback: t={t:.3f}, p={p:.4f}")

    if dummy_df is not None:
        merged = baseline_df.merge(dummy_df,
                                   on=["Subject", "Session_Idx",
                                       "Headstage", "Condition"])
        print()
        print("--- Dummy-CEBRA Control ---")
        for cond in ["Tracking", "Playback"]:
            sub = merged[merged["Condition"] == cond]
            print(f"  {cond}: SR_true={sub['SR'].mean():.4f}, "
                  f"SR_dummy={sub['SR_dummy'].mean():.4f}, "
                  f"R2_drive_true={sub['R2_drive'].mean():.6f}, "
                  f"R2_drive_dummy={sub['R2_drive_dummy'].mean():.6f}")
        gate = (merged["R2_drive"] > merged["R2_drive_dummy"]).sum()
        print(f"  R2_drive gate passed: {gate}/{len(merged)} "
              f"session-conditions")

    # Visualization: SR and R2_drive bar plots
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    # Panel A: SR true vs shuffle vs dummy
    if dummy_df is not None:
        merged = baseline_df.merge(dummy_df,
                                   on=["Subject", "Session_Idx",
                                       "Headstage", "Condition"])
        plot_data = pd.melt(merged,
                            id_vars=["Condition"],
                            value_vars=["SR", "SR_shuffle", "SR_dummy"],
                            var_name="Type", value_name="Skewness Ratio")
    else:
        plot_data = pd.melt(baseline_df,
                            id_vars=["Condition"],
                            value_vars=["SR", "SR_shuffle"],
                            var_name="Type", value_name="Skewness Ratio")
    sns.barplot(data=plot_data, x="Condition", y="Skewness Ratio",
                hue="Type", ax=axes[0],
                palette={"SR": "#440154", "SR_shuffle": "#B2B2B2",
                         "SR_dummy": "#fde725"})
    axes[0].set_title("Skewness Ratio")
    axes[0].legend(fontsize=5)

    # Panel B: R2_drive true vs shuffle
    sns.barplot(data=baseline_df.melt(
        id_vars=["Condition"],
        value_vars=["R2_drive", "R2_drive_shuffle"],
        var_name="Type", value_name="Value"),
        x="Condition", y="Value", hue="Type", ax=axes[1],
        palette={"R2_drive": "#440154", "R2_drive_shuffle": "#B2B2B2"})
    axes[1].set_title("R2_drive (True vs Shuffle)")
    axes[1].legend(fontsize=5)

    # Panel C: Tracking vs Playback per-session scatter
    pivot_plot = baseline_df.pivot_table(
        values="SR", index=["Subject", "Session_Idx", "Headstage"],
        columns="Condition").dropna()
    if len(pivot_plot) > 1:
        axes[2].scatter(pivot_plot["Tracking"], pivot_plot["Playback"],
                        c="#440154", s=20, alpha=0.7)
        lims = [min(pivot_plot.min().min(), 0),
                max(pivot_plot.max().max(), 1)]
        axes[2].plot(lims, lims, '--', c='gray', lw=0.8)
        axes[2].set_xlim(lims); axes[2].set_ylim(lims)
        axes[2].set_xlabel("Tracking SR"); axes[2].set_ylabel("Playback SR")
        axes[2].set_title("Per-Session SR")

    plt.suptitle("Baseline CEBRA-Embedded Lie + Dummy Control",
                 y=1.02, fontsize=9, fontweight="bold")
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(LIE_OUTPUT_DIR,
                                 f"Baseline_DummyControl.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("Skipped: no baseline data available.")
