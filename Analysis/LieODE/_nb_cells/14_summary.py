# ============================================================
# Summary Output
# ============================================================
# Write timestamped summary .txt with all metrics.
# Mirrors the output style of Skieur_LieAlgebra_CEBRA.ipynb Cell 19.

if 'LIE_OUTPUT_DIR' not in dir():
    LIE_OUTPUT_DIR = f"Skieur_LieE2E_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(LIE_OUTPUT_DIR, exist_ok=True)

out_path = os.path.join(LIE_OUTPUT_DIR, "LieE2E_summary.txt")

with open(out_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  SKIEUR End-to-End Lie-ODE -- Summary Report\n")
    f.write("=" * 70 + "\n")
    f.write(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n")
    f.write("--- Parameters ---\n")
    f.write(f"  dt                  = {dt}\n")
    f.write(f"  SESSION_TYPE        = {SESSION_TYPE}\n")
    f.write(f"  USE_MACRO_EPOCH     = {USE_MACRO_EPOCH}\n")
    f.write(f"  MIN_EPOCH_DUR       = {MIN_EPOCH_DUR} s\n")
    f.write(f"  CEBRA_DISTANCE      = {CEBRA_DISTANCE}\n")
    f.write(f"  CEBRA_ARCH          = {CEBRA_ARCH}\n")
    f.write(f"  CEBRA_EMBEDDING_DIM = {CEBRA_EMBEDDING_DIM}\n")
    f.write(f"  TAU_SHIFT           = {TAU_SHIFT} bins\n")
    f.write(f"  LIE_METHOD          = {LIE_METHOD}\n")
    f.write("\n")
    f.write("--- End-to-End Config ---\n")
    f.write(f"  D_LATENT            = {D_LATENT}\n")
    f.write(f"  USE_ODE             = {USE_ODE}\n")
    f.write(f"  ODE_METHOD          = {ODE_METHOD}\n")
    f.write(f"  LAMBDA_DYN          = {LAMBDA_DYN}\n")
    f.write(f"  LAMBDA_DYN_WARMUP   = {LAMBDA_DYN_WARMUP}\n")
    f.write(f"  CONSTRAINED_L       = {CONSTRAINED_L}\n")
    f.write(f"  DRIVE_KEYS          = {DRIVE_KEYS}\n")
    f.write(f"  MINI_TRAJ_LEN       = {MINI_TRAJ_LEN}\n")
    f.write(f"  N_EPOCHS_TRAIN      = {N_EPOCHS_TRAIN}\n")
    f.write(f"  BATCH_SIZE          = {BATCH_SIZE}\n")
    f.write(f"  LR                  = {LR}\n")
    f.write(f"  N_SHUFFLES          = {N_SHUFFLES}\n")
    f.write(f"  TRAIN_VAL_SPLIT     = {TRAIN_VAL_SPLIT}\n")
    f.write(f"  DEVICE              = {DEVICE}\n")
    f.write("\n")
    f.write("--- Sessions ---\n")
    f.write(f"  Total sessions     : {len(n_data_all)}\n")
    f.write(f"  Headstage 0        : {n_hs0}\n")
    f.write(f"  Headstage 1        : {len(n_data_all) - n_hs0}\n")
    try:
        f.write(f"  Sessions trained   : {N_TRAIN_SESSIONS}\n")
    except NameError:
        pass
    f.write("\n")

    # --- Baseline results ---
    f.write("=" * 70 + "\n")
    f.write("  1. Baseline Two-Stage CEBRA-Embedded Lie\n")
    f.write("=" * 70 + "\n")
    try:
        if baseline_df is not None:
            grp = baseline_df.groupby("Condition")[
                ["SR", "R2", "R2_drive", "SR_shuffle", "R2_drive_shuffle",
                 "Eig_Real_Mean", "Eig_Imag_Mean"]].mean().round(4)
            f.write(grp.to_string() + "\n\n")
    except NameError:
        f.write("  (Not run)\n\n")

    # --- Dummy CEBRA ---
    f.write("=" * 70 + "\n")
    f.write("  2. Dummy-CEBRA Negative Control\n")
    f.write("=" * 70 + "\n")
    try:
        if dummy_df is not None:
            merged = baseline_df.merge(dummy_df,
                                       on=["Subject", "Session_Idx",
                                           "Headstage", "Condition"])
            for cond in ["Tracking", "Playback"]:
                sub = merged[merged["Condition"] == cond]
                f.write(f"  {cond}: SR_true={sub['SR'].mean():.4f}, "
                        f"SR_dummy={sub['SR_dummy'].mean():.4f}, "
                        f"R2_drive_true={sub['R2_drive'].mean():.6f}, "
                        f"R2_drive_dummy={sub['R2_drive_dummy'].mean():.6f}\n")
            gate = (merged["R2_drive"] > merged["R2_drive_dummy"]).sum()
            f.write(f"  R2_drive gate passed: {gate}/{len(merged)}\n\n")
        else:
            f.write("  (Not run)\n\n")
    except NameError:
        f.write("  (Not run)\n\n")

    # --- End-to-End results ---
    f.write("=" * 70 + "\n")
    f.write("  3. End-to-End Lie-ODE\n")
    f.write("=" * 70 + "\n")
    try:
        if 'e2e_session_df' in dir() and e2e_session_df is not None:
            f.write("--- Session-level Generator Metrics ---\n")
            f.write(f"  N sessions: {len(e2e_session_df)}\n")
            f.write(f"  Mean SR:           {e2e_session_df['SR'].mean():.4f} "
                    f"(sem={e2e_session_df['SR'].sem():.4f})\n")
            f.write(f"  Mean |Real|:       {e2e_session_df['Eig_Real_Mean'].mean():.4f}\n")
            f.write(f"  Mean |Imag|:       {e2e_session_df['Eig_Imag_Mean'].mean():.4f}\n")
            f.write("\n")
        if e2e_df is not None:
            f.write("--- Per-Condition R2_drive (held-out rollout) ---\n")
            grp = e2e_df.groupby("Condition")[["R2_drive_rollout"]].mean().round(6)
            f.write(grp.to_string() + "\n\n")

            # Paired comparison: R2_drive only
            pivot_e2e = e2e_df.pivot_table(
                values=["R2_drive_rollout"],
                index=["Subject", "Session_Idx", "Headstage"],
                columns="Condition").dropna()
            if len(pivot_e2e) > 1:
                f.write("  Tracking vs Playback paired t-test (R2_drive):\n")
                t, p = ttest_rel(pivot_e2e["R2_drive_rollout"]["Tracking"],
                                 pivot_e2e["R2_drive_rollout"]["Playback"])
                f.write(f"    R2_drive_rollout: t={t:.3f}, p={p:.4f}\n")
            f.write("\n")
        else:
            f.write("  (Not run)\n\n")
    except NameError:
        f.write("  (Not run)\n\n")

    f.write("=" * 70 + "\n")
    f.write("  End of Report\n")
    f.write("=" * 70 + "\n")

print(f"Summary written to: {out_path}")
print(f"All outputs in: {LIE_OUTPUT_DIR}")

# --- Final cleanup ---
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")
