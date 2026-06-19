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
    f.write("  SKIEUR End-to-End Lie Dynamics -- Summary Report\n")
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
    f.write(f"  TEMPERATURE         = {TEMPERATURE}\n")
    f.write(f"  DRIVE_KEYS          = {DRIVE_KEYS}\n")
    f.write(f"  MINI_TRAJ_LEN       = {MINI_TRAJ_LEN}\n")
    f.write(f"  VAL_ROLLOUT_LENS    = {VAL_ROLLOUT_LENS}\n")
    f.write(f"  N_EPOCHS_TRAIN      = {N_EPOCHS_TRAIN}\n")
    f.write(f"  BATCH_SIZE          = {BATCH_SIZE}\n")
    f.write(f"  LR                  = {LR}\n")
    f.write(f"  N_SHUFFLES          = {N_SHUFFLES}\n")
    f.write(f"  N_SEEDS             = {N_SEEDS}\n")
    f.write(f"  RANDOM_SEED         = {RANDOM_SEED}\n")
    f.write(f"  TRAIN_VAL_SPLIT     = {TRAIN_VAL_SPLIT}\n")
    f.write(f"  DEVICE              = {DEVICE}\n")
    f.write("\n")
    f.write("--- Sessions ---\n")
    f.write(f"  Total sessions     : {len(n_data_all)}\n")
    f.write(f"  Headstage 0        : {n_hs0}\n")
    f.write(f"  Headstage 1        : {len(n_data_all) - n_hs0}\n")
    try:
        f.write(f"  Sessions trained   : {n_train}\n")
    except NameError:
        f.write(f"  Sessions trained   : {N_TRAIN_SESSIONS if N_TRAIN_SESSIONS is not None else 'all'}\n")
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
    f.write("  3. End-to-End Lie Dynamics\n")
    f.write("=" * 70 + "\n")
    try:
        if 'e2e_session_df' in dir() and e2e_session_df is not None:
            f.write("--- Session-level Generator Metrics ---\n")
            f.write(f"  N sessions: {len(e2e_session_df)}\n")
            f.write(f"  SR (random-drive diagnostic, N(0,1)): "
                    f"{e2e_session_df['SR'].mean():.4f} "
                    f"sem={e2e_session_df['SR'].sem():.4f}\n")
            if 'SR_Tracking' in e2e_session_df.columns:
                f.write(f"  SR_Tracking (empirical drive):   "
                        f"{e2e_session_df['SR_Tracking'].mean():.4f} "
                        f"sem={e2e_session_df['SR_Tracking'].sem():.4f}\n")
                f.write(f"  SR_Playback (empirical drive):   "
                        f"{e2e_session_df['SR_Playback'].mean():.4f} "
                        f"sem={e2e_session_df['SR_Playback'].sem():.4f}\n")
                common_sr = e2e_session_df[['SR_Tracking','SR_Playback']].dropna()
                if len(common_sr) > 1:
                    t_sr, p_sr = ttest_rel(common_sr['SR_Tracking'],
                                           common_sr['SR_Playback'])
                    f.write(f"  SR TR vs PB paired t-test: t={t_sr:.3f}, p={p_sr:.4f}\n")
            f.write(f"  (Eigenvalues are per-session diagnostics — not cross-session comparable)\n")
            f.write(f"  Per-session |Real|: "
                    f"{e2e_session_df['Eig_Real_Mean'].round(4).tolist()}\n")
            f.write(f"  Per-session |Imag|: "
                    f"{e2e_session_df['Eig_Imag_Mean'].round(4).tolist()}\n")
            f.write("\n")
        if e2e_df is not None:
            f.write("--- Multi-scale R2_drive (held-out rollout) ---\n")
            for wlen in VAL_ROLLOUT_LENS:
                sub = e2e_df[e2e_df["Window_Bins"] == wlen]
                grp = sub.groupby("Condition")["R2_drive_rollout"].mean().round(6)
                f.write(f"  {wlen} bins ({wlen*dt*1000:.0f}ms):\n")
                f.write(f"    {grp.to_string()}\n")
            # Primary-horizon paired test
            sub_p = e2e_df[e2e_df["Window_Bins"] == VAL_ROLLOUT_LENS[0]]
            pivot_e2e = sub_p.pivot_table(
                values=["R2_drive_rollout"],
                index=["Subject", "Session_Idx", "Headstage"],
                columns="Condition").dropna()
            if len(pivot_e2e) > 1:
                f.write("  Tracking vs Playback (primary horizon):\n")
                t, p = ttest_rel(pivot_e2e["R2_drive_rollout"]["Tracking"],
                                 pivot_e2e["R2_drive_rollout"]["Playback"])
                f.write(f"    R2_drive_rollout: t={t:.3f}, p={p:.4f}\n")
            f.write("\n")
        # Ablation
        if ablation_df is not None:
            f.write("--- lambda_dyn=0 Ablation (post-hoc OLS on frozen embedding) ---\n")
            for cond in ["Tracking", "Playback"]:
                sub_a = ablation_df[ablation_df["Condition"] == cond]
                f.write(f"  {cond}: SR_OLS={sub_a['SR_lambda0_ols'].mean():.4f}, "
                        f"R2_drive_OLS={sub_a['R2_drive_lambda0_ols'].mean():.6f}\n")
            f.write("\n")
        # Velocity confound
        try:
            if 'vel_df' in dir():
                f.write("--- Velocity Distribution Check ---\n")
                f.write(vel_df.groupby("Condition")[["RMS","Std","Range"]].mean().round(2).to_string())
                f.write("\n\n")
        except NameError:
            pass
    except NameError:
        f.write("  (Not run)\n\n")

    # --- Wasserstein ---
    f.write("=" * 70 + "\n")
    f.write("  4. Wasserstein Manifold Analysis\n")
    f.write("=" * 70 + "\n")
    try:
        if 'wass_df' in dir() and wass_df is not None:
            f.write(f"  N sessions: {len(wass_df)}\n")
            f.write(f"  W2(TR vs PB) true:       {wass_df['W2_TR_vs_PB'].mean():.4f}"
                    f"  +/- {wass_df['W2_TR_vs_PB'].sem():.4f}\n")
            f.write(f"  W2(TR vs PB) shuffle:    {wass_df['W2_TR_vs_PB_shuffle'].mean():.4f}"
                    f"  +/- {wass_df['W2_TR_vs_PB_shuffle_sem'].mean():.4f}\n")
            f.write(f"  W2 ratio (true/shuffle): {wass_df['W2_ratio'].mean():.2f}"
                    f"  +/- {wass_df['W2_ratio'].sem():.2f}\n")
            f.write(f"  TR self-W2 (half-split): {wass_df['W2_TR_self'].mean():.4f}"
                    f"  +/- {wass_df['W2_TR_self'].sem():.4f}\n")
            f.write(f"  PB self-W2 (half-split): {wass_df['W2_PB_self'].mean():.4f}"
                    f"  +/- {wass_df['W2_PB_self'].sem():.4f}\n")
            if len(wass_df) > 1:
                tr_s = wass_df['W2_TR_self'].values
                pb_s = wass_df['W2_PB_self'].values
                t_s, p_s = ttest_rel(tr_s, pb_s)
                tighter = "Tracking" if np.mean(tr_s) < np.mean(pb_s) else "Playback"
                f.write(f"  TR vs PB self-W2 paired t: t={t_s:.3f}, p={p_s:.4f}\n")
                f.write(f"  → {tighter} manifold is tighter\n")
            f.write(f"  Per-session W2 ratios: {wass_df['W2_ratio'].round(2).tolist()}\n")
            f.write(f"  Physical interpretation:\n")
            f.write(f"    W2 > shuffle → TR/PB occupy different neural state regions\n")
            f.write(f"    TR_self < PB_self → Tracking manifold is more compact (lower entropy)\n")
            f.write(f"    W2 difference → 'control energy' cost of closed-loop dynamics\n")
        else:
            f.write("  (Not run — install POT: pip install POT)\n")
    except NameError:
        f.write("  (Not run)\n")
    f.write("\n")

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
