# ============================================================
# Phase 2b -- Run Joint Training (all sessions, multi-seed)
# ============================================================
# Trains one model per session × seed, with multi-scale validation.
# Optionally runs lambda_dyn=0 ablation for comparison.

e2e_results = []       # per session-condition-seed (R2_drive at multiple scales)
e2e_session = []       # per session-seed (SR, eig)
e2e_histories = []     # per session-seed training curves

# Determine training sessions
n_sessions_available = len(n_data_all)
if N_TRAIN_SESSIONS is None:
    n_train = n_sessions_available
else:
    n_train = min(N_TRAIN_SESSIONS, n_sessions_available)
train_indices = np.random.choice(n_sessions_available, size=n_train, replace=False)
train_indices = sorted(train_indices)
print(f"Training on {n_train}/{n_sessions_available} sessions "
      f"(indices: {train_indices.tolist()})")
print(f"Seeds per session: {N_SEEDS}")
print(f"Validation scales: {VAL_ROLLOUT_LENS} bins")

# ---- Optional lambda_dyn=0 ablation (pure InfoNCE, no dynamics constraint) ----
DO_ABLATION = True  # Set False to skip the ablation pass
ablation_results = []  # per session SR without dynamics loss

for idx in tqdm(train_indices, desc="Joint Training"):
    n_data_session = n_data_all[idx]
    f_df = f_data_all[idx]
    hs_label = "hs0" if idx < n_hs0 else "hs1"
    n_neurons = n_data_session.shape[0]
    d_drive = len(DRIVE_KEYS)

    # ---- Pre-compute pooled TR+PB standardization (matching training) ----
    all_v = np.concatenate([
        f_df[f_df["Condition"] == val]["Velocity_x"].values
        for val in [0.0, 1.0]])
    mu_pool, std_pool = np.mean(all_v), np.std(all_v)
    if std_pool < 1e-9: std_pool = 1.0

    # ---- Multi-scale R2_drive accumulator (averaged across seeds) ----
    seed_r2d = {cond: {s: [] for s in VAL_ROLLOUT_LENS}
                for cond in ["Tracking", "Playback"]}
    seed_r2d_sh = {cond: {s: [] for s in VAL_ROLLOUT_LENS}
                   for cond in ["Tracking", "Playback"]}
    seed_sr, seed_eig_r, seed_eig_i = [], [], []
    seed_sr_tr, seed_sr_pb = [], []  # per-condition SR
    seed_losses = []

    for seed in range(N_SEEDS):
        # Set per-seed randomness
        torch.manual_seed(RANDOM_SEED + idx * 100 + seed)
        np.random.seed(RANDOM_SEED + idx * 100 + seed)

        model = SkieurLieODE(n_neurons, D_LATENT, d_drive,
                             constrained_L=CONSTRAINED_L,
                             use_ode=USE_ODE, ode_method=ODE_METHOD)
        model.to(DEVICE)

        model, history, val_metrics = train_one_session(
            model, n_data_session, f_df, idx)

        e2e_histories.append(history)
        J_avg, L_mat, sr, re, im = model.get_generator_matrices()
        seed_sr.append(sr)
        seed_eig_r.append(re)
        seed_eig_i.append(im)
        seed_losses.append(history['loss_total'][-1]
                           if history['loss_total'] else float('nan'))

        for cond_name in ["Tracking", "Playback"]:
            vm = val_metrics.get(cond_name, {})
            # Multi-scale R2_drive (keyed by window length)
            r2d_dict = vm.get('R2_drive_multiscale', {})
            r2d_sh_dict = vm.get('R2_drive_shuffle_multiscale', {})
            for wlen in VAL_ROLLOUT_LENS:
                seed_r2d[cond_name][wlen].append(
                    r2d_dict.get(wlen, float('nan')))
                seed_r2d_sh[cond_name][wlen].append(
                    r2d_sh_dict.get(wlen, float('nan')))

        # ---- Per-condition SR (pooled standardization, accumulated across seeds) ----
        with torch.no_grad():
            L_np = model.lie_cell.dissipation.get_L_numpy()
            L_fro = np.linalg.norm(L_np)
            for val, cond_name in [(0.0, "Tracking"), (1.0, "Playback")]:
                v_cond = f_df[f_df["Condition"] == val]["Velocity_x"].values
                if len(v_cond) > 100:
                    v_std = (v_cond - mu_pool) / std_pool  # pooled standardization
                    u_samples = torch.tensor(
                        v_std[:1000].reshape(-1, 1),
                        dtype=torch.float32, device=DEVICE)
                    _, J_samples, _ = model.lie_cell.compute_generator(u_samples)
                    J_np = J_samples.cpu().numpy()
                    sr_vals = [np.linalg.norm(J_np[k]) /
                               (np.linalg.norm(J_np[k]) + L_fro + 1e-9)
                               for k in range(len(J_np))]
                    (seed_sr_tr if cond_name == "Tracking" else seed_sr_pb).append(
                        float(np.mean(sr_vals)))
                else:
                    (seed_sr_tr if cond_name == "Tracking" else seed_sr_pb).append(
                        float('nan'))

        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Aggregate across seeds ----
    e2e_session.append({
        "Subject": "SKIEUR", "Session_Idx": idx, "Headstage": hs_label,
        "Space": "E2E_LieDynamics", "D_LATENT": D_LATENT,
        "SR": np.mean(seed_sr), "SR_sem": np.std(seed_sr) / max(1, N_SEEDS)**0.5,
        "SR_Tracking": np.mean(seed_sr_tr) if seed_sr_tr else float('nan'),
        "SR_Playback": np.mean(seed_sr_pb) if seed_sr_pb else float('nan'),
        "Eig_Real_Mean": np.mean(seed_eig_r), "Eig_Imag_Mean": np.mean(seed_eig_i),
        "Loss_final": np.mean(seed_losses),
        "N_Seeds": N_SEEDS, "N_Neurons": n_neurons,
    })

    for cond_name in ["Tracking", "Playback"]:
        for wlen in VAL_ROLLOUT_LENS:
            r2_vals = seed_r2d[cond_name][wlen]
            r2_sh_vals = seed_r2d_sh[cond_name][wlen]
            e2e_results.append({
                "Subject": "SKIEUR", "Session_Idx": idx,
                "Headstage": hs_label, "Condition": cond_name,
                "Window_Bins": wlen,
                "R2_drive_rollout": np.mean(r2_vals) if r2_vals else float('nan'),
                "R2_drive_rollout_sem": np.std(r2_vals) / max(1, len(r2_vals))**0.5
                if r2_vals else float('nan'),
                "R2_drive_shuffle": np.mean(r2_sh_vals) if r2_sh_vals else float('nan'),
            })

    # ---- lambda_dyn=0 ablation: post-hoc OLS Lie on frozen embedding ----
    # The encoder from λ=0 training has NOT seen the dynamics constraint.
    # Post-hoc OLS Lie fit asks: does InfoNCE alone produce rotational
    # embedding structure, without the Lie loss?  If SR_ols_abl ≈ SR_e2e,
    # the rotation comes from InfoNCE, not from the dynamics constraint.
    if DO_ABLATION:
        torch.manual_seed(RANDOM_SEED + idx * 100)
        np.random.seed(RANDOM_SEED + idx * 100)
        model_abl = SkieurLieODE(n_neurons, D_LATENT, d_drive,
                                 constrained_L=CONSTRAINED_L,
                                 use_ode=USE_ODE, ode_method=ODE_METHOD)
        model_abl.to(DEVICE)
        model_abl, hist_abl, _ = train_one_session(
            model_abl, n_data_session, f_df, idx,
            lambda_dyn=0.0, lambda_warmup=0)

        # Post-hoc OLS Lie fit on frozen λ=0 embedding (per-condition)
        for val, cond_name in [(0.0, "Tracking"), (1.0, "Playback")]:
            epochs_n, epochs_l, _ = extract_epochs(
                n_data_session, f_df, val, dt, label_col="Velocity_x")
            if not epochs_n:
                continue
            # Pool condition epochs, encode, OLS fit
            ep_cat = np.concatenate(epochs_n, axis=0)
            lab_cat = np.concatenate(epochs_l, axis=0)
            with torch.no_grad():
                emb_cat = model_abl.encode(
                    torch.tensor(ep_cat, dtype=torch.float32, device=DEVICE)
                ).cpu().numpy()
            J_s, sr_ols, r2, J_ols, r2d = fit_lie_algebra_with_leak(emb_cat, lab_cat)
            re_ols, im_ols = compute_eigenvalue_metrics(J_ols)
            ablation_results.append({
                "Session_Idx": idx, "Headstage": hs_label,
                "Condition": cond_name,
                "SR_lambda0_ols": sr_ols,
                "R2_drive_lambda0_ols": r2d,
                "Eig_Real_lambda0": re_ols,
                "Eig_Imag_lambda0": im_ols,
            })
        del model_abl; gc.collect(); torch.cuda.empty_cache()

e2e_df = pd.DataFrame(e2e_results)
e2e_session_df = pd.DataFrame(e2e_session)
ablation_df = pd.DataFrame(ablation_results) if ablation_results else None

print(f"Trained {n_train} sessions × {N_SEEDS} seeds = "
      f"{n_train * N_SEEDS} models.")
print()

# --- Multi-scale R2_drive summary ---
print("--- Multi-scale R2_drive (mean across sessions × seeds) ---")
r2_pivot = e2e_df.pivot_table(
    values="R2_drive_rollout", index="Window_Bins", columns="Condition",
    aggfunc="mean").round(6)
print(r2_pivot.to_string())

# --- SR summary ---
print("\n--- Skewness Ratio ---")
print(f"  SR (N(0,1) diagnostic): {e2e_session_df['SR'].mean():.4f} "
      f"± {e2e_session_df['SR'].sem():.4f}  [from get_generator_matrices, random-drive]")
if 'SR_Tracking' in e2e_session_df.columns:
    print(f"  SR_Tracking (empirical):  {e2e_session_df['SR_Tracking'].mean():.4f} "
          f"± {e2e_session_df['SR_Tracking'].sem():.4f}  [condition-specific drive]")
    print(f"  SR_Playback (empirical):  {e2e_session_df['SR_Playback'].mean():.4f} "
          f"± {e2e_session_df['SR_Playback'].sem():.4f}  [condition-specific drive]")
    common_sr = e2e_session_df[['SR_Tracking', 'SR_Playback']].dropna()
    if len(common_sr) > 1:
        t_sr, p_sr = ttest_rel(common_sr['SR_Tracking'], common_sr['SR_Playback'])
        print(f"  SR TR vs PB paired t-test: t={t_sr:.3f}, p={p_sr:.4f}")

# --- lambda_dyn=0 ablation summary (post-hoc OLS on frozen embedding) ---
if ablation_df is not None:
    print("\n--- lambda_dyn=0 Ablation (post-hoc OLS Lie on frozen embedding) ---")
    # Compare E2E R2_drive vs OLS R2_drive on λ=0 embedding (shortest window)
    e2e_short = e2e_df[e2e_df["Window_Bins"] == VAL_ROLLOUT_LENS[0]]
    merged_abl = e2e_short.merge(ablation_df, on=["Session_Idx", "Condition"])
    for cond in ["Tracking", "Playback"]:
        sub = merged_abl[merged_abl["Condition"] == cond]
        sr_col = f"SR_{cond}"
        sr_e2e_cond = e2e_session_df[sr_col].mean() if sr_col in e2e_session_df.columns else float('nan')
        print(f"  {cond}: R2_drive_E2E={sub['R2_drive_rollout'].mean():.6f}, "
              f"R2_drive_OLS_λ=0={sub['R2_drive_lambda0_ols'].mean():.6f}, "
              f"SR_E2E={sr_e2e_cond:.4f}, "
              f"SR_OLS_λ=0={sub['SR_lambda0_ols'].mean():.4f}")
    # Paired test: average across conditions per session first (not treating
    # Tracking/Playback from same session as independent)
    common = merged_abl.dropna(subset=["R2_drive_rollout", "R2_drive_lambda0_ols"])
    session_avg = common.groupby("Session_Idx")[
        ["R2_drive_rollout", "R2_drive_lambda0_ols"]].mean()
    if len(session_avg) > 1:
        t_a, p_a = ttest_rel(session_avg["R2_drive_rollout"],
                             session_avg["R2_drive_lambda0_ols"])
        print(f"  Paired R2_drive (E2E vs OLS_λ=0, per-session avg): "
              f"t={t_a:.3f}, p={p_a:.4f}")
        if p_a < 0.05:
            print(f"  -> Dynamics constraint significantly improves R2_drive over")
            print(f"     InfoNCE-only embedding. Rotation is NOT just an InfoNCE artifact.")
        else:
            print(f"  -> No significant difference: InfoNCE alone may produce comparable")
            print(f"     rotational structure. The dynamics constraint adds little.")

# --- Loss curves (first 5 session-seed pairs) ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
n_plot = min(5, len(e2e_histories))
for i in range(n_plot):
    hist = e2e_histories[i]
    color = plt.cm.viridis(i / max(1, n_plot - 1))
    axes[0].plot(hist['loss_total'], c=color, alpha=0.7, lw=0.5)
    axes[1].plot(hist['loss_infonce'], c=color, alpha=0.7, lw=0.5)
    axes[2].plot(hist['loss_dyn'], c=color, alpha=0.7, lw=0.5)
axes[0].set_ylabel("Total Loss"); axes[0].set_xlabel("Epoch")
axes[1].set_ylabel("InfoNCE Loss"); axes[1].set_xlabel("Epoch")
axes[2].set_ylabel("Dynamics MSE"); axes[2].set_xlabel("Epoch")
for ax in axes: ax.set_yscale('log')
plt.suptitle(f"Training Curves ({n_train} sessions × {N_SEEDS} seeds, "
             f"{n_plot} shown)", y=1.02, fontsize=9, fontweight="bold")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"E2E_LossCurves.{fmt}"),
                dpi=150, bbox_inches="tight")
plt.show()

# --- Multi-scale R2_drive decay curve ---
fig, ax = plt.subplots(figsize=(5, 3.5))
for cond, color, marker in [("Tracking", "#440154", "o"),
                              ("Playback", "#21918c", "s")]:
    sub = e2e_df[e2e_df["Condition"] == cond]
    means = sub.groupby("Window_Bins")["R2_drive_rollout"].mean()
    sems = sub.groupby("Window_Bins")["R2_drive_rollout"].sem()
    ax.errorbar(means.index * dt * 1000, means.values,
                yerr=sems.values, c=color, marker=marker, ms=4, lw=1,
                label=cond)
ax.set_xlabel("Rollout Horizon (ms)"); ax.set_ylabel("R2_drive (Rollout)")
ax.axhline(0, c='gray', lw=0.5, ls='--'); ax.legend(fontsize=7)
ax.set_title("R2_drive Decay with Rollout Horizon")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"E2E_MultiScale_R2.{fmt}"),
                dpi=150, bbox_inches="tight")
plt.show()

print("End-to-end training complete.")
