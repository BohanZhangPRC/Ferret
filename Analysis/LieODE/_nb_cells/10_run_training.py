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

    # ---- Multi-scale R2_drive accumulator (averaged across seeds) ----
    seed_r2d = {cond: {s: [] for s in VAL_ROLLOUT_LENS}
                for cond in ["Tracking", "Playback"]}
    seed_r2d_sh = {cond: {s: [] for s in VAL_ROLLOUT_LENS}
                   for cond in ["Tracking", "Playback"]}
    seed_sr, seed_eig_r, seed_eig_i = [], [], []
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

        # ---- Per-condition SR (condition-specific drive distribution) ----
        # The shared generator has ONE L but J(u) varies with drive via ControlNet.
        # We compute SR_TR = E_{u~TR}[||J(u)||/(||J(u)||+||L||)] and same for PB.
        sr_cond = {}
        with torch.no_grad():
            L_np = model.lie_cell.dissipation.get_L_numpy()
            L_fro = np.linalg.norm(L_np)
            for val, cond_name in [(0.0, "Tracking"), (1.0, "Playback")]:
                # Sample drives from this condition's raw data
                v_cond = f_df[f_df["Condition"] == val]["Velocity_x"].values
                if len(v_cond) > 100:
                    # Standardize (matching training)
                    v_std = (v_cond - np.mean(v_cond)) / (np.std(v_cond) + 1e-9)
                    u_samples = torch.tensor(
                        v_std[:1000].reshape(-1, 1),  # up to 1000 samples
                        dtype=torch.float32, device=DEVICE)
                    _, J_samples, _ = model.lie_cell.compute_generator(u_samples)
                    J_np = J_samples.cpu().numpy()
                    sr_vals = [np.linalg.norm(J_np[k]) /
                               (np.linalg.norm(J_np[k]) + L_fro + 1e-9)
                               for k in range(len(J_np))]
                    sr_cond[cond_name] = float(np.mean(sr_vals))
                else:
                    sr_cond[cond_name] = float('nan')

        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Aggregate across seeds ----
    e2e_session.append({
        "Subject": "SKIEUR", "Session_Idx": idx, "Headstage": hs_label,
        "Space": "E2E_LieDynamics", "D_LATENT": D_LATENT,
        "SR": np.mean(seed_sr), "SR_sem": np.std(seed_sr) / max(1, N_SEEDS)**0.5,
        "SR_Tracking": sr_cond.get("Tracking", float('nan')),
        "SR_Playback": sr_cond.get("Playback", float('nan')),
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

    # ---- lambda_dyn=0 ablation (single seed, same session) ----
    if DO_ABLATION:
        torch.manual_seed(RANDOM_SEED + idx * 100)
        np.random.seed(RANDOM_SEED + idx * 100)
        model_abl = SkieurLieODE(n_neurons, D_LATENT, d_drive,
                                 constrained_L=CONSTRAINED_L,
                                 use_ode=USE_ODE, ode_method=ODE_METHOD)
        model_abl.to(DEVICE)
        # Train with lambda_dyn=0 (no dynamics constraint)
        model_abl, hist_abl, _ = train_one_session(
            model_abl, n_data_session, f_df, idx,
            lambda_dyn=0.0, lambda_warmup=0)
        J_a, L_a, sr_a, re_a, im_a = model_abl.get_generator_matrices()
        ablation_results.append({
            "Session_Idx": idx, "Headstage": hs_label,
            "SR_lambda0": sr_a,
            "Loss_final": hist_abl['loss_total'][-1]
            if hist_abl['loss_total'] else float('nan'),
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

# --- lambda_dyn=0 ablation summary ---
if ablation_df is not None:
    print("\n--- lambda_dyn=0 Ablation ---")
    merged_abl = e2e_session_df.merge(ablation_df, on=["Session_Idx", "Headstage"])
    print(f"  SR (lambda={LAMBDA_DYN}): {merged_abl['SR'].mean():.4f} "
          f"± {merged_abl['SR'].sem():.4f}")
    print(f"  SR (lambda=0):          {merged_abl['SR_lambda0'].mean():.4f} "
          f"± {merged_abl['SR_lambda0'].sem():.4f}")
    if len(merged_abl) > 1:
        t_abl, p_abl = ttest_rel(merged_abl["SR"], merged_abl["SR_lambda0"])
        print(f"  Paired t-test: t={t_abl:.3f}, p={p_abl:.4f}")
    print(f"  (If SR survives without dynamics loss, rotation comes from InfoNCE, "
          f"not from the Lie constraint.)")

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
