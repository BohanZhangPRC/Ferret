# ============================================================
# Phase 2b -- Run Joint Training across all sessions
# ============================================================
# Train one model per session, collect loss curves + validation metrics.

e2e_results = []       # per session-condition (R2_drive per condition)
e2e_session = []       # per session (SR, eig — shared generator, same for both conds)
e2e_histories = []     # per session training curves
e2e_val_metrics = []   # per session cross-validated metrics

# Randomly sample sessions (not first N — avoids ordering bias)
n_sessions_available = len(n_data_all)
n_train = min(N_TRAIN_SESSIONS, n_sessions_available)
train_indices = np.random.choice(n_sessions_available, size=n_train, replace=False)
train_indices = sorted(train_indices)
print(f"Training on {n_train}/{n_sessions_available} sessions (randomly sampled): "
      f"{train_indices.tolist()}")

for idx in tqdm(train_indices, desc="Joint Training"):
    n_data_session = n_data_all[idx]
    f_df = f_data_all[idx]
    hs_label = "hs0" if idx < n_hs0 else "hs1"
    n_neurons = n_data_session.shape[0]

    # ---- Initialize model per session ----
    d_drive = len(DRIVE_KEYS)
    model = SkieurLieODE(n_neurons, D_LATENT, d_drive,
                         constrained_L=CONSTRAINED_L,
                         use_ode=USE_ODE, ode_method=ODE_METHOD)
    model.to(DEVICE)

    # ---- Train ----
    model, history, val_metrics = train_one_session(
        model, n_data_session, f_df, idx)

    e2e_histories.append(history)
    e2e_val_metrics.append({
        "Session_Idx": idx, "Headstage": hs_label,
        "N_Neurons": n_neurons,
        **{f"R2_drive_{k}": v['R2_drive_rollout']
           for k, v in val_metrics.items()},
        **{f"n_val_{k}": v['n_val_epochs']
           for k, v in val_metrics.items()},
    })

    # ---- Extract learned metrics ----
    # SR / eigenvalues from per-sample averaging over the drive distribution
    # (per-sample avoids cancellation of odd w_i(u) under symmetric drive).
    J_avg, L_mat, sr, re, im = model.get_generator_matrices()

    e2e_session.append({
        "Subject": "SKIEUR", "Session_Idx": idx,
        "Headstage": hs_label,
        "Space": "E2E_LieODE", "D_LATENT": D_LATENT,
        "SR": sr, "Eig_Real_Mean": re, "Eig_Imag_Mean": im,
        "Loss_final": history['loss_total'][-1]
        if history['loss_total'] else float('nan'),
        "N_Neurons": n_neurons,
    })

    # R2_drive is per-condition (different held-out epochs) — store separately.
    # Also stores E2E's own drive-shuffle null for self-gating.
    for cond_name in ["Tracking", "Playback"]:
        vm = val_metrics.get(cond_name, {})
        e2e_results.append({
            "Subject": "SKIEUR", "Session_Idx": idx,
            "Headstage": hs_label, "Condition": cond_name,
            "R2_drive_rollout": vm.get('R2_drive_rollout', float('nan')),
            "R2_drive_shuffle": vm.get('R2_drive_shuffle', float('nan')),
            "n_val_epochs": vm.get('n_val_epochs', 0),
        })

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

e2e_df = pd.DataFrame(e2e_results)         # per-condition (R2_drive)
e2e_session_df = pd.DataFrame(e2e_session)  # per-session (SR, eig)
e2e_val_df = pd.DataFrame(e2e_val_metrics)
print(f"Trained {N_TRAIN_SESSIONS} sessions end-to-end.")
print("--- Per-session generator metrics ---")
print(e2e_session_df[["Session_Idx", "Headstage", "SR", "Eig_Real_Mean",
                       "Eig_Imag_Mean"]].round(4).to_string())
print("\n--- Per-condition R2_drive ---")
print(e2e_df.groupby("Condition")[["R2_drive_rollout"]].mean().round(6))

# --- Plot loss curves ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for i, hist in enumerate(e2e_histories[:5]):
    color = plt.cm.viridis(i / max(1, len(e2e_histories[:5]) - 1))
    axes[0].plot(hist['loss_total'], c=color, alpha=0.7, lw=0.5,
                 label=f"S{i}")
    axes[1].plot(hist['loss_infonce'], c=color, alpha=0.7, lw=0.5)
    axes[2].plot(hist['loss_dyn'], c=color, alpha=0.7, lw=0.5)

axes[0].set_ylabel("Total Loss"); axes[0].set_xlabel("Epoch")
axes[1].set_ylabel("InfoNCE Loss"); axes[1].set_xlabel("Epoch")
axes[2].set_ylabel("Dynamics MSE"); axes[2].set_xlabel("Epoch")
axes[0].legend(fontsize=4, ncol=2)
for ax in axes:
    ax.set_yscale('log')
plt.suptitle("End-to-End Training Curves (first 5 sessions)",
             y=1.02, fontsize=9, fontweight="bold")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(os.path.join(LIE_OUTPUT_DIR, f"E2E_LossCurves.{fmt}"),
                dpi=150, bbox_inches="tight")
plt.show()

print("End-to-end training complete.")
