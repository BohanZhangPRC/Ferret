# ============================================================
# Phase 2 -- Joint Training Loop (per session)
# ============================================================
# Loss = InfoNCE(z; behavior labels) + lambda_dyn * Dynamics_MSE(z_pred, z_true)
#
# Addressing 3 core challenges:
#   1. Batching: mini-trajectories (T=MINI_TRAJ_LEN) -> InfoNCE on
#      ALL flattened frames; Dynamics on per-trajectory rollout.
#   2. lambda balancing: warmup (lambda=0 for N steps) then linear ramp.
#   3. Dissipation constraint: ablation flag CONSTRAINED_L.


class TrajectoryDataset(Dataset):
    """Dataset of mini-trajectories for joint training."""

    def __init__(self, traj_n, traj_l, traj_d):
        self.traj_n = torch.tensor(traj_n, dtype=torch.float32)
        self.traj_l = torch.tensor(traj_l, dtype=torch.float32)
        self.traj_d = torch.tensor(traj_d, dtype=torch.float32)

    def __len__(self):
        return len(self.traj_n)

    def __getitem__(self, idx):
        return self.traj_n[idx], self.traj_l[idx], self.traj_d[idx]


def get_lambda_dyn(step, warmup, lambda_max):
    """Compute effective lambda_dyn with warmup.

    0 for step < warmup, then linear ramp to lambda_max over warmup steps.
    """
    if step < warmup:
        return 0.0
    if step < 2 * warmup:
        return lambda_max * (step - warmup) / warmup
    return lambda_max


def compute_r2_drive_rollout(model, ep_n, ep_drive, dt=0.005,
                              window_len=None, n_windows=20):
    """Cross-validated R2_drive using SHORT-WINDOW rollout.

    CRITICAL: Full-epoch rollout with unconstrained L diverges exponentially
    over hundreds of timesteps, contaminating R2_drive with numerical instability
    rather than measuring learned dynamics quality.

    Instead, we sample n_windows short windows (each of length window_len,
    matching the training mini-trajectory length) and average R2_drive across
    them. This isolates the model's ability to predict short-term dynamics
    — exactly what the training objective optimizes.

    IMPORTANT — train/val consistency: each window is encoded independently
    from raw neural data (not sliced from a full-epoch encoding).  This
    matches the training pipeline (create_mini_trajectories then encode),
    ensuring identical edge-padding semantics and eliminating a
    distribution shift between training and validation encodings.

    Args:
        model: trained SkieurLieODE
        ep_n: (T_epoch, N) neural data for one epoch
        ep_drive: (T_epoch, D_drive) drive for one epoch
        dt: bin size
        window_len: rollout window length (default: VAL_ROLLOUT_LEN)
        n_windows: number of short windows to sample

    Returns:
        r2_drive: float (averaged across windows)
    """
    if window_len is None:
        window_len = VAL_ROLLOUT_LEN  # from Cell 0 config

    model.eval()
    with torch.no_grad():
        n_t = torch.tensor(ep_n, dtype=torch.float32, device=DEVICE)
        d_t = torch.tensor(ep_drive, dtype=torch.float32, device=DEVICE)
        T_epoch = n_t.shape[0]

        # Precompute leak transition (L is constant, same for all windows)
        L_mat = model.lie_cell.dissipation.forward()       # (D, D)
        exp_L_dt = torch.matrix_exp(L_mat * dt)            # (D, D) — cached once

        if T_epoch < window_len + 1:
            starts = [0]
            n_actual = 1
        else:
            n_actual = min(n_windows, T_epoch - window_len)
            starts = np.random.choice(T_epoch - window_len, size=n_actual,
                                      replace=False)

        mse_full_wins, mse_leak_wins = [], []
        for s in starts:
            end = s + window_len

            # --- Per-window encoding (matches training pipeline) ---
            # Slice RAW neural, then encode — same edge-padding semantics
            # as create_mini_trajectories -> model(batch_n).
            n_win = n_t[s:end]                 # (W, N) raw
            d_win = d_t[s:end]                 # (W, D_drive)
            z_win = model.encode(n_win)        # (W, D) per-window encoding

            # Full rollout: z_0 -> z_1_pred ... z_{W-1}_pred
            z_pred_full = model.rollout(
                z_win[0:1], d_win.unsqueeze(0), dt).squeeze(0)  # (W+1, D)

            # Leak-only rollout (uses cached exp_L_dt)
            z_leak = [z_win[0:1]]
            z_t = z_win[0:1]
            for _ in range(len(d_win)):
                z_t = torch.mm(exp_L_dt, z_t.T).T
                z_leak.append(z_t)
            z_pred_leak = torch.cat(z_leak, dim=0)  # (W+1, D)

            # MSE: align z_pred[1:-1] vs z_win[1:] -> both (W-1, D)
            mse_full_wins.append(
                F.mse_loss(z_pred_full[1:-1], z_win[1:]).item())
            mse_leak_wins.append(
                F.mse_loss(z_pred_leak[1:-1], z_win[1:]).item())
            mse_leak_wins.append(
                F.mse_loss(z_pred_leak[1:-1], z_win[1:]).item())

        if not mse_full_wins:
            return float('nan')

        mse_full_avg = np.mean(mse_full_wins)
        mse_leak_avg = np.mean(mse_leak_wins)
        r2_drive = 1.0 - (mse_full_avg / (mse_leak_avg + 1e-9))
        return r2_drive


def train_one_session(model, n_data_session, f_df, session_idx,
                       n_epochs_train=N_EPOCHS_TRAIN,
                       lambda_dyn=LAMBDA_DYN,
                       lambda_warmup=LAMBDA_DYN_WARMUP):
    """Train the end-to-end model on one session.

    Returns:
        model: trained model
        history: dict with loss/lambda curves
        val_metrics: dict with per-condition held-out metrics
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs_train, eta_min=1e-5)

    # Extract epochs for both conditions
    cond_data = {}
    # Collect all labels first for per-session standardization
    all_labels_session = []
    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        epochs_n, epochs_l, _ = extract_epochs(
            n_data_session, f_df, val, dt, label_col="Velocity_x")
        all_labels_session.extend(epochs_l)

    # Standardize labels per session (they ARE the drive when DRIVE_KEYS=["Velocity_x"])
    if all_labels_session:
        lab_cat = np.concatenate(all_labels_session)
        mu_l, std_l = np.mean(lab_cat), np.std(lab_cat)
    else:
        mu_l, std_l = 0.0, 1.0
    if std_l < 1e-9:
        std_l = 1.0

    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        epochs_n, epochs_l, _ = extract_epochs(
            n_data_session, f_df, val, dt, label_col="Velocity_x")
        # Use standardized labels as drive (correctly aligned + TAU_SHIFT-trimmed)
        # Reshape to (T, 1) for single-drive ControlNet input
        drive_epochs = [((el - mu_l) / std_l).reshape(-1, 1).astype(np.float32)
                        for el in epochs_l]
        valid_idx = [i for i in range(len(epochs_n))
                     if epochs_n[i].shape[0] >= MINI_TRAJ_LEN]
        cond_data[label] = {
            'n': [epochs_n[i] for i in valid_idx],
            'l': [epochs_l[i] for i in valid_idx],
            'd': [drive_epochs[i] for i in valid_idx]
        }

    # Warn if DRIVE_KEYS has more than Velocity_x (not yet supported in aligned path)
    if len(DRIVE_KEYS) > 1 or DRIVE_KEYS != ["Velocity_x"]:
        print(f"  WARNING: drive alignment currently uses Velocity_x labels only. "
              f"DRIVE_KEYS={DRIVE_KEYS} — extra dims ignored.")

    # Train/val split on epochs
    all_train_n, all_train_l, all_train_d = [], [], []
    val_epochs = {}
    for cond_name in ["Tracking", "Playback"]:
        cd = cond_data[cond_name]
        n_ep = len(cd['n'])
        n_train = max(1, int(n_ep * TRAIN_VAL_SPLIT))
        # Training epochs
        for i in range(n_train):
            all_train_n.append(cd['n'][i])
            all_train_l.append(cd['l'][i])
            all_train_d.append(cd['d'][i])
        # Held-out validation epochs
        val_epochs[cond_name] = {
            'n': cd['n'][n_train:],
            'l': cd['l'][n_train:],
            'd': cd['d'][n_train:],
        }

    if not all_train_n:
        print(f"  Session {session_idx}: no training data")
        return model, {}, {}

    history = {'loss_total': [], 'loss_infonce': [], 'loss_dyn': [],
               'lambda_dyn': [], 'grad_norm': []}
    d_drive = len(DRIVE_KEYS)
    global_step = 0

    for epoch in trange(n_epochs_train, desc=f"Train S{session_idx}",
                        leave=False):
        # Sample mini-trajectories
        traj_n, traj_l, traj_d = create_mini_trajectories(
            all_train_n, all_train_l, all_train_d,
            traj_len=MINI_TRAJ_LEN, n_samples_per_epoch=30)

        if traj_n is None:
            continue

        dataset = TrajectoryDataset(traj_n, traj_l, traj_d)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // MINI_TRAJ_LEN,
                                shuffle=True, drop_last=True)

        epoch_loss_total, epoch_loss_info, epoch_loss_dyn = 0.0, 0.0, 0.0
        n_batches = 0

        for batch_n, batch_l, batch_d in dataloader:
            batch_n = batch_n.to(DEVICE)  # (B, T, N)
            batch_l = batch_l.to(DEVICE)  # (B, T)
            batch_d = batch_d.to(DEVICE)  # (B, T, D_drive)

            B, T_len, N_neurons = batch_n.shape

            # ---- Forward pass ----
            z_true, z_pred = model(batch_n, batch_d, dt)  # (B,T,D), (B,T+1,D)

            # ---- InfoNCE loss (flatten ALL frames across batch) ----
            z_flat = z_true.reshape(-1, D_LATENT)  # (B*T, D)
            l_flat = batch_l.reshape(-1)           # (B*T,)
            loss_info = info_nce_loss(z_flat, l_flat)

            # ---- Dynamics MSE (trajectory rollout prediction) ----
            # z_pred: (B, T+1, D) = [z_0_pred, z_1_pred, ..., z_T_pred]
            # z_true: (B, T,   D) = [z_0_true, z_1_true, ..., z_{T-1}_true]
            # Align: z_pred[1:-1] (predicted z_1..z_{T-1}) vs z_true[1:] (true z_1..z_{T-1})
            # Both: (B, T-1, D)
            #
            # CRITICAL: InfoNCE is scale-invariant (F.normalize), so ||z|| has no
            # InfoNCE gradient.  Raw MSE ∝ ||z||^2, so the optimizer can trivially
            # shrink ||z|| -> 0 to reduce dynamics loss without learning any real
            # dynamics.  Normalising by variance removes this degenerate path:
            # both numerator and denominator scale with ||z||^2 -> ratio is
            # scale-invariant, matching the R2_drive evaluation philosophy.
            z_tgt = z_true[:, 1:]                       # (B, T-1, D)
            z_prd = z_pred[:, 1:-1]                     # (B, T-1, D)
            mse_dyn = F.mse_loss(z_prd, z_tgt)
            var_z   = z_tgt.var().clamp_min(1e-6)       # per-element variance
            loss_dyn = mse_dyn / var_z

            # ---- Lambda schedule (warmup) ----
            lam = get_lambda_dyn(global_step, lambda_warmup, lambda_dyn)

            # ---- Total loss ----
            loss = loss_info + lam * loss_dyn

            # ---- Optimization ----
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss_total += loss.item()
            epoch_loss_info += loss_info.item()
            epoch_loss_dyn += loss_dyn.item()
            n_batches += 1
            global_step += 1

        if n_batches > 0:
            history['loss_total'].append(epoch_loss_total / n_batches)
            history['loss_infonce'].append(epoch_loss_info / n_batches)
            history['loss_dyn'].append(epoch_loss_dyn / n_batches)
            history['lambda_dyn'].append(lam)
            history['grad_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)

        scheduler.step()

    # ---- Validation metrics on held-out epochs ----
    val_metrics = {}
    for cond_name in ["Tracking", "Playback"]:
        ve = val_epochs[cond_name]
        r2d_vals = []
        for ep_n, ep_d in zip(ve['n'], ve['d']):
            r2d = compute_r2_drive_rollout(model, ep_n, ep_d, dt)
            r2d_vals.append(r2d)
        val_metrics[cond_name] = {
            'R2_drive_rollout': np.mean(r2d_vals) if r2d_vals else float('nan'),
            'R2_drive_sem': np.std(r2d_vals) / max(1, len(r2d_vals)) ** 0.5
            if r2d_vals else float('nan'),
            'n_val_epochs': len(r2d_vals),
        }

    return model, history, val_metrics


print("Training utilities ready. Ready to train per session.")
print(f"Training config: N_EPOCHS={N_EPOCHS_TRAIN}, LR={LR}, "
      f"BATCH={BATCH_SIZE}, MINI_TRAJ_LEN={MINI_TRAJ_LEN}")
print(f"lambda_dyn={LAMBDA_DYN}, warmup={LAMBDA_DYN_WARMUP} steps")
