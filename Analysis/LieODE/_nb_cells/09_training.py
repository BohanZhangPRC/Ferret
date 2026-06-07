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
        window_len = VAL_ROLLOUT_LENS[0]  # shortest scale by default

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

        if not mse_full_wins:
            return float('nan')

        mse_full_avg = np.mean(mse_full_wins)
        mse_leak_avg = np.mean(mse_leak_wins)
        r2_drive = 1.0 - (mse_full_avg / (mse_leak_avg + 1e-9))
        return r2_drive


def compute_r2_drive_shuffle(model, ep_n, ep_drive, dt=0.005,
                              window_len=None, n_windows=5, n_shuffles=10):
    """E2E null control: circular-shift drive on FIXED trained encoder.

    Uses np.roll (circular shift) rather than np.random.permutation.
    This PRESERVES the autocorrelation structure and power spectrum of the
    drive signal x(t) (which is slowly-varying and highly autocorrelated),
    destroying only the temporal ALIGNMENT with the neural state.  This is
    the correct null: same distribution, same autocorrelation, random timing.

    (cf. lie_algebra_method_description.md section 12.3 — permutation destroys
    both alignment AND spectrum, confounding the null hypothesis.)

    n_windows is reduced to 5 (vs 20 for the true R2_drive) to keep
    validation time manageable: each shuffle is a full per-window
    encode + rollout, and n_shuffles=10 × n_windows multiplies cost.

    Returns:
        r2_drive_shuffle: float (mean across circular-shift realizations)
    """
    if window_len is None:
        window_len = VAL_ROLLOUT_LENS[0]  # shortest scale by default

    T_drive = len(ep_drive)
    if T_drive < 2:
        return float('nan')

    # Minimum shift >= MINI_TRAJ_LEN: shifts smaller than the trajectory
    # length leave the shuffled drive highly correlated with the original
    # (anti-conservative null).  We enforce a gap at least as large as
    # the rollout window so the null genuinely breaks alignment.
    min_shift = max(1, window_len)
    if T_drive <= 2 * min_shift:
        min_shift = max(1, T_drive // 3)

    vals = []
    for _ in range(n_shuffles):
        shift = np.random.randint(min_shift, T_drive - min_shift)
        ep_drive_shuf = np.roll(ep_drive, shift, axis=0)
        r2d = compute_r2_drive_rollout(model, ep_n, ep_drive_shuf,
                                        dt=dt, window_len=window_len,
                                        n_windows=n_windows)
        if not np.isnan(r2d):
            vals.append(r2d)
    return np.mean(vals) if vals else float('nan')


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

    # ---- Epoch extraction ----
    # Step 1: extract drive labels for standardization (DRIVE_KEYS[0])
    # Step 2: extract CEBRA labels for InfoNCE (CEBRA_LABEL)
    # When CEBRA_LABEL == DRIVE_KEYS[0], this is one extraction used for both.
    # When decoupled (e.g. CEBRA_LABEL="Played_frequency"), the embedding is
    # shaped by sensory context while dynamics are driven by motor command.
    cond_data = {}
    all_drive_labels = []
    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        _, drive_l, _ = extract_epochs(
            n_data_session, f_df, val, dt, label_col=DRIVE_KEYS[0])
        all_drive_labels.extend(drive_l)

    # Standardize drive labels (pooled TR+PB)
    if all_drive_labels:
        drv_cat = np.concatenate(all_drive_labels)
        mu_l, std_l = np.mean(drv_cat), np.std(drv_cat)
    else:
        mu_l, std_l = 0.0, 1.0
    if std_l < 1e-9:
        std_l = 1.0

    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        # CEBRA labels (for InfoNCE)
        epochs_n, epochs_l, _ = extract_epochs(
            n_data_session, f_df, val, dt, label_col=CEBRA_LABEL)
        # Drive labels (for dynamics u(t))
        _, drive_l, _ = extract_epochs(
            n_data_session, f_df, val, dt, label_col=DRIVE_KEYS[0])
        # Sanity check: same epoch count and lengths as CEBRA extraction
        assert len(drive_l) == len(epochs_l), \
            f"Mismatch: {len(drive_l)} drive epochs vs {len(epochs_l)} CEBRA epochs"
        for i in range(len(epochs_l)):
            assert len(drive_l[i]) == len(epochs_l[i]), \
                f"Epoch {i}: drive len {len(drive_l[i])} vs CEBRA len {len(epochs_l[i])}"

        drive_epochs = [((el - mu_l) / std_l).reshape(-1, 1).astype(np.float32)
                        for el in drive_l]
        valid_idx = [i for i in range(len(epochs_n))
                     if epochs_n[i].shape[0] >= MINI_TRAJ_LEN]
        cond_data[label] = {
            'n': [epochs_n[i] for i in valid_idx],
            'l': [epochs_l[i] for i in valid_idx],
            'd': [drive_epochs[i] for i in valid_idx]
        }

    # Multi-dim drive not yet implemented in the aligned epoch path.
    # (Would require per-epoch multi-dim drive extraction with TAU_SHIFT alignment.)
    if len(DRIVE_KEYS) > 1 or DRIVE_KEYS != ["Velocity_x"]:
        raise NotImplementedError(
            f"Multi-dim drive not yet implemented. DRIVE_KEYS={DRIVE_KEYS}. "
            f"Set DRIVE_KEYS=['Velocity_x'] for now. "
            f"To implement: extract per-epoch multi-dim drive vectors aligned "
            f"to epoch boundaries with TAU_SHIFT, matching epochs_l alignment.")

    # Train/val split on epochs (requires >=2 epochs per condition for valid split)
    all_train_n, all_train_l, all_train_d = [], [], []
    val_epochs = {}
    n_skipped_val = 0
    for cond_name in ["Tracking", "Playback"]:
        cd = cond_data[cond_name]
        n_ep = len(cd['n'])
        if n_ep < 2:
            print(f"  Session {session_idx} {cond_name}: only {n_ep} epoch(s), "
                  f"skipping (need >=2 for train/val split)")
            n_skipped_val += 1
            val_epochs[cond_name] = {'n': [], 'l': [], 'd': []}
            continue
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

    if n_skipped_val > 0 or not all_train_n:
        print(f"  Session {session_idx}: insufficient epochs for train/val, skipping")
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
            loss_info = info_nce_loss(z_flat, l_flat, temperature=TEMPERATURE)

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

    # ---- Multi-scale validation on held-out epochs ----
    val_metrics = {}
    for cond_name in ["Tracking", "Playback"]:
        ve = val_epochs[cond_name]
        r2d_multiscale = {w: [] for w in VAL_ROLLOUT_LENS}
        r2d_sh_multiscale = {w: [] for w in VAL_ROLLOUT_LENS}
        for ep_n, ep_d in zip(ve['n'], ve['d']):
            for wlen in VAL_ROLLOUT_LENS:
                r2d = compute_r2_drive_rollout(model, ep_n, ep_d, dt,
                                                window_len=wlen)
                r2d_multiscale[wlen].append(r2d)
                # Shuffle null (only at shortest scale for efficiency)
                if wlen == VAL_ROLLOUT_LENS[0]:
                    r2d_sh = compute_r2_drive_shuffle(
                        model, ep_n, ep_d, dt, window_len=wlen,
                        n_shuffles=N_SHUFFLES)
                    r2d_sh_multiscale[wlen].append(r2d_sh)
                else:
                    r2d_sh_multiscale[wlen].append(float('nan'))

        val_metrics[cond_name] = {
            'R2_drive_multiscale': {w: np.mean(r2d_multiscale[w])
                                     if r2d_multiscale[w] else float('nan')
                                     for w in VAL_ROLLOUT_LENS},
            'R2_drive_shuffle_multiscale': {w: np.mean(r2d_sh_multiscale[w])
                                             if r2d_sh_multiscale[w] else float('nan')
                                             for w in VAL_ROLLOUT_LENS},
            # Backward-compat: primary metric at shortest scale
            'R2_drive_rollout': np.mean(r2d_multiscale[VAL_ROLLOUT_LENS[0]])
            if r2d_multiscale[VAL_ROLLOUT_LENS[0]] else float('nan'),
            'R2_drive_shuffle': np.mean(r2d_sh_multiscale[VAL_ROLLOUT_LENS[0]])
            if r2d_sh_multiscale[VAL_ROLLOUT_LENS[0]] else float('nan'),
            'n_val_epochs': len(ve['n']),
        }

    return model, history, val_metrics


print("Training utilities ready. Ready to train per session.")
print(f"Training config: N_EPOCHS={N_EPOCHS_TRAIN}, LR={LR}, "
      f"BATCH={BATCH_SIZE}, MINI_TRAJ_LEN={MINI_TRAJ_LEN}")
print(f"lambda_dyn={LAMBDA_DYN}, warmup={LAMBDA_DYN_WARMUP} steps")
