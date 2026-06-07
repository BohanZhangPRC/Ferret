# ============================================================
# Helper functions: epoch extraction + Lie algebra + new utils
# ============================================================

# --- Reused from existing notebook ---

def preprocess_data(data_list, method="l2"):
    """Preprocess list of (time, neurons) arrays.
    method='l2': per-timepoint L2 normalization (for cosine distance)
    method='zscore': per-neuron Z-score across time (for euclidean distance)
    """
    out = []
    for d in data_list:
        if method == "zscore":
            mean = np.mean(d, axis=0, keepdims=True)
            std = np.std(d, axis=0, keepdims=True)
            std[std == 0] = 1e-9
            out.append(((d - mean) / std).astype(np.float32))
        else:  # l2
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            out.append((d / norms).astype(np.float32))
    return out


def extract_macro_epochs(n_data_session, f_df, condition_val, dt,
                          min_duration=2.0, label_col="Velocity_x"):
    """Extract contiguous macro-epochs of the same Condition."""
    conditions = f_df["Condition"].values
    mask = (conditions == condition_val)
    min_bins = int(min_duration / dt)
    epochs_n, epochs_l = [], []
    in_epoch, start = False, 0
    for i in range(len(mask)):
        if mask[i] and not in_epoch:
            start = i; in_epoch = True
        elif not mask[i] and in_epoch:
            if i - start >= min_bins:
                epochs_n.append(n_data_session[:, start:i].T.astype(np.float32))
                epochs_l.append(f_df[label_col].values[start:i].astype(np.float32))
            in_epoch = False
    if in_epoch and (len(mask) - start) >= min_bins:
        epochs_n.append(n_data_session[:, start:].T.astype(np.float32))
        epochs_l.append(f_df[label_col].values[start:].astype(np.float32))
    return epochs_n, epochs_l, len(epochs_n)


def extract_epochs(n_data_session, f_df, condition_val, dt,
                   label_col="Velocity_x", step=1):
    """Unified extraction: macro-epochs or peri-event windows."""
    if USE_MACRO_EPOCH:
        epochs_n, epochs_l, n_ep = extract_macro_epochs(
            n_data_session, f_df, condition_val, dt,
            min_duration=MIN_EPOCH_DUR, label_col=label_col)
        if step > 1:
            epochs_n = [e[::step].astype(np.float32) for e in epochs_n]
            epochs_l = [l[::step].astype(np.float32) for l in epochs_l]
        method = "l2" if CEBRA_DISTANCE == "cosine" else "zscore"
        epochs_n = preprocess_data(epochs_n, method=method)
        if TAU_SHIFT > 0:
            epochs_n, epochs_l = zip(*[(e_n[TAU_SHIFT:], e_l[:-TAU_SHIFT])
                                        for e_n, e_l in zip(epochs_n, epochs_l)])
            epochs_n, epochs_l = list(epochs_n), list(epochs_l)
        return epochs_n, epochs_l, n_ep
    else:
        trigger_mask = ((f_df["Condition"].values == condition_val) &
                        (f_df["Frequency_changes"].values == 1))
        trigger_indices = np.where(trigger_mask)[0]
        n_pre = int(t_pre / dt)
        n_post = int(t_post / dt)
        windows_n, windows_l = [], []
        for idx in trigger_indices:
            start = idx - n_pre
            end = idx + n_post + 1
            if start >= 0 and end <= n_data_session.shape[1]:
                windows_n.append(n_data_session[:, start:end].T.astype(np.float32))
                windows_l.append(f_df[label_col].values[start:end].astype(np.float32))
        if step > 1:
            windows_n = [w[::step].astype(np.float32) for w in windows_n]
            windows_l = [l[::step].astype(np.float32) for l in windows_l]
        method = "l2" if CEBRA_DISTANCE == "cosine" else "zscore"
        windows_n = preprocess_data(windows_n, method=method)
        if TAU_SHIFT > 0:
            windows_n, windows_l = zip(*[(w_n[TAU_SHIFT:], w_l[:-TAU_SHIFT])
                                          for w_n, w_l in zip(windows_n, windows_l)])
            windows_n, windows_l = list(windows_n), list(windows_l)
        return windows_n, windows_l, len(windows_n)


def _fit_lie_lstsq(r, x_dot, dt=0.005):
    """OLS + skew-symmetrization (original baseline method)."""
    T, N = r.shape
    dr_dt = np.gradient(r, dt, axis=0)
    U_rot = r * x_dot[:, np.newaxis]
    U = np.hstack([U_rot, r])
    weights_T, _, _, _ = np.linalg.lstsq(U, dr_dt, rcond=None)
    weights = weights_T.T
    J_ols = weights[:, :N]
    J_skew = 0.5 * (J_ols - J_ols.T)
    norm_total = np.linalg.norm(J_ols)
    norm_skew = np.linalg.norm(J_skew)
    sr = norm_skew / norm_total if norm_total > 1e-9 else 0
    dR_pred = U_rot @ J_skew.T + r @ weights[:, N:].T
    ss_res = np.sum((dr_dt - dR_pred) ** 2)
    ss_tot = np.sum((dr_dt - np.mean(dr_dt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0
    dR_leak = r @ weights[:, N:].T
    ss_leak = np.sum((dr_dt - dR_leak) ** 2)
    r2_drive = 1 - ss_res / ss_leak if ss_leak > 1e-9 else 0
    return J_skew, sr, r2, J_ols, r2_drive


def _fit_lie_pytorch(r, x_dot, dt=0.005, n_iter=500, lr=1e-3):
    """Constrained optimization: J_skew = W - W^T via PyTorch."""
    T, N = r.shape
    dr_dt = np.gradient(r, dt, axis=0)
    R_t = torch.tensor(r, dtype=torch.float32)
    X_t = torch.tensor(x_dot, dtype=torch.float32).reshape(-1, 1)
    dR_t = torch.tensor(dr_dt, dtype=torch.float32)
    W = torch.zeros(N, N, requires_grad=True)
    L_t = torch.zeros(N, N, requires_grad=True)
    opt = torch.optim.Adam([W, L_t], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        J_t = W - W.T
        dR_pred = (R_t * X_t) @ J_t.T + R_t @ L_t.T
        loss = torch.mean((dR_t - dR_pred) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        J_skew = (W - W.T).numpy()
        L_np = L_t.numpy()
    U = np.hstack([r * x_dot[:, None], r])
    w_T, _, _, _ = np.linalg.lstsq(U, dr_dt, rcond=None)
    J_ols = w_T.T[:, :N]
    sr = (np.linalg.norm(J_skew) / np.linalg.norm(J_ols)
          if np.linalg.norm(J_ols) > 1e-9 else 0)
    dR_pred = (r * x_dot[:, None]) @ J_skew.T + r @ L_np.T
    ss_res = np.sum((dr_dt - dR_pred) ** 2)
    ss_tot = np.sum((dr_dt - np.mean(dr_dt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0
    dR_leak = r @ L_np.T
    ss_leak = np.sum((dr_dt - dR_leak) ** 2)
    r2_drive = 1 - ss_res / ss_leak if ss_leak > 1e-9 else 0
    return J_skew, sr, r2, J_ols, r2_drive


def fit_lie_algebra_with_leak(r, x_dot, dt=0.005, n_iter=500, lr=1e-3):
    """Fit dR/dt = J_skew * R * x_dot + L * R.
    Method controlled by global LIE_METHOD.
    """
    if LIE_METHOD == "lstsq":
        return _fit_lie_lstsq(r, x_dot, dt)
    else:
        return _fit_lie_pytorch(r, x_dot, dt, n_iter, lr)


# --- NEW helper functions ---

def build_drive_vector(f_df, drive_keys=None):
    """Build multi-dimensional drive vector from feature DataFrame."""
    if drive_keys is None:
        drive_keys = DRIVE_KEYS
    components = []
    for key in drive_keys:
        if key in f_df.columns:
            x = f_df[key].values.astype(np.float32)
            mu, std = np.mean(x), np.std(x)
            if std < 1e-9:
                std = 1.0
            components.append((x - mu) / std)
        else:
            print(f"  WARNING: drive key '{key}' not in f_df; skipping")
    if not components:
        x = f_df["Velocity_x"].values.astype(np.float32)
        mu, std = np.mean(x), np.std(x)
        if std < 1e-9:
            std = 1.0
        components.append((x - mu) / std)
    return np.column_stack(components) if len(components) > 1 else components[0]


def create_mini_trajectories(epochs_n, epochs_l, drive_epochs,
                              traj_len=20, n_samples_per_epoch=50):
    """Sample mini-trajectories from epoch data for joint training.

    Each mini-trajectory is a contiguous window of length traj_len,
    sampled uniformly from within the epoch.

    Returns (traj_n, traj_l, traj_d) or (None, None, None) if no valid data.
    """
    all_n, all_l, all_d = [], [], []
    for e_n, e_l, e_d in zip(epochs_n, epochs_l, drive_epochs):
        T_ep = e_n.shape[0]
        if T_ep < traj_len:
            continue
        n_sample = min(n_samples_per_epoch, T_ep - traj_len + 1)
        starts = np.random.choice(T_ep - traj_len + 1, size=n_sample,
                                  replace=False)
        for s in starts:
            all_n.append(e_n[s:s + traj_len])
            all_l.append(e_l[s:s + traj_len] if e_l.ndim == 1
                         else e_l[s:s + traj_len])
            all_d.append(e_d[s:s + traj_len] if e_d.ndim == 2
                         else e_d[s:s + traj_len, None])
    if not all_n:
        return None, None, None
    return np.stack(all_n), np.stack(all_l), np.stack(all_d)


def compute_eigenvalue_metrics(J_full):
    """Compute |Real| and |Imag| from eigenvalues of J_full.

    Returns (mean_abs_real, mean_abs_imag).
    """
    eigvals = np.linalg.eigvals(J_full)
    real_mean = np.mean(np.abs(np.real(eigvals)))
    imag_mean = np.mean(np.abs(np.imag(eigvals)))
    return real_mean, imag_mean


def info_nce_loss(z, labels, temperature=None):
    """InfoNCE loss for contrastive learning on continuous behavioral labels.

    For each anchor, finds k nearest neighbors in label space as positives.
    Uses per-row k-nearest (not global threshold) to avoid collapse from
    self-pair zeros dominating kthvalue on the full flattened matrix.
    """
    if temperature is None:
        temperature = TEMPERATURE  # from Cell 0 config
    z_norm = F.normalize(z, p=2, dim=1)
    sim = z_norm @ z_norm.T / temperature       # (N, N)
    label_diff = torch.abs(labels[:, None] - labels[None, :])  # (N, N)
    N = len(labels)
    k_pos = max(2, int(0.05 * N))

    # Per-row: exclude self (diagonal) and find k-th smallest label difference
    diff_no_self = label_diff + torch.eye(N, device=labels.device) * 1e10
    kth_per_row, _ = torch.kthvalue(diff_no_self, k_pos, dim=1)  # (N,)
    pos_mask = label_diff < kth_per_row[:, None]    # (N, N)
    pos_mask.fill_diagonal_(False)                   # still exclude self

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)           # (N,)
    all_sum = exp_sim.sum(dim=1) - exp_sim.diagonal()           # (N,) exclude self
    loss = -torch.log(pos_sum / (all_sum + 1e-9) + 1e-9).mean()
    return loss


print("Helpers ready: extract_epochs, fit_lie_algebra_with_leak, "
      "build_drive_vector, create_mini_trajectories, info_nce_loss")
