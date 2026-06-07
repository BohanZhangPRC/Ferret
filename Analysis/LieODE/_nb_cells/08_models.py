# ============================================================
# Phase 1 -- PyTorch Model Components
# ============================================================

# --- Skew-symmetric basis ---

def make_skew_basis(dim):
    """Create orthonormal basis for so(dim): dim*(dim-1)/2 skew-symmetric matrices.

    Each basis matrix G_k has exactly two non-zero entries:
    G_k[i,j] = 1/sqrt(2), G_k[j,i] = -1/sqrt(2).
    Normalized so that ||G_k||_F = 1 and <G_i, G_j> = delta_ij.
    """
    n_basis = dim * (dim - 1) // 2
    basis = torch.zeros(n_basis, dim, dim)
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            G = torch.zeros(dim, dim)
            G[i, j] = 1.0 / (2.0 ** 0.5)
            G[j, i] = -1.0 / (2.0 ** 0.5)
            basis[k] = G
            k += 1
    return basis  # (n_basis, dim, dim)


class SkewBasis(nn.Module):
    """Fixed orthonormal skew-symmetric basis for so(dim).

    Holds dim*(dim-1)/2 orthonormal skew matrices G_i as a fixed buffer.
    The actual coefficients w_i(u_t) come from ControlNet, making
    J(u_t) = sum_i w_i(u_t) * G_i skew-symmetric by construction.

    (No learnable parameters — the ControlNet learns the mapping from drive.)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_basis = dim * (dim - 1) // 2
        self.register_buffer('basis', make_skew_basis(dim))  # (n_basis, dim, dim)


class ControlNet(nn.Module):
    """Nonlinear control network: multi-dim drive -> basis weights.

    J(u_t) = sum_i w_i(u_t) * G_i, where w_i come from a small MLP.
    """
    def __init__(self, drive_dim, n_basis, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = CONTROL_HIDDEN
        layers = []
        in_dim = drive_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_basis))
        self.net = nn.Sequential(*layers)
        self.n_basis = n_basis

    def forward(self, u_t):
        """u_t: (*, drive_dim) -> w_i: (*, n_basis)."""
        w = self.net(u_t)
        w = torch.tanh(w)  # smooth, bounded [-1, 1]
        return w


class Dissipation(nn.Module):
    """Leak/dissipation matrix L.

    Two modes:
    - CONSTRAINED_L=True:  L = -C @ C.T (guarantees Re(eig) <= 0)
    - CONSTRAINED_L=False: L is unconstrained (dim, dim) matrix
    """
    def __init__(self, dim, constrained=False):
        super().__init__()
        self.dim = dim
        self.constrained = constrained
        if constrained:
            self.C = nn.Parameter(torch.randn(dim, dim) * 0.1)
        else:
            self.L_raw = nn.Parameter(torch.zeros(dim, dim))

    def forward(self):
        """Return L matrix (dim, dim)."""
        if self.constrained:
            return -self.C @ self.C.T
        return self.L_raw

    def get_L_numpy(self):
        """Return L as numpy array."""
        return self.forward().detach().cpu().numpy()


class LieODECell(nn.Module):
    """Lie-ODE transition cell: dz/dt = (J(u_t) + L) @ z.

    Supports two transition modes:
    - discrete: z_{t+1} = matrix_exp(dt * (J(u_t) + L)) @ z_t
    - ode:      z(t) = odeint(dz/dt, z_0, t_span) (requires torchdiffeq)
    """
    def __init__(self, dim, n_basis, constrained_L=False, use_ode=False,
                 ode_method="rk4"):
        super().__init__()
        self.dim = dim
        self.use_ode = use_ode
        self.ode_method = ode_method
        self.skew_basis = SkewBasis(dim)
        self.control = ControlNet(1, n_basis)
        self.dissipation = Dissipation(dim, constrained=constrained_L)

    def set_control_input_dim(self, drive_dim):
        """Rebuild control net if drive_dim changes."""
        if self.control.net[0].in_features != drive_dim:
            n_basis = self.dim * (self.dim - 1) // 2
            self.control = ControlNet(drive_dim, n_basis)

    def compute_generator(self, u_t):
        """Compute J(u_t), L, and full generator A = J + L.

        Args:
            u_t: (*, drive_dim)

        Returns:
            A: (*, dim, dim) full generator
            J_weighted: (*, dim, dim) skew-symmetric component
            L: (dim, dim) dissipation
        """
        L = self.dissipation.forward()      # (dim, dim)
        w = self.control(u_t)               # (*, n_basis)
        G = self.skew_basis.basis           # (n_basis, dim, dim)
        J_weighted = (w[..., None, None] * G).sum(dim=-3)  # (*, dim, dim)
        A = J_weighted + L
        return A, J_weighted, L

    def forward_discrete(self, z_0, drive_seq, dt=0.005):
        """Rollout using discrete matrix_exp.

        z_0: (B, D)
        drive_seq: (B, T, D_drive)
        Returns: z_seq (B, T+1, D) with z_seq[:, 0] = z_0
        """
        B, T, _ = drive_seq.shape
        z_seq = [z_0]
        z_t = z_0
        for t in range(T):
            A, _, _ = self.compute_generator(drive_seq[:, t, :])
            A_dt = A * dt
            exp_A = torch.matrix_exp(A_dt)
            z_t = torch.bmm(exp_A, z_t.unsqueeze(-1)).squeeze(-1)
            z_seq.append(z_t)
        return torch.stack(z_seq, dim=1)

    def _ode_func(self, t, z, drive_interp):
        """ODE RHS: dz/dt = (J(u_t) + L) @ z."""
        u_t = drive_interp(t)
        A, _, _ = self.compute_generator(u_t)
        return torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)

    def forward_ode(self, z_0, drive_seq, dt=0.005):
        """Rollout using ODE solver (requires torchdiffeq).

        z_0: (B, D)
        drive_seq: (B, T, D_drive)
        Returns: z_seq (B, T+1, D)
        """
        B, T, _ = drive_seq.shape
        t_span = torch.linspace(0, T * dt, T + 1, device=z_0.device)

        def drive_interp(t_val):
            idx = (t_val / dt).long().clamp(0, T - 1)
            return drive_seq[torch.arange(B, device=t_val.device), idx, :]

        z_seq = torchdiffeq.odeint(
            lambda t, z: self._ode_func(t, z, drive_interp),
            z_0, t_span, method=self.ode_method,
            options={'step_size': dt} if self.ode_method == 'rk4' else {}
        )
        return z_seq.permute(1, 0, 2)

    def forward(self, z_0, drive_seq, dt=0.005):
        """Rollout: z_0 -> z_seq."""
        if self.use_ode and HAS_TORCHDIFFEQ:
            if self.ode_method == "dopri5":
                print("WARNING: USE_ODE is experimental; drive is piecewise-constant "
                      "(nearest-neighbor interp). Adaptive dopri5 may be unstable "
                      "at bin boundaries — prefer rk4 or discrete matrix_exp.")
            return self.forward_ode(z_0, drive_seq, dt)
        return self.forward_discrete(z_0, drive_seq, dt)


class Encoder(nn.Module):
    """Temporal conv encoder: offset-window neural -> latent z (preserves T).

    Only the first Conv1d layer uses kernel=offset (default 11 bins = 55ms at
    dt=0.005).  All subsequent layers use kernel=1, so the temporal receptive
    field is exactly `offset` bins — comparable to CEBRA offset10 (~50ms).

    Input:  (B, T, N) or (T, N)
    Output: (B, T, D_LATENT) or (T, D_LATENT)
    """
    def __init__(self, n_neurons, latent_dim, hidden=None, offset=11):
        super().__init__()
        if hidden is None:
            hidden = ENCODER_HIDDEN
        if offset % 2 == 0:
            raise ValueError(f"offset must be odd to preserve T; got {offset}")
        pad = offset // 2   # symmetric padding preserves length
        in_c = n_neurons
        layers = []
        # First layer: temporal context via offset-kernel Conv1d
        layers += [nn.Conv1d(in_c, hidden[0], offset, padding=pad), nn.ReLU()]
        in_c = hidden[0]
        # Subsequent layers: kernel=1 (pointwise, no additional temporal RF)
        for h in hidden[1:]:
            layers += [nn.Conv1d(in_c, h, 1), nn.ReLU()]
            in_c = h
        layers += [nn.Conv1d(in_c, latent_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B,T,N) or (T,N) -> (B,T,D) or (T,D)."""
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)   # (1, T, N)
        # Conv1d expects (B, C, T)
        z = self.net(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        return z.squeeze(0) if single else z


class SkieurLieODE(nn.Module):
    """Full end-to-end model: Encoder + LieODECell.

    forward modes:
    - "encode":  neural -> latent z only
    - "rollout": z_0 + drive_seq -> predicted trajectory
    - "full":    neural -> z -> rollout -> (z_true, z_pred)
    """
    def __init__(self, n_neurons, latent_dim, drive_dim,
                 constrained_L=False, use_ode=False, ode_method="rk4"):
        super().__init__()
        n_basis = latent_dim * (latent_dim - 1) // 2
        self.latent_dim = latent_dim
        self.encoder = Encoder(n_neurons, latent_dim)
        self.lie_cell = LieODECell(latent_dim, n_basis,
                                   constrained_L=constrained_L,
                                   use_ode=use_ode, ode_method=ode_method)
        self.lie_cell.set_control_input_dim(drive_dim)

    def encode(self, neural):
        """neural: (T, N) or (B, T, N) -> z: (T, D) or (B, T, D)."""
        return self.encoder(neural)

    def rollout(self, z_0, drive_seq, dt=0.005):
        """z_0: (B, D), drive_seq: (B, T, D_drive) -> z_seq: (B, T+1, D)."""
        return self.lie_cell(z_0, drive_seq, dt)

    def forward(self, neural_seq, drive_seq, dt=0.005):
        """Full forward pass.

        neural_seq: (B, T, N)
        drive_seq:  (B, T, D_drive)
        Returns: z_true (B, T, D), z_pred (B, T+1, D)
        """
        # Conv encoder preserves T — no reshape/flatten needed
        z_true = self.encode(neural_seq)  # (B, T, D)
        z_pred = self.rollout(z_true[:, 0, :], drive_seq, dt)
        return z_true, z_pred

    def get_generator_matrices(self, n_drive_samples=100):
        """Return representative J_skew, L as numpy arrays for metric reporting.

        The true generator J(u_t) is drive-dependent via ControlNet.
        Rather than reporting J(0) (which may be near-zero if the model
        learns "no rotation at rest"), we sample n_drive_samples from the
        standardized-drive distribution N(0,1) and report ||E_u[J(u)]||.

        This captures the expected rotational structure under the drive
        distribution, making SR and eigenvalues representative of the
        actual operating regime.
        """
        with torch.no_grad():
            dev = next(self.parameters()).device
            d_drive = self.lie_cell.control.net[0].in_features
            # Sample from standardized drive distribution N(0, 1)
            u_samples = torch.randn(n_drive_samples, d_drive, device=dev)
            _, J_samples, L = self.lie_cell.compute_generator(u_samples)
            # ||J|| averaged over drive distribution
            J_norms = torch.norm(J_samples.reshape(n_drive_samples, -1), dim=1)
            J_avg_norm = J_norms.mean()
            # Representative J: the average generator matrix
            J_avg = J_samples.mean(dim=0)
        return J_avg.cpu().numpy(), L.cpu().numpy()


# --- Quick sanity test ---
n_test = n_data_all[0].shape[0]
d_drive = len(DRIVE_KEYS)
model = SkieurLieODE(n_test, D_LATENT, d_drive,
                     constrained_L=CONSTRAINED_L,
                     use_ode=USE_ODE, ode_method=ODE_METHOD)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")
print(f"  Encoder: Conv1d offset-11, {n_test} -> {D_LATENT} (comparable to CEBRA offset10)")
print(f"  Skew basis: {D_LATENT}*(D_LATENT-1)/2 = {D_LATENT*(D_LATENT-1)//2}")
print(f"  Drive dim: {d_drive}")
print(f"  CONSTRAINED_L: {CONSTRAINED_L}")
print(f"  USE_ODE: {USE_ODE}")

# Quick smoke test
B, T = 8, MINI_TRAJ_LEN
x_test = torch.randn(B, T, n_test, device=DEVICE)
d_test = torch.randn(B, T, d_drive, device=DEVICE)
z_true, z_pred = model(x_test, d_test)
print(f"  Smoke test: z_true {tuple(z_true.shape)}, z_pred {tuple(z_pred.shape)}")
# Check skewness
J_s, L_m = model.get_generator_matrices()
J_skewness = np.linalg.norm(J_s) / (np.linalg.norm(J_s) + np.linalg.norm(L_m) + 1e-9)
print(f"  Initial J skewness (before training): {J_skewness:.4f}")
print("Model instantiation OK.")
