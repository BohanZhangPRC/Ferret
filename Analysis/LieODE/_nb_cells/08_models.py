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
    """Naive (unstructured) control network: multi-dim drive -> basis weights.

    J(u_t) = sum_i w_i(u_t) * G_i, where w_i come from a small MLP.

    This is the fallback when N_GATE_DIMS = -1.  The MLP learns an
    arbitrary mapping from all drive dimensions to basis weights — no
    physical constraint on how the motor command gates rotation.
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


class StructuredControlNet(nn.Module):
    """Dual-engine structured control: w_i(t) = gate(u_t) * MLP_i(f(t)).

    gate = α_1*gate_var_1 + ... + α_k*gate_var_k  (nn.Linear, bias=False)
    f(t) = remaining drive dims                    (sensory context)

    The gate is a learned linear combination of multiple drive signals
    (e.g., Velocity_x + Freq_dot for A1 sensorimotor integration).
    bias=False preserves the physical boundary condition:
        all gate signals = 0  =>  gate = 0  =>  J = 0

    In Playback:  Freq_dot ≠ 0 → gate non-zero → rotation continues at v=0
    In Tracking:  Freq_dot ∝ v → both terms resonate → strongest rotation

    When normalize=False (default), ||J|| depends on both gate magnitude and
    the context MLP's output, making SR informative about learned structure.
    When normalize=True, ||J|| ∝ |gate| exactly (SR degenerates).
    """
    def __init__(self, n_gate_dims, ctx_dim, n_basis, hidden=None, normalize=False):
        super().__init__()
        if hidden is None:
            hidden = CONTROL_HIDDEN
        self.n_gate_dims = n_gate_dims
        self.normalize = normalize
        self.gate_linear = nn.Linear(n_gate_dims, 1, bias=False)
        layers = []
        in_dim = ctx_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_basis))
        self.ctx_net = nn.Sequential(*layers)
        if normalize:
            self.log_scale = nn.Parameter(torch.zeros(1))  # softplus -> ~0.69

    def forward(self, u_t):
        """u_t: (*, drive_dim) where u_t[..., :n_gate_dims] = gate vars,
        u_t[..., n_gate_dims:] = sensory context.

        Returns w_i: (*, n_basis).
        """
        gate = self.gate_linear(u_t[..., :self.n_gate_dims])  # (*, 1)
        f = u_t[..., self.n_gate_dims:]                        # (*, ctx_dim)
        h = self.ctx_net(f)                                     # (*, n_basis)
        if self.normalize:
            h_dir = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)
            s = F.softplus(self.log_scale)
            return gate * h_dir * s
        return gate * h


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
    """Lie dynamics transition cell: dz/dt = (J(u_t) + L) @ z.

    Supports two transition modes:
    - discrete: z_{t+1} = matrix_exp(dt * (J(u_t) + L)) @ z_t
    - ode:      z(t) = odeint(dz/dt, z_0, t_span) (requires torchdiffeq)
    """
    def __init__(self, dim, n_basis, constrained_L=False, use_ode=False,
                 ode_method="rk4", n_gate_dims=2, normalize_ctx=False):
        super().__init__()
        self.dim = dim
        self.use_ode = use_ode
        self.ode_method = ode_method
        self.n_gate_dims = n_gate_dims
        self.normalize_ctx = normalize_ctx
        self.skew_basis = SkewBasis(dim)
        # Placeholder — will be replaced by set_control_input_dim
        self.control = ControlNet(1, n_basis)
        self.dissipation = Dissipation(dim, constrained=constrained_L)

    def set_control_input_dim(self, drive_dim):
        """Rebuild control net for the given drive_dim.

        When n_gate_dims >= 1 and drive_dim > n_gate_dims: uses
        StructuredControlNet with ctx_dim = drive_dim - n_gate_dims.
        When n_gate_dims = -1: uses naive ControlNet (fallback).
        """
        n_basis = self.dim * (self.dim - 1) // 2
        self.drive_dim = drive_dim  # stored for get_generator_matrices
        if self.n_gate_dims >= 1 and drive_dim > self.n_gate_dims:
            ctx_dim = drive_dim - self.n_gate_dims
            self.control = StructuredControlNet(
                self.n_gate_dims, ctx_dim, n_basis, normalize=self.normalize_ctx)
        else:
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
        """x: (B,T,N) or (T,N) -> (B,T,D) or (T,D).

        Output is L2-normalised per timepoint.  This pins the manifold to the
        unit sphere centred at the origin — a mathematical necessity for the
        Lie algebra rotation J·Z: if the manifold were off-centre, any pure
        rotation would produce catastrophic displacement rather than rotation.

        The normalisation is compatible with the dynamics because J is
        skew-symmetric, so exp(J·dt) is orthogonal (norm-preserving).
        The rollout stays on the sphere automatically — no train/val mismatch.
        """
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)   # (1, T, N)
        # Conv1d expects (B, C, T)
        z = self.net(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        # Pin manifold to origin-centred unit sphere
        z = F.normalize(z, p=2, dim=-1)
        return z.squeeze(0) if single else z


class SkieurLieODE(nn.Module):
    """Full end-to-end model: Encoder + LieODECell.

    forward modes:
    - "encode":  neural -> latent z only
    - "rollout": z_0 + drive_seq -> predicted trajectory
    - "full":    neural -> z -> rollout -> (z_true, z_pred)
    """
    def __init__(self, n_neurons, latent_dim, drive_dim,
                 constrained_L=False, use_ode=False, ode_method="rk4",
                 n_gate_dims=2, normalize_ctx=False):
        super().__init__()
        n_basis = latent_dim * (latent_dim - 1) // 2
        self.latent_dim = latent_dim
        self.encoder = Encoder(n_neurons, latent_dim)
        self.lie_cell = LieODECell(latent_dim, n_basis,
                                   constrained_L=constrained_L,
                                   use_ode=use_ode, ode_method=ode_method,
                                   n_gate_dims=n_gate_dims,
                                   normalize_ctx=normalize_ctx)
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
        """Return metrics from the drive-dependent generator J(u) = sum w_i(u) G_i.

        CRITICAL: a successful model learns w_i(u) ≈ -w_i(-u) (rotation direction
        follows velocity sign).  With symmetric drive distribution N(0,1), the
        MATRIX average E_u[J(u)] ≈ 0 cancels out — exactly when rotation is
        strongest.  We therefore compute metrics PER SAMPLE and average:

          SR      = mean_u [ ||J(u)|| / (||J(u)|| + ||L||) ]
          |Real|  = mean_u [ |Real(eig(J(u) + L))| ]
          |Imag|  = mean_u [ |Imag(eig(J(u) + L))| ]

        Returns:
            J_avg:  (D, D) mean matrix (may be near zero — do NOT use for SR/eig)
            L:      (D, D) dissipation matrix
            sr_mean:     float, per-sample-averaged skewness ratio
            eig_real:    float, per-sample-averaged |Real(eigenvalue)|
            eig_imag:    float, per-sample-averaged |Imag(eigenvalue)|
        """
        with torch.no_grad():
            dev = next(self.parameters()).device
            d_drive = self.lie_cell.drive_dim  # stored by set_control_input_dim
            u_samples = torch.randn(n_drive_samples, d_drive, device=dev)
            _, J_samples, L = self.lie_cell.compute_generator(u_samples)
            L_np = L.cpu().numpy()
            L_fro = np.linalg.norm(L_np)

            # Per-sample metrics
            J_np = J_samples.cpu().numpy()  # (n_samples, D, D)
            sr_vals = []
            eig_real_vals, eig_imag_vals = [], []
            for k in range(n_drive_samples):
                J_k = J_np[k]
                j_norm = np.linalg.norm(J_k)
                sr_vals.append(j_norm / (j_norm + L_fro + 1e-9))
                eigvals = np.linalg.eigvals(J_k + L_np)
                eig_real_vals.append(np.mean(np.abs(np.real(eigvals))))
                eig_imag_vals.append(np.mean(np.abs(np.imag(eigvals))))

            sr_mean = float(np.mean(sr_vals))
            eig_real = float(np.mean(eig_real_vals))
            eig_imag = float(np.mean(eig_imag_vals))

            # Average matrix (for reference only — may cancel for odd w(u))
            J_avg = J_samples.mean(dim=0).cpu().numpy()

        return J_avg, L_np, sr_mean, eig_real, eig_imag


# --- Quick sanity test ---
n_test = n_data_all[0].shape[0]
d_drive = len(DRIVE_KEYS)
model = SkieurLieODE(n_test, D_LATENT, d_drive,
                     constrained_L=CONSTRAINED_L,
                     use_ode=USE_ODE, ode_method=ODE_METHOD,
                     n_gate_dims=N_GATE_DIMS,
                     normalize_ctx=NORMALIZE_CTX)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
control_type = type(model.lie_cell.control).__name__
print(f"Model: {n_params:,} parameters")
print(f"  ControlNet: {control_type} (n_gate_dims={N_GATE_DIMS}, normalize_ctx={NORMALIZE_CTX})")
print(f"  Encoder: Conv1d offset-11, {n_test} -> {D_LATENT} (comparable to CEBRA offset10)")
print(f"  Skew basis: {D_LATENT}*(D_LATENT-1)/2 = {D_LATENT*(D_LATENT-1)//2}")
print(f"  Drive dim: {d_drive} = {DRIVE_KEYS}")
print(f"  CONSTRAINED_L: {CONSTRAINED_L}")
print(f"  USE_ODE: {USE_ODE}")

# Quick smoke test
B, T = 8, MINI_TRAJ_LEN
x_test = torch.randn(B, T, n_test, device=DEVICE)
d_test = torch.randn(B, T, d_drive, device=DEVICE)
z_true, z_pred = model(x_test, d_test)
print(f"  Smoke test: z_true {tuple(z_true.shape)}, z_pred {tuple(z_pred.shape)}")
# Check skewness (per-sample average, NOT matrix-average)
J_avg, L_m, sr, eig_r, eig_i = model.get_generator_matrices()
print(f"  Initial SR (per-sample avg): {sr:.4f}, |Real|: {eig_r:.4f}, |Imag|: {eig_i:.4f}")
print(f"  (J_avg norm: {np.linalg.norm(J_avg):.4f} — may be near-zero for odd w(u); SR/eig use per-sample avg)")
print("Model instantiation OK.")
