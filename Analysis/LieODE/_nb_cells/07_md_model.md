## 2. End-to-End Lie-ODE Model

PyTorch implementation of the joint encoder + Lie generator pipeline.

**Architecture:**
1. **Encoder** (Conv1d, first-layer kernel=11 ≈55ms RF; subsequent layers kernel=1):
   neural window -> latent z (dim D_LATENT) — temporal RF matches CEBRA offset10 (~50ms)
2. **SkewBasis**: D(D-1)/2 fixed orthonormal skew-symmetric basis matrices G_i
3. **ControlNet** (small MLP): multi-dim drive u_t -> weights w_i
   -> J(u_t) = sum_i w_i(u_t) * G_i (skew-symmetric by construction)
4. **Dissipation**: L = -C @ C.T (stable) OR unconstrained (ablation flag)
5. **Transition**: dz/dt = (J(u_t) + L) @ z -> discrete matrix_exp (default) or ODE (**experimental**)

**Key design decisions:**
- **Discrete matrix_exp default** (fast, robust, no torchdiffeq dependency)
- **Dynamics loss variance-normalized**: `mse / var(z_true)` — prevents trivial `||z|| -> 0` solution that InfoNCE's scale-invariance would otherwise allow
- **ODE mode is experimental** (requires torchdiffeq): drive interpolation is piecewise-constant (nearest-neighbor); continuous-time claims not yet realized. Prefer `rk4` over `dopri5` if enabled — adaptive stepper may be unstable at bin boundaries
- **Short-window validation**: held-out R2_drive uses random windows of `VAL_ROLLOUT_LEN` bins (not full-epoch rollout), preventing long-range divergence from contaminating the headline metric
- **CONSTRAINED_L flag**: allows ablation between stable (`-CC^T`) and unconstrained dissipation
- **lambda_dyn warmup**: 0 for first N steps, then linear ramp (stabilizes training)
