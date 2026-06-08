# Structured ControlNet: From Naive E2E to Physically-Constrained Dynamics

## Status

Proposed — not yet implemented in `Skieur_EndToEnd_LieODE.ipynb`.
**Revised** following peer review of the initial draft (v1 → v2).

---

## 0. Revision Notes (v2)

This version addresses five substantive concerns identified in review of v1:

| # | Concern | Resolution |
|---|---------|------------|
| 1 | Trap 2 mechanism misdiagnosed: ~−3×10⁻⁴ is *near-zero*, not "exponential divergence" | Reframed: the failure is that J contributes negligible incremental predictive power relative to leak; short-window validation (20–100 bins) was designed to prevent divergence, so near-zero negative R² is the correct observable for "J ≈ useless" |
| 2 | Trap 1 fix (decoupling) and Trap 2 fix (gating) were conflated as one axis | Clarified as two orthogonal design dimensions with a 2×2 table; Structured E2E only fixes Trap 2 — Trap 1 still requires `CEBRA_LABEL ≠` gate variable |
| 3 | SR degenerates to $|v|/(|v|+||L||)$ under unit-norm gating, losing all information about learned rotational structure | Acknowledged; replaced unit-norm with soft-normalization; recommended per-sample SR averaging $\mathbb{E}_u[\mathrm{SR}(J(u))]$ not SR of $\mathbb{E}_u[J(u)]$ |
| 4 | $\mathbb{E}[J]=0$ (zero-mean $v$) → SR of $\mathbb{E}[J]$ = 0, contradicting companion method doc | SR now computed as $\mathbb{E}_u[||J(u)||]/(\mathbb{E}_u[||J(u)||]+||L||)$ — per-sample norm averaging, not norm of average |
| 5 | Implementation prerequisites (multi-drive TAU_SHIFT alignment, NotImplementError guard) glossed over | Explicitly listed as §4.9 prerequisites; noted that notebook already supports multi-drive extraction via `extract_epochs` per DRIVE_KEYS column |

---

## 1. The Double Trap: Two Independent Failure Modes in Neural Dynamics Modeling

The rotational-dynamics literature in systems neuroscience faces two distinct,
non-overlapping failure modes. Our pipeline has encountered both sequentially, and
understanding their **independence** is the key to designing the correct architecture.

### Trap 1: Circular Reasoning (Double Dipping)

| | |
|---|---|
| **Who falls in** | All two-stage pipelines where the same behavioral variable both shapes the embedding (via InfoNCE) and drives the dynamics (via Lie/OLS fit). |
| **Mechanism** | The embedding manifold is pre-aligned with the driver. A high Skewness Ratio may reflect the InfoNCE objective's imprinting of a rotational topology rather than genuine neural rotational dynamics. |
| **How to detect** | **Dummy-CEBRA control**: train CEBRA on `shuffle(labels)`, fit Lie on the dummy embedding. If $R^2_{\text{drive}}$ survives, it's an InfoNCE artifact. **OR: variable decoupling** — use a different variable for embedding (e.g., Position) than for dynamics (e.g., Velocity) — a structurally simpler solution. |
| **How to fix** | Either (a) **variable decoupling** — set `CEBRA_LABEL ≠ DRIVE_KEYS[0]`, so the embedding-shaping and dynamics-driving variables are distinct; or (b) **Dummy-CEBRA** — the gold-standard negative control that destroys label structure before embedding. |

**Why GhostLie OLS partially avoids Trap 1:** Short-window OLS on raw neural data (no CEBRA embedding) has no InfoNCE step, so there is no embedding-shaping variable to create circularity. However, the CEBRA-embedded GhostLie variant *is* vulnerable.

### Trap 2: Negligible J Contribution (The Incremental Prediction Problem)

| | |
|---|---|
| **Who falls in** | Unconstrained E2E models where a free-form ControlNet MLP maps drive → rotation weights. |
| **Mechanism** | The model minimizes $\mathcal{L} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \mathcal{L}_{\text{dynamics}}$. If the learning problem is hard (small dynamic range, noisy gradients, competing objectives), the optimizer may find a solution where the rotation term $J$ contributes almost nothing to the rollout prediction — the dynamics loss is dominated by the leak term $L$, and the learned $J$ is effectively vestigial. The result is $R^2_{\text{rollout}} \approx 0$ (or slightly negative, ~$-10^{-4}$, from finite-sample MSE noise). |
| **How to detect** | $R^2_{\text{rollout}}$ near zero for most sessions despite training convergence (loss decreases). The model *learns something*, but the rotation term contributes negligible incremental predictive power over the leak-only baseline. |
| **How to fix** | Impose a **Structured ControlNet** that forces the motor command to multiplicatively gate the rotation: $w_i(t) = v(t) \cdot \mathrm{MLP}_i(f(t))$. This structural constraint prevents the optimizer from "ignoring J" — the motor gate $v$ *must* modulate the rotation amplitude, making the J term structurally non-negotiable in the dynamics. |

**Important distinction from v1:** The quantitative signature of Trap 2 is *near-zero* $R^2_{\text{rollout}}$ (~−3×10⁻⁴), **not** large-magnitude negative values. The short-window validation (20–100 bins) in `compute_r2_drive_rollout` was specifically designed to prevent exponential divergence. A truly diverging rollout would produce $R^2 \ll -1$ (e.g., −10³). The near-zero values indicate that $J$ simply contributes almost nothing — the leak term $L$ already captures nearly all the predictable dynamics at these timescales.

### Why They're Independent — and Why That Matters

Trap 1 (circular reasoning) is a **static identifiability** problem: it corrupts whether
the rotational signal *exists* in the first place. Trap 2 (negligible J contribution)
is a **learning dynamics** problem: it corrupts whether the model *discovers* the
rotational signal during training.

These are **orthogonal design dimensions**. A model can fall into zero, one, or both:

| | Trap 2: J ≈ 0 | Trap 2: J significant |
|---|---|---|
| **Trap 1: Circular** | CEBRA-embedded GhostLie (worst: both) | GhostLie OLS raw-space (J real but same-variable) |
| **Trap 1: Decoupled** | Naive E2E with `CEBRA_LABEL≠gate` (this notebook) | **Target: Structured E2E** (decoupled + gated) |

**Key insight:** "Structured E2E" fixes **only Trap 2**. Whether it also fixes Trap 1
depends on an **independent** design choice — setting `CEBRA_LABEL ≠ DRIVE_KEYS[0]`.
The v1 of this plan incorrectly conflated the two axes. In the notebook's default
configuration, `CEBRA_LABEL = "Velocity_x"` and `DRIVE_KEYS[0] = "Velocity_x"` —
the structured gate alone does not break the circularity. Decoupling is a separate,
equally important configuration decision.

---

## 2. Neuroscientific Motivation for the Structured Gate

The constraint $w_i(t) = v(t) \cdot \mathrm{MLP}_i(f(t))$ implements the canonical
computational architecture of a **forward model** as formalized by Wolpert,
Ghahramani & Jordan (1995, *Science*):

> *The motor command gates the predictive update; the sensory context determines
> which internal model is engaged.*

| Component | Mathematical role | Neuroscientific interpretation |
|-----------|------------------|-------------------------------|
| $v(t)$ | Multiplicative gate | **Efference copy** of the motor command (head velocity). At $v=0$, no prediction update is needed — the system should not rotate. |
| $\mathrm{MLP}(f(t))$ | Context-dependent coefficients | **Sensory context** (played frequency). Determines *which* rotational planes are engaged — the auditory state selects the relevant internal model. |
| $w_i(t) = v(t) \cdot \mathrm{MLP}_i(f(t))$ | Structured basis coefficients | The motor command *gates* the rotation; the sensory context *steers* it. |
| $J(t) = \sum_i w_i(t) \cdot G_i$ | Drive-dependent generator | Pure rotation at angular velocity $\propto |v|$, on planes selected by $f$. |

**Key physical constraint:** $v(t) = 0 \Rightarrow J(t) = 0$. The neural state does
not rotate when the animal is not moving, regardless of what sound is playing. This
is the boundary condition that distinguishes a *motor-driven forward model* from a
*stimulus-driven dynamical system*.

**Anatomical plausibility:** Nelson, Schneider & Mooney (2013, *J. Neurosci.*)
demonstrated monosynaptic M2→A1 projections carrying motor-related signals. In the
structured architecture, $v(t)$ models the *gain* of this projection — motor signals
modulate the amplitude of auditory-cortical state rotations, but do not determine
their direction. The direction is set by the auditory context $f(t)$, consistent
with the known columnar and tonotopic organization of A1.

---

## 3. Mathematical Formulation

### 3.1 Current (Naive) ControlNet

```python
# Unconstrained: MLP learns arbitrary mapping [v, f, ...] -> w
w_i = tanh(MLP([v(t), f(t), ...]))    # (*, n_basis)
J(t) = sum_i w_i(t) * G_i
A(t) = J(t) + L
z_{t+1} = matrix_exp(A(t) * dt) @ z_t
```

**Observed behavior:** $R^2_{\text{rollout}} \approx -3 \times 10^{-4}$ (near zero)
for most sessions, despite training loss decreasing. The rotation term $J$
contributes negligible incremental predictive power relative to the leak-only
baseline. The optimizer may be finding solutions where $||J|| \ll ||L||$, making
the dynamics essentially leak-dominated.

### 3.2 Proposed (Structured) ControlNet

The ControlNet is split into two sub-modules with a multiplicative gate:

**Sensory encoder (for steering direction):**
$$h(t) = \mathrm{MLP}_{\text{ctx}}(f(t)) \in \mathbb{R}^{n_{\text{basis}}}$$

A small MLP that maps the sensory context (frequency, or any set of non-motor
variables) to a vector of basis coefficients. This determines *which rotational
planes* are engaged and in what *relative proportion*.

**Motor gate (for rotation amplitude):**
$$w_i(t) = v(t) \cdot h_i(t)$$

The motor command $v(t)$ multiplicatively gates the sensory-derived coefficients.
This enforces:
- $v(t) = 0 \Rightarrow w_i(t) = 0 \Rightarrow J(t) = 0$ (no rotation at rest)
- $\mathrm{sign}(v(t))$ determines rotation direction (left vs right head movement)
- $|v(t)|$ determines angular velocity (faster movement → faster rotation)

**Full generator:**
$$J(t) = \sum_{i=1}^{n_{\text{basis}}} w_i(t) \cdot G_i = v(t) \cdot \sum_{i=1}^{n_{\text{basis}}} h_i(t) \cdot G_i$$

The generator factorizes into a scalar motor amplitude $v(t)$ and a context-dependent
matrix $\tilde{J}(f) = \sum_i h_i(f) \cdot G_i$:

$$J(v, f) = v \cdot \tilde{J}(f)$$

This factorization uses the linearity of the Lie algebra: each $G_i$ is
skew-symmetric, so any linear combination is skew-symmetric. $\tilde{J}(f)$ is
skew-symmetric for all $f$ — the Lie algebra structure is preserved *by construction*
at every timestep, without post-hoc projection.

### 3.3 Normalization Strategy (Revised from v1)

**v1 proposed unit-norm:** $h_{\text{norm}} = h / (||h|| + \varepsilon)$. This has a
critical degeneracy: with orthonormal basis $\{G_i\}$,
$||J(v,f)||_F = |v| \cdot ||\sum_i h_i^{\text{norm}} G_i||_F = |v|$. The rotation
magnitude becomes **purely a function of velocity**, making $\mathrm{SR} = |v|/(|v|+||L||)$
— independent of what the context MLP learned. SR loses all discriminative power,
and $\mathrm{SR}_{\text{TR}}$ vs $\mathrm{SR}_{\text{PB}}$ differences reduce to
velocity-amplitude confounds.

**v2 proposal — soft normalization with learnable scale:**

```python
h = self.ctx_net(f)                          # (*, n_basis)
h_norm = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)  # unit direction
scale = softplus(self.log_scale)             # learnable scalar > 0
w = v.unsqueeze(-1) * h_norm * scale         # (*, n_basis)
```

This preserves the directional normalization (so $v$ controls amplitude)
while allowing the context MLP to modulate the **effective rotation magnitude**
via a learnable `log_scale` parameter. The scale can differ across conditions
(Tracking vs Playback), giving SR a non-trivial signal.

**Alternative (preferred for simplicity): drop the normalization entirely**
and rely on the multiplicative gate alone:

```python
h = self.ctx_net(f)             # (*, n_basis) — un-normalized
w = v.unsqueeze(-1) * h         # (*, n_basis) — v gates amplitude
```

The gate $v$ still enforces $v=0 \Rightarrow J=0$. Without normalization, $||h||$
can vary with $f$, allowing the context to modulate both direction *and* magnitude.
The risk — that $\mathrm{MLP}_{\text{ctx}}$ learns to output large $h$ to compensate
for small $v$ — is mitigated by the fact that $v$ passes through zero frequently in
natural behavior (velocity oscillates around zero). The network cannot rely on a
constant large $h$ to "fake" rotation because at $v=0$, $J=0$ regardless of $h$.

**Recommendation:** Start with the un-normalized variant (simpler, fewer hyperparams).
If $||h||$ drifts to extreme values during training, add the soft-normalization.

### 3.4 Leak Term (Unchanged)

The dissipation $L$ remains as-is: either $L = -C C^T$ (stable, $\text{Re}(\lambda) \leq 0$)
or unconstrained. $L$ captures autonomous dynamics not gated by the motor command.

### 3.5 State Transition (Unchanged)

$$z_{t+1} = \exp\left( \left[ v(t) \cdot \tilde{J}(f(t)) + L \right] \cdot \Delta t \right) \cdot z_t$$

where $\Delta t = 0.005$ s (the bin size).

---

## 4. Implementation Plan

### 4.1 New Module: `StructuredControlNet`

```python
class StructuredControlNet(nn.Module):
    """Structured control: w_i(t) = v(t) * MLP_i(f(t)).

    v(t): motor gate (scalar, zero-centered) — multiplicative amplitude
    f(t): sensory context (one or more variables) — determines direction

    At v=0, J=0 for all f — the physical boundary condition.
    """
    def __init__(self, ctx_dim, n_basis, hidden=None, normalize=False):
        super().__init__()
        if hidden is None:
            hidden = CONTROL_HIDDEN  # e.g., [32]
        self.normalize = normalize
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
        """u_t: (*, drive_dim) where u_t[..., 0] = v (motor gate).

        Returns w_i: (*, n_basis).
        """
        v = u_t[..., 0:1]            # (*, 1) — motor gate
        f = u_t[..., 1:]             # (*, ctx_dim) — sensory context
        h = self.ctx_net(f)          # (*, n_basis) — direction + magnitude
        if self.normalize:
            h_dir = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)
            s = F.softplus(self.log_scale)
            return v * h_dir * s
        return v * h                 # (*, n_basis) — multiplicative gate
```

### 4.2 Design Decision: `normalize=False` as Default

| | `normalize=False` | `normalize=True` |
|---|---|---|
| SR | $||J||$ reflects both $v$ and $f$ context | $||J|| \propto |v|$ only (degenerate) |
| Risk | $h$ may drift to large values | SR loses discriminative power |
| Mitigation | $v=0$ frequent in natural behavior → gate enforces zero rotation | Per-sample SR averaging still needed |
| Recommendation | **Default** | Ablation only |

### 4.3 Config Changes (Cell 0)

Add / modify:

```python
# Drive configuration for Structured ControlNet:
DRIVE_KEYS = ["Velocity_x", "Played_frequency"]
#   DRIVE_KEYS[0] = motor gate (multiplicative; signed, zero-centered)
#   DRIVE_KEYS[1:] = sensory context (determines rotation direction/magnitude)

# Decoupling flag (independent axis — fixes Trap 1):
CEBRA_LABEL = "Velocity_x"          # InfoNCE contrastive label
# When CEBRA_LABEL != DRIVE_KEYS[0]: *** Decoupled mode ***
#   Embedding shaped by CEBRA_LABEL, dynamics driven by DRIVE_KEYS[0]

MOTOR_GATE_IDX = 0                  # index of the multiplicative gate in DRIVE_KEYS
# Set MOTOR_GATE_IDX = -1 to fall back to naive (unstructured) ControlNet
```

### 4.4 Changes to `LieODECell` (Cell 8)

```python
class LieODECell(nn.Module):
    def __init__(self, dim, n_basis, constrained_L=False, use_ode=False,
                 ode_method="rk4", motor_gate_idx=0, normalize_ctx=False):
        ...
        self.motor_gate_idx = motor_gate_idx
        drive_dim = 1  # will be set by set_control_input_dim
        ...

    def set_control_input_dim(self, drive_dim):
        if self.motor_gate_idx >= 0:
            ctx_dim = drive_dim - 1  # the rest after motor gate
            self.control = StructuredControlNet(ctx_dim, self.dim * (self.dim - 1) // 2)
        else:
            self.control = ControlNet(drive_dim, self.dim * (self.dim - 1) // 2)
```

`compute_generator` needs **no change** — it already calls `self.control(u_t)`
and uses the returned weights. The structured constraint is encapsulated in
`StructuredControlNet.forward()`.

### 4.5 Changes to `SkieurLieODE` (Cell 8)

```python
class SkieurLieODE(nn.Module):
    def __init__(self, n_neurons, latent_dim, drive_dim,
                 constrained_L=False, use_ode=False, ode_method="rk4",
                 motor_gate_idx=0):
```

Passed through to `LieODECell`.

### 4.6 SR Computation (Revised from v1)

**v1 proposed (incorrect):** $\mathrm{SR} = ||\mathbb{E}[J]|| / (||\mathbb{E}[J]|| + ||L||)$.  
**Problem:** $\mathbb{E}[v] = 0$ (zero-mean standardization) ⇒ $\mathbb{E}[J] = 0$ ⇒ SR = 0 always.

**v2 (correct):** Compute SR per-sample, then average:

$$\mathrm{SR} = \mathbb{E}_{u \sim p(u)}\left[ \frac{||J(u)||}{||J(u)|| + ||L||} \right]$$

This is implemented in `get_generator_matrices()`:

```python
def get_generator_matrices(self, n_drive_samples=100):
    with torch.no_grad():
        u_samples = torch.randn(n_drive_samples, d_drive, device=dev)
        _, J_samples, L = self.lie_cell.compute_generator(u_samples)
        # Per-sample SR: ||J(u)|| / (||J(u)|| + ||L||)
        J_norms = torch.norm(J_samples.reshape(n_drive_samples, -1), dim=1)
        L_norm = torch.norm(L)
        sr_per_sample = J_norms / (J_norms + L_norm + 1e-9)  # (n_samples,)
        sr = sr_per_sample.mean().item()
        # Representative J (used for eigenvalues only):
        J_avg = J_samples.mean(dim=0)
        return J_avg.cpu().numpy(), L.cpu().numpy(), sr
```

**Caveat — SR:** With unit-norm gating, $\mathrm{SR}_{\text{per-sample}} = |v|/(|v|+\|L\|)$, making SR purely a function of $|v|$ — see §3.3. With `normalize=False`, $\|J\|$ depends on both $v$ and $f$, making SR informative about learned rotational structure.

**Caveat — eigenvalue degeneracy under zero-mean drive:** `J_avg = J_samples.mean(dim=0)` is approximately the zero matrix when $v$ is zero-mean (standardized), because $J(v,f) = v \cdot \tilde{J}(f)$ is antisymmetric in $v$. The eigenvalues of $J_{\text{avg}} + L$ therefore reflect only the leak term $L$, not the full generator $J(u)+L$. For meaningful eigenvalue reporting, either (a) restrict sampling to $|v| > 0$ (e.g., $|v| > 1$ std), or (b) compute eigenvalues per-sample and report the distribution: $\mathbb{E}_u[|\mathrm{Re}(\lambda(J(u)+L))|]$ and $\mathbb{E}_u[|\mathrm{Im}(\lambda(J(u)+L))|]$. The current implementation reports eigenvalues of $\mathbb{E}[J] + L \approx L$ and should be interpreted with this caveat.

### 4.7 Verification Criteria (Revised)

1. **$R^2_{\text{rollout}}$ moves from near-zero negative to small positive**
   for a majority of sessions. The structured gate forces $J$ to contribute to the
   rollout, so the incremental prediction over leak-only should become detectable.
   Expected magnitude: small positive (~10⁻⁴ to 10⁻³), not dramatic — the dynamic
   range of rotation over 20–100 bins is inherently limited.

2. **Unit test: $J(v=0, f) = 0$ for all $f$.**
   Pass `u = [0.0, f_random]` through `compute_generator` and verify
   $||J|| < 10^{-6}$.

3. **$\lambda_{\text{dyn}} = 0$ ablation produces $R^2_{\text{rollout}} \approx 0$**
   (the correct null — without dynamics training, J is random and contributes
   nothing).

4. **SR is not trivially $|v|/(|v|+||L||)$.**
   Verify that $\mathrm{corr}(|v|, \text{SR})$ across drive samples is not
   perfect ($r < 0.99$). With `normalize=False`, the context MLP should contribute
   non-trivial variance to $||J||$.

5. **Kinematic confound check (§7.6 of companion doc).**
   Verify that $\mathrm{SR}_{\text{TR}} > \mathrm{SR}_{\text{PB}}$ (if observed)
   is not explained by $|v|_{\text{TR}} > |v|_{\text{PB}}$. Compute the KS
   statistic for the velocity distributions and, if significantly different,
   report SR conditioned on velocity bins.

6. **Dummy-CEBRA gate still holds** (Cell 4 — unaffected by this change).

7. **Single-drive fallback produces results comparable to Naive E2E.**
   Regression test: with `DRIVE_KEYS = ["Velocity_x"]` and `MOTOR_GATE_IDX = -1`,
   the model should reproduce the original Naive E2E behavior.

### 4.8 Diagnostic Experiments

**Before implementing the structured gate**, run these diagnostics on the Naive E2E
to quantify the current failure mode:

1. **Velocity-ablation test:**
   Set $v=0$ in the drive during validation (replace `Velocity_x` with zeros).
   If $R^2_{\text{rollout}}$ does **not** drop significantly, the model has
   indeed learned to ignore $v$ — the structured gate is well-motivated.
   If it **does** drop, $v$ is being used and the near-zero $R^2$ has a
   different cause (e.g., dynamic range too small).

2. **Single-step rotation magnitude analysis:**
   Compute $||J(u_t) \cdot dt||$ across the test drive distribution.
   If typical single-step rotations are ~0.005 rad and 20-step cumulative
   rotation is ~0.1 rad, the signal is intrinsically small — even a perfect
   model would produce near-zero $R^2_{\text{rollout}}$ because the leak term
   already captures most of the variance at these timescales.

### 4.9 Implementation Prerequisites

These must be in place before the Structured ControlNet can be tested:

1. **Multi-drive per-epoch extraction with TAU_SHIFT alignment.**
   Implemented in commit `97de7f0`: `train_one_session` calls `extract_epochs`
   separately for each `DRIVE_KEYS` column (since `extract_macro_epochs` uses
   Condition boundaries, all extractions produce identical epoch boundaries),
   standardizes each dimension independently (pooled TR+PB via per-dimension
   `drive_mu[dk]`/`drive_std[dk]` dicts), and stacks them via `np.column_stack`.
   Sanity checks assert identical epoch counts and lengths across dimensions.
   **The training path does NOT call `build_drive_vector`** — that function is
   used only in the OLS baseline path (Cell 4).
   **Status: implemented** (commit `97de7f0`; verified — no `NotImplementedError`
   guard exists for `len(DRIVE_KEYS) > 1`).

2. **Decoupled configuration.** Set `CEBRA_LABEL ≠ DRIVE_KEYS[0]` explicitly
   when testing the structured gate, to ensure Trap 1 and Trap 2 fixes are
   evaluated independently. **Status: supported via config in Cell 0; default
   is coupled (both `= "Velocity_x"`).**

3. **Regression test baseline.** Record Naive E2E metrics ($R^2_{\text{rollout}}$,
   SR, eigenvalues) on ≥3 sessions with `MOTOR_GATE_IDX = -1` before switching,
   to quantify the structured gate's effect.

---

## 5. Relationship to the Companion Documents

| Document | Role |
|----------|------|
| `lie_algebra_method_description.md` | Foundation: Lie groups, OLS fitting, SR/R² metrics, eigenvalue interpretation, limitations of the two-stage pipeline |
| `lie_e2e_method_description.md` | Full description of the E2E notebook: architecture, training, batching, metrics, controls, limitations |
| `label_drive_decoupling_plan.md` | The variable-decoupling strategy (Trap 1 fix) — independent axis from the structured gate |
| **This document** | The structured-gate strategy (Trap 2 fix) — rationale, math, implementation |

---

## 6. References

1. **Wolpert, D.M., Ghahramani, Z. and Jordan, M.I.** (1995) 'An internal model for sensorimotor integration', *Science*, 269(5232), pp. 1880–1882.
2. **Nelson, A., Schneider, D.M. and Mooney, R.** (2013) 'A circuit for motor cortical modulation of auditory cortical activity', *Journal of Neuroscience*, 33(36), pp. 14342–14353.
3. **Schneider, D.M. and Mooney, R.** (2018) 'How movement modulates hearing', *Annual Review of Neuroscience*, 41, pp. 553–572.
4. **Schneider, S., Lee, J.H. and Mathis, M.W.** (2023) 'Learnable latent embeddings for joint behavioural and neural analysis', *Nature*, 617(7960), pp. 360–368.
5. **Keller, G.B. and Mrsic-Flogel, T.D.** (2018) 'Predictive processing: a canonical cortical computation', *Neuron*, 100(2), pp. 424–435.
