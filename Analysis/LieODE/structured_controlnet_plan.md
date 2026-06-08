# Structured ControlNet: From Naive E2E to Physically-Constrained Dynamics

## Status

Proposed — not yet implemented in `Skieur_EndToEnd_LieODE.ipynb`.

## 1. The Double Trap: Two Independent Failure Modes in Neural Dynamics Modeling

The rotational-dynamics literature in systems neuroscience faces two distinct,
non-overlapping failure modes. Our pipeline has encountered both sequentially, and
understanding their independence is the key to designing the correct architecture.

### Trap 1: Circular Reasoning (Double Dipping)

| | |
|---|---|
| **Who falls in** | All two-stage OLS pipelines, including GhostLie |
| **Mechanism** | The same behavioral variable (e.g., Position) is used to (a) shape the CEBRA embedding via InfoNCE, then (b) drive the Lie algebra fit. The embedding manifold is *pre-aligned* with the driver — high SR and significant TR > PB differences may be algorithmic artifacts, not neural facts. |
| **How to detect** | Dummy-CEBRA control: train CEBRA on `shuffle(labels)`, fit Lie on the dummy embedding. If $R^2_{\text{drive}}$ survives, it's an artifact. |
| **How to fix** | **Variable decoupling**: separate embedding-shaping variables from dynamics-driving variables. |

GhostLie OLS passes the Dummy-CEBRA gate in the local window regime (short-window OLS
on raw neural data avoids the embedding circularity), but its CEBRA-embedded variant
is structurally vulnerable to Trap 1.

### Trap 2: Integration Collapse (The Rollout Dilemma)

| | |
|---|---|
| **Who falls in** | Naive E2E models with unconstrained ControlNet MLPs |
| **Mechanism** | A free-form MLP `ControlNet([v, f, ...]) → w_i` optimizes only the rollout MSE. If one drive component (e.g., auditory frequency $f$) has larger variance and longer autocorrelation than another (e.g., velocity $v$), the MLP learns $w \approx \text{MLP}(f)$, effectively discarding $v$. The model then predicts **non-zero rotation even at zero velocity** ($v=0$). Over 20–100 steps of `matrix_exp`, persistent spurious rotation causes the rollout to diverge exponentially — the headline $R^2_{\text{rollout}}$ metric becomes dominated by integration error, not dynamics quality. |
| **How to detect** | $R^2_{\text{rollout}} < 0$ for most sessions (observed: `-0.000320`, `-0.000652` across Tracking and Playback). The model *converges* (loss decreases) but the held-out metric is noise. |
| **How to fix** | **Structured ControlNet**: impose the physical constraint that motor commands *gate* rotation — $w_i(t) = v(t) \cdot \text{MLP}_i(f(t))$. At $v=0$, rotation must vanish, regardless of sensory context. |

### Why They're Independent

Trap 1 (circular reasoning) is a **static identifiability** problem: it corrupts the
*existence* of the rotational signal. Trap 2 (integration collapse) is a **dynamic
stability** problem: it corrupts the *predictability* of the rotational signal.

- GhostLie OLS avoids Trap 2 (short-window OLS doesn't integrate) but is vulnerable to Trap 1 (same variable does both jobs).
- Naive E2E avoids Trap 1 (decoupled variables) but falls into Trap 2 (unconstrained MLP learns to ignore the motor gate).
- **Structured E2E avoids both**: decoupled variables break the circularity; the multiplicative gate $w = v \cdot \text{MLP}(f)$ enforces the physical boundary condition $v=0 \Rightarrow J=0$, preventing integration collapse.

```
                       Avoids Trap 1?     Avoids Trap 2?
                       (no circularity)    (no integration collapse)
GhostLie OLS           ✗ (same variable)   ✓ (short-window OLS)
Naive E2E (this nb)    ✓ (decoupled vars)  ✗ (MLP discards v)
Structured E2E         ✓ (decoupled vars)  ✓ (v · MLP(f) gate)
```

---

## 2. Neuroscientific Motivation for the Structured Gate

The constraint $w_i(t) = v(t) \cdot \text{MLP}_i(f(t))$ is not an arbitrary
regularization — it implements the canonical computational architecture of a
**forward model** as formalized by Wolpert, Ghahramani & Jordan (1995, *Science*):

> *The motor command gates the predictive update; the sensory context determines
> which internal model is engaged.*

| Component | Mathematical role | Neuroscientific interpretation |
|-----------|------------------|-------------------------------|
| $v(t)$ | Multiplicative gate | **Efference copy** of the motor command (head velocity). At $v=0$, no prediction update is needed — the system should not rotate. |
| $\text{MLP}(f(t))$ | Context-dependent weights | **Sensory context** (played frequency). Determines *which* rotational plane is engaged — the auditory state selects the relevant internal model. |
| $w_i(t) = v(t) \cdot \text{MLP}_i(f(t))$ | Structured basis coefficients | The motor command *gates* the rotation; the sensory context *steers* it. |
| $J(t) = \sum_i w_i(t) \cdot G_i$ | Drive-dependent generator | Pure rotation at angular velocity $\propto |v|$, on planes selected by $f$. |

**Key physical constraint:** $v(t) = 0 \Rightarrow J(t) = 0$. The neural state
does not rotate when the animal is not moving, regardless of what sound is playing.
This is the boundary condition that distinguishes a *motor-driven forward model*
from a *stimulus-driven dynamical system*.

**Anatomical plausibility:** Nelson, Schneider & Mooney (2013, *J. Neurosci.*)
demonstrated monosynaptic M2→A1 projections carrying motor-related signals. In the
structured architecture, $v(t)$ models the *gain* of this projection path — motor
signals modulate the amplitude of auditory-cortical state rotations, but do not
determine their direction. The direction is set by the auditory context $f(t)$,
consistent with the known columnar and tonotopic organization of A1.

---

## 3. Mathematical Formulation

### 3.1 Current (Naive) ControlNet

```python
# Unconstrained: MLP learns arbitrary mapping [v, f] -> w
w_i = tanh(MLP([v(t), f(t)]))     # (*, n_basis)
J(t) = sum_i w_i(t) * G_i
A(t) = J(t) + L
z_{t+1} = matrix_exp(A(t) * dt) @ z_t
```

Failure mode: the MLP can learn $w \approx \text{MLP}(f)$, making $J(t)$
approximately independent of $v(t)$. At $v=0$, rotation persists → rollout diverges.

### 3.2 Proposed (Structured) ControlNet

The ControlNet is split into two sub-modules:

**Sensory encoder (for steering direction):**
$$h(t) = \text{MLP}_{\text{ctx}}(f(t)) \in \mathbb{R}^{n_{\text{basis}}}$$
A small MLP that maps the sensory context (frequency) to a vector of basis
coefficients. This determines *which rotational planes* are engaged and in what
*relative proportion* — the "shape" of the rotation.

**Motor gate (for rotation amplitude):**
$$w_i(t) \;=\; v(t) \;\cdot\; h_i(t)$$
The motor command $v(t)$ multiplicatively gates the sensory-derived coefficients.
This enforces:
- $v(t) = 0 \Rightarrow w_i(t) = 0 \Rightarrow J(t) = 0$ (no rotation at rest)
- $\text{sign}(v(t))$ determines rotation direction (left vs right head movement)
- $|v(t)|$ determines angular velocity (faster movement → faster rotation)

**Full generator:**
$$J(t) = \sum_{i=1}^{n_{\text{basis}}} w_i(t) \cdot G_i = v(t) \cdot \sum_{i=1}^{n_{\text{basis}}} h_i(t) \cdot G_i$$

The generator *factorizes* into a scalar motor amplitude $v(t)$ and a
context-dependent matrix $\tilde{J}(f) = \sum_i h_i(f) \cdot G_i$:

$$J(v, f) = v \cdot \tilde{J}(f)$$

This factorization has an important property: $\tilde{J}(f)$ is skew-symmetric
(because each $G_i$ is skew-symmetric), so $J(v, f)$ is skew-symmetric for any $v$.
The Lie algebra structure is preserved *by construction* at every timestep, not
enforced post-hoc.

### 3.3 Leak Term (Unchanged)

The dissipation $L$ remains as-is: either $L = -C C^T$ (stable, $\text{Re}(\lambda) \leq 0$)
or unconstrained. $L$ captures autonomous dynamics that are *not* gated by the
motor command — baseline decay, drift, or restoring forces.

### 3.4 State Transition

The state evolution uses the discrete matrix exponential (unchanged from the
current implementation):

$$z_{t+1} = \exp\left( \left[ v(t) \cdot \tilde{J}(f(t)) + L \right] \cdot \Delta t \right) \cdot z_t$$

where $\Delta t = 0.005$ s (the bin size).

---

## 4. Implementation Plan

### 4.1 New Module: `StructuredControlNet`

Replace the current `ControlNet` class with a structured variant:

```python
class StructuredControlNet(nn.Module):
    """Structured control: w_i(t) = v(t) * MLP_i(f(t)).

    v(t): motor command (scalar velocity) — multiplicative gate
    f(t): sensory context (frequency / position / ...) — determines direction

    The motor gate enforces J(v=0, f) = 0 regardless of f.
    """
    def __init__(self, ctx_dim, n_basis, hidden=None):
        # ctx_dim: number of sensory context variables
        # n_basis: D*(D-1)/2 skew basis matrices
        super().__init__()
        if hidden is None:
            hidden = CONTROL_HIDDEN  # e.g., [32]
        layers = []
        in_dim = ctx_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_basis))
        self.ctx_net = nn.Sequential(*layers)  # MLP_ctx(f) -> h

    def forward(self, u_t):
        """u_t: (*, drive_dim) where drive_dim = 1 (v) + ctx_dim (f, ...).

        Returns w_i: (*, n_basis).
        """
        v = u_t[..., 0:1]            # (*, 1) — motor gate (velocity)
        f = u_t[..., 1:]             # (*, ctx_dim) — sensory context
        h = self.ctx_net(f)          # (*, n_basis) — direction from context
        h_norm = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)  # unit direction
        return v * h_norm            # (*, n_basis) — gated by velocity
```

**Key design decisions:**
1. **`DRIVE_KEYS[0]` = motor gate** (required; must be a signed, zero-centered variable like `Velocity_x`). The first element of the drive vector is always the multiplicative gate.
2. **`DRIVE_KEYS[1:]` = sensory context** (optional; can be `["Played_frequency"]`, `["Position"]`, or empty for velocity-only). These determine the rotation direction via `ctx_net`.
3. **`h_norm` normalization**: constrains the direction vector to unit length, so $v$ alone controls amplitude. Without this, `ctx_net` could learn to output large $h$ to compensate for small $v$, undermining the gate.
4. **Sign preservation**: `v` retains its sign, so leftward vs rightward movement produce opposite rotations.

### 4.2 Config Changes (Cell 0)

Add / modify these parameters:

```python
DRIVE_KEYS = ["Velocity_x", "Played_frequency"]  
# [0] = motor gate (multiplicative), [1:] = sensory context

MOTOR_GATE_IDX = 0    # index of the multiplicative gate in DRIVE_KEYS
CTX_INDICES = [1]     # indices of sensory context variables in DRIVE_KEYS
```

And update the description comment for `DRIVE_KEYS`:

```python
# DRIVE_KEYS format for Structured ControlNet:
#   [0] = motor gate (signed, zero-centered, e.g., Velocity_x)
#   [1:] = sensory context (determines rotation direction, e.g., Played_frequency)
# For naive (unstructured) ControlNet, set MOTOR_GATE_IDX = -1.
```

### 4.3 Changes to `LieODECell` (Cell 8)

The `LieODECell.__init__` and `set_control_input_dim` need to distinguish between
naive and structured modes:

```python
class LieODECell(nn.Module):
    def __init__(self, dim, n_basis, constrained_L=False, use_ode=False,
                 ode_method="rk4", motor_gate_idx=0):
        ...
        self.motor_gate_idx = motor_gate_idx
        if motor_gate_idx >= 0:
            # Structured: motor gate + sensory context
            ctx_dim = drive_dim - 1  # rest after motor gate
            self.control = StructuredControlNet(ctx_dim, n_basis)
        else:
            # Naive fallback: unconstrained MLP
            self.control = ControlNet(drive_dim, n_basis)
```

`compute_generator` needs no change — it already calls `self.control(u_t)` and
uses the returned weights. The structured-vs-naive behavior is encapsulated in
the `ControlNet`/`StructuredControlNet` forward pass.

### 4.4 Changes to `SkieurLieODE` (Cell 8)

The `__init__` signature gains `motor_gate_idx`:

```python
class SkieurLieODE(nn.Module):
    def __init__(self, n_neurons, latent_dim, drive_dim,
                 constrained_L=False, use_ode=False, ode_method="rk4",
                 motor_gate_idx=0):
```

Passed through to `LieODECell`.

### 4.5 Drive Vector Construction (Cell 2)

`build_drive_vector` already handles multi-dimensional drives. The only change:
standardize each component independently (already done), with the motor gate
(`Velocity_x`) standardized to zero mean (critical for the multiplicative gate).

### 4.6 Training Loop (Cell 9) — No Changes

The training loop is unchanged. The structured constraint is in the architecture,
not in the loss function. The optimizer still minimizes `InfoNCE + λ * dynamics_MSE`;
it just cannot "cheat" by learning $w \approx \text{MLP}(f)$ because the
multiplicative gate makes that impossible.

### 4.7 Validation (Cell 9) — No Changes

`compute_r2_drive_rollout` uses `model.rollout()` which calls
`model.lie_cell.compute_generator()` — the structured gate propagates
automatically.

### 4.8 Metrics (Cell 10) — Minor Change

`get_generator_matrices()` already samples from the drive distribution (N3 fix).
With the structured gate, $J(0, f) = 0$ regardless of $f$, so sampling from
$\mathcal{N}(0,1)$ for $v$ and the empirical distribution for $f$ gives the
correct representative $J$:

```python
# Sampling should respect the structured semantics:
# u_samples[:, 0] ~ N(0, 1)       motor gate (velocity-like)
# u_samples[:, 1:] ~ N(0, 1)      sensory context
```

Since both are standardized to $\mathcal{N}(0,1)$ by `build_drive_vector`, this
already works correctly with the existing sampling code.

---

## 5. Verification Criteria

After implementing the Structured ControlNet, the following should be true:

1. **$R^2_{\text{rollout}}$ becomes positive** for a majority of sessions.
   The structured gate prevents integration collapse, so the headline metric
   should shift from noise (negative) to a small positive signal.

2. **$R^2_{\text{rollout}}$ at $\lambda_{\text{dyn}} = 0$ is near zero.**
   With the InfoNCE-only ablation ($\lambda_{\text{dyn}} = 0$), the encoder still
   learns an embedding, but without dynamics training the rollout should not
   predict the trajectory → $R^2_{\text{rollout}} \approx 0$. This is the correct
   null, in contrast to the Naive E2E where $R^2_{\text{rollout}} < 0$ even after
   dynamics training.

3. **$J(v=0, f) = 0$ for all $f$.** A unit test: pass
   `u = [0.0, f_random]` through `compute_generator` and verify $||J|| < 10^{-6}$.

4. **SR at the average drive is interpretable.** Since $J(v, f) = v \cdot \tilde{J}(f)$,
   SR should be computed from $\mathbb{E}_{v,f}[J(v, f)]$, which reflects the
   *typical* rotation under the drive distribution.

5. **Dummy-CEBRA gate still holds.** The Dummy-CEBRA control (Cell 4) is unaffected
   by this change — it operates on the two-stage pipeline, not the E2E one.

---

## 6. Relationship to the Three-Stage Narrative

This implementation completes the third stage of the argument:

| Stage | Model | Trap 1 (Circular) | Trap 2 (Integration) | Key Result |
|-------|-------|-------------------|---------------------|------------|
| 1. GhostLie OLS | Two-stage (CEBRA → OLS) | ✗ Vulnerable | ✓ Short-window avoids | High SR in local windows |
| 2. Naive E2E | End-to-end, unconstrained MLP | ✓ Decoupled variables | ✗ MLP discards motor gate | $R^2_{\text{rollout}} < 0$ |
| 3. **Structured E2E** | End-to-end, $w = v \cdot \text{MLP}(f)$ | ✓ Decoupled variables | ✓ Physical gate | $R^2_{\text{rollout}} > 0$, interpretable $J$ |

The negative $R^2_{\text{rollout}}$ from Stage 2 is not a failure — it is the
**necessary evidence** that the unconstrained MLP fails, motivating the
structured constraint in Stage 3. Without Stage 2's negative results, Stage 3's
constraint would appear ad hoc. With them, it is a theoretically-motivated
architectural necessity.

---

## 7. References

1. **Wolpert, D.M., Ghahramani, Z. and Jordan, M.I.** (1995) 'An internal model for sensorimotor integration', *Science*, 269(5232), pp. 1880–1882.
2. **Nelson, A., Schneider, D.M. and Mooney, R.** (2013) 'A circuit for motor cortical modulation of auditory cortical activity', *Journal of Neuroscience*, 33(36), pp. 14342–14353.
3. **Schneider, D.M. and Mooney, R.** (2018) 'How movement modulates hearing', *Annual Review of Neuroscience*, 41, pp. 553–572.
4. **Schneider, S., Lee, J.H. and Mathis, M.W.** (2023) 'Learnable latent embeddings for joint behavioural and neural analysis', *Nature*, 617(7960), pp. 360–368.
5. **Keller, G.B. and Mrsic-Flogel, T.D.** (2018) 'Predictive processing: a canonical cortical computation', *Neuron*, 100(2), pp. 424–435.
