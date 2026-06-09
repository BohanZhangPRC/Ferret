# End-to-End Lie Dynamics: A Joint Encoder–Generator Framework for Characterising Rotational Structure in Neural Population Dynamics

This document describes the mathematical foundation, modelling pipeline, metrics, controls, and limitations of the **end-to-end (E2E) Lie Dynamics framework** implemented in `Skieur_EndToEnd_LieODE.ipynb`. It is written as a companion to the original two-stage CEBRA → OLS-Lie pipeline described in `lie_algebra_method_description.md`. Explicit comparisons between the two pipelines are provided throughout.

---

## 1. Motivation: What the Two-Stage Pipeline Could Not Resolve

The original pipeline (`Skieur_LieAlgebra_CEBRA.ipynb`) operates in two separate stages:

1. **CEBRA embedding**: train a contrastive model (InfoNCE loss) to produce a low-dimensional latent representation $z(t) \in \mathbb{R}^D$ of the neural population state, using a behavioural variable $x(t)$ (e.g. head velocity) as the contrastive label.
2. **Post-hoc Lie algebra fit**: on the frozen embedding, fit the linear input-driven dynamical system

$$\frac{dz}{dt} = J_{\mathrm{skew}} \cdot z \cdot x(t) + L \cdot z$$

via OLS regression with post-hoc skew-symmetrisation $J_{\mathrm{skew}} = {\frac{1}{2}}\bigl(J_{\mathrm{ols}} - {J_{\mathrm{ols}}}^{T}\bigr)$.

This two-stage approach carries several structural limitations (documented in `lie_algebra_method_description.md` §12):

| Limitation | Consequence |
|-----------|-------------|
| **Circular reasoning** (§12.2) | CEBRA embedding is shaped by $x(t)$; the Lie algebra is then fit using the same $x(t)$. High SR may be an algorithmic artefact. |
| **Post-hoc projection** (§12.2, 13.2) | $J_{\mathrm{skew}} = {\frac{1}{2}}\bigl(J_{\mathrm{ols}} - {J_{\mathrm{ols}}}^{T}\bigr)$ is the projection of the unconstrained optimum, not the constrained optimum. |
| **Finite-difference noise** (§13.3) | $\frac{dz}{dt}$ is estimated via `np.gradient`, amplifying high-frequency noise. |
| **Scalar linear gating** (§12.8) | $J_{\mathrm{skew}} \cdot x(t)$ assumes a single behavioural variable linearly gates one fixed rotation generator. |
| **Missing Dummy-CEBRA control** (§9) | No negative control where CEBRA is trained on shuffled labels, leaving the possibility that any contrastive embedding produces apparent rotational structure. |
| **Permutation null** (§12.3) | Shuffled-label null uses `np.random.permutation`, destroying the autocorrelation structure of $x(t)$ and confounding temporal alignment with spectral distortion. |

The E2E framework addresses these limitations through **joint optimisation**, **strict Lie parameterisation**, **trajectory-rollout validation**, and **expanded controls**.

---

## 2. End-to-End Model Architecture

### 2.1 Overview

The E2E model is a single PyTorch module that jointly learns:

1. A **temporal encoder** $f_\theta: \mathbb{R}^{T \times N} \to \mathbb{R}^{T \times D}$ mapping neural population activity to a latent trajectory $z(t)$;
2. A **drive-dependent Lie generator** $J(u_t) = \sum_k w_k(u_t) \cdot G_k$ where $G_k$ are fixed orthonormal skew-symmetric basis matrices and $w_k(u_t)$ is a learned nonlinear function of the behavioural drive;
3. A **dissipation (leak) matrix** $L \in \mathbb{R}^{D \times D}$ capturing autonomous dynamics.

The latent state evolves according to:

$$z(t + \Delta t) = \exp\big(\Delta t \cdot [J(u_t) + L]\big) \cdot z(t)$$

using the **matrix exponential** (discrete `torch.matrix_exp`, the default and primary transition mode). An experimental ODE mode (`torchdiffeq.odeint`) is available but disabled by default.

### 2.2 Component Details

#### Encoder

A temporal Conv1d network preserving sequence length $T$:

- **First layer**: `Conv1d(N_neurons → hidden[0], kernel=11, padding=5)`. The kernel size of 11 bins (55 ms at dt = 0.005 s) gives a temporal receptive field comparable to CEBRA's `offset10` configuration (≈50 ms), ensuring fair comparison.
- **Subsequent layers**: `Conv1d(hidden[i] → hidden[i+1], kernel=1)` — pointwise convolutions that add nonlinear depth without expanding the temporal receptive field.
- **Output**: `Conv1d(hidden[-1] → D_LATENT, kernel=1)`.

The encoder accepts both `(T, N)` and `(B, T, N)` inputs, handling single-trajectory and batched encoding.

#### Skew-Symmetric Basis (SkewBasis)

$D(D-1)/2$ fixed orthonormal skew-symmetric basis matrices $G_k \in \mathbb{R}^{D \times D}$, each satisfying $G_k^T = -G_k$, $\lVert G_k\rVert_F = 1$, and $\langle G_i, G_j \rangle_F = \delta_{ij}$. These span the Lie algebra $\mathfrak{so}(D)$.

The basis is constructed with exactly two non-zero entries per matrix: $G_k[i,j] = 1/\sqrt{2}$, $G_k[j,i] = -1/\sqrt{2}$.

**Key difference from the two-stage pipeline**: the basis matrices are fixed buffers with **no learnable parameters**. All learnable rotation structure comes from the Control Network.

#### Control Network — Dual-Engine Structured Gate (StructuredControlNet)

The default control architecture (`N_GATE_DIMS = 2`, `MOTOR_GATE_IDX = -1` for naive fallback) implements an A1-biologically-motivated **dual-engine gate** that separates the drive into two functional streams:

**Gate variables** $u_t[0:2] = [v(t), \dot{f}(t)]$ (Velocity_x, Freq_dot):
$$\text{gate}(t) = \alpha \cdot v(t) + \beta \cdot \dot{f}(t) \in \mathbb{R}^1$$

where $\alpha, \beta$ are learned via `nn.Linear(2, 1, bias=False)`. The `bias=False` constraint preserves the physical boundary condition: **both signals zero $\Rightarrow$ gate $= 0 \Rightarrow J = 0$**.

**Sensory context** $u_t[2:] = [f(t)]$ (Played_frequency):
$$h(t) = \text{MLP}_{\text{ctx}}(f(t)) \in \mathbb{R}^{D(D-1)/2}$$

The full generator:
$$J(u_t) = \text{gate}(t) \cdot \sum_{k=1}^{D(D-1)/2} h_k(t) \cdot G_k = (\alpha v + \beta \dot{f}) \cdot \tilde{J}(f)$$

**Why dual-engine for A1 auditory cortex?** A1 is not a pure motor area — its neural manifold rotates under two independent forces:
1. **Acoustic drive** ($\dot{f}$): sound frequency changes push the manifold forward regardless of movement. Dominant in Playback, where $v \approx 0$ but the acoustic sequence continues.
2. **Motor efference** ($v$): head movement modulates the rotation amplitude. Dominant in Tracking, where $\dot{f} \propto v$.

A pure motor gate ($w = v \cdot \text{MLP}(f)$) fails in Playback: $v = 0 \Rightarrow J = 0$, but the auditory manifold is actively rotating with the acoustic stream. The dual-engine gate resolves this by letting $\beta \cdot \dot{f}$ sustain rotation when the animal is still but sound continues.

**In Tracking**, $\dot{f} \propto v$ (sound tracks movement) — both terms resonate, producing the strongest structured rotation. **In Playback**, $\dot{f}$ and $v$ are decoupled — the network must learn to weight the acoustic drive more heavily.

Naive fallback (`N_GATE_DIMS = -1`): uses the original `ControlNet` — an unstructured MLP with $\tanh$ output. Provided for ablation and regression testing.

Because each $G_k$ is skew-symmetric by construction, $J(u_t)$ is **guaranteed skew-symmetric** throughout training — no post-hoc projection needed. This is the strict Lie parameterisation (improvement #2). The dual-engine gate is improvement #4 (nonlinear multidim forward model) with biological structure imposed.

#### Dissipation (Leak)

Two modes, controlled by the `CONSTRAINED_L` flag:

- **Unconstrained** (default): $L \in \mathbb{R}^{D \times D}$ is a free parameter matrix. The model may learn any real eigenvalues — stable, unstable, or oscillatory.
- **Constrained**: $L = -C C^T$ where $C \in \mathbb{R}^{D \times D}$ is the learned parameter. This guarantees all eigenvalues of $L$ have $\mathrm{Re}(\lambda) \leq 0$ (stable dissipation), ensuring the autonomous dynamics are contracting.

**Key difference from the two-stage pipeline**: the two-stage pipeline fits $L$ via OLS with no stability constraint. The E2E pipeline learns $L$ via gradient descent and offers a stability-guaranteed variant for ablation.

#### Transition Modes

- **Discrete matrix_exp** (default): $z_{t+1} = \exp(\Delta t \cdot [J(u_t) + L]) \cdot z_t$. Fast, robust, no external dependencies.
- **ODE** (experimental, `USE_ODE=True`): $z(t) = \mathrm{odeint}(dz/dt, z_0, t_{\mathrm{span}})$ via `torchdiffeq`. The drive is interpolated piecewise-constant (nearest-neighbour); continuous-time claims are not yet realised with this interpolation scheme. Use `rk4` rather than adaptive `dopri5` if enabled.

---

## 3. Joint Training Objective

The total loss combines a contrastive embedding objective and a dynamics prediction objective:

$$\mathcal{L} = \mathcal{L}_{\mathrm{InfoNCE}}(z; x_{\mathrm{label}}) + \lambda_{\mathrm{dyn}} \cdot \mathcal{L}_{\mathrm{dynamics}}(z_{\mathrm{pred}}, z_{\mathrm{true}})$$

### 3.1 InfoNCE Loss (Embedding)

The InfoNCE loss encourages the latent representation $z(t)$ to reflect the behavioural structure:

$$\mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N} \log\frac{\sum_{j \in \mathcal{P}_i} \exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k) / \tau)}$$

where $\mathcal{P}_i$ is the set of positive pairs for anchor $i$, defined as the $k$-nearest neighbours in behavioural label space (per-row $k$-nearest, not global threshold). The temperature $\tau = 1.5$ matches the CEBRA baseline default.

**Critical design choice**: the InfoNCE loss uses `F.normalize(z)` (scale-invariant). If the dynamics loss were raw MSE $\propto \|z\|^2$, the optimiser could trivially reduce $\mathcal{L}_{\mathrm{dynamics}}$ by shrinking $\|z\| \to 0$ without learning any real dynamics. To prevent this degenerate solution, the dynamics loss is **variance-normalised** (see §3.2).

**Key difference from the two-stage pipeline**: in the two-stage pipeline, InfoNCE and dynamics are optimised in separate stages with no joint constraint. In the E2E pipeline, they are jointly optimised, and the encoder must simultaneously satisfy contrastive and dynamical objectives.

### 3.2 Dynamics Loss (Variance-Normalised Trajectory MSE)

$$\mathcal{L}_{\mathrm{dynamics}} = \frac{\mathrm{MSE}(z_{\mathrm{pred}}[1:T-1], z_{\mathrm{true}}[1:T-1])}{\mathrm{Var}(z_{\mathrm{true}}[1:T-1]) + 10^{-6}}$$

where $z_{\mathrm{pred}}$ is the rolled-out trajectory from $z_0$ using the learned generator, and $z_{\mathrm{true}}$ is the encoder output. Both numerator and denominator scale with $\|z\|^2$, making the loss **scale-invariant** — matching the InfoNCE term and preventing the trivial $\|z\| \to 0$ solution.

**Alignment note**: the rollout produces $T+1$ states $[z_0^{\mathrm{pred}}, z_1^{\mathrm{pred}}, \ldots, z_T^{\mathrm{pred}}]$, while the encoder produces $T$ states $[z_0^{\mathrm{true}}, z_1^{\mathrm{true}}, \ldots, z_{T-1}^{\mathrm{true}}]$. The aligned comparison is $z_{\mathrm{pred}}[1:T-1]$ vs. $z_{\mathrm{true}}[1:]$ (both $T-1$ frames).

### 3.3 Lambda Schedule

$\lambda_{\mathrm{dyn}}$ follows a warmup schedule: $\lambda_{\mathrm{eff}} = 0$ for the first `LAMBDA_DYN_WARMUP` steps (pure InfoNCE establishes the embedding), then linearly ramps to the target $\lambda_{\mathrm{dyn}}$ over the next `LAMBDA_DYN_WARMUP` steps. This prevents the dynamics loss from dominating before the embedding has stabilised.

---

## 4. Batching Strategy: Mini-Trajectories

### 4.1 Problem

The InfoNCE loss operates on **discrete positive/negative pairs** sampled across the entire dataset, while the dynamics loss requires **continuous time slices** for trajectory rollout. These two objectives have conflicting batching requirements.

### 4.2 Solution

Instead of per-frame sampling, the dataloader samples **mini-trajectories** — contiguous segments of length `MINI_TRAJ_LEN` (default 20 bins = 100 ms) — from macro-epochs. Within each batch:

- **InfoNCE**: all frames across all trajectories are flattened into a single pool. Pairwise similarities are computed across the **entire batch** (not within-trajectory only), preserving the contrastive objective's ability to find global structure.
- **Dynamics**: each trajectory is independently rolled out from $t=0$ to $t=T$, and the MSE between predicted and true latent states is computed per-trajectory.

This hybrid batching satisfies both objectives simultaneously.

---

## 5. Validation: Multi-Scale Short-Window Rollout

### 5.1 Why Short Windows?

With unconstrained $L$ (default `CONSTRAINED_L = False`), the matrix exponential $\exp(L \cdot \Delta t)$ can diverge or collapse over hundreds of timesteps. A full-epoch rollout would contaminate the validation metric $R^2_{\mathrm{drive}}$ with numerical instability rather than measuring learned dynamics quality.

### 5.2 Multi-Scale Protocol

Validation $R^2_{\mathrm{drive}}$ is computed at multiple window lengths `VAL_ROLLOUT_LENS = [20, 50, 100]` bins (100 ms, 250 ms, 500 ms):

1. Sample $n_{\mathrm{windows}} = 20$ random start positions from the held-out epoch.
2. For each window, **encode the raw neural slice** (per-window encoding, matching the training pipeline's edge-padding semantics — see §7.1).
3. Rollout the full generator $J(u_t) + L$ and the leak-only baseline $L$ for the window duration.
4. Compute $R^2_{\mathrm{drive}}$ per window, average across windows.

This produces a **decay curve** of $R^2_{\mathrm{drive}}$ as a function of rollout horizon, revealing how far into the future the learned dynamics generalise.

---

## 6. Metrics

### 6.1 Skewness Ratio (SR)

$$\mathrm{SR} = \mathbb{E}_{u \sim p(u)}\left[\frac{\|J(u)\|_F}{\|J(u)\|_F + \|L\|_F}\right]$$

where the expectation is taken over the drive distribution. This **per-sample averaging** (compute SR at each sampled drive, then average) is essential: a successful model learns $w_k(u) \approx -w_k(-u)$ (rotation direction follows velocity sign), so the matrix average $\mathbb{E}_u[J(u)] \approx 0$ would cancel out. Per-sample averaging avoids this cancellation artefact.

**Three variants are reported** (all computed per-sample, then averaged):

| Variant | Drive distribution | Purpose |
|---------|-------------------|---------|
| **Random-drive diagnostic** | $u \sim \mathcal{N}(0, I)$ | Diagnostic; comparable across all sessions without data access. Computed by `get_generator_matrices()`. |
| **Condition-specific SR** ($SR_{\mathrm{Tracking}}$, $SR_{\mathrm{Playback}}$) | $u$ sampled separately from each condition's empirical velocity distribution, standardised with the **pooled** TR+PB mean and std (matching training) | Tests whether Tracking specifically enhances rotational structure over Playback. Per-condition, per-seed, then averaged. |
| **Empirical pooled SR** | Arithmetic mean of the two condition-specific SRs: $\frac{1}{2}(SR_{\mathrm{Tracking}} + SR_{\mathrm{Playback}})$, computed per session | Primary session-level metric; reflects the model's operating regime under both conditions equally. Note: this is the mean-of-means, not SR computed from a single pooled-drive sample — the two are close but not identical because SR is nonlinear in the drive and conditions may have unequal sample counts. |

**Key difference from the two-stage pipeline**: the two-stage SR is $\lVert J_{\mathrm{skew}}\rVert / \lVert J_{\mathrm{ols}}\rVert$, computed from a single post-hoc matrix. The E2E SR is a **distributional expectation** over the drive-dependent $J(u_t)$, reflecting the fact that the generator is not a single matrix but a function of behavioural state.

### 6.2 Drive-Specific $R^2_{\mathrm{drive}}$ (Trajectory-Rollout)

$$R^2_{\mathrm{drive}} = 1 - \frac{\mathrm{MSE}(z_{\mathrm{pred}}, z_{\mathrm{true}})}{\mathrm{MSE}(z_{\mathrm{leak}}, z_{\mathrm{true}})}$$

where $z_{\mathrm{pred}}$ is the full-generator rollout and $z_{\mathrm{leak}}$ is the leak-only ($L$-only, $J = 0$) rollout. Both are computed over the **same short windows** using the **same per-window encoding**.

$R^2_{\mathrm{drive}} > 0$ means the rotational component $J(u_t)$ improves trajectory prediction beyond what the autonomous leak dynamics alone achieve. $R^2_{\mathrm{drive}} \approx 0$ means the leak term dominates prediction. $R^2_{\mathrm{drive}} < 0$ means the full model is worse than leak-only (e.g. $J(u_t)$ adds noise rather than structure).

**Key difference from the two-stage pipeline**: the two-stage $R^2_{\mathrm{drive}}$ is an **instantaneous derivative** metric ($dR/dt$ prediction). The E2E $R^2_{\mathrm{drive}}$ is an **integrated trajectory** metric (rollout state prediction). These are different estimators of conceptually related quantities and are **not directly comparable on the same scale**. Cross-pipeline comparison is valid only for within-pipeline null gating (each vs. its own null), not for direct numerical comparison.

### 6.3 Eigenvalue Analysis

Eigenvalues of the full generator $J(u_t) + L$ are computed **per drive sample**, then averaged:

$$|\overline{\mathrm{Re}}| = \mathbb{E}_{u}\left[|\mathrm{Re}(\mathrm{eig}(J(u) + L))|\right], \quad |\overline{\mathrm{Im}}| = \mathbb{E}_{u}\left[|\mathrm{Im}(\mathrm{eig}(J(u) + L))|\right]$$

**Important caveat**: eigenvalues are reported in the encoder's latent coordinate system. Since each session trains an independent encoder, the latent space scale is arbitrary and **eigenvalues are not comparable across sessions**. They are provided as per-session diagnostics only. The Skewness Ratio, being a scale-invariant ratio, is the primary cross-session comparable metric.

---

## 7. Controls and Null Models

### 7.1 Train/Validation Encoding Consistency

**Problem**: the Conv1d encoder uses `kernel=11, padding=5`, meaning the first and last 5 frames of any window see zero-padded neighbours. If validation encoded the full epoch and then sliced windows, the window edges would use real neighbours — a distribution shift from training.

**Solution**: validation windows are encoded **per-window from raw neural slices** (`n_win = n_t[s:end]; z_win = model.encode(n_win)`), exactly matching the training pipeline (`create_mini_trajectories` → `model(batch_n)`). This eliminates the train/val encoding gap.

### 7.2 E2E Drive-Shuffle Null (Circular Shift)

The primary null control for the E2E pipeline: the trained encoder and generator are held **fixed**, and the drive sequence is **circularly shifted** (`np.roll`) by a random offset. This:

- **Preserves** the autocorrelation structure and power spectrum of $x(t)$ (the drive is slowly-varying and highly autocorrelated).
- **Destroys only** the temporal alignment between drive and neural state.
- Uses a **minimum shift** $\geq$ `MINI_TRAJ_LEN` to prevent the null from being anti-conservative (small shifts leave the drive highly correlated with the original).

**This is the correct null**: same distribution, same autocorrelation, random timing — exactly the improvement over permutation shuffles recommended in `lie_algebra_method_description.md` §12.3.

$N_{\mathrm{SHUFFLES}} = 50$ realisations per epoch (screening-level; $\geq 500$ recommended for formal permutation inference). For computational efficiency, the shuffle null uses $n_{\mathrm{windows}} = 5$ (vs. 20 for the true $R^2_{\mathrm{drive}}$) — each shuffle requires a full per-window encode + rollout. Reported as $R^2_{\mathrm{drive, shuffle}}$ alongside the true $R^2_{\mathrm{drive}}$.

The E2E gate is: $R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, shuffle}}$ at the primary horizon.

### 7.3 Baseline Shuffle Null (also Circular Shift)

The baseline two-stage pipeline's shuffle null has been upgraded from `np.random.permutation` to `np.roll` (circular shift), matching the E2E null family. This ensures both pipelines' null distributions are constructed from the same principle.

### 7.4 Dummy-CEBRA Negative Control

**Purpose**: test whether CEBRA's InfoNCE loss alone imprints rotational topology onto the embedding, independent of real neural dynamics.

**Protocol**:
1. Train a CEBRA model on the same neural data but with **fully permuted behavioural labels** (`np.random.permutation`). This is a "strong negative control" — it destroys all label structure including autocorrelation.
2. Fit the Lie algebra via OLS in this dummy embedding space.
3. Compare $R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, dummy}}$.

If the true embedding's $R^2_{\mathrm{drive}}$ significantly exceeds the dummy embedding's, the rotational structure is a genuine property of the behaviourally-aligned neural dynamics, not an InfoNCE artefact.

**Note**: this uses full permutation (not circular shift) because the goal is to destroy **all** label structure — a stronger test than the autocorrelation-preserving null. A circular-shift-label CEBRA dummy would be a useful intermediate control for future work.

### 7.5 $\lambda_{\mathrm{dyn}} = 0$ Ablation (InfoNCE-Only)

**Purpose**: test whether the dynamics constraint induces rotational structure beyond what InfoNCE alone produces.

**Protocol**:
1. Train the E2E model with $\lambda_{\mathrm{dyn}} = 0$ (pure InfoNCE, no dynamics loss).
2. **Freeze the encoder**. Post-hoc fit the Lie algebra via OLS (`fit_lie_algebra_with_leak`) on the frozen embedding — this uses the **same estimator** as the baseline pipeline, enabling direct comparison.
3. Compare $SR_{\mathrm{E2E}}$ vs. $SR_{\lambda=0}^{\mathrm{OLS}}$ and ${R^2}_{\mathrm{drive},\,\mathrm{E2E}}$ vs. ${R^2}_{\mathrm{drive},\,\lambda=0}^{\mathrm{OLS}}$.

If $SR_{\mathrm{E2E}} > SR_{\lambda=0}^{\mathrm{OLS}}$, the dynamics constraint genuinely induces rotational structure. If they are comparable, InfoNCE alone may be sufficient.

### 7.6 Kinematic Confound Check

The behavioural drive $x(t)$ (head velocity) may have different distributions in Tracking vs. Playback. If the animal moves less during Playback, the drive dynamic range is smaller, mechanically reducing $R^2_{\mathrm{drive}}$ independent of neural computation.

**Diagnostic**: velocity distributions are compared between conditions using:
- Per-session histograms
- RMS, standard deviation, and range per condition
- Paired t-tests on velocity metrics
- **Kolmogorov–Smirnov (KS) tests** per session

If a majority of sessions show significantly different velocity distributions ($p < 0.05$ on the KS test), a warning is issued: TR/PB $R^2_{\mathrm{drive}}$ differences cannot be attributed to neural mechanisms without covariate correction.

---

## 8. Comparison: Two-Stage vs. End-to-End Pipeline

| Aspect | Two-Stage (baseline) | End-to-End (this work) |
|--------|---------------------|----------------------|
| **Embedding** | CEBRA InfoNCE, trained separately | Jointly trained with dynamics constraint |
| **Generator** | Single $J_{\mathrm{skew}}$, scalar-gated by $x(t)$ | $J(u_t) = \sum_k w_k(u_t) G_k$, nonlinear drive-dependent |
| **Skew constraint** | Post-hoc projection $J_{\mathrm{skew}} = \frac{1}{2}(J - {J}^T)$ | Strict: $J(u_t)$ is skew-symmetric by construction |
| **Dynamics fitting** | OLS on $\frac{dz}{dt}$ (`np.gradient`) | Trajectory rollout via `matrix_exp`, integrated prediction |
| **$R^2_{\mathrm{drive}}$** | Instantaneous derivative prediction | Integrated trajectory prediction (multi-scale) |
| **SR computation** | $\|J_{\mathrm{skew}}\| / \|J_{\mathrm{ols}}\|$ (single matrix) | $\mathbb{E}_u[\|J(u)\| / (\|J(u)\| + \|L\|)]$ (distributional) |
| **Leak term** | OLS-fitted, no stability constraint | Learned via gradient descent; optional $-CC^T$ constraint |
| **Encoder** | CEBRA offset10 (50 ms RF) | Conv1d kernel=11 (55 ms RF, matched) |
| **Null model** | Permutation (original); circular shift (upgraded) | Circular shift (autocorrelation-preserving) |
| **Dummy CEBRA** | Not implemented in original | Implemented: shuffled-label CEBRA → Lie fit |
| **$\lambda=0$ ablation** | Not applicable (two-stage) | Implemented: frozen encoder + post-hoc OLS Lie fit |
| **Kinematic check** | Not performed | Velocity distribution KS test per session |
| **Validation** | In-sample (same epochs) | Held-out epochs, multi-scale (100/250/500 ms) |
| **Multi-seed** | Single seed | 3 seeds per session, seed variance reported |
| **Temperature** | $\tau = 1.5$ (CEBRA default) | $\tau = 1.5$ (matched) |
| **$N_{\mathrm{SHUFFLES}}$** | 10 (original); 50 (upgraded) | 50 (screening-level) |

---

## 9. Statistical Design

### 9.1 Session Sampling

All available sessions are used (`N_TRAIN_SESSIONS = None`). Sessions are randomly ordered (not first-N) to avoid ordering bias.

### 9.2 Multi-Seed Training

Each session is trained with $N_{\mathrm{SEEDS}} = 3$ different random seeds. Per-session metrics are reported as mean ± SEM across seeds. This provides a lower bound on metric variance from stochastic optimisation.

### 9.3 Paired Comparisons

Tracking vs. Playback comparisons use **paired t-tests at the session level** for $R^2_{\mathrm{drive}}$ (which is per-condition, computed on condition-specific held-out epochs). SR comparisons use per-condition empirical-drive SR ($SR_{\mathrm{Tracking}}$, $SR_{\mathrm{Playback}}$).

For the $\lambda=0$ ablation, conditions are averaged per session before the paired test (avoiding treating Tracking and Playback from the same session as independent samples).

### 9.4 Multiple Comparison Note

Multi-scale validation ($[20, 50, 100]$ bins) produces three $R^2_{\mathrm{drive}}$ values per session-condition. Each horizon is tested independently. The primary endpoint is the shortest horizon (`VAL_ROLLOUT_LENS[0] = 20` bins), which matches the training trajectory length.

---

## 10. Methodological Evolution & Troubleshooting

The final E2E architecture is the result of systematic troubleshooting to resolve sequential biological and geometric failures observed during development. Below is a summary of the iterative improvements:

### 10.1 Initial Baseline (GhostLie V2.0)
- **Method**: Global continuous CEBRA embedding followed by OLS-Lie fitting with tight event-triggered masks and a global circular-shift shuffle null.
- **Result**: Demonstrated significant Skewness Ratio (SR) in both Tracking and Playback conditions, resolving earlier issues with integration dilution and null model spectrum.
- **Failure**: When driven by `Velocity_x`, $R^2_{\mathrm{drive}}$ was near zero. Linear OLS assumes a constant rotation matrix, failing to capture state-dependent non-linear routing. When driven by `Position`, predictability emerged, but circular reasoning remained a severe confound.

### 10.2 Single-Drive E2E & Drive-Label Decoupling
- **Method**: Transitioned to joint training (E2E) using `Velocity_x` for both the InfoNCE contrastive label and the dynamical drive.
- **Discovery**: This formulation suffered from the same severe "circular reasoning" as the two-stage baseline. If the embedding is shaped purely by velocity, predicting velocity-dependent dynamics from it becomes an algorithmic artefact.
- **Fix**: Decoupled the embedding from the drive. Switched InfoNCE to use `Played_frequency` (sensory context) as the label, forcing the embedding to reflect the acoustic state rather than motor commands.

### 10.3 Naive End-to-End Joint Training (Unstructured MLP)
- **Method**: Retained the decoupled embedding (`Played_frequency` for InfoNCE), but fed both `[Velocity_x, Played_frequency]` into a naive, unstructured ControlNet MLP to generate the rotation matrix.
- **Result**: Severe "Integration Dilemma". The network learned to ignore the noisy `Velocity_x` signal and simply output constant rotation based on frequency. When the animal stopped ($v=0$), rotation continued unabated, causing trajectory rollout prediction errors to explode ($R^2_{\mathrm{drive\_rollout}} \ll 0$).

### 10.4 Structured ControlNet (Dual-Engine Motor-Sensory Gate)
- **Method**: Imposed a strict physical gating constraint: $J(u_t) = g(t) \cdot \mathrm{MLP}(f(t))$. Initially, the gate was strictly motor: $g(t) = v(t)$.
- **Discovery**: A1 is a primary sensory cortex. In Playback, the animal can be stationary ($v=0$), but the continuous acoustic progression ($df/dt$) still drives the rotation. Forcing rotation to stop when $v=0$ violates biological reality.
- **Refinement**: Upgraded the gate to a learned linear combination of motor and sensory drives: $g(t) = \alpha \cdot v(t) + \beta \cdot \dot{f}(t)$.
- **Result**: Perfectly learned a conservative rotational generator (SR = 0.9999, $|\mathrm{Real}| = 0.0$). However, $R^2_{\mathrm{drive\_rollout}}$ surprisingly remained negative.

### 10.5 The Off-Center Manifold Crisis (Final Geometric Fix)
- **Method**: Traced the integration failure to a fundamental topological mismatch. Euclidean InfoNCE places the embedded sequence arbitrarily far from the coordinate origin. Applying a pure Lie rotation $J \cdot z$ to an off-center manifold produces a catastrophic tangential translation, flinging the rollout trajectory into deep space.
- **Fix** (commit `597b00a`): Pinned the manifold strictly to the origin-centered unit sphere by applying $L_2$ normalisation (`F.normalize(z, p=2, dim=-1)`) to the output of the temporal encoder. Because $\exp(J \Delta t)$ is an orthogonal (norm-preserving) matrix, the rollout trajectory natively glides along the spherical manifold without translation errors. Results pending at time of writing.

---

## 11. Limitations

### 11.1 Single-Animal, Single-Brain-Region

All results are from one animal (SKIEUR). Session-level paired t-tests do not support population-level inference. Multi-animal hierarchical modelling or explicit single-case-study framing is needed for publication-level claims. (Same as `lie_algebra_method_description.md` §12.14.)

### 11.2 Circular Reasoning — Partially Mitigated, Not Default-Resolved

The default configuration uses `CEBRA_LABEL = "Velocity_x"` as the InfoNCE contrastive label, while the dual-engine gate uses `[Velocity_x, Freq_dot, Played_frequency]` as the dynamics drive. Because `Velocity_x` appears in both, the embedding may still be partially shaped by the same behavioural variable that drives the dynamics. **Decoupled mode is available** (`CEBRA_LABEL = "Played_frequency"`, the sensory context variable) and eliminates Trap 1 (circular reasoning) entirely — but it is not the default. The Dummy-CEBRA control and $\lambda=0$ ablation provide partial validation; explicit decoupling (`CEBRA_LABEL ≠ ` any gate variable) is recommended for publication results.

### 11.3 SR Is Not Coordinate-Invariant

The skew/symmetric decomposition is only physically meaningful as "rotation vs. stretch" under an isotropic metric. CEBRA/Conv1d coordinates are neither orthogonal nor endowed with a canonical metric. SR values should be interpreted as useful but coordinate-frame-dependent descriptors. (Same as `lie_algebra_method_description.md` §12.1.)

### 11.4 Eigenvalue Cross-Session Comparability

With the L2-normalised encoder output (`F.normalize(z, p=2, dim=-1)`, commit `597b00a`), the latent manifold is pinned to the origin-centred unit sphere across all sessions. This removes the arbitrary per-session scale factor that previously made $J(u_t)$ and $L$ magnitudes incomparable. Cross-session eigenvalue statistics are now mathematically meaningful. **Caveat:** the normalisation constrains $\lVert z\rVert = 1$, which implicitly rescales the learned generator. $|\mathrm{Im}|$ values now have a natural interpretation as angular velocity on the unit sphere (radians per timestep).

### 11.5 Short-Window Validation Scope

$R^2_{\mathrm{drive}}$ is validated on windows of 20–100 bins (100–500 ms). The model is trained on 20-bin trajectories. While multi-scale validation probes generalisation to longer horizons, the model has not been trained to capture epoch-scale dynamics. Positive $R^2_{\mathrm{drive}}$ at 500 ms does not imply the model captures the full dynamical structure of the system.

### 11.6 $N_{\mathrm{SHUFFLES}} = 50$ Is Screening-Level

With 50 shuffle realisations, the minimum resolvable permutation $p$-value is approximately $1/51 \approx 0.02$. This is sufficient for screening but not for formal inference with multiple-comparison correction. $\geq 500$ realisations are recommended for publication.

### 11.7 ODE Mode Is Experimental

The default transition is discrete `matrix_exp`. When `USE_ODE = True`, the drive is interpolated piecewise-constant (nearest-neighbour), meaning the "continuous-time ODE" claim is not yet realised. Adaptive solvers (`dopri5`) may be unstable at bin boundaries. The ODE mode is provided for future exploration and should not be cited as a methodological contribution in the current form.

### 11.7b L2-Normalised Manifold Constraint

The encoder output is L2-normalised per timepoint (`F.normalize(z, p=2, dim=-1)`, commit `597b00a`) to pin the manifold to the origin-centred unit sphere — a mathematical necessity for pure Lie algebra rotation $J \cdot z$. This constraint has two implications: (a) the manifold is forced onto $\mathbb{S}^{D-1}$, which may not reflect the true neural population geometry; (b) the dynamics loss operates on normalised $z$, which reduces the variance available for MSE differentiation. The variance-based normalisation of the dynamics loss (§5.2) partially mitigates (b).

### 11.8 Multi-Dim Drive (Implemented)

Multi-dimensional drive extraction is fully implemented (commit `97de7f0`, cf454df). `train_one_session` calls `extract_epochs` separately for each `DRIVE_KEYS` column (since `extract_macro_epochs` uses Condition boundaries, all extractions produce identical epoch boundaries), standardises each dimension independently (pooled TR+PB via per-dimension `drive_mu[dk]`/`drive_std[dk]` dicts), and stacks them via `np.column_stack` into a `(T_i, d_drive)` array. Sanity checks assert identical epoch counts and lengths across dimensions.

The default `DRIVE_KEYS = ["Velocity_x", "Freq_dot", "Played_frequency"]` feeds the dual-engine StructuredControlNet. `build_drive_vector()` is defined but never called in the training path (dead code; retained for OLS baseline compatibility).

**Important caveat for multi-scale $R^2$ comparisons**: as the gate dimensionality changes between configurations (e.g. single-gate `[Velocity_x]` vs. dual-gate `[Velocity_x, Freq_dot, Played_frequency]`), the ControlNet architecture and parameter count change. Cross-configuration $R^2$ comparisons should control for parameter count (e.g. by matching the total number of trainable parameters in the control pathway).

### 11.9 Kinematic Confound

Velocity distributions may differ between Tracking and Playback. If the animal moves less during Playback, the drive dynamic range is smaller, mechanically reducing $R^2_{\mathrm{drive}}$ independent of neural computation. Velocity distribution diagnostics (KS test, RMS comparison) are reported, but no formal covariate correction (e.g. velocity-range matching or ANCOVA) is performed.

### 11.10 Baseline vs. E2E $R^2_{\mathrm{drive}}$ Are Different Estimators

The baseline $R^2_{\mathrm{drive}}$ (derivative-based OLS) and E2E $R^2_{\mathrm{drive}}$ (trajectory-rollout MSE) are fundamentally different quantities. Direct numerical comparison (e.g. on a $y = x$ scatter) is **not valid**. Each pipeline's $R^2_{\mathrm{drive}}$ is gated only against its **own null distribution**.

---

## 12. Reproducibility

- **Random seed**: `RANDOM_SEED = 42` for `numpy`, `torch`, and CUDA.
- **Multi-seed**: 3 seeds per session with per-seed `RANDOM_SEED + idx * 100 + seed`.
- **All config in Cell 0**: no hidden parameters.
- **Output directory**: timestamped `Skieur_LieE2E_YYYYMMDD_HHMMSS/` with PDF + PNG figures and summary `.txt`.
- **Build infrastructure**: the notebook is assembled from modular `_nb_cells/*.py` source files via `_build_notebook_final.py`.

---

## 13. References

The same reference list as `lie_algebra_method_description.md` applies, with the following additions relevant to the E2E framework:

1. **Chen, R.T.Q., Rubanova, Y., Bettencourt, J. and Duvenaud, D.** (2018) 'Neural Ordinary Differential Equations', *NeurIPS 2018*.
2. **Kidger, P.** (2021) 'On Neural Differential Equations', *DPhil thesis, University of Oxford*.
3. **Gu, A. and Dao, T.** (2023) 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces', *arXiv preprint*, arXiv:2312.00752.

For the full reference list (Churchland et al. 2012, Schneider et al. 2023, Wolpert et al. 1995, etc.), see `lie_algebra_method_description.md` §References.
