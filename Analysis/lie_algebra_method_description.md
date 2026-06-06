# Lie Algebra and Rotational Dynamics for Sensorimotor Integration

This document describes the mathematical foundation, modeling pipeline, and physical interpretation of the Lie Algebra / Rotational Dynamics method used to analyze neural population data in this project.

## 1. Project Context & Motivation

The overarching goal is to understand how the primary auditory cortex (A1) and premotor cortex (PMC) implement an internal **forward model** during a continuous sensorimotor task. A ferret operates in a **closed-loop (Tracking)** condition where its continuous head velocity directly controls the frequency of an acoustic note, contrasted with an **open-loop (Playback)** condition where the same acoustic sequence is replayed without motor control.

A functional forward model would need to accomplish two things simultaneously:
1. **Continuous Prediction Updating:** Use the ongoing motor command (head kinematics) to continuously update the prediction of the upcoming sensory state (acoustic frequency).
2. **Sensory Attenuation:** Suppress the predictable sensory consequences of self-generated movement to filter out reafferent noise.

The mathematical framework below captures both phenomena geometrically within the neural population's state space. Note that eigenvalue |Real| (dissipation) and |Imag| (rotation) can be interpreted as signatures of sensory attenuation and predictive updating respectively, but these are **hypotheses to be tested**, not confirmed facts.

---

## 2. Foundational Math & Model

We model the neural population dynamics as an input-driven continuous linear dynamical system:

$$
\frac{dR}{dt} = J_{\mathrm{skew}} \cdot R \cdot x(t) + L \cdot R
$$

- **$R(t)$**: The continuous neural population state — either raw firing rates (`(time, neurons)`) or a low-dimensional CEBRA embedding (typically 3D).
- **$x(t)$**: The continuous external behavioral drive (e.g., instantaneous head velocity `Velocity_x` or cumulative `Position`).
- **$J_{\mathrm{skew}}$**: The skew-symmetric (rotational) component of the fitted generator — responsible for rotating the neural state in response to the behavioral drive.
- **$L$**: An unconstrained linear "leak" term that captures baseline decay, drift, or restoring forces independent of the behavioral drive. It handles the natural dynamics of the neural system when $x(t) = 0$.

**How the fitting works (OLS):**

The target variable is the empirical derivative of the neural state, computed via discrete gradient:

```python
dr_dt = np.gradient(r, dt, axis=0)
```

The predictors are two design matrices:
- **Rotation term:** $R(t) \cdot x(t)$ — the neural state gated by the behavioral drive
- **Leak term:** $R(t)$ — the neural state alone

OLS simultaneously fits both:

```python
U = np.hstack([r * x_dot[:, None], r])   # [rotation predictor, leak predictor]
weights, _, _, _ = np.linalg.lstsq(U, dr_dt, rcond=None)
```

From the fitted weights:
- **$J_{\mathrm{ols}}$** = `weights[:, :N]` — the raw (unconstrained) generator matrix
- **$J_{\mathrm{skew}}$** = `0.5 * (J_{\mathrm{ols}} - J_{\mathrm{ols}}^T)` — the skew-symmetric component, which isolates pure rotation
- **$L$** = `weights[:, N:]` — the leak matrix

The model prediction is:

```python
dR_pred = (r * x_dot[:, None]) @ J_skew.T + r @ L.T
```

A PyTorch variant (`_fit_lie_pytorch`) performs constrained optimization where `J_skew = W - W^T` is enforced during gradient descent rather than post-hoc, but it is functionally equivalent for the metrics.

---

## 3. Metrics: Skewness Ratio, R², and R²_drive

### A. Skewness Ratio (SR)

$$
\mathrm{SR} = \frac{\|J_{\mathrm{skew}}\|}{\|J_{\mathrm{ols}}\|}
$$

Measures what fraction of the fitted generator is purely rotational (skew-symmetric). SR ≈ 1 means the behavioral drive primarily rotates the neural manifold; SR ≈ 0 means it primarily scales or dissipates the state.

### B. Total R²

$$
R^2 = 1 - \frac{\sum(dR/dt - dR_{\mathrm{pred}})^2}{\sum(dR/dt - \overline{dR/dt})^2}
$$

Measures how well the **full model** (rotation + leak) predicts the neural derivative. **Important caveat:** the leak term $L \cdot R$ can explain a large fraction of the derivative variance even without any behavioral drive. Therefore $R^2_{\mathrm{true}} \approx R^2_{\mathrm{shuffle}}$ does NOT mean the rotation is absent — it means the leak term dominates the total variance.

### C. Drive-Specific R²_drive (Partial R²)

To isolate the contribution of the behavioral drive, we compute the incremental variance explained by the rotation term over a leak-only baseline:

$$
R^2_{\mathrm{drive}} = 1 - \frac{\sum(dR/dt - dR_{\mathrm{full}})^2}{\sum(dR/dt - dR_{\mathrm{leak}})^2}
$$

where $dR_{\mathrm{leak}} = L \cdot R$ and $dR_{\mathrm{full}} = J_{\mathrm{skew}} \cdot R \cdot x + L \cdot R$.

**This is the stricter test:** $R^2_{\mathrm{drive, true}} \gg R^2_{\mathrm{drive, shuffle}}$ indicates that the time-aligned behavioral drive adds explanatory power beyond what the autonomous leak dynamics already capture.

---

## 4. Eigenvalue Analysis

The eigenvalues of **$J_{\mathrm{ols}}$** (the full unconstrained generator, **not** $J_{\mathrm{skew}}$) are computed:

```python
eigvals = np.linalg.eigvals(J_ols)
real_mean = np.mean(np.abs(np.real(eigvals)))
imag_mean = np.mean(np.abs(np.imag(eigvals)))
```

**Why $J_{\mathrm{ols}}$ and not $J_{\mathrm{skew}}$?** $J_{\mathrm{skew}}$ is skew-symmetric by construction, so its eigenvalues are always **purely imaginary** ($\lambda = \pm i\omega$). Computing eigenvalues of $J_{\mathrm{skew}}$ would give $|Real| \equiv 0$ as a mathematical identity, not a biological finding. $J_{\mathrm{ols}}$ retains the full dissipative structure.

**Why take absolute values?** Eigenvalues of a real matrix come in complex-conjugate pairs $a \pm ib$. Without absolute values, imaginary parts would cancel (mean ≈ 0), and real parts (which are negative for stable dissipative systems) would be unintuitive — stronger dissipation → more negative → smaller mean. Taking absolute values gives:

- **|Imag|**: Mean rotation frequency — how fast the neural state oscillates/rotates in response to the behavioral drive
- **|Real|**: Mean dissipation rate — how strongly the behavioral drive contracts/decays the neural state amplitude

A higher $|Imag|$ in Tracking would suggest active movement **drives stronger rotational dynamics** (consistent with predictive updating). A higher $|Real|$ in Tracking would suggest active movement **enhances state contraction** (consistent with sensory attenuation).

---

## 5. Pipeline: Data Preprocessing

### Epoch Extraction

The pipeline uses **macro-epoch extraction** by default (`USE_MACRO_EPOCH = True`): contiguous segments of the same Condition (`Tracking` or `Playback`) lasting at least `MIN_EPOCH_DUR` (typically 2.0 s). These segments preserve temporal continuity, avoiding derivative artifacts at artificial boundaries.

### Preprocessing

Before CEBRA training, neural data is preprocessed depending on the CEBRA distance metric:
- **`cosine` distance:** Per-timepoint L2 normalization (`method='l2'`) — normalizes each timepoint to unit length
- **`euclidean` distance:** Per-neuron z-score across time (`method='zscore'`) — standardizes each neuron's firing rate distribution

### Temporal Shift

`TAU_SHIFT = 6` (≈ 30 ms at 5 ms bins) shifts the neural data forward relative to the behavioral labels. This accounts for the physiological latency between neural activity and the behavioral readout.

---

## 6. CEBRA-Embedded Lie Algebra Pipeline

When using CEBRA embeddings (Sections 1.3–1.4 in the analysis notebook):

1. **Per-session CEBRA:** One pooled CEBRA model is trained per session, with both Tracking and Playback epochs included. This ensures both conditions share the same latent coordinate system, making their Lie generators comparable.

2. **Per-epoch Lie fit:** After transforming each epoch through the CEBRA model, `fit_lie_algebra_with_leak` is applied **separately to each epoch** (not to concatenated data). This avoids derivative discontinuities at epoch boundaries.

3. **Metrics are averaged across epochs** within each session-condition to produce a single value per session.

4. **Time-shuffled control:** For each epoch, the behavioral labels are randomly permuted 10 times (`N_SHUFFLES = 10`), and the Lie algebra is re-fit on each permutation (same CEBRA embedding, shuffled labels). The CEBRA model is **not** retrained — the embedding stays fixed, and only the OLS Lie algebra fit is repeated. This is the critical design choice: it tests whether the *temporal alignment* between the behavioral variable and the neural state matters, without confounding from re-embedding. The shuffle metrics (`SR_shuffle`, `R2_shuffle`, `R2_drive_shuffle`) are averaged across the 10 realizations to reduce baseline noise.

5. **R² gate:** A session-condition is considered to have meaningful rotational structure only if $R^2_{\mathrm{true}} > R^2_{\mathrm{shuffle}}$ (total R²) **and** $R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, shuffle}}$ (drive-specific R²). The drive-specific gate is stricter and more informative.

6. **Paired comparisons:** Tracking vs. Playback comparisons use paired t-tests at the session level. Since each session has one CEBRA model shared across conditions, per-condition statistics are reported separately.

---

## 7. Summary of Reported Metrics

| Metric | What it measures | Computed from |
|--------|-----------------|---------------|
| **Skewness Ratio (SR)** | Fraction of generator that is rotational | $J_{\mathrm{skew}}$, $J_{\mathrm{ols}}$ |
| **SR_shuffle** | Baseline SR with shuffled labels | Same, on permuted labels |
| **R²** | Total derivative prediction quality | Full model vs. null |
| **R²_shuffle** | Baseline R² with shuffled labels | Same, on permuted labels |
| **R²_drive** | Rotation-only over leak-only baseline | $dR_{\mathrm{full}}$ vs. $dR_{\mathrm{leak}}$ |
| **R²_drive_shuffle** | Baseline R²_drive with shuffled labels | Same, on permuted labels |
| **|Imag|** | Mean rotation frequency | Eigenvalues of $J_{\mathrm{ols}}$ |
| **|Real|** | Mean dissipation/contraction rate | Eigenvalues of $J_{\mathrm{ols}}$ |

---

## 8. Comparison with jPCA (Churchland et al., 2012)

### jPCA
jPCA fits an **autonomous** rotational dynamics approximation in PCA-reduced space:
$$ \dot{x} = M_{\mathrm{skew}} \cdot x $$
There is no external behavioral drive. It assumes the neural state rotates autonomously due to intrinsic network dynamics (e.g., motor cortex during reaching). The skew-symmetric constraint is enforced during fitting.

### This Analysis (Velocity-Modulated Lie Algebra)
This analysis fits an **input-driven** system:
$$ \dot{R} = J \cdot R \cdot x_{\mathrm{drive}} + L \cdot R $$
The behavioral variable ($x_{\mathrm{drive}}$) explicitly gates the rotation and dissipation. The generator $J$ is unconstrained during the OLS fit, allowing simultaneous capture of rotation ($J_{\mathrm{skew}}$ / imaginary eigenvalues) and dissipation (real eigenvalues of $J_{\mathrm{ols}}$). This makes it uniquely suited for studying continuous closed-loop sensorimotor integration, where external actions continuously force the neural state.

---

## 9. Understanding the Shuffle Control

### What the shuffle actually does

For each epoch, the behavioral label vector is randomly reordered via `np.random.permutation(e_l)`. For example:

```
True:     [0.2, 0.5, 0.8, 0.3, -0.1, 0.6, ...]   (time-ordered velocity)
Shuffled: [0.6, -0.1, 0.2, 0.8, 0.5, 0.3, ...]   (random order)
```

Only the rotation predictor `U_rot = R * x_dot` changes between true and shuffle fits. Everything else is identical:

| Step | True | Shuffle |
|------|------|---------|
| CEBRA embedding `emb` | used as-is | **identical** — CEBRA is NOT retrained |
| `dr_dt` = `np.gradient(emb)` | computed from `emb` | **identical** |
| `U_rot` = `emb * e_l` | true label order | `emb * e_l_shuf` ← **only difference** |
| Leak predictor `emb` in `U` | unchanged | **identical** |
| `J_skew` (rotation weights) | fitted to true labels | different (time alignment destroyed) |
| `L` (leak weights) | fitted to `emb` columns | nearly identical (same `emb` columns) |
| `ss_tot` (R² denominator) | `Σ(dr_dt - mean(dr_dt))²` | **identical** |

### What this destroys vs preserves

**Destroys:** The temporal/causal relationship between the behavioral variable and the neural state. If the true velocity at time *t* is 0.5 m/s and the embedding at time *t* responds specifically to that velocity, shuffling breaks this coupling.

**Preserves:** The label distribution (same mean, variance, range), the CEBRA embedding, the leak term's contribution, and the total variance of `dr_dt`.

### Why total R² is nearly identical between true and shuffle

This is **not a bug** — it is a direct consequence of the model design. The total R² measures how well **rotation + leak** together predict `dr_dt`. In low-dimensional embeddings (e.g., 3D CEBRA), the leak term `L·R` — a 3×3 autoregressive matrix — already explains the vast majority of the derivative variance. The rotation term adds a small correction that changes the generator's *structure* (reflected in SR) but barely moves the total R² needle.

**Analogy:** Adding a pinch of salt to a bowl of rice. The salt changes the flavor profile (SR detects it), but measuring the total weight of the bowl before and after (total R²) shows no difference — the rice dominates the measurement.

**The meaningful metrics are SR and R²_drive**, which specifically isolate the drive-gated component.

### Why 10 shuffles per epoch

A single shuffle is one random pairing of labels and neural states, and can be "lucky" or "unlucky" — producing a shuffle SR that is accidentally high or low due to sampling noise. Averaging across 10 independent permutations reduces the baseline variance by a factor of ~10 and gives a stable estimate of the null distribution.

---

## 10. OLS as Multi-Output Regression

### How it differs from ordinary single-variable regression

Standard linear regression predicts one scalar outcome:

$$ y = a \cdot x + b $$

where $y$ is a vector of length $T$ (timepoints), $x$ is a single predictor, and $a$ is a scalar coefficient.

Here the OLS solves for **N targets simultaneously**:

$$ dr/dt = J \cdot (R \cdot x_{\mathrm{drive}}) + L \cdot R $$

- `dr_dt` is $(T \times N)$ — one derivative time series per neuron
- The design matrix `U` is $(T \times 2N)$ — two predictor blocks side by side
- `weights` from `lstsq(U, dr_dt)` is $(2N \times N)$ — **a matrix, not a scalar**

### Design matrix construction

```
U = [ R * x_dot  |  R ]
     ─── N cols ──   ── N cols ──

One row (timepoint t):

[ r[t,1]·x[t]  r[t,2]·x[t]  ...  r[t,N]·x[t]  |  r[t,1]  r[t,2]  ...  r[t,N] ]
 ─────────── rotation features (N) ────────────    ────── leak features (N) ─────
```

### What the weight matrix means

After fitting, `weights` is a $(2N \times N)$ matrix. Extracted into blocks:

```python
J_ols = weights[:N, :]     # (N×N) — first N rows
L     = weights[N:, :]     # (N×N) — last N rows
```

`J_ols[i, j]` means: **how much does source neuron `i` (gated by velocity `x`) contribute to the derivative of target neuron `j`?**

This is fundamentally different from a scalar regression coefficient. It describes an **internal interaction graph** of the neural population — which neurons drive which others, and with what sign, in response to the behavioral variable. The skew-symmetrization step (`J_skew = 0.5*(J_ols - J_ols^T)`) then extracts the purely rotational component of this interaction graph.

---

## 11. CEBRA vs PCA / jPCA / dPCA and the Dimensionality Tradeoff

The choice of dimensionality reduction method and embedding dimension fundamentally determines what dynamical structure can be observed. This section explains why CEBRA at 3D is specifically chosen over alternatives, and what would happen at different embedding sizes.

---

### 11.1 CEBRA vs PCA

**PCA** finds directions in neural state space with maximum variance:

$$ \max \ \mathrm{Var}(X \cdot w), \quad \|w\| = 1 $$

It does not look at behavioral labels. A dimension with large variance from global arousal or reward expectation will dominate; a dimension with small raw variance that perfectly encodes velocity may be discarded entirely — shuffled into the 15th or 20th principal component and lost.

**CEBRA** uses contrastive learning (InfoNCE loss):

$$ L_{\mathrm{CEBRA}} = -\log \frac{\exp(\mathrm{sim}(z_i, z_i^+))}{\sum_j \exp(\mathrm{sim}(z_i, z_j^-))} $$

- **Positive pairs** (nearby timepoints or same behavioral context) are pulled together
- **Negative pairs** (distant timepoints or different behavioral context) are pushed apart

The behavioral variable (velocity, position) **drives the training**. CEBRA nonlinearly reshapes the embedding to extract structure relevant to that variable — even if the relevant signal has tiny raw variance (e.g., raw-space R²_drive ≈ 0.0002). It works like a scalpel: regardless of how weak the tracking signal is in the raw neural code, CEBRA isolates it into a low-dimensional manifold.

| | PCA | CEBRA |
|---|---|---|
| Input | Neural data only | Neural data + behavioral labels |
| Optimizes | Reconstruction error | Contrastive (InfoNCE) loss |
| Preserves | High-variance dimensions | Behaviorally relevant dimensions |
| Nonlinear? | Linear only | Can use deep (nonlinear) networks |
| Example failure | Keeps heart-rate artifact, drops velocity code | Keeps velocity code, drops heart-rate artifact |
| Topology | May scramble manifold curvature | Preserves geometric curvature and topology |

#### Why PCA is "time-blind" — and why that matters for dynamics

The most fundamental difference between PCA and CEBRA is **whether the algorithm cares about time**.

**PCA is time-blind.** If you take the entire experimental recording, randomly shuffle the time order of all data points (putting second 1 at minute 10, second 2 at hour 3, etc.), PCA will compute **exactly the same principal axes and the same reduced representation**. The covariance matrix — PCA's only input — is invariant to any permutation of time. PCA sees neural activity as a cloud of static points in space and asks: "which directions have the most scatter?" If the largest-variance signal is resting-state baseline drift, or time-independent global arousal, PCA will fill its top components with these static features — and velocity-coding neurons, which may have tiny variance but precise temporal structure, will be discarded.

**Dynamics require time.** The Lie algebra framework is fundamentally temporal: it fits the derivative $dR/dt$ — how the current state transitions to the next state. It asks "does the behavioral drive *cause* the neural state to rotate?" This causal-temporal question cannot be answered in a space built by a time-blind algorithm. The extracted axes must encode not just what the brain is doing *on average*, but how it moves from one state to the next over time.

**CEBRA is time- and behavior-aware.** Its contrastive learning pairs data points based on their **temporal and behavioral proximity**: if time $t$ and $t+1$ share similar velocity/position, their neural states are pulled together in the embedding; if they are behaviorally far apart, they are pushed apart. This means CEBRA's embedding intrinsically encodes the **continuous temporal evolution** of the neural state in relation to behavior. The resulting latent space is not a static snapshot of variance — it is a **continuous geometric movie** of the sensorimotor transformation.

**Analogy:**

| | PCA | CEBRA |
|---|---|---|
| **What it produces** | A static "variance X-ray" of the brain | A continuous geometric animation guided by behavior |
| **Time** | Ignored — shuffle the data, same result | Essential — defines positive/negative pairs |
| **What survives** | Whatever has largest amplitude | Whatever has clearest behavioral structure |
| **Suitable for** | Static population coding analysis | Dynamical systems analysis (Lie algebra, jPCA-like rotation detection, state-space trajectory modeling) |

This is the foundational reason why the Lie algebra rotational dynamics fit successfully in CEBRA space but would be swamped by static noise in raw or PCA-reduced space.

---

### 11.2 CEBRA vs jPCA (behavior-driven vs variance-driven rotation) (behavior-driven vs variance-driven rotation)

**jPCA** is an extension of PCA specifically designed for rotational dynamics. It works in two steps:

1. **PCA:** Reduce to top K principal components (variance-maximizing)
2. **Skew-symmetric fit:** Fit an autonomous rotational system $\dot{x} = M_{\mathrm{skew}} \cdot x$ in the PCA-reduced space

The critical weakness is **Step 1**. If the behaviorally relevant signal does not dominate the raw variance — which is common when global state variables (arousal, anesthesia depth, reward) fluctuate strongly — it will be submerged in the discarded PCA dimensions before jPCA ever sees it. jPCA's rotational fit then operates on dimensions dominated by non-behavioral variance, and the tracking-related rotation may be entirely invisible.

**CEBRA avoids this entirely.** Because the behavioral label drives the embedding from the start, CEBRA extracts the sensorimotor manifold regardless of its variance rank in raw neural space. This is why CEBRA-embedded Lie analysis (Sections 1.3–1.4) often shows higher and cleaner rotational structure than the raw-space analysis (Sections 1.1–1.2).

---

### 11.3 CEBRA vs dPCA (preserving topology vs flattening manifolds)

**dPCA (Demixed PCA)** is also supervised — it uses behavioral labels to find "demixed" axes (e.g., X-axis = pure position, Y-axis = pure time). But it achieves this via **linear orthogonal decomposition**: it forces the latent dimensions to be mutually orthogonal and purely linear combinations of the input.

The problem: real neural manifolds are often **curved**. Neurons encode continuous variables through phase differences, forming rings, tori, or other curved topologies. dPCA takes this curved manifold and **flattens it into orthogonal straight lines**. Once the curvature is destroyed:

- The rotational structure of the dynamics is lost
- Lie algebra fitting can no longer detect a clean $J_{\mathrm{skew}}$
- The skewness ratio drops toward noise levels

**CEBRA's key advantage for Lie algebra analysis** is that its loss functions (Euclidean or cosine distance) **preserve geometric curvature and topology**. A ring-shaped neural code remains a ring in the embedding; a toroidal code remains a torus. This is the rigid prerequisite for fitting rotational ($J_{\mathrm{skew}}$) dynamics — you cannot detect rotation on a flattened manifold any more than you can detect the curvature of a crumpled map.

**Summary of the three-way comparison:**

| | CEBRA | jPCA | dPCA |
|---|---|---|---|
| **Supervision** | Behavioral labels | None (PCA step) | Behavioral labels |
| **Nonlinear?** | Yes | No | No |
| **Signal extraction** | Precise — pulls out behavior-relevant structure regardless of variance rank | Blind — only keeps top-variance PCs; behavior signal may be lost | Moderate — uses labels but limited to linear decomposition |
| **Topology preservation** | Yes — preserves rings, tori, curvature | Partial — PCA preserves linear structure but discards nonlinear curvature | **No** — flattens curved manifolds into orthogonal axes, destroying rotation geometry |
| **Why it works or fails for Lie algebra** | **Best suited:** extracts the sensorimotor manifold while preserving the continuous geometry needed for rotational dynamics | **Fails** when behavior signal has low raw variance | **Fails** because flattened manifolds have no rotational structure to detect |

CEBRA is currently the only method that simultaneously achieves **precision filtering of behaviorally relevant signals** and **faithful preservation of continuous geometric topology** — the two necessary conditions for fitting interpretable Lie algebra rotational dynamics.

---

### 11.4 What happens at different embedding dimensions?

The current pipeline uses `CEBRA_EMBEDDING_DIM = 3`. This choice has specific mathematical and physical consequences.

#### 3D: the minimal sufficient embedding for a 1D continuous variable

A 1D continuous behavioral variable (e.g., horizontal velocity or position) has an intrinsic manifold that is topologically a **circle** (for cyclic position/frequency) or a **line segment** (for bounded velocity/position ranges). The minimal embedding that can fully contain a 2D rotational plane — one complex-conjugate eigenvalue pair — is **3 dimensions** (2 for the rotation plane + 1 for drift/leak orthogonal to it).

In 3D, the $J$ matrix is 3×3 with 9 parameters, and mathematically supports at most **1 rotational plane** (one pair of conjugate complex eigenvalues). This is the cleanest setting: the single rotation maps directly to the behavioral drive, with minimal confounding.

**Advantages of 3D:**
- Physically interpretable — the rotation can be visualized as a single plane
- Minimal overfitting risk — 9 J parameters + 9 L parameters = 18 total
- Rotation is "concentrated" — SR is typically at its highest because CEBRA forces all behaviorally relevant structure into 3 dimensions
- The leak term's 3×3 autoregressive matrix can exhaustively model simple autonomous dynamics

**Disadvantages:**
- R²_drive is often near zero (as observed) because 3 dimensions leave little residual variance beyond what the leak term captures
- Any secondary oscillatory signals (breathing, licking frequency, pupil fluctuations) are squeezed out or merged with the primary drive

#### 6D–8D: multiple rotational planes emerge

With 6 or 8 dimensions, $J$ becomes 36–64 parameters. This supports **2–4 independent rotational planes** (pairs of complex-conjugate eigenvalues), each potentially at a different frequency.

**What you would likely observe:**

1. **Multiple rotational planes:** The primary plane remains velocity/position-driven. Secondary planes may capture other oscillatory behavioral signals — breathing rhythm, whisking, licking — that are partially correlated with the primary drive but have distinct frequencies.

2. **R² increases (both true and shuffle):** More parameters = better fit. Total R² will rise, but so will shuffle R². The gap between them (R²_drive) may or may not widen.

3. **Overfitting risk:** At higher dimensions, the model can fit noise structure into its rotational planes. This makes the **R²_drive gate** (true > shuffle for the drive-specific component) more important, not less. You must verify that the rotation you detect is beyond what random label permutations would produce.

4. **SR dilution:** In 3D, CEBRA forces all behaviorally relevant structure into those 3 dimensions, concentrating the rotational signal (SR often > 0.8). In higher dimensions, the extra dimensions are dominated by non-rotational drift or noise, and the overall SR — averaged across all dimensions — will drop, even though the rotational planes themselves remain strong.

#### 12D+: diminishing returns and noise amplification

At 12+ dimensions (144 J parameters), the model can fit essentially any structure in the data. The risk becomes:

- Shuffle R² catches up to true R² as noise-fitting capacity increases
- Eigenvalue spectra become diffuse and hard to interpret
- Tracking vs. Playback differences may be swamped by high-dimensional noise
- The elegant geometric interpretation of J_skew (a single rotational plane gated by behavior) breaks down into a tangle of weakly rotating components

**Practical recommendation:** 3D is the right setting for establishing the existence and strength of the primary rotational structure. If R²_drive ≈ 0 in 3D (as currently observed), testing at 6D or 8D can determine whether this is a compression artifact or a genuine feature of the neural dynamics. If R²_drive remains near zero even at 8D, the rotational structure is real but fundamentally small in magnitude relative to the autonomous leak dynamics — a meaningful scientific result in its own right.

---

### 11.5 The dimensionality tradeoff — summary

| Dimension | J parameters | Rotational planes | SR | R²_drive | Overfitting risk | Interpretability |
|-----------|-------------|-------------------|-----|----------|-----------------|-----------------|
| **3D** | 9 | 1 | Highest | ~0 | Minimal | Best — single rotation plane, directly visualizable |
| **6–8D** | 36–64 | 2–4 | Moderate | May rise | Moderate — need strict R²_drive gate | Good — primary + secondary oscillatory modes |
| **12D+** | 144+ | Many | Diluted | May rise further | High — noise planes can mimic rotation | Poor — diffuse eigenvalue spectra |

**The key insight:** 3D CEBRA's R²_drive ≈ 0 (with high SR) is not a failure — it is the expected behavior when a minimal embedding concentrates rotational structure into few parameters, leaving the leak term dominant. This is analogous to compressing a high-resolution image: the essential shape (SR) is preserved, but the fine-grained variance (R²_drive) is lost to compression. Higher embedding dimensions can recover this variance at the cost of interpretability and increased overfitting risk.
