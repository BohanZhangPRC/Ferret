# Lie Algebra and Rotational Dynamics for Sensorimotor Integration

This document describes the mathematical foundation, modeling pipeline, and physical interpretation of the Lie Algebra / Rotational Dynamics method used to analyze neural population data in this project.

## 1. Project Context & Motivation

The overarching goal is to understand how the primary auditory cortex (A1) and premotor cortex (PMC) implement an internal **forward model** during a continuous sensorimotor task. A ferret operates in a **closed-loop (Tracking)** condition where its continuous head velocity directly controls the frequency of an acoustic note, contrasted with an **open-loop (Playback)** condition where the same acoustic sequence is replayed without motor control.

Modeling the neural state evolution $\frac{dR}{dt}$ as a system driven by an external motor command $x(t)$ follows directly from the classical computational framework of **continuous state estimation for sensorimotor integration**, formalized by **Wolpert, Ghahramani, & Jordan (1995, *Science*)**. In this framework, the brain continuously uses efference copies of motor commands to predict and cancel the sensory consequences of self-generated actions. The Lie algebra model operationalizes this idea geometrically: the motor command does not merely modulate neural gain — it actively *steers* the neural state along a structured manifold via the skew-symmetric generator $J_{\mathrm{skew}}$.

A functional forward model would need to accomplish two things simultaneously:
1. **Continuous Prediction Updating:** Use the ongoing motor command (head kinematics) to continuously update the prediction of the upcoming sensory state (acoustic frequency) — a canonical cortical computation formalized by **Keller & Mrsic-Flogel (2018, *Neuron*)** as predictive processing.
2. **Sensory Attenuation:** Suppress the predictable sensory consequences of self-generated movement to filter out reafferent noise — first demonstrated at the single-neuron level in primate auditory cortex by **Eliades & Wang (2008, *Nature*)**.

The mathematical framework below captures both phenomena geometrically within the neural population's state space. Note that eigenvalue |Real| (dissipation) and |Imag| (rotation) can be interpreted as signatures of sensory attenuation and predictive updating respectively, but these are **hypotheses to be tested**, not confirmed facts.

**Behavioral context:** In the closed-loop sensorimotor paradigm used here (cf. Shamma et al., 2021+, on active sensing and sensorimotor interactions), the auditory cortex is not a passive feature extractor — it is an active hub dynamically reshaped by motor feedback. **Schneider & Mooney (2018, *Annu. Rev. Neurosci.*)** reviewed the converging evidence that motor-related signals globally modulate auditory cortical processing across species, establishing the biological plausibility of motor-to-auditory transformations as a general principle. Complementary engineering models such as **MirrorNet** (which learns audio synthesizer controls inspired by sensorimotor interaction) demonstrate that closed-loop motor-auditory architectures can self-organize structured internal representations. The contrast between Tracking and Playback conditions is therefore not merely a control for attention or arousal; it is a direct test of whether the motor-to-sensory transformation operates as a structured geometric operation (Lie group rotation) rather than unstructured gain modulation.

---

## 2. Foundational Math & Model

We model the neural population dynamics as an input-driven continuous linear dynamical system, following the dynamical systems perspective on cortical computation articulated by **Shenoy, Sahani, & Churchland (2013, *Annu. Rev. Neurosci.*)**:

$$
\frac{dR}{dt} = J_{\mathrm{skew}} \cdot R \cdot x(t) + L \cdot R
$$

- **$R(t)$**: The continuous neural population state — either raw firing rates (`(time, neurons)`) or a low-dimensional CEBRA embedding (typically 3D).
- **$x(t)$**: The continuous external behavioral drive (e.g., instantaneous head velocity `Velocity_x` or cumulative `Position`).
- **$J_{\mathrm{skew}}$**: The skew-symmetric (rotational) component of the fitted generator — responsible for rotating the neural state in response to the behavioral drive.
- **$L$**: An unconstrained linear "leak" term that captures baseline decay, drift, or restoring forces independent of the behavioral drive. It handles the natural dynamics of the neural system when $x(t) = 0$.

**Anatomical basis for the motor drive $x(t)$:** The biological plausibility of a motor-command signal directly gating auditory cortical dynamics is supported by established neuroanatomy. **Nelson, Schneider, & Mooney (2013, *J. Neurosci.*)** demonstrated that the secondary motor cortex (M2 / PMC) sends direct, monosynaptic excitatory projections to the primary auditory cortex (A1). These projections carry motor-related signals — including corollary discharge of movement commands — that can act as the biophysical implementation of $x(t)$: a behaviorally-gated input that dynamically modulates the gain and phase of auditory representations. In the Lie algebra framework, this anatomical pathway provides the physical conduit through which head velocity becomes the gating variable that steers the generator $J_{\mathrm{skew}}$.

---

### Lie Groups and Lie Algebras — what they are and why they matter here

This section unpacks the terms "Lie group" and "Lie algebra" from first principles so that the connection to neural dynamics becomes concrete and intuitive.

#### The core idea, in one sentence

> A **Lie group** is a smooth curved space of *states*. A **Lie algebra** is the flat tangent space of *velocities* at each point.

Everything that follows is an elaboration of this one sentence, with examples and neural context.

#### 1. What is a Lie group?

A Lie group is a set of transformations that is simultaneously a **smooth manifold** (you can move continuously through it) and a **group** (you can compose and invert transformations).

**Example: rotations in 2D.** The set of all rotation angles $\theta \in [0, 2\pi)$ forms a circle $S^1$. You can rotate from any angle to any other angle continuously (manifold). You can compose two rotations ($\theta_1 + \theta_2$) and invert a rotation ($-\theta$) — this is the group structure. The circle is a 1-dimensional Lie group.

**Example: rotations in 3D.** The set of all 3D rotation matrices forms the Lie group $SO(3)$. It is a 3-dimensional curved space (not a flat 3D volume — it is a projective space where opposite points are identified, like folding a ball in half). Each point on this manifold is a complete orientation.

**In the neural context**, we are asking: *does the neural population state $R(t)$ live on (or near) a Lie group manifold?* If so, the brain is using rotational transformations to encode and update sensory predictions — moving the neural state along a curved manifold in a structured, reversible way rather than pushing it arbitrarily through state space.

#### 2. What is a Lie algebra?

The Lie algebra is the **tangent space** at the identity element of a Lie group. It describes all possible *infinitesimal* transformations — the velocities you can have while staying on the manifold.

**Example: 2D rotations.** The Lie group is the circle of angles $\theta$. The Lie algebra is the tangent line at $\theta = 0$ — it has one dimension (angular velocity $\omega = d\theta/dt$). Any rotation can be reached by integrating constant angular velocity: $\theta(t) = \theta(0) + \omega \cdot t$.

**Example: 3D rotations.** The Lie algebra $\mathfrak{so}(3)$ is the set of all $3 \times 3$ **skew-symmetric** matrices (matrices satisfying $M^T = -M$). Any such matrix has the form:

$$
M = \begin{bmatrix} 0 & -c & b \\ c & 0 & -a \\ -b & a & 0 \end{bmatrix}
$$

This matrix has exactly 3 independent parameters $(a, b, c)$ — matching the 3 degrees of freedom of 3D rotation (pitch, yaw, roll). The action of this matrix on a vector $\mathbf{v}$ produces $\mathbf{v} \times (a, b, c)$ — the cross product, which rotates $\mathbf{v}$ around the axis $(a, b, c)$ at angular speed $\|(a, b, c)\|$.

**The defining property of a Lie algebra is the commutator (Lie bracket):**

$$ [A, B] = AB - BA $$

This measures how much two transformations fail to commute — how much the result depends on the order in which you apply them. For skew-symmetric matrices, $[A, B]$ is also skew-symmetric. The Lie bracket captures the non-commutative geometry of the manifold.

#### 3. How they connect: exponential map

The bridge from Lie algebra to Lie group is the **matrix exponential**:

$$ R(\theta) = \exp(\theta \cdot M) = I + \theta M + \frac{\theta^2}{2!} M^2 + \cdots $$

- The Lie algebra element $M$ (an infinitesimal rotation) is exponentiated to produce a finite rotation $R(\theta)$ on the Lie group.
- Conversely, the Lie algebra can be recovered by taking the derivative at $\theta = 0$: $M = \left.\frac{dR}{d\theta}\right|_{\theta=0}$.

This is exactly the relationship between position and velocity in physics: integrate velocity (Lie algebra) to get position (Lie group); differentiate position to get velocity. The key difference is that on a curved manifold, the velocity at each point lives in a tangent space that is *tilted* relative to the tangent space at other points — you cannot simply add velocities from different locations.

#### 4. Why this matters for the neural analysis

The model equation $dR/dt = J_{\mathrm{skew}} \cdot R \cdot x(t) + L \cdot R$ can now be read with group-theoretic eyes:

| Term | Mathematical role | Physical meaning |
|------|------------------|-----------------|
| $R(t)$ | Point on (or near) a Lie group manifold | Neural population state at time $t$ |
| $dR/dt$ | Tangent vector — an element of the Lie algebra | How the neural state is changing right now |
| $J_{\mathrm{skew}}$ | **Generator** — maps behavioral drive $x(t)$ into the Lie algebra | The "steering matrix": how head velocity translates into neural rotation |
| $L \cdot R$ | Non-rotational flow (drift, decay, leak) | Autonomous dynamics not driven by behavior |

The crucial structural constraint is that $J_{\mathrm{skew}}$ is **skew-symmetric** ($J^T = -J$). This is not an arbitrary choice:

- Skew-symmetric matrices are exactly the Lie algebra elements of the rotation group $SO(N)$.
- When $x(t) \neq 0$, the term $J_{\mathrm{skew}} \cdot R \cdot x(t)$ generates a *pure rotation* of the neural state — it moves $R(t)$ along the manifold without changing its norm or stretching the geometry.
- The matrix exponential $\exp(J_{\mathrm{skew}} \cdot x \cdot \Delta t)$ is a rotation matrix. Integrating this over time traces a curved trajectory on the manifold.

**If $J_{\mathrm{skew}}$ is large (high Skewness Ratio)**, the behavioral drive primarily rotates the neural state — the brain uses a Lie-group-like rotational code for sensorimotor prediction. **If $J_{\mathrm{skew}}$ is small (low SR)**, the drive mostly scales or dissipates the state — closer to a leaky integrator than a rotational transformer.

#### 5. Analogy: a car's steering wheel

| Concept | Car analogy | Neural analogy |
|---------|------------|----------------|
| **Lie group** | The angle of the steering wheel (a circle $S^1$) | The neural population state $R(t)$ on its manifold |
| **Lie algebra** | Angular velocity of the wheel (1 number: $\omega$) | $J_{\mathrm{skew}} \cdot R \cdot x(t)$ — the rotational push from behavior |
| **Generator $J$** | The steering column — converts hand torque ($x$) into wheel rotation | How neural circuits convert head velocity into a structured rotation of the population state |
| **Skew-symmetry** | A steering wheel turns left and right symmetrically — turning left by $\theta$ and then right by $\theta$ returns you to the start | $J_{\mathrm{skew}}^T = -J_{\mathrm{skew}}$ ensures the rotation is *reversible*: forward and backward movements cancel |
| **Leak $L$** | Friction in the steering column — the wheel slowly returns to center if you let go | Passive decay of neural activity back toward baseline |
| **SR ≈ 1** | The steering column is well-lubricated: your hand movement goes almost entirely into rotation | The behavioral drive is efficiently coupled to rotational neural dynamics |
| **SR ≈ 0** | The steering column is rusty: most force goes into friction and vibration instead of rotation | The drive mostly produces non-rotational drift or noise |

#### 6. Why not just use PCA or a generic linear model?

A generic linear model $dR/dt = A \cdot R$ would fit *any* matrix $A$ — symmetric, skew-symmetric, or a mix. It would capture rotational dynamics, but it would also fit scaling, shearing, and pure noise with equal enthusiasm. By explicitly decomposing $J_{\mathrm{ols}}$ into its skew-symmetric ($J_{\mathrm{skew}}$) and symmetric components, and computing SR and eigenvalues, we ask a specific structured question: *is the behavioral drive generating a group-like rotation of the neural manifold?*

This is a stronger scientific claim than "the drive modulates neural activity." It asserts that the modulation follows a specific geometric form — a Lie group action — which carries implications for how the brain organizes sensorimotor transformations: as structured, reversible, manifold-preserving operations rather than arbitrary gain modulation.

---

**How the fitting works (OLS) — step by step:**

#### Step 1: Compute the target (dependent variable)

The derivative of the neural state is estimated via discrete finite differences:

```python
dr_dt = np.gradient(r, dt, axis=0)   # shape (T, N)
# dr_dt[t, j] ≈ (R[t+1, j] - R[t-1, j]) / (2 * dt)
```

This is what the model must predict — how each neuron's activity changes from one moment to the next.

**Smoothing note:** `np.gradient` on raw spike-rate data amplifies high-frequency Poisson noise. While the current pipeline operates directly on binned and CEBRA-smoothed embeddings (which mitigates this), a principled upgrade for raw-space analyses is to pre-smooth neural trajectories via **Gaussian Process Factor Analysis (vGPFA)** or **Kalman smoothing** before derivative estimation. These methods extract continuous latent trajectories under a Poisson-like observation model, providing low-noise $R(t)$ and analytically derived $\dot{R}(t)$ without finite-difference artifacts.

#### Step 2: Build the design matrix (independent variables)

The model equation $dR/dt = J \cdot (R \cdot x) + L \cdot R$ says the derivative at time $t$ depends on two things:
- **$R \cdot x$** — the neural state at time $t$, gated (multiplied) by the behavioral drive at time $t$
- **$R$** — the neural state at time $t$, acting autonomously

These become the two predictor blocks, stacked side-by-side:

```python
U_rot = r * x_dot[:, None]      # (T, N): each column = r[:, j] * x_dot
U     = np.hstack([U_rot, r])   # (T, 2N): rotation block + leak block
```

Visually, one row of the design matrix `U` (a single timepoint $t$) looks like:

```
Column indices:    0        1        ...   N-1    |    N     N+1    ...   2N-1
                 ───────── rotation features ─────   ───────── leak features ─────────

U[t, :]  =  [ r[t,0]·x[t]  r[t,1]·x[t]  ...  r[t,N-1]·x[t]  |  r[t,0]  r[t,1]  ...  r[t,N-1] ]
```

The first $N$ columns ask: "how does each neuron's activity, weighted by the current velocity, predict the derivative?" The second $N$ columns ask: "how does each neuron's unweighted activity predict the derivative?"

#### Step 3: Solve the multi-output linear system

```python
weights, residuals, rank, singular_values = np.linalg.lstsq(U, dr_dt, rcond=None)
# weights: (2N, N)  — the full solution matrix
# We ignore residuals, rank, and singular_values (not needed for metrics)
```

`lstsq` solves $U \cdot W = dr\_dt$ for $W$, minimizing the sum of squared residuals across **all $N$ target neurons simultaneously**. Each of the $N$ columns of `dr_dt` is a separate regression target, and each of the $2N$ columns of `U` is a separate predictor. The result `weights` is a $(2N \times N)$ matrix — it tells you how to combine the $2N$ predictors to best predict each of the $N$ targets.

Equivalent formulation — for each target neuron $j$ (column of `dr_dt`), `lstsq` finds the $2N$ coefficients in `weights[:, j]` that minimize:

$$ \sum_t \left( dr\_dt[t, j] - \sum_{k=0}^{N-1} J_{k,j} \cdot r[t,k] \cdot x[t] - \sum_{k=0}^{N-1} L_{k,j} \cdot r[t,k] \right)^2 $$

This is **not $N$ separate scalar regressions** — `lstsq` solves all $N$ target columns in a single matrix operation, leveraging any shared structure across the predictor matrix $U$.

#### Step 4: Extract the generator and leak matrices

```python
N = r.shape[1]                     # number of neurons / embedding dimensions

J_ols  = weights[:N, :]            # (N, N) — first N rows = rotation generator
J_skew = 0.5 * (J_ols - J_ols.T)   # (N, N) — skew-symmetric (purely rotational) component
L      = weights[N:, :]            # (N, N) — remaining N rows = leak matrix
```

**Why the skew-symmetrization?** Any real square matrix can be uniquely decomposed into a symmetric part and a skew-symmetric part:

$$ J_{\mathrm{ols}} = \underbrace{\frac{1}{2}(J_{\mathrm{ols}} - J_{\mathrm{ols}}^T)}_{\text{skew-symmetric (rotation)}} + \underbrace{\frac{1}{2}(J_{\mathrm{ols}} + J_{\mathrm{ols}}^T)}_{\text{symmetric (scaling/dissipation)}} $$

The skew-symmetric part $J_{\mathrm{skew}}$ satisfies $J_{\mathrm{skew}}^T = -J_{\mathrm{skew}}$. Its diagonal is always zero (a neuron does not rotationally drive *itself*), and each off-diagonal pair encodes a rotational coupling: $J_{\mathrm{skew}}[i,j] = -J_{\mathrm{skew}}[j,i]$ means neurons $i$ and $j$ form part of a rotational plane. The symmetric part is discarded because it represents scaling, stretching, or pure dissipation — not rotation.

$L$ captures the autonomous dynamics: it tells you how the neural state evolves on its own, independent of the behavioral drive $x(t)$.

#### Step 5: Reconstruct the prediction and compute metrics

```python
# Prediction from the full model (rotation + leak)
dR_pred = (r * x_dot[:, None]) @ J_skew.T + r @ L.T   # (T, N)

# Residuals and total variance
ss_res = np.sum((dr_dt - dR_pred) ** 2)                # sum of squared residuals
ss_tot = np.sum((dr_dt - np.mean(dr_dt, axis=0)) ** 2) # total sum of squares

# Total R²
r2 = 1 - ss_res / ss_tot

# Drive-specific R² (over leak-only baseline)
dR_leak = r @ L.T                                      # prediction from L alone
ss_leak = np.sum((dr_dt - dR_leak) ** 2)
r2_drive = 1 - ss_res / ss_leak

# Skewness ratio
sr = np.linalg.norm(J_skew) / np.linalg.norm(J_ols)
```

The OLS variant (`LIE_METHOD = "lstsq"`) and the PyTorch variant (`"pytorch"`) differ only in how $J_{\mathrm{skew}}$ and $L$ are obtained (post-hoc skew-symmetrization vs. constrained gradient descent with $J = W - W^T$). Both produce functionally equivalent $J_{\mathrm{skew}}$, SR, R², and R²_drive metrics. The current pipeline uses `"lstsq"` by default for speed and reproducibility.

**Optimization Caveat — post-hoc skew-symmetrization vs. constrained fitting:** The OLS approach solves for an unconstrained $J_{\mathrm{ols}}$ and then extracts $J_{\mathrm{skew}} = 0.5 \cdot (J_{\mathrm{ols}} - J_{\mathrm{ols}}^T)$. This yields the *skew-symmetric projection of the unconstrained optimum*, which is **not guaranteed to be the constrained global optimum** — i.e., the best possible skew-symmetric matrix under the Lie algebra constraint may differ from the post-hoc projection. The PyTorch variant (`_fit_lie_pytorch`) addresses this by parameterizing $J = W - W^T$ directly and optimizing under the constraint via gradient descent. A planned future upgrade is to use PyTorch-constrained fitting as the default, solving $\min_{J^T = -J, L} \|dR/dt - J \cdot (R \cdot x) - L \cdot R\|^2$ exactly on the Lie algebra manifold.

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

**Biological mechanism for $|Real|$ (dissipation):** The interpretation of the real eigenvalue component as sensory attenuation has a concrete synaptic-level candidate. **Schneider, Nelson, & Mooney (2014, *Nature*)** demonstrated that motor-corollary discharge signals recruit local parvalbumin-positive (PV+) inhibitory interneurons in A1, producing precisely timed suppression of auditory responses to self-generated sounds. In the geometric framework, this inhibition corresponds to the negative real part of $J_{\mathrm{ols}}$ eigenvalues — an active contraction of the neural state amplitude along specific manifold directions, gated by the motor command $x(t)$. The $|Real|$ metric is thus not an abstract mathematical quantity but a population-level readout of this disynaptic inhibitory motif.

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

5. **R² gate:** A session-condition is considered to have meaningful rotational structure only if $R^2_{\mathrm{true}} > R^2_{\mathrm{shuffle}}$ (total R²) **and** $R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, shuffle}}$ (drive-specific R²). **The drive-specific gate ($R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, shuffle}}$) is the primary validation criterion.** In low-dimensional embeddings (e.g., 3D CEBRA), the leak term $L \cdot R$ dominates the total R², making the total R² gate weak and potentially misleading. The $R^2_{\mathrm{drive}}$ gate isolates the behavioral drive contribution and is the stricter, more interpretable threshold: it directly answers whether the time-aligned motor command explains additional derivative variance beyond what the autonomous leak dynamics already capture.

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

### Extending to Multi-Input Bilinear Dynamics (Bilinear LDS)

The current model treats the behavioral drive $x(t)$ as a 1D scalar (e.g., head velocity). This is a deliberate simplification that maximizes interpretability — a single rotational plane with a single gating variable. However, real sensorimotor behavior involves multiple simultaneous signals: velocity, acceleration, cumulative position, frequency error, and their lagged histories. A natural extension that preserves geometric interpretability while adding richness is the **multi-input bilinear dynamical system (Bilinear LDS)**:

$$ \dot{z} = A_0 z + \sum_{k} u_k(t) \cdot A_k z $$

where:
- $z(t)$ is the latent neural state (e.g., CEBRA embedding)
- $A_0$ is the autonomous dynamics matrix (analogous to $L$)
- $u_k(t)$ are multiple behavioral inputs (velocity, acceleration, position, error, etc.)
- Each $A_k$ is independently decomposable into skew-symmetric ($J_k$) and symmetric ($S_k$) components: $A_k = J_k + S_k$

This framework generalizes the single-drive Lie algebra model to a **multi-channel behavioral gating system** while preserving the core geometric structure: each $A_k$ has its own SR, eigenvalue spectrum, and can be tested for rotational dominance independently. It connects to the modern literature on **adaptive unitary state-space models** (cf. NeurIPS 2025 unitary SSM lineage) and continuous-time bilinear recurrent neural networks, positioning the current 1D-drive model as the minimal interpretable case within a broader class of structured sensorimotor dynamics models.

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

### The strongest negative control: Shuffle-Label CEBRA

The time-shuffled control described above re-fits only the OLS Lie algebra step on a fixed CEBRA embedding. This leaves a potential vulnerability: **CEBRA itself is trained with behavioral labels**, so the embedding manifold may have been shaped by the same behavioral structure that the Lie algebra then detects. A skeptic could argue that any contrastive embedding trained on velocity labels will produce a manifold that *looks* rotational under a Lie algebra fit, regardless of whether the underlying neural dynamics are genuinely rotational.

The gold-standard negative control to address this is the **Shuffle-Label CEBRA Control**:

1. **Train a dummy CEBRA model** on the same neural data but with **fully shuffled behavioral labels** — the time structure and behavioral contingency are destroyed before the embedding is learned.
2. **Fit the Lie algebra** in this dummy embedding space, computing SR, $R^2$, and $R^2_{\mathrm{drive}}$ exactly as in the true pipeline.
3. **Compare:** The true-embedding metrics must be significantly higher than the dummy-embedding metrics for the rotational dynamics claim to hold.

If $SR_{\mathrm{true}} > SR_{\mathrm{dummy}}$ **and** $R^2_{\mathrm{drive, true}} > R^2_{\mathrm{drive, dummy}}$, then the rotational structure is a genuine property of the neural population dynamics, not an artifact of the InfoNCE loss function imprinting rotational topology onto the embedding. This control is computationally expensive (requires retraining CEBRA for each session) and is planned as a confirmatory analysis for the final publication version of this pipeline.

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

**CEBRA** (**Schneider, Lee, & Mathis, 2023, *Nature***) uses contrastive learning (InfoNCE loss):

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

### 11.2 CEBRA vs jPCA (behavior-driven vs variance-driven rotation)

**jPCA** (**Churchland et al., 2012, *Nature***) is an extension of PCA specifically designed for rotational dynamics. It fits an **autonomous** (no external drive) rotational system $\dot{x} = M_{\mathrm{skew}} \cdot x$ after PCA-based dimensionality reduction. It works in two steps:

1. **PCA:** Reduce to top K principal components (variance-maximizing)
2. **Skew-symmetric fit:** Fit the autonomous rotational system in the PCA-reduced space

The critical weakness is **Step 1**. If the behaviorally relevant signal does not dominate the raw variance — which is common when global state variables (arousal, anesthesia depth, reward) fluctuate strongly — it will be submerged in the discarded PCA dimensions before jPCA ever sees it. jPCA's rotational fit then operates on dimensions dominated by non-behavioral variance, and the tracking-related rotation may be entirely invisible.

**CEBRA avoids this entirely.** Because the behavioral label drives the embedding from the start, CEBRA extracts the sensorimotor manifold regardless of its variance rank in raw neural space. This is why CEBRA-embedded Lie analysis (Sections 1.3–1.4) often shows higher and cleaner rotational structure than the raw-space analysis (Sections 1.1–1.2).

---

### 11.3 dPCA (Demixed PCA — linear supervised dimensionality reduction)

dPCA (**Kobak et al., 2016, *eLife***) is a **linear supervised dimensionality reduction** method. It uses task labels (stimulus, decision, time) to find "demixed" low-dimensional axes — e.g., the first dPC isolates stimulus-dependent variance, the second isolates decision-dependent variance, and so on. Unlike standard PCA (unsupervised, variance-maximizing), dPCA incorporates the experimental design into the decomposition.

**Why it is not used in place of CEBRA here:** dPCA achieves demixing via **regularized linear regression** — the latent axes are constrained to be orthogonal linear combinations of the input neurons. This linearity has two consequences relevant to this pipeline:

1. **It preserves only linear structure.** Real neural manifolds encoding continuous variables (velocity, position) are often curved — neurons form rings, tori, or helical trajectories through phase-coded representations. A purely linear decomposition will not fully capture this curvature, and the rotational geometry that Lie algebra fitting relies on may be weakened or distorted.

2. **It is still dimensionality reduction.** dPCA reduces from N neurons to K dPCs — it is directly in the same category as PCA, jPCA, and CEBRA-embedding for the purpose of building a low-dimensional manifold. The key difference is *what* it optimizes for (demixed variance partitioning) and *how* (linear orthogonal axes).

**Comparison to CEBRA for Lie algebra analysis:**

| | CEBRA | dPCA |
|---|---|---|
| **Reduction type** | Nonlinear (deep network) | Linear (regularized regression) |
| **Supervision** | Behavioral labels (velocity, position, time) | Task parameter labels |
| **Topology preservation** | Yes — preserves rings, tori, curved manifolds | Partial — linear transformation cannot reproduce nonlinear curvature |
| **Suitability for Lie algebra** | Excellent — extracts behaviorally relevant manifold without distorting its geometry | Limited — linear axes may not capture the full rotational structure of curved sensorimotor manifolds |

**dPCA can complement this pipeline as a downstream analysis:** after CEBRA embedding and Lie algebra fitting, dPCA could be applied to the raw neural data to quantify what fraction of total population variance is explained by velocity vs. position vs. condition. This would enrich the interpretation without replacing CEBRA as the preprocessing step.

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

---

## 12. Limitations

This framework offers a geometrically interpretable window into sensorimotor neural dynamics, but it carries several limitations that should be weighed when interpreting results or planning follow-up analyses.

### 12.1 Circular reasoning risk between CEBRA embedding and Lie algebra fitting

**The concern:** CEBRA's contrastive learning (InfoNCE loss) uses the same behavioral variable $x(t)$ (e.g., velocity or position) that subsequently drives the Lie algebra model. The embedding manifold is thus *shaped in advance* by the behavioral labels. Fitting $J_{\mathrm{skew}}$ with the same $x(t)$ on this behaviorally-shaped manifold can produce high Skewness Ratios as an algorithmic artifact — the embedding geometry and the rotational fit are not independent. The question is whether the observed rotational structure reflects intrinsic neural population dynamics or the InfoNCE loss imprinting rotational topology onto the manifold.

**Mitigations in current pipeline:**

| Defense | How it helps | Residual concern |
|---------|-------------|-----------------|
| Time-shuffled control (Sec 9) | Destroys temporal alignment between $x(t)$ and $R(t)$ within the fixed embedding; tests whether the rotation is label-dependent | Does not rule out embedding-level artifacts — the manifold itself may have been shaped by behavioral labels during CEBRA training |
| Raw-space analysis (Sec 1.1–1.2) | Lie algebra fit in unsmoothed, unreduced neural space — no embedding bias | Raw space has lower SR and higher noise; results may differ in magnitude but should show consistent direction |
| Shuffle-Label CEBRA Control (Sec 9) | Retrains CEBRA on fully shuffled labels, then fits Lie algebra in the dummy embedding | Computationally expensive; planned as confirmatory analysis, not yet implemented in current pipeline |

**Comparison to standard methods:** jPCA (Churchland et al., 2012) searches for rotational dynamics in **unsupervised** PCA space — it avoids the circularity risk entirely by not using behavioral labels during dimensionality reduction, but can miss low-variance behaviorally relevant signals. CEBRA inverts this tradeoff: it recovers weak but behaviorally crucial signals at the cost of potential embedding bias.

### 12.2 Post-hoc skew-symmetrization is not the constrained optimum

The OLS pipeline solves for an unconstrained $J_{\mathrm{ols}}$ and then extracts $J_{\mathrm{skew}} = 0.5 \cdot (J_{\mathrm{ols}} - J_{\mathrm{ols}}^T)$. This yields the *skew-symmetric projection of the unconstrained optimum*, which is not guaranteed to be the best skew-symmetric matrix under the Lie algebra constraint. In rigorous Lie group dynamical systems (e.g., $SO(N)$ or $SE(N)$ models), the state is constrained *during* optimization, not post-hoc. The current approach minimizes the total prediction error, not the prediction error under the rotation-only hypothesis.

### 12.3 Low $R^2_{\mathrm{drive}}$ weakens causal interpretation

In the 3D CEBRA pipeline, $R^2_{\mathrm{drive}}$ is typically near zero — the behavioral drive explains negligible *additional* derivative variance beyond the autonomous leak term. While this can be partially attributed to compression (Sec 11.4), it remains a vulnerability: in standard input-driven dynamical systems modeling, a drive term that contributes near-zero incremental $R^2$ faces scrutiny as to whether it is genuinely the mechanism propelling the system. The current framework is better characterized as a **geometric feature extractor** (quantifying the rotational *shape* of the generator) than a **generative physical dynamics model** (predicting state trajectories from the drive).

### 12.4 Scalar drive and linear gating assumptions

The model $dR/dt = J_{\mathrm{skew}} \cdot R \cdot x(t) + L \cdot R$ assumes:
- $x(t)$ is a **1D scalar** (e.g., head velocity)
- The gating is **purely multiplicative and linear** ($R \cdot x$)

Real sensorimotor integration is high-dimensional and nonlinear. A1 and PMC receive multidimensional proprioceptive feedback, top-down predictive error signals, and corollary discharge — not a single kinematic scalar. The scalar linear gating assumption is a deliberate simplification for interpretability, but it may miss structure (e.g., nonlinear gain modulation, multi-input interactions) that higher-capacity models would capture.

### 12.5 Lack of explicit noise and uncertainty modeling

The pipeline uses deterministic finite differences (`np.gradient`) and deterministic OLS regression. Neural spiking data has substantial Poisson noise and trial-to-trial variability. The gold standard in modern computational neuroscience (GPFA, LFADS, Kalman filtering variants) uses **probabilistic generative models** that explicitly separate process noise (uncertainty in latent state evolution) from observation noise (stochastic mapping from latent state to spikes). Computing derivatives directly on binned spike counts or embedding trajectories amplifies high-frequency noise, and OLS fits on noisy derivatives inherit this noise.

---

## 13. Future Directions

The following improvements would address the limitations identified above and move the framework from geometric feature extraction toward generative physical modeling.

### 13.1 End-to-end joint optimization of embedding and dynamics

**Current:** CEBRA embedding → Lie algebra fit (two-stage pipeline, shared behavioral labels).

**Proposed:** A single PyTorch model with a joint loss function:

$$ \mathcal{L} = \mathcal{L}_{\mathrm{InfoNCE}} + \lambda \cdot \mathcal{L}_{\mathrm{Dynamics}} $$

where $\mathcal{L}_{\mathrm{Dynamics}}$ penalizes deviations from $dR/dt = J_{\mathrm{skew}} \cdot R \cdot x + L \cdot R$. The encoder (CEBRA-like MLP) learns a latent representation $R$ that simultaneously satisfies the contrastive behavioral objective *and* the Lie algebra dynamical constraint. This breaks the circular reasoning concern: the embedding is not merely shaped by behavioral labels, but also by the requirement that its dynamics follow a physically interpretable equation.

### 13.2 Strict Lie group parameterization and constrained optimization

**Current:** OLS + post-hoc $J_{\mathrm{skew}} = 0.5 \cdot (J_{\mathrm{ols}} - J_{\mathrm{ols}}^T)$.

**Proposed:** Parameterize $J$ directly on the Lie algebra $\mathfrak{so}(N)$ (the space of skew-symmetric $N \times N$ matrices) using only the $N(N-1)/2$ independent parameters. Optimize under the constraint $J^T = -J$ using manifold-aware optimizers (e.g., Geoopt library). The state evolution can be expressed via the matrix exponential:

$$ R(t + \Delta t) = \exp(J_{\mathrm{skew}} \cdot x(t) \cdot \Delta t) \cdot R(t) $$

using `torch.matrix_exp`. This ensures the rotational geometry is preserved *throughout* optimization rather than enforced after the fact, and the learned $J_{\mathrm{skew}}$ is the true constrained optimum.

### 13.3 Neural ODE for continuous-time, noise-robust fitting

**Current:** Discrete `np.gradient` derivatives → OLS fitting.

**Proposed:** Replace the discrete gradient + OLS pipeline with a **Neural ODE** framework. Define the state evolution as:

$$ \frac{dR}{dt} = f_\theta(R(t), x(t)) $$

where $f_\theta$ contains the structured generator ($J_{\mathrm{skew}}$ and $L$) within a differentiable ODE solver (e.g., RK4 or Dormand-Prince). The ODE is integrated forward in time and the loss is computed against the observed neural trajectory. This approach:
- Eliminates finite-difference derivative noise
- Handles non-uniform sampling and missing data gracefully
- Fits parameters through the integrated trajectory, not instantaneous derivatives
- Provides a natural bridge to probabilistic extensions (Latent ODEs, variational inference over $R(t)$)

### 13.4 Nonlinear multi-dimensional forward model

**Current:** Scalar $x(t)$ linearly gates a single fixed $J_{\mathrm{skew}}$.

**Proposed:** Replace $J_{\mathrm{skew}} \cdot x(t)$ with a learned control network that maps multi-dimensional behavioral inputs to a linear combination of Lie algebra basis generators:

$$ J(t) = \sum_{k} w_k(\mathbf{x}_t) \cdot G_k $$

where $\mathbf{x}_t$ is a vector of kinematic variables (velocity, acceleration, position, error, and their lagged histories), $G_k$ are fixed skew-symmetric basis matrices spanning $\mathfrak{so}(N)$, and $w_k$ are scalar weights produced by a lightweight neural network. This generalizes the model from "a single steering wheel turning at variable speed" to "a cockpit with multiple control surfaces, each engaged differently depending on the behavioral context." The increase in $R^2_{\mathrm{drive}}$ from this expansion would directly test whether the current low drive variance is due to the scalar linear gating assumption rather than the absence of genuine motor-to-sensory drive.

### 13.5 Probabilistic generative extension

Embed the structured Lie dynamics within a probabilistic state-space model:

$$ R_{t+1} \sim \mathcal{N}(\exp(J_{\mathrm{skew}} \cdot x_t \cdot \Delta t) \cdot R_t + L \cdot R_t \cdot \Delta t, \; Q) $$
$$ y_t \sim \mathrm{Poisson}(f(R_t)) $$

where $Q$ is the process noise covariance and $f$ is a learned or fixed observation model mapping from latent state to spike counts. Inference via Kalman smoothing or variational autoencoders would provide uncertainty estimates on $J_{\mathrm{skew}}$ and allow model comparison (e.g., rotation model vs. non-rotation baseline) via marginal likelihood rather than point-estimate metrics.

---

## References

1. **Churchland, M.M., Cunningham, J.P., Kaufman, M.T., Foster, J.D., Nuyujukian, P., Ryu, S.I. and Shenoy, K.V.** (2012) 'Neural population dynamics during reaching', *Nature*, 487(7405), pp. 51–56. doi:10.1038/nature11129.

2. **Eliades, S.J. and Wang, X.** (2008) 'Neural substrates of vocalization-feedback monitoring in primate auditory cortex', *Nature*, 453(7198), pp. 1102–1106. doi:10.1038/nature06910.

3. **Keller, G.B. and Mrsic-Flogel, T.D.** (2018) 'Predictive processing: a canonical cortical computation', *Neuron*, 100(2), pp. 424–435. doi:10.1016/j.neuron.2018.10.003.

4. **Kobak, D., Brendel, W., Constantinidis, C., Feierstein, C.E., Kepecs, A., Mainen, Z.F., Qi, X.L., Romo, R., Uchida, N. and Machens, C.K.** (2016) 'Demixed principal component analysis of neural population data', *eLife*, 5, p. e10989. doi:10.7554/eLife.10989.

5. **Nelson, A., Schneider, D.M. and Mooney, R.** (2013) 'A circuit for motor cortical modulation of auditory cortical activity', *Journal of Neuroscience*, 33(36), pp. 14342–14353. doi:10.1523/JNEUROSCI.0935-13.2013.

6. **Schneider, D.M. and Mooney, R.** (2018) 'How movement modulates hearing', *Annual Review of Neuroscience*, 41, pp. 553–572. doi:10.1146/annurev-neuro-072116-031215.

7. **Schneider, D.M., Nelson, A. and Mooney, R.** (2014) 'A synaptic and circuit basis for corollary discharge in the auditory cortex', *Nature*, 513(7517), pp. 189–194. doi:10.1038/nature13724.

8. **Schneider, S., Lee, J.H. and Mathis, M.W.** (2023) 'Learnable latent embeddings for joint behavioural and neural analysis', *Nature*, 617(7960), pp. 360–368. doi:10.1038/s41586-023-06031-6.

9. **Shamma, S., Patel, P., Mukherjee, S., Marion, G., Khalighinejad, B., Han, C., Herrero, J., Bickel, S., Mehta, A. and Mesgarani, N.** (2021) 'Learning speech production and perception through sensorimotor interactions', *Cerebral Cortex Communications*, 2(1), p. tgaa091. doi:10.1093/texcom/tgaa091.

10. **Shenoy, K.V., Sahani, M. and Churchland, M.M.** (2013) 'Cortical control of arm movements: a dynamical systems perspective', *Annual Review of Neuroscience*, 36, pp. 337–359. doi:10.1146/annurev-neuro-062111-150509.

11. **Wolpert, D.M., Ghahramani, Z. and Jordan, M.I.** (1995) 'An internal model for sensorimotor integration', *Science*, 269(5232), pp. 1880–1882. doi:10.1126/science.7569931.

12. **Siriwardena, Y.M., Marion, G. and Shamma, S.** (2022) 'The MirrorNet: learning audio synthesizer controls inspired by sensorimotor interaction', *ICASSP 2022 — IEEE International Conference on Acoustics, Speech and Signal Processing*, pp. 946–950. doi:10.1109/ICASSP43922.2022.9747358.
