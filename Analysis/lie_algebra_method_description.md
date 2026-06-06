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

4. **Time-shuffled control:** For each epoch, the behavioral labels are randomly permuted 10 times (`N_SHUFFLES = 10`), and the Lie algebra is re-fit on each permutation (same CEBRA embedding, shuffled labels). The shuffle metrics (`SR_shuffle`, `R2_shuffle`, `R2_drive_shuffle`) are averaged across the 10 realizations to reduce baseline noise.

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
