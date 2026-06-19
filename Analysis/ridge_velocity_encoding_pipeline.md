# Ridge Velocity Encoding: Pipeline Documentation

## Scientific Question

Does changing the frequency-to-position mapping (tracking_only → mapping_change) alter how well individual neurons encode movement velocity?

- **MP1 (tracking_only)**: animal controls frequency directly; no playback
- **MP2 (mapping_change)**: frequency-to-position mapping is changed; contains both tracking and playback phases

For each pair of sessions (same recording, two phases), we compare velocity encoding strength (R²) for the same neurons under both mappings.

---

## Data Flow

```
Google Sheet (SKIEUR tab, use=yes)
    └── 11 pairs: tracking_only ↔ mapping_change (via paired_session column)
          │
          ▼
NAS merged pickles (headstage 0 only):
    SKIEUR_hs_0_Tracking_only_0.005_data_ss    (spike-sorted)
    SKIEUR_hs_0_mapping_change_0.005_data_ss
          │
          ▼
Filter to tracking condition only:
    MP1: condition == -1  (~900 s tracking)
    MP2: condition == 0   (~1200 s tracking)
          │
          ▼
Per-neuron Ridge encoding (velocity → spike count):
    Build features: Speed_x ± 50 ms temporal lags (21 features)
    Block-shuffled 5-fold CV (1 s blocks) to prevent temporal leakage
          │
          ▼
Paired comparison: R²_mp1 vs R²_mp2 (same neuron, Wilcoxon signed-rank)
```

---

## Condition Encoding

| Session Type | Tracking | Playback | Notes |
|---|---|---|---|
| tracking_only | **-1** | — | Mislabeled by `create_tt`; all ~900 s are tracking |
| mapping_change | **0** | 1 | Cleanly separated: tracking first, then playback |

Only tracking time bins are used in both MP1 and MP2, for an apple-to-apple comparison.

---

## Ridge Encoding Model

### Design Matrix

- **Target (y)**: single neuron spike count (sqrt-transformed for variance stabilisation)
- **Features (X)**: `Speed_x` at times `[t−50ms, ..., t, ..., t+50ms]` → 21 columns
- Column `j` = lag `j − N_LAGS`:
  - `lag < 0`: velocity *after* spike (predictive)
  - `lag = 0`: velocity at spike time
  - `lag > 0`: velocity *before* spike (causal)

### Cross-Validation: Block-Shuffled

1. Cut the time series into 1-second blocks (200 bins at 5 ms)
2. Shuffle the blocks randomly
3. Split shuffled blocks 80/20 for each of 5 folds

This prevents temporal leakage (shuffled blocks are not temporally adjacent), while keeping train/test splits balanced and covering the full recording duration.

### Model Selection

- `RidgeCV` with 20 alpha values (log-spaced 0.01 → 10,000)
- R² computed by concatenating predictions across all 5 test folds

---

## Key Results (n = 46 paired neurons)

### Overall

| Metric | MP1 | MP2 |
|---|---|---|
| Mean R² | 0.0029 | 0.0024 |
| Median R² | 0.0017 | 0.0015 |
| R² > 0 | 42/46 (91%) | 43/46 (93%) |

| Test | Statistic | p-value |
|---|---|---|
| Wilcoxon signed-rank | 509 | 0.74 |
| Paired t-test | −0.79 | 0.43 |
| Cohen's d | −0.12 | — |

**Conclusion**: No significant difference in velocity encoding between MP1 and MP2 at the population level.

### Speed-Encoding Subset (R² > 0.005, n = 12)

| Metric | MP1 | MP2 |
|---|---|---|
| Mean R² | 0.0067 | 0.0036 |
| MP2 > MP1 | 4/12 (33%) ||

| Test | Statistic | p-value |
|---|---|---|
| Wilcoxon signed-rank | — | 0.23 |
| Paired t-test | — | 0.19 |
| Cohen's d | −0.41 | — |

**Conclusion**: A trend toward weaker encoding in MP2 (d = −0.41, small-medium effect), but not statistically significant with n = 12. Would need ~47 speed-encoding neurons to detect this effect at 80% power.

---

## Notebook Usage

1. Open `Skieur_RidgeVelocity_Encoding.ipynb` in VS Code
2. Select **Anaconda3** kernel
3. Set `SPIKE_SORTED = True` or `False` in Cell 1
4. **Restart Kernel → Run All** (takes ~20 minutes at full resolution)

### Output Files

| File | Content |
|---|---|
| `ridge_velocity_encoding_results.csv` | Per-neuron R² values |
| `ridge_velocity_encoding.png` | Scatter + per-pair bar chart |
| `ridge_velocity_encoding_summary.txt` | Full text summary |

---

## Dependencies

- Python 3.13+, NumPy, Pandas, SciPy, Matplotlib, Seaborn
- scikit-learn (`RidgeCV`, `TimeSeriesSplit`)
- tqdm
- NAS access (`//129.199.81.18/data5/eTheremin`)

---

## Caveats

1. **Neuron identity** across MP1/MP2 is assumed by cluster index but not verified via waveform matching. Different `good_clusters` sets between phases may break this assumption.
2. **R² values are small** — typical for single-neuron encoding of a continuous behavioral variable. Population-level decoding would yield much higher R².
3. **Block-shuffled CV** introduces artificial discontinuities at block boundaries (1 s blocks). The ±50 ms lag window is well within a single block, so cross-block leakage is prevented.
4. **Speed_x is absolute velocity** — deceleration and acceleration are not distinguished. Using signed velocity components may reveal directional tuning.
