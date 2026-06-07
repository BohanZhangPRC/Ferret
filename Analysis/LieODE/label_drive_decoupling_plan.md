# Plan: Decouple InfoNCE Label from Dynamics Drive

## Why

Limitation #2 (circular reasoning) states: *"the InfoNCE label and the dynamics drive are both derived from Velocity_x — the embedding is shaped by the same variable it is asked to 'explain' dynamically."*

The feature DataFrame already contains two additional columns that can break this circularity:

| Column | Content | Role |
|--------|---------|------|
| `Played_frequency` | Acoustic frequency heard by the ferret | **InfoNCE label** — sensory context |
| `Frequency_changes` | Boolean: whether frequency changed in this bin | **Event anchor** for peri-event extraction |

In the **Playback** condition, `Played_frequency` (a replayed acoustic sequence) and `Velocity_x` (the animal's head movement) are **independent signals** — the animal hears whatever was recorded, regardless of whether it moves. This means:

- **Tracking**: motor command and sensory consequence are causally linked (partial circularity)
- **Playback**: motor command and sensory input are **fully decoupled** (zero circularity)

If rotational structure (high SR, positive R2_drive) persists in Playback when InfoNCE uses `Played_frequency` and dynamics uses `Velocity_x`, the rotation **cannot be explained by circular reasoning** — the embedding was shaped by an acoustic signal, yet the motor drive still produces structured rotation. This is the strongest single-animal test of limitation #2 available without PMC recordings.

## Changes

### 1. Cell 0 — New config constants

```
# Before:
#   InfoNCE label = Velocity_x (implicit, same as DRIVE_KEYS[0])
#   Dynamics drive = DRIVE_KEYS = ["Velocity_x"]

# After:
CEBRA_LABEL = "Velocity_x"    # column used for InfoNCE contrastive labels
DRIVE_KEYS  = ["Velocity_x"]  # column(s) used for dynamics drive u(t)
```

**Default behaviour unchanged** (`CEBRA_LABEL == DRIVE_KEYS[0] == "Velocity_x"` — identical to current pipeline). Decoupling is opt-in: set `CEBRA_LABEL = "Played_frequency"` to activate.

When `CEBRA_LABEL != DRIVE_KEYS[0]`, print a notice that decoupled mode is active.

### 2. Cell 2 — No changes

`extract_epochs` already accepts `label_col` as a parameter. `build_drive_vector` already constructs multi-dim drive from arbitrary `DRIVE_KEYS`. No helper modifications needed.

### 3. Cell 4 (Baseline + Dummy CEBRA) — Use CEBRA_LABEL

```
# Before:
extract_epochs(..., label_col="Velocity_x")

# After:
extract_epochs(..., label_col=CEBRA_LABEL)
```

This ensures the baseline CEBRA model uses the same label as E2E InfoNCE, keeping the comparison fair.

### 4. Cell 9 (train_one_session) — Dual extraction

The core change. Currently:

```
epochs_n, epochs_l, _ = extract_epochs(..., label_col="Velocity_x")
# epochs_l is used BOTH as InfoNCE label AND as dynamics drive
```

After:

```
# Step 1: extract neural data + CEBRA labels
epochs_n, epochs_l, _ = extract_epochs(..., label_col=CEBRA_LABEL)

# Step 2: extract drive labels (same epoch boundaries, different column)
_, drive_labels_raw, _ = extract_epochs(..., label_col=DRIVE_KEYS[0])

# Sanity check: same number of epochs, same lengths
assert len(epochs_n) == len(drive_labels_raw)
for i in range(len(epochs_n)):
    assert len(epochs_n[i]) == len(drive_labels_raw[i])

# Standardize drive labels (pooled TR+PB) → drive_epochs
# InfoNCE uses epochs_l (CEBRA_LABEL)
# Dynamics uses drive_epochs (DRIVE_KEYS[0])
```

**Why the double extraction works**: `extract_macro_epochs` uses **Condition** boundaries (0.0 = Tracking, 1.0 = Playback), not the label column. So calling it twice with different `label_col` values produces **identical epoch boundaries** — the same time segments, just different values extracted from the feature DataFrame. The only difference is which column's values populate `epochs_l`.

**Computational cost**: preprocesses the neural data twice. For N=10-20 sessions this is negligible (< 1 second). The sanity check catches any boundary mismatch.

### 5. Cell 10 (Phase 2b) — No changes

The multi-seed training loop calls `train_one_session` which now handles the decoupling internally. No changes to the outer loop.

### 6. Method description — New section in limitations

Update §10.2:

```
Before: "The InfoNCE label and the dynamics drive are both derived from x(t)..."
After:  "When CEBRA_LABEL == DRIVE_KEYS[0], the InfoNCE label and dynamics
         drive share the same source.  Setting CEBRA_LABEL = 'Played_frequency'
         decouples them in the Playback condition (acoustic replay is independent
         of head movement).  This partially addresses the circularity but does
         not eliminate it in Tracking, where motor and sensory signals remain
         causally linked."
```

## Experiments enabled

| Experiment | CEBRA_LABEL | DRIVE_KEYS | Tests |
|-----------|-------------|------------|-------|
| **Default (current)** | `Velocity_x` | `["Velocity_x"]` | Baseline — circular |
| **Sensory-context decoupling** | `Played_frequency` | `["Velocity_x"]` | **#2 test**: does motor drive rotate a sensory embedding? |
| **Motor-embedding, sensory-drive** | `Velocity_x` | `["Played_frequency"]` | Reverse: does sensory input rotate a motor embedding? |
| **Frequency-change anchored** | `Frequency_changes` | `["Velocity_x"]` | Does rotation peak at frequency transitions? (set `USE_MACRO_EPOCH=False`) |

## Verification

1. Default config (`CEBRA_LABEL="Velocity_x"`, `DRIVE_KEYS=["Velocity_x"]`) produces identical results to current notebook (regression test)
2. Decoupled config (`CEBRA_LABEL="Played_frequency"`) runs without error
3. Sanity check: epoch counts and lengths match between the two `extract_epochs` calls
4. In Playback: SR and R2_drive with decoupled labels should remain above shuffle null if rotation is genuine (key scientific result)

## What this does NOT fix

- Tracking condition: motor and sensory are still causally linked → residual circularity
- Does not provide a truly independent signal source (that requires PMC efference copy)
- `Played_frequency` and `Velocity_x` may still be correlated in Tracking (the animal tracks frequency with head movement)
- Does not address SR coordinate-dependence (#3) or cross-session eigenvalue comparability (#4)
