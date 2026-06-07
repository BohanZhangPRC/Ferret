## 1. Baseline Two-Stage Pipeline + Dummy-CEBRA Control

Re-run the existing two-stage CEBRA -> OLS-Lie pipeline for reference,
then add the missing **Dummy-CEBRA negative control** (train CEBRA on shuffled labels).

**Key comparison:**
- `SR_true` / `R2_drive_true`: true labels, real embedding -> genuine rotational structure
- `SR_dummy` / `R2_drive_dummy`: shuffled labels, dummy embedding -> InfoNCE artifact baseline
- **Gate:** `R2_drive_true > R2_drive_dummy` signifies rotational structure is a genuine property
  of neural population dynamics, not an artifact of InfoNCE imprinting rotational topology.
