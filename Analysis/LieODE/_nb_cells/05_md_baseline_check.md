### Baseline vs Dummy-CEBRA Quick Check

If rotational structure is genuine (not an InfoNCE artifact):
- `SR_true > SR_dummy` (true embedding has more rotational structure)
- `R2_drive_true >> R2_drive_dummy` (drive explains variance only when labels are real)
- `R2_drive_dummy ~ 0` (random labels provide no predictive power for dynamics)

This is the key negative control that the original notebook acknowledges is missing.
