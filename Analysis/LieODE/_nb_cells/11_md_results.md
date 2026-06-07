## 3. Results: Comparison + Controls

Compare end-to-end metrics against baseline two-stage pipeline and Dummy-CEBRA control.

**Controls grid:**
- (a) Dummy-encoder: shuffled-label CEBRA -> `R2_drive_dummy`
- (b) `lambda_dyn=0` ablation: pure InfoNCE without dynamics constraint
- (c) Drive time-shuffle: permute drive on fixed trained encoder
- (d) Embedding dimension sweep: 3/6/8 (set D_LATENT in Cell 0)

**Gate:** `R2_drive_real > R2_drive_dummy` AND `R2_drive_real > R2_drive_drive_shuffle`
