import json

path = r'C:\Users\PenPen\Ferret\Analysis\Skieur_LieAlgebra_CEBRA.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_map = {cell['id']: i for i, cell in enumerate(nb['cells'])}

# ============================================================
# 1. Update markdown — add R² gate info + remaining caveats
# ============================================================
md_idx = cell_map['cebra-lie-md']
nb['cells'][md_idx]['source'] = (
    '### 1.3 CEBRA-Embedded Lie Algebra (exploratory)\n\n'
    '**Caveat:** CEBRA is trained with the same behavioral variable (Velocity/Position) used as the Lie algebra '
    'drive — so a high skewness ratio may partly reflect the supervised embedding objective rather than intrinsic '
    'rotational dynamics. This section tests whether nonlinear dimensionality reduction makes rotational structure '
    'more apparent, but does **not** constitute topological evidence (cf. Gardner 2022, Hermansen 2024).\n\n'
    '**Design:**\n'
    '- One pooled CEBRA model per session (Tracking + Playback share coordinate system)\n'
    '- Lie algebra fit **per epoch** (no concatenation → no derivative jumps at boundaries)\n'
    '- **Time-shuffled control:** 10 random label permutations per epoch → baseline skewness/R²\n'
    '- **R² gate:** skewness ratio is only treated as meaningful when true R² exceeds shuffle R²\n'
    '- Per-condition statistics reported separately (Tracking and Playback share CEBRA model)\n\n'
    '**Remaining limitations:**\n'
    '- *Shuffle-CEBRA control:* the strongest negative control would train CEBRA on shuffled labels '
    'and then fit Lie algebra — this is not done here (computational cost, exploratory scope)\n'
    '- *Session embeddings not comparable:* each session has its own latent coordinate system; '
    'interpret only within-session True-vs-Shuffle differences, not cross-session absolute values\n'
    '- *Dimensionality:* results may depend on CEBRA embedding dimension (3D here); '
    'toroidal structure requires testing across 2D/3D/6D'
)

# ============================================================
# 2. Velocity code — add R² gate after t-test
# ============================================================
vel_idx = cell_map['cebra-lie-vel']
vel_src = ''.join(nb['cells'][vel_idx]['source'])

# Find the section after True-vs-Shuffle t-test output and add R² gate
insert_marker = '''        # Also report at session level: average Tracking+PB per session'''

r2_gate_block = '''
        # ---- R² gate: only meaningful if true R² > shuffle R² ----
        print()
        print("  R² gate check (skewness meaningful only when R²_true > R²_shuffle):")
        for cond in ["Tracking", "Playback"]:
            sub = cebra_lie_vel_df[cebra_lie_vel_df["Condition"] == cond]
            n_pass = (sub["R2"] > sub["R2_shuffle"]).sum()
            n_total = len(sub)
            sr_pass = sub[sub["R2"] > sub["R2_shuffle"]]["Skewness_Ratio"].mean() if n_pass > 0 else float('nan')
            sr_all = sub["Skewness_Ratio"].mean()
            print(f"    {cond:<12}: {n_pass}/{n_total} sessions pass R² gate "
                  f"(SR_pass={sr_pass:.4f}, SR_all={sr_all:.4f})")
        print("    ↑ If R²_true ≤ R²_shuffle, the Lie generator does not explain the trajectory")
        print("      derivative — high skewness in those cases is noise, not signal.")'''

vel_src = vel_src.replace(insert_marker, r2_gate_block + '\n' + insert_marker)

# ============================================================
# 3. Position code — same R² gate
# ============================================================
pos_idx = cell_map['cebra-lie-pos']
pos_src = ''.join(nb['cells'][pos_idx]['source'])

# Add R² gate after t-test section
insert_marker_pos = '''        # Barplot'''
r2_gate_pos = '''
        # ---- R² gate check ----
        print()
        print("  R² gate check (skewness meaningful only when R²_true > R²_shuffle):")
        for cond in ["Tracking", "Playback"]:
            sub = cebra_lie_pos_df[cebra_lie_pos_df["Condition"] == cond]
            n_pass = (sub["R2"] > sub["R2_shuffle"]).sum()
            n_total = len(sub)
            sr_pass = sub[sub["R2"] > sub["R2_shuffle"]]["Skewness_Ratio"].mean() if n_pass > 0 else float('nan')
            sr_all = sub["Skewness_Ratio"].mean()
            print(f"    {cond:<12}: {n_pass}/{n_total} sessions pass R² gate "
                  f"(SR_pass={sr_pass:.4f}, SR_all={sr_all:.4f})")'''

pos_src = pos_src.replace(insert_marker_pos, r2_gate_pos + '\n' + insert_marker_pos)

# ============================================================
# Apply
# ============================================================
nb['cells'][md_idx]['source'] = nb['cells'][md_idx]['source']  # already set above
nb['cells'][vel_idx]['source'] = vel_src.splitlines(True)
nb['cells'][pos_idx]['source'] = pos_src.splitlines(True)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done: updated markdown with caveats + R² gate in both Velocity and Position cells.')
