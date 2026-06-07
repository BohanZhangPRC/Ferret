import json, ast, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
NB = r'C:\Users\PenPen\Ferret\Analysis\Skieur_analysis.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ---- Fix Cell 5: add gc >= 10 filter ----
src5 = ''.join(nb['cells'][5]['source'])

# Replace the combine + validation section at the end
old_end = """# Combine headstages
n_data_all = list(n_data_hs0) + list(n_data_hs1)
f_data_all = list(f_data_hs0) + list(f_data_hs1)
n_hs0 = len(n_data_hs0)

print(f"Total spike-sorted sessions: {len(n_data_all)} (hs0={n_hs0}, hs1={len(n_data_hs1)})")

# Quick validation
example = f_data_all[0]
print(f"Example session: {example.shape[0]} timepoints, columns={list(example.columns)}")
print(f"  Condition values: {example[\"Condition\"].unique()}")
print(f"  Tracking pts: {(example[\"Condition\"]==0).sum()}, Playback pts: {(example[\"Condition\"]==1).sum()}")
print(f"  Velocity_x range: [{example[\"Velocity_x\"].min():.1f}, {example[\"Velocity_x\"].max():.1f}]")"""

new_end = """# Combine headstages
n_data_all_raw = list(n_data_hs0) + list(n_data_hs1)
f_data_all_raw = list(f_data_hs0) + list(f_data_hs1)
n_hs0 = len(n_data_hs0)

print(f"Total spike-sorted sessions: {len(n_data_all_raw)} (hs0={n_hs0}, hs1={len(f_data_hs1)})")

# Filter: keep only sessions with >= 10 good clusters
MIN_GC = 10
n_data_all = []
f_data_all = []
n_hs0_filtered = 0
for i, nd in enumerate(n_data_all_raw):
    n_gc = nd.shape[0]
    if n_gc >= MIN_GC:
        n_data_all.append(nd)
        f_data_all.append(f_data_all_raw[i])
        if i < n_hs0:
            n_hs0_filtered += 1

print(f"After gc >= {MIN_GC} filter: {len(n_data_all)} sessions (hs0={n_hs0_filtered}, hs1={len(n_data_all)-n_hs0_filtered})")
print(f"Removed {len(n_data_all_raw) - len(n_data_all)} sessions with < {MIN_GC} good clusters")
n_hs0 = n_hs0_filtered

# Quick validation
example = f_data_all[0]
print(f"Example session: {example.shape[0]} timepoints, gc_count={n_data_all[0].shape[0]}")
print(f"  Condition values: {example[\"Condition\"].unique()}")
print(f"  Tracking pts: {(example[\"Condition\"]==0).sum()}, Playback pts: {(example[\"Condition\"]==1).sum()}")
print(f"  Velocity_x range: [{example[\"Velocity_x\"].min():.1f}, {example[\"Velocity_x\"].max():.1f}]")"""

assert old_end in src5, 'Could not find end section in Cell 5'
src5 = src5.replace(old_end, new_end)

try:
    ast.parse(src5)
    print('Cell 5 syntax: OK')
except SyntaxError as e:
    print(f'Cell 5 syntax error L{e.lineno}: {e.msg}')
    sys.exit(1)

nb['cells'][5]['source'] = [src5]

# ---- Fix Cell 9: sample 5 sessions ----
src9 = ''.join(nb['cells'][9]['source'])

old_exec = """    # ---- Execute ----
    save_directory = r\"C:\\\\Users\\\\PenPen\\\\Desktop\\\\Ferret\\\\Results&PLots\\\\Web\"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f\"Created: {save_directory}\")

    print(\"=\" * 60)
    print(\"SKIEUR 3D Manifold - using signed Velocity_x\")
    print(\"Output: \" + save_directory)
    print(\"=\" * 60)

    cebra_results = run_cebra_skieur(n_data_all, f_data_all, n_hs0, step=5)
    visualize_skieur_webgl(cebra_results, save_directory)

    print(\"Done! Open CEBRA_Manifold_SKIEUR.html in a browser.\")"""

new_exec = """    # ---- Sample 5 sessions chronologically ----
    N_SAMPLE = 5
    total = len(n_data_all)
    if total <= N_SAMPLE:
        indices = list(range(total))
    else:
        indices = [int(round(i * (total - 1) / (N_SAMPLE - 1))) for i in range(N_SAMPLE)]
        indices = list(dict.fromkeys(indices))  # deduplicate

    n_subset = [n_data_all[i] for i in indices]
    f_subset = [f_data_all[i] for i in indices]
    hs0_subset = sum(1 for i in indices if i < n_hs0)

    print(f\"Sampled {len(indices)} / {total} sessions chronologically:\")
    for j, i in enumerate(indices):
        print(f\"  {j+1}. session {i}: {n_data_all[i].shape[0]} neurons, {n_data_all[i].shape[1]} timepoints\")

    # ---- Execute CEBRA ----
    save_directory = r\"C:\\\\Users\\\\PenPen\\\\Desktop\\\\Ferret\\\\Results&PLots\\\\Web\"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f\"Created: {save_directory}\")

    print(\"=\" * 60)
    print(\"SKIEUR 3D Manifold - signed Velocity_x, 5-session sample\")
    print(\"Output: \" + save_directory)
    print(\"=\" * 60)

    cebra_results = run_cebra_skieur(n_subset, f_subset, hs0_subset, step=5)
    visualize_skieur_webgl(cebra_results, save_directory)

    print(\"Done! Open CEBRA_Manifold_SKIEUR.html in a browser.\")"""

assert old_exec in src9, 'Could not find execute section in Cell 9'
src9 = src9.replace(old_exec, new_exec)

try:
    ast.parse(src9)
    print('Cell 9 syntax: OK')
except SyntaxError as e:
    print(f'Cell 9 syntax error L{e.lineno}: {e.msg}')
    sys.exit(1)

nb['cells'][9]['source'] = [src9]

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\nDone: Cell 5 (+gc filter), Cell 9 (5-session sample)')
