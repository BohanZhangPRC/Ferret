import json, ast, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
NB = r'C:\Users\PenPen\Ferret\Analysis\Skieur_analysis.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

L = []
L.append('# List all SKIEUR NAS session directories in chronological order')
L.append('# Beginner = first half, Expert = second half (per session type within each pickle)')
L.append('import os')
L.append('')
L.append("nas = r'\\\\\\\\129.199.81.18\\\\data5\\\\eTheremin\\\\SKIEUR'")
L.append("all_dirs = sorted([")
L.append("    d for d in os.listdir(nas)")
L.append('    if os.path.isdir(os.path.join(nas, d)) and "SESSION" in d')
L.append("])")
L.append("")
L.append("print(f'Total SKIEUR session dirs on NAS: {len(all_dirs)}')")
L.append("print()")
L.append("# The pickle files (playback, mapping_change_only, etc.) each contain a SUBSET of these dirs")
L.append("# The order within each pickle matches the Google Sheet query order,")
L.append("# which typically sorts chronologically by session date/number.")
L.append("# Beginner = first half of that pickle's sessions, Expert = second half.")
L.append("")
L.append("for i, d in enumerate(all_dirs):")
L.append("    print(f'  [{i:3d}] {d}')")
L.append("")
L.append("# Also check: how many playback sessions?")
L.append("# (Playback sessions have both TR and PB conditions)")
L.append('pb_count = 0')
L.append('for d in all_dirs:')
L.append('    hs0_path = os.path.join(nas, d, "headstage_0")')
L.append('    feat_path = os.path.join(hs0_path, "features_0.005.npy")')
L.append('    if os.path.exists(feat_path):')
L.append('        try:')
L.append('            import numpy as np')
L.append('            feat = np.load(feat_path, allow_pickle=True)')
L.append('            conds = set()')
L.append('            for item in feat:')
L.append("                conds.add(item.get('Condition', -1))")
L.append('            if 0 in conds and 1 in conds:  # has both TR and PB')
L.append('                pb_count += 1')
L.append('        except:')
L.append('            pass')
L.append("")
L.append("print(f'\\\\nSessions with BOTH Track+Playback in headstage_0: {pb_count}')")

src = '\n'.join(L)
try:
    ast.parse(src)
    print('Syntax: OK')
except SyntaxError as e:
    print(f'Syntax error L{e.lineno}: {e.msg}')

# Insert after Cell 2 (index 2)
import uuid
new_cell = {
    'cell_type': 'code',
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'source': [src],
    'outputs': []
}
nb['cells'].insert(3, new_cell)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Inserted session list cell at position 3. Total cells: {len(nb[\"cells\"])}')
