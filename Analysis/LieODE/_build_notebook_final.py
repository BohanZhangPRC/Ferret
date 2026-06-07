"""
Build Skieur_EndToEnd_LieODE.ipynb from _nb_cells/ directory.
Run: /c/Users/PenPen/anaconda3/envs/Latest/python.exe _build_notebook_final.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS_DIR = os.path.join(HERE, "_nb_cells")
OUTPUT = os.path.join(HERE, "Skieur_EndToEnd_LieODE.ipynb")

# Cell definitions: (filename, type)
# type: "code" -> .py file, "markdown" -> .md file
CELLS = [
    ("00_config.py", "code"),
    ("01_load_data.py", "code"),
    ("02_helpers.py", "code"),
    ("03_md_section1.md", "markdown"),
    ("04_baseline_dummy.py", "code"),
    ("05_md_baseline_check.md", "markdown"),
    ("06_baseline_stats.py", "code"),
    ("15_cleanup_after_baseline.py", "code"),
    ("07_md_model.md", "markdown"),
    ("08_models.py", "code"),
    ("09_training.py", "code"),
    ("10_run_training.py", "code"),
    ("11_md_results.md", "markdown"),
    ("12_results_comparison.py", "code"),
    ("13_tracking_vs_playback.py", "code"),
    ("14_summary.py", "code"),
    ("16_cleanup_final.py", "code"),
]

nb = {
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5,
    'cells': []
}

for filename, cell_type in CELLS:
    filepath = os.path.join(CELLS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"WARNING: {filename} not found, skipping.")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into lines (preserving trailing newlines)
    lines = content.rstrip('\n').split('\n')

    if cell_type == "code":
        nb['cells'].append({
            'cell_type': 'code',
            'metadata': {},
            'source': lines,
            'outputs': []
        })
    else:
        nb['cells'].append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': lines
        })

    print(f"  Added {filename} ({cell_type}, {len(lines)} lines)")

# Write notebook
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook written to: {OUTPUT}")
print(f"Total cells: {len(nb['cells'])}")
