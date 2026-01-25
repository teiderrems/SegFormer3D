import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / 'pipeline_notebook.ipynb'
print(f'Cleaning outputs in: {nb_path}')

with nb_path.open('r', encoding='utf-8') as f:
    content = f.read()

# Try parsing as JSON first; if that fails, try simple string replace of common patterns
try:
    nb = json.loads(content)
    changed = False
    if 'cells' in nb:
        for cell in nb['cells']:
            if 'outputs' in cell and cell['outputs']:
                cell['outputs'] = []
                changed = True
            if 'execution_count' in cell and cell['execution_count'] is not None:
                cell['execution_count'] = None
                changed = True
    if changed:
        with nb_path.open('w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print('Notebook outputs cleared (JSON mode).')
    else:
        print('No outputs found to clear (JSON mode).')
except Exception as e:
    print(f'JSON parse failed: {e}. Trying text cleanup...')
    # Fallback: remove literal occurrences of 'TransUNet3D:\n' and empty outputs arrays
    new_content = content.replace('TransUNet3D:\n', '')
    new_content = new_content.replace('"outputs": [\n\n      ],', '"outputs": [],')
    if new_content != content:
        with nb_path.open('w', encoding='utf-8') as f:
            f.write(new_content)
        print('Notebook cleaned by text replacement.')
    else:
        print('No changes made by text replacement.')
