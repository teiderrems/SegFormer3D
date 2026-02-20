import os
import sys
import csv
import types
from pathlib import Path

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_inference_all as ria


def test_force_cli_propagated_to_child(tmp_path, monkeypatch):
    # Prepare a fake CSV with one case
    csv_path = tmp_path / 'test.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['data_path', 'case_name'])
        w.writeheader()
        w.writerow({'data_path': str(tmp_path), 'case_name': 'case_001'})

    # Create a dummy checkpoint file and set module globals
    ckpt = tmp_path / 'best_model.pth'
    ckpt.write_text('dummy')
    monkeypatch.setattr(ria, 'CSV', csv_path)
    monkeypatch.setattr(ria, 'CHECKPOINT', ckpt)

    # Ensure INFER_SCRIPT exists (point to repository file) and results dir points to tmp
    monkeypatch.setattr(ria, 'INFER_SCRIPT', Path(__file__).parent.parent / 'inference_simple.py')
    monkeypatch.setattr(ria, 'DEFAULT_RESULTS_DIR', tmp_path / 'results')

    captured = {}

    class DummyProc:
        def __init__(self):
            self.returncode = 0
            self.stdout = ''
            self.stderr = ''

    def fake_run(cmd, capture_output=True, text=True):
        # Save the command for assertions
        captured['cmd'] = cmd
        return DummyProc()

    monkeypatch.setattr('subprocess.run', fake_run)

    # Call main() with --force-cli in argv
    monkeypatch.setattr('sys.argv', ['run_inference_all.py', '--force-cli'])
    ria.main()

    # Assert the child command included --force-cli
    assert '--force-cli' in captured['cmd']
