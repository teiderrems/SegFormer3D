import os
import sys
import csv

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline


def test_pipeline_run_inference_uses_test_csv(monkeypatch, tmp_path):
    # Create a fake preprocessed folder with two patients but CSV listing only one
    preproc = tmp_path / 'preprocessed'
    preproc.mkdir()
    p1 = preproc / 'patient_a'
    p2 = preproc / 'patient_b'
    p1.mkdir()
    p2.mkdir()

    csv_path = preproc / 'test.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['data_path', 'case_name'])
        # write absolute path for patient_b only
        writer.writerow([str(p2.resolve()), 'patient_b'])

    # Prepare config and inject preprocessed path
    cfg, user_cfg = pipeline.load_pipeline_config(None, return_user_config=True)
    cfg['paths']['preprocessed_data_dir'] = str(preproc)

    # Monkeypatch load_pipeline_config so pipeline.main picks our config
    monkeypatch.setattr(pipeline, 'load_pipeline_config', lambda *a, **k: (cfg, {}))

    # Capture run_command calls instead of executing subprocesses
    calls = []

    def fake_run_command(cmd, cwd=None, description=""):
        calls.append(cmd)
        return True

    monkeypatch.setattr(pipeline, 'run_command', fake_run_command)

    # Call run_inference directly
    ok = pipeline.run_inference('SegFormer3D', 'configs/config_segformer3d.yaml', 'checkpoints/best_model.pth', str(preproc), str(tmp_path / 'results'))
    assert ok is True
    # Only patient_b (listed in CSV) should have been processed
    assert any('patient_b' in c for c in calls)
    assert not any('patient_a' in c for c in calls)
    # Default YAML enables save_nifti -> pipeline should have passed the flag to inference_simple
    assert any('--save_nifti' in c for c in calls), 'pipeline should request NIfTI save by default'