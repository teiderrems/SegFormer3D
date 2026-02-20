import os
import sys
import types
import numpy as np

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import inference_simple as inf


def test_save_prediction_as_nifti_uses_nibabel_and_affine(tmp_path, monkeypatch):
    # Simulate that nibabel is available and record calls to save
    called = {}

    def fake_save(img, path):
        called['path'] = path
        called['img'] = img
        # create a placeholder file to simulate actual save
        open(path, 'wb').write(b'0')

    # Provide a minimal Nifti1Image factory (store data + affine)
    def fake_nifti_ctor(data, affine):
        return ('nifti', np.asarray(data), np.asarray(affine))

    monkeypatch.setattr(inf, 'HAS_NIBABEL', True)
    monkeypatch.setattr(inf, 'nib', types.SimpleNamespace(Nifti1Image=fake_nifti_ctor, save=fake_save))

    # Create a dummy prediction (numpy) and metadata with affine
    pred = np.zeros((10, 12, 8), dtype=np.uint8)
    metadata = {'original_affine': np.eye(4)}
    out_path = str(tmp_path / 'pred.nii.gz')

    ok = inf.save_prediction_as_nifti(pred, metadata, out_path)
    assert ok is True
    assert 'path' in called and called['path'] == out_path
    assert called['img'][0] == 'nifti'
    assert (called['img'][1].shape == pred.shape)


def test_save_prediction_as_nifti_fails_without_metadata(monkeypatch, tmp_path):
    # nibabel available but metadata missing -> returns False
    monkeypatch.setattr(inf, 'HAS_NIBABEL', True)
    monkeypatch.setattr(inf, 'nib', types.SimpleNamespace(Nifti1Image=lambda d,a: None, save=lambda *a, **k: None))

    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    ok = inf.save_prediction_as_nifti(pred, None, str(tmp_path / 'out.nii.gz'))
    assert ok is False