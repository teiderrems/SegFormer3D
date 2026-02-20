import os
import sys
from types import SimpleNamespace

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline


def test_skip_training_skips_train_model_and_runs_inference(monkeypatch, tmp_path):
    calls = {'preprocess': False, 'train': False, 'inference': False}

    # Stub preprocess_data to succeed
    monkeypatch.setattr(pipeline, 'preprocess_data', lambda *a, **k: True)

    # Stub train_model to mark if called
    def fake_train(arch, cfg, *args, **kwargs):
        calls['train'] = True
        return True
    monkeypatch.setattr(pipeline, 'train_model', fake_train)

    # Stub run_inference to mark if called and return True
    def fake_infer(arch, cfg_path, ckpt_path, test_data_dir, out_dir):
        calls['inference'] = True
        return True
    monkeypatch.setattr(pipeline, 'run_inference', fake_infer)

    # Prepare CLI args: skip training, enable visualize so pipeline will try inference
    monkeypatch.setattr('sys.argv', ['pipeline.py', '--skip_training', '--visualize', '--architectures', 'SegFormer3D'])

    # Run pipeline.main() — should not call train_model but should call run_inference
    pipeline.main()

    assert not calls['train'], 'train_model should NOT be called when --skip_training is set'
    assert calls['inference'], 'run_inference should be called even when training is skipped'


def test_default_runs_training_when_no_skip(monkeypatch):
    calls = {'train': False}

    def fake_train(arch, cfg, *args, **kwargs):
        calls['train'] = True
        return True
    monkeypatch.setattr(pipeline, 'train_model', fake_train)
    monkeypatch.setattr(pipeline, 'preprocess_data', lambda *a, **k: True)
    monkeypatch.setattr(pipeline, 'run_inference', lambda *a, **k: True)

    monkeypatch.setattr('sys.argv', ['pipeline.py', '--architectures', 'SegFormer3D'])
    pipeline.main()

    assert calls['train'], 'train_model should be called by default (when --skip_training is not provided)'
