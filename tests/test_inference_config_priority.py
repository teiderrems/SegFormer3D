import os
import sys
import yaml

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import inference_simple as inf


class Args:
    pass


def test_yaml_overrides_cli_for_inference_params():
    a = Args()
    a.verbosity = 'normal'
    a.device = 'cpu'
    a.batch_size = 1
    a.save_predictions = False
    a.save_probabilities = False
    a.save_nifti = False
    a.threshold = 0.5

    cfg = {'inference_parameters': {'device': 'cuda', 'batch_size': 4, 'save_predictions': True, 'save_nifti': True, 'threshold': 0.75, 'verbosity': 'debug'}}

    resolved = inf.resolve_inference_params(a, cfg)
    assert resolved['device'] == 'cuda'
    assert resolved['batch_size'] == 4
    assert resolved['save_predictions'] is True
    assert resolved['save_nifti'] is True
    assert abs(resolved['threshold'] - 0.75) < 1e-6
    assert resolved['verbosity'] == 'debug'


def test_cli_used_when_yaml_missing():
    a = Args()
    a.verbosity = 'normal'
    a.device = 'cpu'
    a.batch_size = 2
    a.save_predictions = True
    a.save_probabilities = False
    a.save_nifti = None
    a.threshold = 0.6

    cfg = {}  # no inference_parameters
    resolved = inf.resolve_inference_params(a, cfg)
    assert resolved['device'] == 'cpu'
    assert resolved['batch_size'] == 2
    assert resolved['save_predictions'] is True
    assert resolved['save_nifti'] is True
    assert abs(resolved['threshold'] - 0.6) < 1e-6
    assert resolved['verbosity'] == 'normal'


def test_force_cli_allows_cli_override(monkeypatch):
    # YAML defines device=cuda but user passed --device cpu along with --force-cli
    a = Args()
    a.verbosity = 'normal'
    a.device = 'cpu'
    a.batch_size = 2
    a.save_predictions = False
    a.save_probabilities = False
    a.threshold = 0.6
    a.force_cli = True

    cfg = {'inference_parameters': {'device': 'cuda', 'batch_size': 4, 'save_predictions': True, 'threshold': 0.75, 'verbosity': 'debug'}}

    # Simulate that CLI explicitly provided --device and --batch_size and --save_predictions
    monkeypatch.setattr('sys.argv', ['inference_simple.py', '--device', 'cpu', '--batch_size', '2', '--save_predictions', '--save_nifti', '--force-cli'])

    resolved = inf.resolve_inference_params(a, cfg)
    assert resolved['device'] == 'cpu'        # CLI wins because force-cli + explicit CLI
    assert resolved['batch_size'] == 2
    assert resolved['save_predictions'] is True  # CLI explicitly provided --save_predictions
    assert resolved['save_nifti'] is True       # CLI explicitly provided --save_nifti
    assert abs(resolved['threshold'] - 0.75) < 1e-6  # YAML still used for threshold (no explicit CLI)
    assert resolved['verbosity'] == 'debug'   # YAML verbosity remains (no explicit CLI)