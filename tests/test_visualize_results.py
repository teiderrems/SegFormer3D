import yaml
import types
import os
import sys

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import visualize_results as vr


def make_args(**kwargs):
    class A: pass
    a = A()
    # defaults
    a.config = 'configs/config_segformer3d.yaml'
    a.prediction = 'pred.pt'
    a.input_dir = './data/patient_001'
    a.output_dir = 'visualizations'
    a.volume_vis = False
    a.interactive = False
    a.compute_errors = False
    a.voxel_spacing = None
    a.verbosity = 'normal'
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def test_yaml_overrides_cli_for_visualization_options(tmp_path):
    # Create a temporary config file with visualization settings
    cfg_file = tmp_path / 'viz_cfg.yaml'
    cfg = {'visualization': {'volume_vis': True, 'verbosity': 'debug', 'compute_errors': True, 'output_dir': str(tmp_path / 'out')}}
    cfg_file.write_text(yaml.safe_dump(cfg))

    args = make_args(volume_vis=False, verbosity='normal', compute_errors=False, output_dir='visualizations')
    # Apply merge
    merged = vr.apply_visualization_config_over_args(args, cfg)

    assert merged.volume_vis is True
    assert merged.compute_errors is True
    assert merged.verbosity == 'debug'
    assert merged.output_dir == str(tmp_path / 'out')


def test_cli_remains_when_yaml_missing():
    args = make_args(volume_vis=True, verbosity='normal')
    merged = vr.apply_visualization_config_over_args(args, {})
    assert merged.volume_vis is True
    assert merged.verbosity == 'normal'


def test_force_cli_prefers_cli_over_yaml(monkeypatch):
    # YAML sets verbosity=debug and volume_vis=false, CLI explicitly passes --verbosity normal and --volume_vis
    cfg = {'visualization': {'verbosity': 'debug', 'volume_vis': False}}

    args = make_args(volume_vis=False, verbosity='normal')
    args.force_cli = True

    # Simulate explicit CLI flags
    monkeypatch.setattr('sys.argv', ['visualize_results.py', '--verbosity', 'normal', '--volume_vis', '--force-cli'])

    merged = vr.apply_visualization_config_over_args(args, cfg)
    # CLI values should be preserved because --force-cli + explicit CLI provided
    assert merged.volume_vis is True
    assert merged.verbosity == 'normal'