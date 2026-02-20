import os
import sys
# Ensure repository root is importable for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tempfile
import yaml
import pipeline


def test_default_contains_test_data_dir():
    cfg = pipeline.load_pipeline_config(None)
    assert 'paths' in cfg
    assert 'test_data_dir' in cfg['paths']


def test_user_config_overrides_test_data_dir(tmp_path):
    # Create a temporary YAML file with an explicit test_data_dir
    tmp_yaml = tmp_path / "tmp_pipeline.yaml"
    user_cfg = {
        'paths': {
            'test_data_dir': './data/my_test_set'
        }
    }
    tmp_yaml.write_text(yaml.safe_dump(user_cfg))

    cfg = pipeline.load_pipeline_config(str(tmp_yaml))
    assert cfg['paths']['test_data_dir'] == './data/my_test_set'


def test_yaml_priority_prevents_cli_override_for_paths(tmp_path):
    # YAML définit test_data_dir -> CLI ne doit pas l'écraser
    tmp_yaml = tmp_path / "tmp_pipeline.yaml"
    user_cfg = {'paths': {'test_data_dir': './data/my_test_set'}}
    tmp_yaml.write_text(yaml.safe_dump(user_cfg))

    cfg, user_cfg_loaded = pipeline.load_pipeline_config(str(tmp_yaml), return_user_config=True)

    class Args: pass
    args = Args()
    args.test_data_dir = './data/cli_test_set'
    # other args default to None/False
    args.disable_augmentations = False
    args.architectures = None
    args.raw_data_dir = None
    args.preprocessed_data_dir = None
    args.config_dir = None
    args.checkpoint_dir = None
    args.results_dir = None
    args.split_type = None
    args.train_ratio = None
    args.val_ratio = None
    args.test_ratio = None
    args.k_folds = None
    args.random_seed = None
    args.target_size = None
    args.skip_preprocess = False
    args.crop_to_prostate = False
    args.crop_margin = None

    pipeline.apply_cli_overrides_with_yaml_priority(cfg, user_cfg_loaded, args)
    assert cfg['paths']['test_data_dir'] == './data/my_test_set'


def test_cli_applies_when_key_not_in_yaml(tmp_path):
    # YAML n'a pas target_size -> CLI doit s'appliquer
    tmp_yaml = tmp_path / "tmp_pipeline.yaml"
    user_cfg = {}  # empty YAML
    tmp_yaml.write_text(yaml.safe_dump(user_cfg))

    cfg, user_cfg_loaded = pipeline.load_pipeline_config(str(tmp_yaml), return_user_config=True)

    class Args: pass
    args = Args()
    args.target_size = 128
    # set other required attrs for the helper (defaults)
    args.disable_augmentations = False
    args.architectures = None
    args.raw_data_dir = None
    args.preprocessed_data_dir = None
    args.test_data_dir = None
    args.config_dir = None
    args.checkpoint_dir = None
    args.results_dir = None
    args.split_type = None
    args.train_ratio = None
    args.val_ratio = None
    args.test_ratio = None
    args.k_folds = None
    args.random_seed = None
    args.skip_preprocess = False
    args.crop_to_prostate = False
    args.crop_margin = None

    pipeline.apply_cli_overrides_with_yaml_priority(cfg, user_cfg_loaded, args)
    assert cfg['preprocessing']['target_size'] == 128


def test_force_cli_allows_cli_override_for_paths(tmp_path, monkeypatch):
    # YAML définit test_data_dir -> normalement YAML gagne, mais --force-cli + CLI explicite doit permettre l'override
    tmp_yaml = tmp_path / "tmp_pipeline.yaml"
    user_cfg = {'paths': {'test_data_dir': './data/my_test_set'}}
    tmp_yaml.write_text(yaml.safe_dump(user_cfg))

    cfg, user_cfg_loaded = pipeline.load_pipeline_config(str(tmp_yaml), return_user_config=True)

    class Args: pass
    args = Args()
    args.test_data_dir = './data/cli_test_set'
    args.force_cli = True
    # other args default to None/False
    args.disable_augmentations = False
    args.architectures = None
    args.raw_data_dir = None
    args.preprocessed_data_dir = None
    args.config_dir = None
    args.checkpoint_dir = None
    args.results_dir = None
    args.split_type = None
    args.train_ratio = None
    args.val_ratio = None
    args.test_ratio = None
    args.k_folds = None
    args.random_seed = None
    args.target_size = None
    args.skip_preprocess = False
    args.crop_to_prostate = False
    args.crop_margin = None

    # Simulate that CLI explicitly provided --test_data_dir
    monkeypatch.setattr('sys.argv', ['pipeline.py', '--test_data_dir', './data/cli_test_set', '--force-cli'])

    pipeline.apply_cli_overrides_with_yaml_priority(cfg, user_cfg_loaded, args)
    assert cfg['paths']['test_data_dir'] == './data/cli_test_set'  # CLI should win when force_cli is True
