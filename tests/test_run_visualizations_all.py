import os
import sys
from pathlib import Path
import types

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_visualizations_all as rva
import yaml

def test_get_test_csv_from_args_test_data_dir(tmp_path):
    d = tmp_path / 'preprocessed_data_test'
    d.mkdir()
    (d / 'test.csv').write_text('patient_001')
    class A: pass
    args = A()
    args.test_csv = None
    args.test_dir = None
    args.test_data_dir = str(d)
    args.config = None
    res = rva.get_test_csv_from_args_and_config(args, config_path=rva.CONFIG, default=rva.TEST_CSV)
    assert res == d / 'test.csv'


def test_get_test_csv_from_config(tmp_path):
    # Make a fake config file pointing to a test dataset
    d = tmp_path / 'cfg_data'
    d.mkdir()
    (d / 'test.csv').write_text('patient_002')
    cfg_file = tmp_path / 'cfg.yaml'
    cfg = {'dataset_parameters': {'test_dataset_args': {'root': str(d)}}}
    cfg_file.write_text(yaml.safe_dump(cfg))

    class A: pass
    args = A()
    args.test_csv = None
    args.test_dir = None
    args.test_data_dir = None

    res = rva.get_test_csv_from_args_and_config(args, config_path=str(cfg_file), default=rva.TEST_CSV)
    assert res == d / 'test.csv'


def test_config_takes_precedence_over_cli(tmp_path):
    # If both CLI and YAML provide a test dataset, YAML must be preferred
    d_cfg = tmp_path / 'cfg_data'
    d_cfg.mkdir()
    (d_cfg / 'test.csv').write_text('patient_cfg')

    d_cli = tmp_path / 'cli_data'
    d_cli.mkdir()
    (d_cli / 'test.csv').write_text('patient_cli')

    cfg_file = tmp_path / 'cfg.yaml'
    cfg = {'dataset_parameters': {'test_dataset_args': {'root': str(d_cfg)}}}
    cfg_file.write_text(yaml.safe_dump(cfg))

    class A: pass
    args = A()
    args.test_csv = None
    args.test_dir = None
    args.test_data_dir = str(d_cli)  # CLI points elsewhere

    res = rva.get_test_csv_from_args_and_config(args, config_path=str(cfg_file), default=rva.TEST_CSV)
    assert res == d_cfg / 'test.csv'  # YAML should win


def test_force_cli_allows_cli_override_of_config(tmp_path):
    # If --force-cli is used, CLI should override the YAML config
    d_cfg = tmp_path / 'cfg_data'
    d_cfg.mkdir()
    (d_cfg / 'test.csv').write_text('patient_cfg')

    d_cli = tmp_path / 'cli_data'
    d_cli.mkdir()
    (d_cli / 'test.csv').write_text('patient_cli')

    cfg_file = tmp_path / 'cfg.yaml'
    cfg = {'dataset_parameters': {'test_dataset_args': {'root': str(d_cfg)}}}
    cfg_file.write_text(yaml.safe_dump(cfg))

    class A: pass
    args = A()
    args.test_csv = None
    args.test_dir = None
    args.test_data_dir = str(d_cli)
    args.config = str(cfg_file)
    args.force_cli = True

    res = rva.get_test_csv_from_args_and_config(args, config_path=str(cfg_file), default=rva.TEST_CSV)
    assert res == d_cli / 'test.csv'  # CLI should win when force_cli is True