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