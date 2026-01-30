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
