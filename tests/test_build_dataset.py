import os
import sys
# Ensure repository root is on sys.path so tests can import local packages
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import types
import dataloaders.build_dataset as bd


def _inject_dummy_prostate_module(monkeypatch):
    """Inject a dummy `dataloaders.prostate_seg` module with a simple
    ProstateSegDataset class so `build_dataset` can be tested without
    external data or heavy dependencies.
    """
    dummy_mod = types.SimpleNamespace()

    class DummyProstateSegDataset:
        def __init__(self, root_dir, is_train, transform, split_file, target_size, debug_augment=False):
            self.root_dir = root_dir
            self.is_train = is_train
            self.transform = transform
            self.split_file = split_file
            self.target_size = target_size
            self.debug_augment = debug_augment

    dummy_mod.ProstateSegDataset = DummyProstateSegDataset
    monkeypatch.setitem(sys.modules, "dataloaders.prostate_seg", dummy_mod)

    # Avoid invoking MONAI transforms during the test
    monkeypatch.setattr(bd, "build_augmentations", lambda train: "DUMMY_TRANSFORM")


def test_train_dataset_prints_augmentations_enabled(monkeypatch, capsys):
    _inject_dummy_prostate_module(monkeypatch)
    dataset_args = {"root": "./data/patient_001", "train": True, "augmentations": True, "target_size": 96}
    ds = bd.build_dataset("prostate_seg", dataset_args)
    captured = capsys.readouterr()
    assert "[INFO] Dataset 'train'" in captured.out
    assert "augmentations: ENABLED" in captured.out


def test_val_dataset_prints_augmentations_disabled(monkeypatch, capsys):
    _inject_dummy_prostate_module(monkeypatch)
    dataset_args = {"root": "./data/patient_001", "train": False, "augmentations": False, "target_size": 96}
    ds = bd.build_dataset("prostate_seg", dataset_args)
    captured = capsys.readouterr()
    assert "[INFO] Dataset 'val'" in captured.out
    assert "augmentations: DISABLED" in captured.out
