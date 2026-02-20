import os
import sys

# Ensure project root is importable for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline


def test_skip_preprocess_if_test_csv_exists(monkeypatch, tmp_path):
    """Si `test.csv` est présent dans `preprocessed_data_dir`, le prétraitement
    doit être ignoré automatiquement (même sans `--skip_preprocess`)."""
    # Préparer un répertoire prétraité contenant test.csv
    preproc = tmp_path / 'preprocessed_data_128_128_128'
    preproc.mkdir()
    (preproc / 'test.csv').write_text('patient_001')

    # Charger config et forcer paths.preprocessed_data_dir vers notre dossier temporaire
    cfg, user_cfg = pipeline.load_pipeline_config(None, return_user_config=True)
    cfg['paths']['preprocessed_data_dir'] = str(preproc)

    # Remplacer load_pipeline_config pour utiliser notre config (user_cfg vide)
    monkeypatch.setattr(pipeline, 'load_pipeline_config', lambda *a, **k: (cfg, {}))

    calls = {'preprocess': False, 'inference': False}

    # Stub preprocess_data pour détecter s'il est appelé
    def fake_preprocess(*a, **k):
        calls['preprocess'] = True
        return True
    monkeypatch.setattr(pipeline, 'preprocess_data', fake_preprocess)

    # Stub run_inference pour que la pipeline puisse continuer
    def fake_infer(arch, cfg_path, ckpt_path, test_data_dir, out_dir):
        calls['inference'] = True
        return True
    monkeypatch.setattr(pipeline, 'run_inference', fake_infer)

    # Simuler CLI: skip_training pour éviter l'entraînement réel
    monkeypatch.setattr('sys.argv', ['pipeline.py', '--skip_training', '--architectures', 'SegFormer3D'])

    pipeline.main()

    assert not calls['preprocess'], 'preprocess_data ne doit PAS être appelé quand test.csv existe'
    assert calls['inference'], 'run_inference doit être appelé ensuite'