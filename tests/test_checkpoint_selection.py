import os
import sys
import time

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline


def touch(p):
    p.write_text('x')
    # ensure distinct mtime ordering
    os.utime(p, None)


def test_pipeline_chooses_model_file_over_info_txt(tmp_path, monkeypatch):
    """Si les deux fichiers 'best_model_info.txt' et 'best_model.pth' existent,
    la pipeline doit choisir le fichier binaire de modèle (.pth), pas le .txt d'info."""
    # Préparer un répertoire de checkpoints temporaire
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    info_txt = ckpt_dir / "best_model_info.txt"
    pth_file = ckpt_dir / "best_model.pth"

    # Créer les deux fichiers et définir mtime de façon à ce que le .txt soit plus récent
    touch(pth_file)
    # Sleep to guarantee different timestamps on some filesystems
    time.sleep(0.01)
    touch(info_txt)

    # Charger la config par défaut et injecter le checkpoint_dir personnalisé
    cfg = pipeline.load_pipeline_config(None)
    cfg['paths']['checkpoint_dir'] = str(ckpt_dir)

    # Monkeypatch load_pipeline_config pour retourner notre config (avec user_cfg vide)
    monkeypatch.setattr(pipeline, 'load_pipeline_config', lambda *a, **k: (cfg, {}))

    # Stub preprocess_data pour éviter opérations réelles
    monkeypatch.setattr(pipeline, 'preprocess_data', lambda *a, **k: True)

    selected = {}

    def fake_run_inference(arch, cfg_path, ckpt_path, test_data_dir, out_dir):
        selected['ckpt'] = ckpt_path
        return True

    monkeypatch.setattr(pipeline, 'run_inference', fake_run_inference)

    # Simulate CLI: skip training + explicit 'best_model'
    monkeypatch.setattr('sys.argv', ['pipeline.py', '--skip_training', '--checkpoints', 'best_model', '--architectures', 'SegFormer3D'])

    pipeline.main()

    assert 'ckpt' in selected, 'run_inference should have been called'
    assert selected['ckpt'].endswith('.pth'), 'La pipeline doit sélectionner un fichier .pth (pas le .txt)'
