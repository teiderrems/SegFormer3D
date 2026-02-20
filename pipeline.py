#!/usr/bin/env python3
"""
Pipeline automatisée d'entraînement et de test pour les architectures 3D
Allant du prétraitement à l'inférence sur les données de test
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path
# sklearn import moved into generate_csv_splits_sklearn to avoid hard dependency at module import time
import random
import yaml

# Logger unifié
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scripts.logger import get_logger, log_pipeline_step, log_section

pipeline_logger = get_logger("pipeline", level="INFO")


def load_pipeline_config(config_path, return_user_config=False):
    """
    Charge la configuration de la pipeline depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier de configuration YAML
        return_user_config: si True, retourne aussi le dictionnaire chargé depuis le YAML

    Returns:
        Dictionnaire de configuration avec valeurs par défaut (et optionnellement le user_config)
    """
    # Configuration par défaut
    default_config = {
        'preprocessing': {
            'target_size': 96,
            'normalize_method': 'minmax',
            'skip_existing': True,
            'crop_to_prostate': False,
            'crop_margin': 2
        },
        'splits': {
            'split_type': 'fixed',
            'train_ratio': 0.8,
            'val_ratio': 0.2,
            'test_ratio': 0.0,
            'k_folds': 5,
            'random_seed': 42,
            'stratified': True
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 2,
            'learning_rate': 0.001,
            'device': 'cuda'
        },
        'architectures': {
            'enabled': ['SegFormer3D']
        },
        'paths': {
            'raw_data_dir': './data/raw_prostate',
            'preprocessed_data_dir': './data/preprocessed_data_128_128_128',
            'test_data_dir': './data/preprocessed_data_128_128_128',
            'config_dir': './configs',
            'checkpoint_dir': './checkpoints',
            'results_dir': './results'
        },
        'augmentations': {
            # Par défaut les augmentations sont activées (comportement historique)
            'enabled': True
        },
        'dataset_parameters': {
            'dataset_type': 'prostate_seg',
            'train_dataset_args': {
                'train': True,
                'debug_augment': False
            },
            'val_dataset_args': {
                'train': False,
                'debug_augment': False
            },
            'test_dataset_args': {
                'train': False,
                'debug_augment': False
            }
        },
        'dataloader': {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True
        },
        'advanced': {
            'skip_preprocess': False,
            'verbosity': 1
        }
    }

    # Charge la configuration depuis le fichier si elle existe
    user_config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}

            # Fusionne avec la configuration par défaut
            def merge_configs(default, user):
                if not isinstance(default, dict) or not isinstance(user, dict):
                    return user if user is not None else default
                result = default.copy()
                for key, value in user.items():
                    if key in result and isinstance(result[key], dict):
                        result[key] = merge_configs(result[key], value)
                    else:
                        result[key] = value
                return result

            config = merge_configs(default_config, user_config)

            # Normalisation: si l'utilisateur **n'a pas** défini explicitement
            # `paths.test_data_dir` dans le YAML, on l'aligne sur
            # `paths.preprocessed_data_dir` pour éviter des chemins divergents
            # entre `preprocessed_data_dir` (souvent personnalisé) et la valeur
            # par défaut statique de `test_data_dir`.
            # Ne pas écraser si l'utilisateur a explicitement défini `test_data_dir`.
            try:
                user_paths = (user_config or {}).get('paths', {}) if isinstance(user_config, dict) else {}
                cfg_paths = config.setdefault('paths', {})
                if 'test_data_dir' not in user_paths:
                    cfg_paths['test_data_dir'] = cfg_paths.get('preprocessed_data_dir', cfg_paths.get('test_data_dir'))
            except Exception:
                # En cas d'erreur inattendue, ne pas empêcher le chargement de la config
                pipeline_logger.debug('Impossible de normaliser test_data_dir à partir de preprocessed_data_dir')

            pipeline_logger.info(f"Configuration chargée depuis: {config_path}")
            return (config, user_config) if return_user_config else config

        except Exception as e:
            pipeline_logger.error(f"Erreur lors du chargement de {config_path}: {e}")
            pipeline_logger.warning("Utilisation de la configuration par défaut")
            return (default_config, {}) if return_user_config else default_config
    else:
        pipeline_logger.warning("Fichier de configuration non trouvé, utilisation des paramètres par défaut")
        return (default_config, {}) if return_user_config else default_config


def apply_cli_overrides_with_yaml_priority(config, user_cfg, args):
    """
    Applique les arguments CLI au dictionnaire `config` **seulement si** la clé
    correspondante n'est pas explicitement définie dans `user_cfg` (YAML),
    sauf si l'utilisateur a passé `--force-cli` : dans ce cas les options CLI
    explicites prennent priorité.

    - `config` est modifié in-place.
    - `user_cfg` doit être le dict tel que lu depuis le YAML (vide si aucun YAML).
    """
    import sys

    def user_cfg_has(user_cfg_dict, key_path):
        cur = user_cfg_dict or {}
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                return False
            cur = cur[k]
        return True

    # Helper pour savoir si une option CLI a été explicitement fournie
    def cli_provided(option_name):
        for a in sys.argv[1:]:
            if a == option_name or a.startswith(option_name + "="):
                return True
        return False

    force = getattr(args, 'force_cli', False)

    # Appliquer les arguments CLI **seulement si** la clé correspondante n'est pas définie dans le YAML
    # ou si --force-cli a été activé ET l'option CLI a été explicitement fournie.
    if getattr(args, 'disable_augmentations', False) and (force and cli_provided('--disable_augmentations') or not user_cfg_has(user_cfg, ['augmentations', 'enabled'])):
        config['augmentations']['enabled'] = False

    if getattr(args, 'architectures', None) and (force and cli_provided('--architectures') or not user_cfg_has(user_cfg, ['architectures', 'enabled'])):
        config['architectures']['enabled'] = args.architectures
    if getattr(args, 'raw_data_dir', None) and (force and cli_provided('--raw_data_dir') or not user_cfg_has(user_cfg, ['paths', 'raw_data_dir'])):
        config['paths']['raw_data_dir'] = args.raw_data_dir
    if getattr(args, 'preprocessed_data_dir', None) and (force and cli_provided('--preprocessed_data_dir') or not user_cfg_has(user_cfg, ['paths', 'preprocessed_data_dir'])):
        config['paths']['preprocessed_data_dir'] = args.preprocessed_data_dir
    if getattr(args, 'test_data_dir', None) and (force and cli_provided('--test_data_dir') or not user_cfg_has(user_cfg, ['paths', 'test_data_dir'])):
        config['paths']['test_data_dir'] = args.test_data_dir
    if getattr(args, 'config_dir', None) and (force and cli_provided('--config_dir') or not user_cfg_has(user_cfg, ['paths', 'config_dir'])):
        config['paths']['config_dir'] = args.config_dir
    if getattr(args, 'checkpoint_dir', None) and (force and cli_provided('--checkpoint_dir') or not user_cfg_has(user_cfg, ['paths', 'checkpoint_dir'])):
        config['paths']['checkpoint_dir'] = args.checkpoint_dir
    if getattr(args, 'results_dir', None) and (force and cli_provided('--results_dir') or not user_cfg_has(user_cfg, ['paths', 'results_dir'])):
        config['paths']['results_dir'] = args.results_dir
    if getattr(args, 'split_type', None) and (force and cli_provided('--split_type') or not user_cfg_has(user_cfg, ['splits', 'split_type'])):
        config['splits']['split_type'] = args.split_type
    if getattr(args, 'train_ratio', None) is not None and (force and cli_provided('--train_ratio') or not user_cfg_has(user_cfg, ['splits', 'train_ratio'])):
        config['splits']['train_ratio'] = args.train_ratio
    if getattr(args, 'val_ratio', None) is not None and (force and cli_provided('--val_ratio') or not user_cfg_has(user_cfg, ['splits', 'val_ratio'])):
        config['splits']['val_ratio'] = args.val_ratio
    if getattr(args, 'test_ratio', None) is not None and (force and cli_provided('--test_ratio') or not user_cfg_has(user_cfg, ['splits', 'test_ratio'])):
        config['splits']['test_ratio'] = args.test_ratio
    if getattr(args, 'k_folds', None) and (force and cli_provided('--k_folds') or not user_cfg_has(user_cfg, ['splits', 'k_folds'])):
        config['splits']['k_folds'] = args.k_folds
    if getattr(args, 'random_seed', None) and (force and cli_provided('--random_seed') or not user_cfg_has(user_cfg, ['splits', 'random_seed'])):
        config['splits']['random_seed'] = args.random_seed
    if getattr(args, 'target_size', None) and (force and cli_provided('--target_size') or not user_cfg_has(user_cfg, ['preprocessing', 'target_size'])):
        config['preprocessing']['target_size'] = args.target_size
    if getattr(args, 'skip_preprocess', False) and (force and cli_provided('--skip_preprocess') or not user_cfg_has(user_cfg, ['advanced', 'skip_preprocess'])):
        config['advanced']['skip_preprocess'] = True
    if getattr(args, 'crop_to_prostate', False) and (force and cli_provided('--crop_to_prostate') or not user_cfg_has(user_cfg, ['preprocessing', 'crop_to_prostate'])):
        config['preprocessing']['crop_to_prostate'] = True
    if getattr(args, 'crop_margin', None) is not None and (force and cli_provided('--crop_margin') or not user_cfg_has(user_cfg, ['preprocessing', 'crop_margin'])):
        config['preprocessing']['crop_margin'] = args.crop_margin

    return config

def generate_csv_splits(preprocessed_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42, architecture="SegFormer3D"):
    """
    Génère les fichiers CSV pour les splits train/validation/test ou validation croisée
    Utilise le script create_prostate_splits.py pour une stratification avancée par classe dominante
    """
    preprocessed_path = Path(preprocessed_dir)
    if not preprocessed_path.exists():
        pipeline_logger.error(f"Répertoire prétraité non trouvé: {preprocessed_path}")
        return False

    # Chemin vers le script create_prostate_splits.py centralisé
    script_path = Path(__file__).parent / "data" / "prostate_raw_data" / "create_prostate_splits.py"
    if not script_path.exists():
        pipeline_logger.warning(f"Script create_prostate_splits.py non trouvé, utilisation de sklearn")
        return generate_csv_splits_sklearn(preprocessed_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed)

    pipeline_logger.info(f"Génération des CSV avec stratification par classe dominante")

    # Déterminer les arguments pour le script
    if split_type == "fixed":
        # Pour split fixe, on utilise train.csv et validation.csv (80-20 par défaut)
        test_size = val_ratio + test_ratio  # Proportion pour validation+test
        cmd = [
            sys.executable, str(script_path),
            "--input_dir", str(preprocessed_path),
            "--output_dir", str(preprocessed_path),
            "--test_size", str(test_size),
            "--test_ratio", str(test_ratio),
            "--random_state", str(random_seed),
            "--stratified", "true"
        ]
    elif split_type == "kfold":
        # Pour k-fold, on génère les fichiers fold_*.csv
        cmd = [
            sys.executable, str(script_path),
            "--input_dir", str(preprocessed_path),
            "--output_dir", str(preprocessed_path),
            "--kfold", str(k_folds),
            "--random_state", str(random_seed),
            "--stratified", "true"
        ]
    else:
        pipeline_logger.error(f"Type de split inconnu: {split_type}. Utilisez 'fixed' ou 'kfold'")
        return False

    # Exécuter le script
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pipeline_logger.debug(result.stdout.strip())
        if result.stderr:
            pipeline_logger.warning(result.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        pipeline_logger.error(f"Erreur create_prostate_splits.py: {e}")
        pipeline_logger.warning("Utilisation de la méthode sklearn comme fallback")
        return generate_csv_splits_sklearn(preprocessed_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed)


def generate_csv_splits_sklearn(preprocessed_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42):
    """Méthode fallback utilisant sklearn pour les splits (sans stratification avancée)

    Note: sklearn import is done locally so the module can be imported even when
    sklearn is not installed (useful for lightweight unit tests).
    """
    # Import sklearn locally to avoid hard import-time dependency
    try:
        from sklearn.model_selection import train_test_split, KFold
    except Exception as e:
        pipeline_logger.error(f"sklearn requis: {e}")
        return False

    preprocessed_path = Path(preprocessed_dir)
    if not preprocessed_path.exists():
        pipeline_logger.error(f"Répertoire prétraité non trouvé: {preprocessed_path}")
        return False

    # Lister tous les patients prétraités
    patients = []
    for item in preprocessed_path.iterdir():
        if item.is_dir() and (item / f"{item.name}_modalities.pt").exists():
            patients.append(item.name)

    if not patients:
        pipeline_logger.error(f"Aucun patient prétraité trouvé dans {preprocessed_path}")
        return False

    patients.sort()  # Pour reproductibilité
    random.seed(random_seed)

    pipeline_logger.info(f"Génération CSV (sklearn) pour {len(patients)} patients, split='{split_type}'")

    if split_type == "fixed":
        # Split fixe train/val/test
        train_patients, temp_patients = train_test_split(patients, train_size=train_ratio, random_state=random_seed)
        val_patients, test_patients = train_test_split(temp_patients, train_size=val_ratio/(val_ratio+test_ratio), random_state=random_seed)

        # Créer les DataFrames
        train_df = pd.DataFrame({
            'data_path': [str(preprocessed_path / p) for p in train_patients],
            'case_name': train_patients
        })
        val_df = pd.DataFrame({
            'data_path': [str(preprocessed_path / p) for p in val_patients],
            'case_name': val_patients
        })
        test_df = pd.DataFrame({
            'data_path': [str(preprocessed_path / p) for p in test_patients],
            'case_name': test_patients
        })

        # Sauvegarder les CSV
        train_df.to_csv(preprocessed_path / "train.csv", index=False)
        val_df.to_csv(preprocessed_path / "validation.csv", index=False)
        test_df.to_csv(preprocessed_path / "test.csv", index=False)

        pipeline_logger.info(f"CSV générés: train ({len(train_patients)}), val ({len(val_patients)}), test ({len(test_patients)})")

    elif split_type == "kfold":
        # Validation croisée k-fold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
            train_patients_fold = [patients[i] for i in train_idx]
            val_patients_fold = [patients[i] for i in val_idx]

            # Créer les DataFrames pour ce fold
            train_df = pd.DataFrame({
                'data_path': [str(preprocessed_path / p) for p in train_patients_fold],
                'case_name': train_patients_fold
            })
            val_df = pd.DataFrame({
                'data_path': [str(preprocessed_path / p) for p in val_patients_fold],
                'case_name': val_patients_fold
            })

            # Sauvegarder les CSV pour ce fold
            train_df.to_csv(preprocessed_path / f"train_fold_{fold+1}.csv", index=False)
            val_df.to_csv(preprocessed_path / f"validation_fold_{fold+1}.csv", index=False)

        pipeline_logger.info(f"CSV générés pour {k_folds} folds")

    else:
        pipeline_logger.error(f"Type de split inconnu: {split_type}")
        return False

    return True

def run_command(command, cwd=None, description=""):
    """Exécute une commande et affiche le résultat"""
    log_section(pipeline_logger, description)
    pipeline_logger.debug(f"Commande: {command}")
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            pipeline_logger.debug(result.stdout.strip())
        if result.stderr:
            pipeline_logger.warning(result.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        pipeline_logger.error(f"Erreur lors de l'exécution: {e}")
        if e.stdout:
            pipeline_logger.debug(f"Sortie: {e.stdout.strip()}")
        if e.stderr:
            pipeline_logger.error(f"Erreur: {e.stderr.strip()}")
        return False

def preprocess_data(architecture, input_dir, output_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42, target_size=96, normalize_method="minmax", skip_existing=True, crop_to_prostate=False, crop_margin=2):
    """Prétraite les données pour une architecture donnée et génère les CSV"""
    # Vérifier le script de prétraitement centralisé
    preprocess_script = Path("data") / "prostate_raw_data" / "prostate_preprocess.py"
    if not preprocess_script.exists():
        pipeline_logger.error(f"Script de prétraitement non trouvé: {preprocess_script}")
        return False

    # Créer le répertoire de sortie si nécessaire
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exécuter le prétraitement avec les paramètres configurables
    command = f"{sys.executable} {preprocess_script} --input_dir {input_dir} --output_dir {output_dir} --target_size {target_size} --normalize_method {normalize_method}"
    if skip_existing:
        command += " --skip_existing"
    if crop_to_prostate:
        command += f" --crop_to_prostate --crop_margin {crop_margin}"

    if not run_command(command, cwd=str(Path(__file__).parent), description=f"Prétraitement des données pour {architecture} (taille: {target_size}x{target_size}x{target_size})"):
        return False

    # Générer les CSV après prétraitement réussi
    if not generate_csv_splits(output_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed, architecture):
        pipeline_logger.error(f"Échec de la génération des CSV pour {architecture}")
        return False

    return True

def train_model(architecture, config_path, checkpoint_path=None, resume_checkpoint=None):
    """Entraîne le modèle pour une architecture donnée.
    
    Args:
        architecture: Nom de l'architecture.
        config_path: Chemin vers la config YAML.
        checkpoint_path: Checkpoint pour le fine-tuning (reset optimiseur).
        resume_checkpoint: Checkpoint pour reprendre l'entraînement (restaure tout).
    """
    # Trouver le script d'entraînement centralisé
    train_script = Path("train_scripts") / "trainer_ddp.py"
    if not train_script.exists():
        pipeline_logger.error(f"Script d'entraînement non trouvé: {train_script}")
        return False

    # Commande d'entraînement
    absolute_config = Path(config_path).resolve()
    command = f"{sys.executable} {train_script} --config {absolute_config}"
    if resume_checkpoint:
        absolute_resume = Path(resume_checkpoint).resolve()
        command += f" --resume {absolute_resume}"
    elif checkpoint_path:
        absolute_checkpoint = Path(checkpoint_path).resolve()
        command += f" --checkpoint {absolute_checkpoint}"

    log_section(pipeline_logger, f"Entraînement {architecture}")
    if resume_checkpoint:
        pipeline_logger.info(f"Reprise de l'entraînement depuis: {resume_checkpoint}")
    elif checkpoint_path:
        pipeline_logger.info(f"Fine-tuning depuis: {checkpoint_path}")
    pipeline_logger.debug(f"Commande: {command}")
    result = subprocess.run(command, shell=True, cwd=str(Path(__file__).parent))
    if result.returncode == 0:
        pipeline_logger.info("Entraînement terminé avec succès")
        return True
    else:
        pipeline_logger.error(f"Erreur d'entraînement (code: {result.returncode})")
        return False

def run_inference(architecture, config_path, checkpoint_path, test_data_dir, output_dir):
    """Exécute l'inférence sur les données de test"""
    # Trouver le script d'inférence centralisé
    inference_script = Path("inference_simple.py")
    if not inference_script.exists():
        pipeline_logger.error(f"Script d'inférence non trouvé: {inference_script}")
        return False

    # Charger la configuration pour récupérer les paramètres d'inférence
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    inference_params = config.get('inference_parameters', {})
    batch_size = inference_params.get('batch_size', 1)
    device = inference_params.get('device', 'cuda')
    save_predictions = inference_params.get('save_predictions', True)
    save_probabilities = inference_params.get('save_probabilities', False)
    save_nifti = inference_params.get('save_nifti', True)
    threshold = inference_params.get('threshold', 0.5)

    # Créer le répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Traiter les patients de test — si un `test.csv` est présent, l'utiliser
    test_data_path = Path(test_data_dir)
    if not test_data_path.exists():
        pipeline_logger.error(f"Répertoire de test non trouvé: {test_data_path}")
        return False

    # Si test.csv existe, respecter la liste de cas indiquée (supporte colonnes `data_path` et/ou `case_name`)
    csv_path = test_data_path / 'test.csv'
    patients = []
    if csv_path.exists():
        import csv
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Priorité à data_path si fourni
                    dp = row.get('data_path', '').strip() if row.get('data_path') is not None else ''
                    case = row.get('case_name', '').strip() if row.get('case_name') is not None else ''

                    candidate = None
                    if dp:
                        p = Path(dp)
                        if not p.is_absolute():
                            # essayer relatif au repo root, puis relatif à test_data_path
                            repo_root = Path(__file__).resolve().parents[1]
                            p_repo = (repo_root / dp).resolve()
                            p_local = (test_data_path / dp).resolve()
                            if p_repo.exists():
                                candidate = p_repo
                            elif p_local.exists():
                                candidate = p_local
                            else:
                                candidate = p.resolve()
                        else:
                            candidate = p.resolve()
                    elif case:
                        candidate = (test_data_path / case).resolve()

                    if candidate and candidate.exists() and candidate.is_dir():
                        patients.append(candidate)
                    else:
                        pipeline_logger.warning(f"Patient listé dans test.csv introuvable ou non-dossier: {row}")
        except Exception as e:
            pipeline_logger.error(f"Impossible de lire {csv_path}: {e}")
            return False
    else:
        # Ancien comportement: lister tous les dossiers présents dans le répertoire prétraité
        patients = [p for p in test_data_path.iterdir() if p.is_dir()]

    if not patients:
        pipeline_logger.error(f"Aucun patient trouvé pour inférence dans {test_data_path}")
        return False

    success_count = 0
    from tqdm import tqdm
    for patient_dir in tqdm(patients, desc=f"Inférence {architecture}", unit="patient"):
        patient_output_dir = output_dir / patient_dir.name
        patient_output_dir.mkdir(exist_ok=True)

        # Chemins relatifs depuis le répertoire de l'architecture
        # Utiliser chemins absolus pour l'inférence
        cfg = Path(config_path).resolve()
        chk = Path(checkpoint_path).resolve()
        inp = Path(patient_dir).resolve()
        outp = Path(patient_output_dir).resolve()

        # Commande d'inférence pour ce patient
        command = f"{sys.executable} {inference_script} --config {cfg} --checkpoint {chk} --input_dir {inp} --output_dir {outp} --batch_size {batch_size} --device {device}"
        if save_predictions:
            command += " --save_predictions"
        if save_probabilities:
            command += " --save_probabilities"
        if save_nifti:
            command += " --save_nifti"
        command += f" --threshold {threshold}"
        if run_command(command, cwd=str(Path(__file__).parent), description=f"Inférence pour {architecture} - {patient_dir.name}"):
            success_count += 1
        else:
            pipeline_logger.warning(f"Échec pour le patient {patient_dir.name}")

    pipeline_logger.info(f"Inférence terminée: {success_count}/{len(patients)} patients traités")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline automatisée d'entraînement et test pour architectures 3D"
    )
    parser.add_argument("--config", type=str, default="./pipeline_config.yaml",
                       help="Fichier de configuration YAML (défaut: ./pipeline_config.yaml)")
    parser.add_argument("--architectures", nargs="+",
                       help="Liste des architectures à traiter (remplace la config)")
    parser.add_argument("--raw_data_dir",
                       help="Répertoire des données brutes (remplace la config)")
    parser.add_argument("--preprocessed_data_dir",
                       help="Répertoire pour les données prétraitées (remplace la config)")
    parser.add_argument("--config_dir",
                       help="Répertoire des configurations (remplace la config)")
    parser.add_argument("--checkpoint_dir",
                       help="Répertoire des checkpoints (remplace la config)")
    parser.add_argument("--results_dir",
                       help="Répertoire des résultats d'inférence (remplace la config)")
    parser.add_argument("--skip_preprocess", action="store_true",
                       help="Sauter l'étape de prétraitement")
    parser.add_argument("--skip_training", action="store_true",
                       help="Sauter l'étape d'entraînement (exécute seulement inférence + visualisations)")
    parser.add_argument("--split_type", choices=["fixed", "kfold"],
                       help="Type de split pour les données (remplace la config)")
    parser.add_argument("--train_ratio", type=float,
                       help="Ratio pour l'ensemble d'entraînement (remplace la config)")
    parser.add_argument("--val_ratio", type=float,
                       help="Ratio pour l'ensemble de validation (remplace la config)")
    parser.add_argument("--test_ratio", type=float,
                       help="Ratio pour l'ensemble de test (remplace la config)")
    parser.add_argument("--k_folds", type=int,
                       help="Nombre de folds pour la validation croisée (remplace la config)")
    parser.add_argument("--random_seed", type=int,
                       help="Graine aléatoire pour la reproductibilité (remplace la config)")
    parser.add_argument("--target_size", type=int,
                       help="Taille cible pour le resampling des volumes (remplace la config)")
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help="Liste de checkpoints à inférer (ex: best_model final_model). Si non fourni, utilise la détection automatique")
    parser.add_argument('--finetune_checkpoint', type=str, default=None,
                        help="Chemin vers un checkpoint pour le fine-tuning (remplace l'entraînement from scratch)")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre l'entraînement à l'époque d'interruption (restaure modèle, optimiseur, scheduler, métriques)")
    parser.add_argument('--visualize', action='store_true', help='Générer visualisations et métriques après inférence')
    parser.add_argument('--test_data_dir', type=str, help='Répertoire des données de test (remplace la config)')
    parser.add_argument('--skip_volume', action='store_true', help='Ignorer les visualisations volumétriques 3D pour la génération des visualisations')
    parser.add_argument('--vis_timeout', type=int, default=600, help='Timeout (s) pour chaque visualisation de patient')
    parser.add_argument('--arch_config', type=str, default=None, help='Fichier de configuration d\'architecture à utiliser au lieu de configs/config_<arch>.yaml')

    # Option pour activer/désactiver les augmentations de données au niveau pipeline
    parser.add_argument('--disable_augmentations', action='store_true', help='Désactiver les augmentations de données pendant l\'entraînement')
    parser.add_argument('--crop_to_prostate', action='store_true',
                       help='Supprimer les slices axiales sans prostate avant le resampling')
    parser.add_argument('--crop_margin', type=int,
                       help='Nombre de slices de marge autour de la prostate lors du cropping (défaut: 2, remplace la config)')
    parser.add_argument('--force-cli', action='store_true', help='Forcer les arguments CLI à remplacer les valeurs du YAML (par défaut: YAML > CLI)')

    args = parser.parse_args()

    # Charger la configuration (renvoie aussi user_config pour prioriser YAML)
    cfg_ret = load_pipeline_config(args.config, return_user_config=True)
    if isinstance(cfg_ret, tuple):
        config, user_cfg = cfg_ret
    else:
        config, user_cfg = cfg_ret, {}

    # Appliquer les arguments CLI en respectant la priorité YAML > CLI
    apply_cli_overrides_with_yaml_priority(config, user_cfg, args)

    # Créer les répertoires nécessaires
    Path(config['paths']['preprocessed_data_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)

    # Afficher la configuration utilisée
    pipeline_logger.info("Configuration de la pipeline:")
    pipeline_logger.info(f"  Architectures: {config['architectures']['enabled']}")
    pipeline_logger.info(f"  Données brutes: {config['paths']['raw_data_dir']}")
    pipeline_logger.info(f"  Données prétraitées: {config['paths']['preprocessed_data_dir']}")
    ts = config['preprocessing']['target_size']
    pipeline_logger.info(f"  Taille des volumes: {ts}x{ts}x{ts}")
    crop_status = 'Activé' if config['preprocessing'].get('crop_to_prostate', False) else 'Désactivé'
    pipeline_logger.info(f"  Crop prostate: {crop_status}" + (f" (marge={config['preprocessing'].get('crop_margin', 2)} slices)" if config['preprocessing'].get('crop_to_prostate', False) else ""))
    pipeline_logger.info(f"  Type de split: {config['splits']['split_type']}")
    if config['splits']['split_type'] == 'fixed':
        pipeline_logger.info(f"  Ratios: Train={config['splits']['train_ratio']}, Val={config['splits']['val_ratio']}, Test={config['splits']['test_ratio']}")
    else:
        pipeline_logger.info(f"  Nombre de folds: {config['splits']['k_folds']}")
    pipeline_logger.info(f"  Stratification: {'Activée' if config['splits']['stratified'] else 'Désactivée'}")

    success_count = 0

    # Normaliser architectures en liste (supporte string ou liste dans la config)
    archs = config['architectures']['enabled']
    if isinstance(archs, str):
        archs = [archs]
    config['architectures']['enabled'] = archs

    for arch in config['architectures']['enabled']:
        log_section(pipeline_logger, f"ARCHITECTURE: {arch}")

        # 1. Prétraitement
        # --- Comportement ajouté : si un `test.csv` existe déjà dans le répertoire
        # prétraité, on considère le prétraitement comme déjà fait et on le saute
        # automatiquement. L'utilisateur peut toujours forcer le saut via
        # `--skip_preprocess` ou la config YAML `advanced.skip_preprocess`.
        preprocessed_dir = Path(config['paths']['preprocessed_data_dir'])
        test_csv_path = preprocessed_dir / 'test.csv'

        # Déterminer si l'on doit exécuter le prétraitement
        if config.get('advanced', {}).get('skip_preprocess', False) or getattr(args, 'skip_preprocess', False):
            pipeline_logger.info("--skip_preprocess demandé: saut du prétraitement")
            do_preprocess = False
        elif test_csv_path.exists():
            pipeline_logger.info(f"Fichier de split détecté ({test_csv_path}) — saut automatique du prétraitement")
            do_preprocess = False
        else:
            do_preprocess = True

        if do_preprocess:
            if not preprocess_data(arch, config['paths']['raw_data_dir'], config['paths']['preprocessed_data_dir'],
                                 config['splits']['split_type'], config['splits']['train_ratio'],
                                 config['splits']['val_ratio'], config['splits']['test_ratio'],
                                 config['splits']['k_folds'], config['splits']['random_seed'],
                                 config['preprocessing']['target_size'], config['preprocessing']['normalize_method'],
                                 config['preprocessing']['skip_existing'],
                                 config['preprocessing'].get('crop_to_prostate', False),
                                 config['preprocessing'].get('crop_margin', 2)):
                pipeline_logger.error(f"Échec du prétraitement pour {arch}")
                continue

        # 2. Entraînement
        # Allow overriding the architecture config via CLI (useful for debug runs)
        if args.arch_config:
            config_path = Path(args.arch_config)
        else:
            config_path = Path("configs") / f"config_{arch.lower()}.yaml"

        if not config_path.exists():
            pipeline_logger.error(f"Configuration non trouvée: {config_path}")
            continue

        # If user requested to skip training, bypass the training step and proceed to inference
        if args.skip_training:
            pipeline_logger.info(f"--skip_training activé: sauter l'entraînement pour {arch} et passer à l'inférence")
            train_cfg_to_use = str(config_path)
        else:
            # Load architecture config (and inject pipeline-level preferences)
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    arch_cfg = yaml.safe_load(f) or {}

                # Assurer l'existence des champs attendus
                arch_cfg.setdefault('dataset_parameters', {})
                train_args = arch_cfg['dataset_parameters'].setdefault('train_dataset_args', {})
                val_args = arch_cfg['dataset_parameters'].setdefault('val_dataset_args', {})
                test_args = arch_cfg['dataset_parameters'].setdefault('test_dataset_args', {})

                # Injecter le flag d'augmentations (train: par défaut selon pipeline, val/test: désactivées)
                train_args['augmentations'] = bool(config.get('augmentations', {}).get('enabled', True))
                val_args['augmentations'] = False
                test_args['augmentations'] = False

                # Injecter les chemins root et split_file depuis la config pipeline
                preprocessed_dir = str(Path(config['paths']['preprocessed_data_dir']).resolve())
                train_args.setdefault('root', preprocessed_dir)
                train_args.setdefault('split_file', str(Path(preprocessed_dir) / 'train.csv'))
                val_args.setdefault('root', preprocessed_dir)
                val_args.setdefault('split_file', str(Path(preprocessed_dir) / 'validation.csv'))
                test_data_dir = str(Path(config['paths'].get('test_data_dir', config['paths']['preprocessed_data_dir'])).resolve())
                test_args.setdefault('root', test_data_dir)
                test_args.setdefault('split_file', str(Path(test_data_dir) / 'test.csv'))

                # Injecter la taille cible (target_size) depuis la config pipeline
                ts = config['preprocessing']['target_size']
                train_args.setdefault('target_size', ts)
                val_args.setdefault('target_size', ts)
                test_args.setdefault('target_size', ts)

                # Injecter dataset_type
                pipeline_ds = config.get('dataset_parameters', {})
                if 'dataset_type' in pipeline_ds:
                    arch_cfg['dataset_parameters'].setdefault('dataset_type', pipeline_ds['dataset_type'])

                # Injecter debug_augment si défini dans pipeline config
                for split_key, split_args in [('train_dataset_args', train_args), ('val_dataset_args', val_args), ('test_dataset_args', test_args)]:
                    pipeline_split = pipeline_ds.get(split_key, {})
                    if 'debug_augment' in pipeline_split:
                        split_args.setdefault('debug_augment', pipeline_split['debug_augment'])

                # Injecter les paramètres du dataloader depuis la config pipeline
                pipeline_dl = config.get('dataloader', {})
                if pipeline_dl:
                    arch_cfg.setdefault('dataloader', {})
                    for key in ['batch_size', 'shuffle', 'num_workers', 'drop_last']:
                        if key in pipeline_dl:
                            arch_cfg['dataloader'].setdefault(key, pipeline_dl[key])

                # Injecter batch_size et d'autres paramètres training si définis dans pipeline
                pipeline_training = config.get('training', {})
                if pipeline_training:
                    arch_cfg.setdefault('training_parameters', {})
                    if 'batch_size' in pipeline_training:
                        arch_cfg['training_parameters'].setdefault('batch_size', pipeline_training['batch_size'])
                        arch_cfg.setdefault('dataloader', {}).setdefault('batch_size', pipeline_training['batch_size'])
                    if 'num_epochs' in pipeline_training:
                        arch_cfg['training_parameters'].setdefault('num_epochs', pipeline_training['num_epochs'])
                    if 'learning_rate' in pipeline_training:
                        arch_cfg['training_parameters'].setdefault('learning_rate', pipeline_training['learning_rate'])
                    if 'device' in pipeline_training:
                        arch_cfg['training_parameters'].setdefault('device', pipeline_training['device'])

                # Écrire dans un fichier temporaire pour éviter d'écraser la config d'origine
                import tempfile
                tmp_cfg_file = Path(tempfile.gettempdir()) / f"config_{arch.lower()}_pipeline_tmp.yaml"
                with open(tmp_cfg_file, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(arch_cfg, f, default_flow_style=False, allow_unicode=True)

                train_cfg_to_use = str(tmp_cfg_file)
            except Exception as e:
                pipeline_logger.error(f"Erreur injection augmentations: {e}")
                train_cfg_to_use = str(config_path)

            if not train_model(arch, train_cfg_to_use, args.finetune_checkpoint, args.resume_checkpoint):
                pipeline_logger.error(f"Échec de l'entraînement pour {arch}")
                continue

        # 3. Inférence
        # Chercher les checkpoints demandés et exécuter l'inférence pour chacun
        checkpoint_dir_arch = Path(config['paths']['checkpoint_dir']) / arch
        repo_ckpt_dir = Path(config['paths']['checkpoint_dir'])

        # Répertoire des données de test (peut être différent des données prétraitées pour train/val)
        # Toujours initialiser `test_data_dir` afin d'éviter UnboundLocalError lorsque
        # des checkpoints sont fournis explicitement via CLI.
        test_data_dir = Path(config['paths'].get('test_data_dir', config['paths']['preprocessed_data_dir']))

        # Déterminer la liste de checkpoints à exécuter (argument CLI > défaut)
        if args.checkpoints:
            requested_ckpts = args.checkpoints
        else:
            # Par défaut on essaie 'best_model' puis 'final_model'
            requested_ckpts = ['best_model', 'final_model']

        base_results_dir = Path(config['paths']['results_dir']) / arch

        any_success_for_arch = False
        for ckpt_name in requested_ckpts:
            # Rechercher des fichiers correspondant au nom du checkpoint (ex: best_model*) dans plusieurs emplacements
            pattern = f"{ckpt_name}*"

            # Collecte brute (peut contenir des fichiers .txt d'information)
            raw_candidates = list(repo_ckpt_dir.glob(pattern))
            if checkpoint_dir_arch.exists():
                raw_candidates += list(checkpoint_dir_arch.glob(pattern))

            # Ne conserver que des fichiers de modèles reconnus (éviter .txt / .json)
            allowed_exts = {'.pth', '.pt', '.ckpt', '.tar'}
            candidates = [p for p in raw_candidates if p.suffix.lower() in allowed_exts]

            # Si aucun fichier valide pour le pattern, fallback vers .pth/.pt présents dans les répertoires
            if not candidates:
                candidates = list(repo_ckpt_dir.glob("*.pth")) + list(repo_ckpt_dir.glob("*.pt"))
                if checkpoint_dir_arch.exists():
                    candidates += list(checkpoint_dir_arch.glob("*.pth")) + list(checkpoint_dir_arch.glob("*.pt"))

            if not candidates:
                pipeline_logger.warning(f"Aucun checkpoint pour '{ckpt_name}'. Ignoré.")
                continue

            # Choisir le plus récent parmi les candidats valides
            checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)
            pipeline_logger.info(f"Checkpoint '{ckpt_name}': {checkpoint_path}")

            # Résultats séparés par tag (ex: results/SegFormer3D/best_model/)
            results_dir = base_results_dir / ckpt_name
            results_dir.mkdir(parents=True, exist_ok=True)

            # Lancer l'inférence
            ok = run_inference(arch, str(config_path), str(checkpoint_path), str(test_data_dir), str(results_dir))
            if not ok:
                pipeline_logger.error(f"Échec inférence {arch} avec {checkpoint_path}")
                continue

            any_success_for_arch = True

            # Après inférence, générer les visualisations/métriques si demandé
            if args.visualize:
                vis_cmd = f"{sys.executable} scripts/run_visualizations_all.py --verbosity normal --results_subdir {ckpt_name} --vis_tag {ckpt_name}"
                if args.skip_volume:
                    vis_cmd += " --skip_volume"
                if args.vis_timeout and args.vis_timeout > 0:
                    vis_cmd += f" --timeout {args.vis_timeout}"

                pipeline_logger.info(f"Visualisations pour {arch} / {ckpt_name}")
                run_command(vis_cmd, cwd=str(Path(__file__).parent), description=f"Visualisations {arch} ({ckpt_name})")

        if not any_success_for_arch:
            pipeline_logger.warning(f"Aucune inférence réussie pour {arch}")
            continue

        success_count += 1
        pipeline_logger.info(f"Pipeline terminée avec succès pour {arch}")

    pipeline_logger.info("")
    pipeline_logger.info("=" * 60)
    pipeline_logger.info(f"  PIPELINE TERMINÉE: {success_count}/{len(config['architectures']['enabled'])} architectures")
    pipeline_logger.info("=" * 60)

if __name__ == "__main__":
    main()