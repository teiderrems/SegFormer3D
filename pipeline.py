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
from sklearn.model_selection import train_test_split, KFold
import random
import yaml


def load_pipeline_config(config_path):
    """
    Charge la configuration de la pipeline depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier de configuration YAML

    Returns:
        Dictionnaire de configuration avec valeurs par défaut
    """
    # Configuration par défaut
    default_config = {
        'preprocessing': {
            'target_size': 96,
            'normalize_method': 'minmax',
            'skip_existing': True
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
            'config_dir': './configs',
            'checkpoint_dir': './checkpoints',
            'results_dir': './results'
        },
        'advanced': {
            'skip_preprocess': False,
            'verbosity': 1
        }
    }

    # Charge la configuration depuis le fichier si elle existe
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

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
            print(f"Configuration chargée depuis: {config_path}")
            return config

        except Exception as e:
            print(f"Erreur lors du chargement de la configuration {config_path}: {e}")
            print("Utilisation de la configuration par défaut")
            return default_config
    else:
        print("Fichier de configuration non trouvé, utilisation des paramètres par défaut")
        return default_config

def generate_csv_splits(preprocessed_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42, architecture="SegFormer3D"):
    """
    Génère les fichiers CSV pour les splits train/validation/test ou validation croisée
    Utilise le script create_prostate_splits.py pour une stratification avancée par classe dominante
    """
    preprocessed_path = Path(preprocessed_dir)
    if not preprocessed_path.exists():
        print(f"Répertoire prétraité non trouvé: {preprocessed_path}")
        return False

    # Chemin vers le script create_prostate_splits.py centralisé
    script_path = Path(__file__).parent / "data" / "prostate_raw_data" / "create_prostate_splits.py"
    if not script_path.exists():
        print(f"Script create_prostate_splits.py non trouvé: {script_path}")
        print("Utilisation de la méthode sklearn simple...")
        return generate_csv_splits_sklearn(preprocessed_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed)

    print(f"Génération des CSV pour les données prétraitées avec stratification par classe dominante")
    print(f"Utilisation du script: {script_path}")

    # Déterminer les arguments pour le script
    if split_type == "fixed":
        # Pour split fixe, on utilise train.csv et validation.csv (80-20 par défaut)
        test_size = val_ratio + test_ratio  # Proportion pour validation+test
        cmd = [
            sys.executable, str(script_path),
            "--input_dir", str(preprocessed_path),
            "--output_dir", str(preprocessed_path),
            "--test_size", str(test_size),
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
        print(f"Type de split inconnu: {split_type}. Utilisez 'fixed' ou 'kfold'")
        return False

    # Exécuter le script
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Sortie du script de génération de splits:")
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script create_prostate_splits.py: {e}")
        print("Sortie d'erreur:", e.stderr)
        print("Utilisation de la méthode sklearn simple comme fallback...")
        return generate_csv_splits_sklearn(preprocessed_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed)


def generate_csv_splits_sklearn(preprocessed_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42):
    """Méthode fallback utilisant sklearn pour les splits (sans stratification avancée)"""
    preprocessed_path = Path(preprocessed_dir)
    if not preprocessed_path.exists():
        print(f"Répertoire prétraité non trouvé: {preprocessed_path}")
        return False

    # Lister tous les patients prétraités
    patients = []
    for item in preprocessed_path.iterdir():
        if item.is_dir() and (item / f"{item.name}_modalities.pt").exists():
            patients.append(item.name)

    if not patients:
        print(f"Aucun patient prétraité trouvé dans {preprocessed_path}")
        return False

    patients.sort()  # Pour reproductibilité
    random.seed(random_seed)

    print(f"Génération des CSV (méthode sklearn) pour {len(patients)} patients avec split_type='{split_type}'")

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

        print(f"CSV générés: train ({len(train_patients)}), validation ({len(val_patients)}), test ({len(test_patients)})")

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

        print(f"CSV générés pour {k_folds} folds de validation croisée")

    else:
        print(f"Type de split inconnu: {split_type}. Utilisez 'fixed' ou 'kfold'")
        return False

    return True

def run_command(command, cwd=None, description=""):
    """Exécute une commande et affiche le résultat"""
    print(f"\n=== {description} ===")
    print(f"Commande: {command}")
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, check=True)
        print("Sortie standard:")
        print(result.stdout)
        if result.stderr:
            print("Erreurs:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution: {e}")
        print(f"Sortie standard: {e.stdout}")
        print(f"Erreurs: {e.stderr}")
        return False

def preprocess_data(architecture, input_dir, output_dir, split_type="fixed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, k_folds=5, random_seed=42, target_size=96, normalize_method="minmax", skip_existing=True):
    """Prétraite les données pour une architecture donnée et génère les CSV"""
    # Vérifier le script de prétraitement centralisé
    preprocess_script = Path("data") / "prostate_raw_data" / "prostate_preprocess.py"
    if not preprocess_script.exists():
        print(f"Script de prétraitement non trouvé: {preprocess_script}")
        return False

    # Créer le répertoire de sortie si nécessaire
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exécuter le prétraitement avec les paramètres configurables
    command = f"{sys.executable} {preprocess_script} --input_dir {input_dir} --output_dir {output_dir} --target_size {target_size} --normalize_method {normalize_method}"
    if skip_existing:
        command += " --skip_existing"

    if not run_command(command, cwd=str(Path(__file__).parent), description=f"Prétraitement des données pour {architecture} (taille: {target_size}x{target_size}x{target_size})"):
        return False

    # Générer les CSV après prétraitement réussi
    if not generate_csv_splits(output_dir, split_type, train_ratio, val_ratio, test_ratio, k_folds, random_seed, architecture):
        print(f"Échec de la génération des CSV pour {architecture}")
        return False

    return True

def train_model(architecture, config_path):
    """Entraîne le modèle pour une architecture donnée"""
    # Trouver le script d'entraînement centralisé
    train_script = Path("train_scripts") / "trainer_ddp.py"
    if not train_script.exists():
        print(f"Script d'entraînement non trouvé: {train_script}")
        return False

    # Commande d'entraînement
    absolute_config = Path(config_path).resolve()
    command = f"{sys.executable} {train_script} --config {absolute_config}"

    print(f"\n=== Entraînement du modèle pour {architecture} ===")
    print(f"Commande: {command}")
    result = subprocess.run(command, shell=True, cwd=str(Path(__file__).parent))
    if result.returncode == 0:
        print("Entraînement terminé avec succès")
        return True
    else:
        print(f"Erreur lors de l'entraînement: {result.returncode}")
        return False

def run_inference(architecture, config_path, checkpoint_path, test_data_dir, output_dir):
    """Exécute l'inférence sur les données de test"""
    # Trouver le script d'inférence centralisé
    inference_script = Path("inference_simple.py")
    if not inference_script.exists():
        print(f"Script d'inférence non trouvé: {inference_script}")
        return False

    # Créer le répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Traiter chaque patient dans le répertoire de test
    test_data_path = Path(test_data_dir)
    if not test_data_path.exists():
        print(f"Répertoire de données de test non trouvé: {test_data_path}")
        return False

    patients = [p for p in test_data_path.iterdir() if p.is_dir()]
    if not patients:
        print(f"Aucun patient trouvé dans {test_data_path}")
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
        command = f"{sys.executable} {inference_script} --config {cfg} --checkpoint {chk} --input_dir {inp} --output_dir {outp}"
        if run_command(command, cwd=str(Path(__file__).parent), description=f"Inférence pour {architecture} - {patient_dir.name}"):
            success_count += 1
        else:
            print(f"Échec pour le patient {patient_dir.name}")

    print(f"Inférence terminée: {success_count}/{len(patients)} patients traités")
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

    args = parser.parse_args()

    # Charger la configuration
    config = load_pipeline_config(args.config)

    # Remplacer les paramètres de configuration par les arguments en ligne de commande
    if args.architectures:
        config['architectures']['enabled'] = args.architectures
    if args.raw_data_dir:
        config['paths']['raw_data_dir'] = args.raw_data_dir
    if args.preprocessed_data_dir:
        config['paths']['preprocessed_data_dir'] = args.preprocessed_data_dir
    if args.config_dir:
        config['paths']['config_dir'] = args.config_dir
    if args.checkpoint_dir:
        config['paths']['checkpoint_dir'] = args.checkpoint_dir
    if args.results_dir:
        config['paths']['results_dir'] = args.results_dir
    if args.split_type:
        config['splits']['split_type'] = args.split_type
    if args.train_ratio is not None:
        config['splits']['train_ratio'] = args.train_ratio
    if args.val_ratio is not None:
        config['splits']['val_ratio'] = args.val_ratio
    if args.test_ratio is not None:
        config['splits']['test_ratio'] = args.test_ratio
    if args.k_folds:
        config['splits']['k_folds'] = args.k_folds
    if args.random_seed:
        config['splits']['random_seed'] = args.random_seed
    if args.target_size:
        config['preprocessing']['target_size'] = args.target_size
    if args.skip_preprocess:
        config['advanced']['skip_preprocess'] = True

    # Créer les répertoires nécessaires
    Path(config['paths']['preprocessed_data_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)

    # Afficher la configuration utilisée
    print("Configuration de la pipeline:")
    print(f"  Architectures: {config['architectures']['enabled']}")
    print(f"  Données brutes: {config['paths']['raw_data_dir']}")
    print(f"  Données prétraitées: {config['paths']['preprocessed_data_dir']}")
    print(f"  Taille des volumes: {config['preprocessing']['target_size']}x{config['preprocessing']['target_size']}x{config['preprocessing']['target_size']}")
    print(f"  Type de split: {config['splits']['split_type']}")
    if config['splits']['split_type'] == 'fixed':
        print(f"  Ratios: Train={config['splits']['train_ratio']}, Val={config['splits']['val_ratio']}, Test={config['splits']['test_ratio']}")
    else:
        print(f"  Nombre de folds: {config['splits']['k_folds']}")
    print(f"  Stratification: {'Activée' if config['splits']['stratified'] else 'Désactivée'}")
    print()

    success_count = 0

    for arch in config['architectures']['enabled']:
        print(f"\n{'='*50}")
        print(f"TRAITEMENT DE L'ARCHITECTURE: {arch}")
        print(f"{'='*50}")

        # 1. Prétraitement
        if not config['advanced']['skip_preprocess']:
            if not preprocess_data(arch, config['paths']['raw_data_dir'], config['paths']['preprocessed_data_dir'],
                                 config['splits']['split_type'], config['splits']['train_ratio'],
                                 config['splits']['val_ratio'], config['splits']['test_ratio'],
                                 config['splits']['k_folds'], config['splits']['random_seed'],
                                 config['preprocessing']['target_size'], config['preprocessing']['normalize_method'],
                                 config['preprocessing']['skip_existing']):
                print(f"Échec du prétraitement pour {arch}")
                continue

        # 2. Entraînement
        config_path = Path("configs") / f"config_{arch.lower()}.yaml"
        if not config_path.exists():
            print(f"Configuration non trouvée: {config_path}")
            continue

        if not train_model(arch, str(config_path)):
            print(f"Échec de l'entraînement pour {arch}")
            continue

        # 3. Inférence
        # Trouver le checkpoint le plus récent
        checkpoint_dir = Path(config['paths']['checkpoint_dir']) / arch
        if not checkpoint_dir.exists():
            print(f"Répertoire de checkpoints non trouvé: {checkpoint_dir}")
            continue

        checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            print(f"Aucun checkpoint trouvé dans {checkpoint_dir}")
            continue

        # Prendre le checkpoint le plus récent
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        test_data_dir = Path(config['paths']['preprocessed_data_dir'])  # Les patients sont directement dans preprocessed_data pour l'inférence
        results_dir = Path(config['paths']['results_dir']) / arch

        if not run_inference(arch, str(config_path), str(checkpoint_path), str(test_data_dir), str(results_dir)):
            print(f"Échec de l'inférence pour {arch}")
            continue

        success_count += 1
        print(f"Pipeline terminée avec succès pour {arch}")

    print(f"\n{'='*50}")
    print(f"PIPELINE TERMINÉE: {success_count}/{len(config['architectures']['enabled'])} architectures traitées avec succès")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()