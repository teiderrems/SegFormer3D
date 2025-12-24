"""
Script pour générer les fichiers train.csv et validation.csv pour données de prostate.

Après exécution de prostate_preprocess.py, utilisez ce script pour:
1. Scanner les répertoires prétraités
2. Générer les fichiers train.csv et validation.csv (avec stratification par classe)
3. Optionnel: générer les fichiers CSV pour k-fold cross-validation

Utilisation:
    # Train/Val simple (80-20 split avec stratification par classe)
    python create_prostate_splits.py \\
        --input_dir ./data/prostate_preprocessed \\
        --output_dir ./data/prostate_data \\
        --test_size 0.2 \\
        --stratified true

    # Avec cross-validation (5-fold avec stratification)
    python create_prostate_splits.py \\
        --input_dir ./data/prostate_preprocessed \\
        --output_dir ./data/prostate_data \\
        --kfold 5 \\
        --stratified true

    # Sans stratification (simple random split)
    python create_prostate_splits.py \\
        --input_dir ./data/prostate_preprocessed \\
        --output_dir ./data/prostate_data \\
        --test_size 0.2 \\
        --stratified false
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import warnings


def get_case_classes(case_name: str, preprocessed_dir: str, num_classes: int = 3) -> int:
    """
    Détermine la classe dominante pour un cas (utilisé pour stratification).
    
    Stratégie: Classe avec le plus de voxels (sauf 0=fond).
    
    Args:
        case_name: Nom du patient
        preprocessed_dir: Répertoire contenant les données
        num_classes: Nombre total de classes
    
    Returns:
        Classe dominante (1 à num_classes-1), ou 0 si fond seul
    """
    label_path = os.path.join(preprocessed_dir, case_name, f"{case_name}_label.pt")
    
    try:
        label = torch.load(label_path)
        
        # Compte les voxels par classe
        class_counts = []
        for c in range(num_classes):
            count = (label == c).sum().item()
            class_counts.append(count)
        
        # Trouve la classe dominante (sauf 0=fond)
        # Classe dominante = classe avec plus de voxels parmi les classes non-fond
        non_bg_classes = [(c, count) for c, count in enumerate(class_counts) if c > 0]
        
        if non_bg_classes:
            dominant_class = max(non_bg_classes, key=lambda x: x[1])[0]
            return dominant_class
        else:
            return 0  # Fond seul
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la lecture {label_path}: {e}")
        return 0


def get_case_names(preprocessed_dir: str) -> List[str]:
    """
    Récupère la liste des noms de patients.
    
    Args:
        preprocessed_dir: Répertoire contenant les données prétraitées
    
    Returns:
        Liste des noms de patients (trié)
    """
    cases = []
    
    for item in os.listdir(preprocessed_dir):
        item_path = os.path.join(preprocessed_dir, item)
        if os.path.isdir(item_path):
            # Vérifie que les fichiers .pt existent
            modality_file = os.path.join(item_path, f"{item}_modalities.pt")
            label_file = os.path.join(item_path, f"{item}_label.pt")
            
            if os.path.exists(modality_file) and os.path.exists(label_file):
                cases.append(item)
    
    return sorted(cases)


def train_val_split(
    cases: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessed_dir: str = None,
    stratified: bool = True,
    num_classes: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée une split train/val simple (avec stratification optionnelle).
    
    Args:
        cases: Liste des noms de patients
        test_size: Proportion pour validation (défaut: 0.2 = 80-20)
        random_state: Seed pour reproductibilité
        preprocessed_dir: Répertoire des données (pour data_path)
        stratified: Si True, stratifie par classe dominante
        num_classes: Nombre de classes pour stratification
    
    Returns:
        (train_df, val_df)
    """
    np.random.seed(random_state)
    
    if stratified:
        # Stratification par classe
        class_indices = {}
        for c in range(num_classes):
            class_indices[c] = []
        
        for case in cases:
            dominant_class = get_case_classes(case, preprocessed_dir, num_classes)
            class_indices[dominant_class].append(case)
        
        train_cases = []
        val_cases = []
        
        # Pour chaque classe, split avec proportions
        for c in range(num_classes):
            class_cases = class_indices[c]
            if len(class_cases) == 0:
                continue
            
            shuffled = np.random.permutation(class_cases)
            split_idx = max(1, int(len(class_cases) * (1 - test_size)))
            
            train_cases.extend(shuffled[:split_idx])
            val_cases.extend(shuffled[split_idx:])
        
        train_cases = np.array(train_cases)
        val_cases = np.array(val_cases)
    else:
        # Split simple sans stratification
        shuffled = np.random.permutation(cases)
        split_idx = int(len(cases) * (1 - test_size))
        
        train_cases = shuffled[:split_idx]
        val_cases = shuffled[split_idx:]
    
    # Crée les DataFrames
    def create_df(case_list, base_path):
        data = []
        for case in sorted(case_list):
            data.append({
                "data_path": f"{base_path}/{case}",
                "case_name": case
            })
        return pd.DataFrame(data)
    
    train_df = create_df(train_cases, preprocessed_dir)
    val_df = create_df(val_cases, preprocessed_dir)
    
    return train_df, val_df


def kfold_split(
    cases: List[str],
    k: int = 5,
    random_state: int = 42,
    preprocessed_dir: str = None,
    stratified: bool = True,
    num_classes: int = 3
) -> Tuple[dict, dict]:
    """
    Crée des splits k-fold pour cross-validation (avec stratification optionnelle).
    
    Args:
        cases: Liste des noms de patients
        k: Nombre de folds (défaut: 5)
        random_state: Seed pour reproductibilité
        preprocessed_dir: Répertoire des données
        stratified: Si True, stratifie par classe dominante dans chaque fold
        num_classes: Nombre de classes pour stratification
    
    Returns:
        (train_dfs_dict, val_dfs_dict) - Chacun contient k DataFrames
    """
    np.random.seed(random_state)
    
    if stratified:
        # Stratification par classe
        class_indices = {}
        for c in range(num_classes):
            class_indices[c] = []
        
        for case in cases:
            dominant_class = get_case_classes(case, preprocessed_dir, num_classes)
            class_indices[dominant_class].append(case)
        
        # Pour chaque classe, divise en k folds
        class_folds = {}
        for c in range(num_classes):
            class_cases = class_indices[c]
            if len(class_cases) == 0:
                class_folds[c] = [[] for _ in range(k)]
                continue
            
            shuffled = np.random.permutation(class_cases)
            fold_size = len(shuffled) // k
            
            folds = []
            for i in range(k):
                start = i * fold_size
                end = start + fold_size if i < k - 1 else len(shuffled)
                folds.append(list(shuffled[start:end]))
            
            class_folds[c] = folds
        
        # Combine les folds de toutes les classes
        folds = [[] for _ in range(k)]
        for c in range(num_classes):
            for fold_idx in range(k):
                folds[fold_idx].extend(class_folds[c][fold_idx])
    else:
        # Split simple sans stratification
        shuffled = np.random.permutation(cases)
        fold_size = len(shuffled) // k
        
        folds = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)
            folds.append(list(shuffled[start:end]))
    
    def create_df(case_list, base_path):
        data = []
        for case in sorted(case_list):
            data.append({
                "data_path": f"{base_path}/{case}",
                "case_name": case
            })
        return pd.DataFrame(data)
    
    # Pour chaque fold, le fold_i est validation, les autres sont train
    train_dfs = {}
    val_dfs = {}
    
    for fold_idx in range(k):
        val_cases = folds[fold_idx]
        train_cases = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        
        train_dfs[fold_idx] = create_df(train_cases, preprocessed_dir)
        val_dfs[fold_idx] = create_df(val_cases, preprocessed_dir)
    
    return train_dfs, val_dfs


def main():
    parser = argparse.ArgumentParser(
        description="Génère les splits train/val pour données de prostate prétraitées"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/prostate_preprocessed",
        help="Répertoire contenant les données prétraitées"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/prostate_data",
        help="Répertoire de sortie pour les CSV"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion pour validation en train/val split (défaut: 0.2)"
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="Nombre de folds pour cross-validation (optionnel)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Si True, stratifie par classe dominante (défaut: True)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Nombre de classes pour stratification (défaut: 3)"
    )
    
    args = parser.parse_args()
    
    # Crée le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Récupère les noms de patients
    cases = get_case_names(args.input_dir)
    
    if not cases:
        print(f"❌ Aucun patient trouvé dans {args.input_dir}")
        print("   Assurez-vous que prostate_preprocess.py a été exécuté correctement")
        return
    
    print(f"\n📊 Génération des splits pour {len(cases)} patients")
    print(f"   Répertoire source: {args.input_dir}")
    print(f"   Répertoire de sortie: {args.output_dir}")
    print(f"   Stratification: {'Activée (par classe dominante)' if args.stratified else 'Désactivée'}")
    print(f"   Nombre de classes: {args.num_classes}\n")
    
    if args.kfold:
        # K-fold cross-validation
        print(f"🔄 Création de {args.kfold}-fold cross-validation...")
        
        train_dfs, val_dfs = kfold_split(
            cases,
            k=args.kfold,
            random_state=args.random_state,
            preprocessed_dir=args.input_dir,
            stratified=args.stratified,
            num_classes=args.num_classes
        )
        
        # Sauvegarde les fichiers CSV
        for fold_idx in range(args.kfold):
            train_file = os.path.join(args.output_dir, f"train_fold_{fold_idx+1}.csv")
            val_file = os.path.join(args.output_dir, f"validation_fold_{fold_idx+1}.csv")
            
            train_dfs[fold_idx].to_csv(train_file, index=False)
            val_dfs[fold_idx].to_csv(val_file, index=False)
            
            print(f"   ✅ Fold {fold_idx+1}: {len(train_dfs[fold_idx])} train, {len(val_dfs[fold_idx])} val")
        
        print(f"\n✅ Fichiers CSV créés:")
        for fold_idx in range(args.kfold):
            print(f"   - train_fold_{fold_idx+1}.csv")
            print(f"   - validation_fold_{fold_idx+1}.csv")
    
    else:
        # Simple train/val split
        print(f"📊 Création d'une split train/val ({1-args.test_size:.0%}/{args.test_size:.0%})...")
        
        train_df, val_df = train_val_split(
            cases,
            test_size=args.test_size,
            random_state=args.random_state,
            preprocessed_dir=args.input_dir,
            stratified=args.stratified,
            num_classes=args.num_classes
        )
        
        train_file = os.path.join(args.output_dir, "train.csv")
        val_file = os.path.join(args.output_dir, "validation.csv")
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        print(f"\n✅ Fichiers CSV créés:")
        print(f"   - train.csv ({len(train_df)} patients)")
        print(f"   - validation.csv ({len(val_df)} patients)")
        
        # Affiche un aperçu
        print(f"\n📋 Aperçu train.csv:")
        print(train_df.head(3).to_string(index=False))
    
    print(f"\n✨ Prêt pour l'entraînement!")


if __name__ == "__main__":
    main()
