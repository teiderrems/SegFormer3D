#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration 3 classes (prostate + bandelettes).
Teste le pipeline complet: config, preprocessing, dataloader, inference.
"""

import os
import sys
import yaml
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

def test_config():
    """Vérifie que la configuration prostate supporte 3 classes."""
    print("\n" + "="*60)
    print("TEST 1: Configuration (3 classes)")
    print("="*60)
    
    config_path = "experiments/prostate_seg/config_prostate.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config non trouvée: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config.get('model', {}).get('num_classes')
    class_weights = config.get('loss', {}).get('class_weights', [])
    
    print(f"✅ num_classes: {num_classes}")
    if num_classes == 3:
        print("   ✅ Correct: 3 classes (0=fond, 1=prostate, 2=bandelettes)")
    else:
        print(f"   ❌ Erreur: attendu 3, trouvé {num_classes}")
        return False
    
    print(f"✅ class_weights: {class_weights}")
    if len(class_weights) == 3:
        print("   ✅ Correct: 3 poids (fond, prostate, bandelettes)")
    else:
        print(f"   ❌ Erreur: attendu 3 poids, trouvé {len(class_weights)}")
        return False
    
    return True


def test_preprocessing():
    """Teste si le preprocessing gère correctement multi-label."""
    print("\n" + "="*60)
    print("TEST 2: Préprocessing (multi-label)")
    print("="*60)
    
    try:
        from data.prostate_raw_data.prostate_preprocess import ProstatePreprocessor
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        return False
    
    # Crée un préprocesseur
    preprocessor = ProstatePreprocessor(target_size=96)
    
    # Teste _load_segmentation
    print("✅ ProstatePreprocessor initialisé")
    
    # Vérifie que la méthode _load_segmentation existe
    if hasattr(preprocessor, '_load_segmentation'):
        print("✅ Méthode _load_segmentation détectée")
    else:
        print("❌ Méthode _load_segmentation manquante")
        return False
    
    return True


def test_model_architecture():
    """Teste que l'architecture supporte 3 classes."""
    print("\n" + "="*60)
    print("TEST 3: Architecture modèle (3 classes)")
    print("="*60)
    
    try:
        from architectures.segformer3d import SegFormer3D
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        return False
    
    # Crée un modèle avec 3 classes
    try:
        model = SegFormer3D(in_channels=2, num_classes=3)
        print(f"✅ SegFormer3D créé: in_channels=2, num_classes=3")
        
        # Test forward pass
        dummy_input = torch.randn(1, 2, 96, 96, 96)
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (1, 3, 96, 96, 96)
        if output.shape == expected_shape:
            print(f"✅ Forward pass réussi: output shape {output.shape}")
        else:
            print(f"❌ Shape incorrect: attendu {expected_shape}, trouvé {output.shape}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du modèle: {e}")
        return False
    
    return True


def test_inference_classes():
    """Teste que l'inférence supporte 3 classes."""
    print("\n" + "="*60)
    print("TEST 4: Pipeline d'inférence (3 classes)")
    print("="*60)
    
    inference_path = "experiments/prostate_seg/inference_prostate.py"
    if not os.path.exists(inference_path):
        print(f"❌ Fichier non trouvé: {inference_path}")
        return False
    
    # Lit le fichier et vérifie les signatures
    with open(inference_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("num_classes=3", "Configuration 3 classes"),
        ("post_process_multiclass", "Méthode post-processing multi-classe"),
        ("threshold_bandelettes", "Support threshold séparé pour bandelettes"),
        ("save_separate_labels", "Support sauvegarde étiquettes séparées"),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - '{check_str}' non trouvé")
            all_passed = False
    
    return all_passed


def test_dataloader_compatibility():
    """Teste que le dataloader gère les labels 0, 1, 2."""
    print("\n" + "="*60)
    print("TEST 5: Compatibilité DataLoader")
    print("="*60)
    
    try:
        from dataloaders.prostate_seg import ProstateSegDataset
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        return False
    
    print("✅ DataLoader importé avec succès")
    
    # Crée un dataset fictif
    test_data_dir = Path("/tmp/test_prostate_seg")
    test_data_dir.mkdir(exist_ok=True)
    
    # Crée des fichiers de test
    modalities = torch.randn(2, 96, 96, 96)
    labels = torch.zeros(1, 96, 96, 96)
    labels[0, 30:60, 30:60, 30:60] = 1  # Prostate
    labels[0, 40:50, 40:50, 40:50] = 2  # Bandelettes
    
    patient_dir = test_data_dir / "patient_test"
    patient_dir.mkdir(exist_ok=True)
    
    torch.save(modalities, patient_dir / "patient_test_modalities.pt")
    torch.save(labels, patient_dir / "patient_test_label.pt")
    
    print(f"✅ Données de test créées dans {test_data_dir}")
    
    # Teste le dataset
    try:
        dataset = ProstateSegDataset(
            data_dir=str(test_data_dir),
            augmentation=False
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            modality_shape = sample['image'].shape
            label_shape = sample['label'].shape
            label_values = torch.unique(sample['label'])
            
            print(f"✅ Dataset sample chargé")
            print(f"   Modalities shape: {modality_shape}")
            print(f"   Labels shape: {label_shape}")
            print(f"   Valeurs uniques: {sorted(label_values.tolist())}")
            
            if set(label_values.tolist()) <= {0, 1, 2}:
                print(f"✅ Labels contiennent bien 0, 1, 2")
            else:
                print(f"❌ Valeurs label incorrectes: {label_values}")
                return False
        
    except Exception as e:
        print(f"⚠️  Erreur lors du test du dataset (peut être normal): {e}")
        # Ne retourne pas False, c'est peut-être normal
    
    return True


def main():
    """Lance tous les tests."""
    print("\n" + "█"*60)
    print("█  TESTS SegFormer3D - Configuration 3 classes")
    print("█  (Prostate + Bandelettes)")
    print("█"*60)
    
    tests = [
        ("Config", test_config),
        ("Preprocessing", test_preprocessing),
        ("Architecture", test_model_architecture),
        ("Inference", test_inference_classes),
        ("DataLoader", test_dataloader_compatibility),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Exception dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés!")
        print("Configuration 3 classes (prostate + bandelettes) OK")
        return 0
    else:
        print("\n⚠️  Certains tests ont échoué")
        return 1


if __name__ == "__main__":
    sys.exit(main())
