SegFormer3D - Segmentation de Prostate avec NII.GZ
================================================

✨ ADAPTATION POUR PROSTATE (Fichiers NII.GZ)

Ce projet contient tous les fichiers nécessaires pour utiliser SegFormer3D 
(Vision Transformer 3D) pour la segmentation de la prostate à partir de 
fichiers IRM au format NIfTI (nii.gz).

📁 FICHIERS CRÉÉS POUR PROSTATE
================================

1. DATALOADERS:
   ✅ dataloaders/prostate_seg.py
      - ProstateSegDataset: Charge données prétraitées en .pt
      - ProstateSegDatasetMultiModal: Support modalités variables

2. PREPROCESSEMENT:
   ✅ data/prostate_raw_data/prostate_preprocess.py
      - Convertit nii.gz → PyTorch tensors
      - Resampling à 96×96×96
      - Normalisation intensités
      - Classe: ProstatePreprocessor

3. SPLIT DONNÉES:
   ✅ data/prostate_raw_data/create_prostate_splits.py
      - Génère train.csv / validation.csv
      - Support 80-20 split ou k-fold
      - CSV: data_path, case_name

4. CONFIGURATION:
   ✅ experiments/prostate_seg/config_prostate.yaml
      - in_channels: 2 (T2, ADC)
      - num_classes: 2 (background, prostate)
      - Augmentations optimisées prostate
      - Hyperparamètres pré-réglés

5. INFÉRENCE:
   ✅ experiments/prostate_seg/inference_prostate.py
      - Classe ProstateInferencer
      - Sliding window pour volumes larges
      - Post-traitement automatique
      - Sauvegarde nii.gz

6. DOCUMENTATION:
   ✅ GUIDE_PROSTATE_COMPLETE_FR.md
      - Guide complet en français
      - Exemples pratiques
      - Dépannage

🚀 DÉMARRAGE RAPIDE
====================

1. PRÉPARER LES DONNÉES:
   
   Structure attendue:
   data/prostate_raw_data/
   ├── patient_001/
   │   ├── T2.nii.gz
   │   ├── ADC.nii.gz
   │   └── segmentation.nii.gz
   ├── patient_002/
   └── ...

2. PRÉTRAITEMENT:
   
   python data/prostate_raw_data/prostate_preprocess.py \
       --input_dir ./data/prostate_raw_data \
       --output_dir ./data/prostate_data/preprocessed

3. SPLITS:
   
   python data/prostate_raw_data/create_prostate_splits.py \
       --input_dir ./data/prostate_data/preprocessed \
       --output_dir ./data/prostate_data

4. ENTRAÎNEMENT:
   
   python train_scripts/trainer_ddp.py \
       --config experiments/prostate_seg/config_prostate.yaml

5. INFÉRENCE:
   
   python experiments/prostate_seg/inference_prostate.py \
       --model_path ./experiments/prostate_seg/checkpoints/best.pt \
       --input_dir ./test_data/raw \
       --output_dir ./test_data/predictions

📊 DIFFÉRENCES PAR RAPPORT À BRATS
===================================

┌──────────────┬─────────────────┬──────────────┐
│ Aspect       │ BraTS (Original)│ Prostate     │
├──────────────┼─────────────────┼──────────────┤
│ Entrée       │ 4 modalités     │ 2 modalités  │
│              │ (T1,T1CE,T2,FL) │ (T2, ADC)    │
├──────────────┼─────────────────┼──────────────┤
│ Classes      │ 3 classes       │ 2 classes    │
│              │ (ED, NCR, TC)   │ (bg, prostate)
├──────────────┼─────────────────┼──────────────┤
│ Format       │ Tenseurs .pt    │ Fichiers     │
│              │ 128×128×128     │ nii.gz 96³   │
├──────────────┼─────────────────┼──────────────┤
│ Dataset      │ brats2021_seg   │ prostate_seg │
│ Type         │ brats2017_seg   │              │
└──────────────┴─────────────────┴──────────────┘

⚙️ MODIFICATIONS CLÉS DU CODE
=============================

1. dataloaders/build_dataset.py:
   + Ajouté support "prostate_seg"
   + Importe ProstateSegDataset
   + Paramètres flexibles (split_file optionnel)

2. architectures/segformer3d.py:
   - Inchangé (architecture flexible)
   - Accepte in_channels=2, num_classes=2

3. config_prostate.yaml:
   + in_channels: 2 (au lieu de 4)
   + num_classes: 2 (au lieu de 3)
   + Augmentations adaptées prostate
   + Class weights pour déséquilibre

📋 CLASSES PRINCIPALES
======================

ProstateSegDataset (dataloaders/prostate_seg.py):
  - Charge données prétraitées en .pt
  - Compatible MONAI transforms
  - Supporte modalités variables

ProstatePreprocessor (data/prostate_raw_data/prostate_preprocess.py):
  - Charge nii.gz avec nibabel
  - Resample 96×96×96
  - Normalisation (minmax ou zscore)
  - Export PyTorch .pt

ProstateInferencer (experiments/prostate_seg/inference_prostate.py):
  - Charge modèle pré-entraîné
  - Sliding window inference
  - Post-traitement automatique
  - Sauvegarde nii.gz

🔧 DÉPENDANCES SUPPLÉMENTAIRES
===============================

# NIfTI I/O
pip install nibabel

# Traitement images (resampling, filtrage)
pip install scikit-image scipy

# Déjà installé
# torch, monai, pandas, numpy

✅ INSTALLATION:
pip install nibabel scikit-image scipy

📚 DOCUMENTATION COMPLÈTE
=========================

Lisez GUIDE_PROSTATE_COMPLETE_FR.md pour:
  ✓ Pipeline complet étape par étape
  ✓ Exemples pratiques (Python, bash)
  ✓ Configuration avancée
  ✓ Dépannage détaillé
  ✓ Métriques d'évaluation
  ✓ Architecture expliquée

🎯 OBJECTIFS DE PERFORMANCE
============================

Dice Score:       > 0.85 (excellent)
Hausdorff Dist:   < 5 mm (bon)
Spécificité:      > 0.95
Sensibilité:      > 0.80

💡 CONSEILS PRATIQUES
====================

1. DONNÉES:
   - Minimum 30-50 patients pour entraînement
   - Assurez-vous que les segmentations sont correctes
   - Vérifiez intensités nii.gz avant preprocessing

2. ENTRAÎNEMENT:
   - Commencez avec 50 epochs pour tester
   - Utilisez GPU (CUDA > 4GB recommandé)
   - Monitorez loss et Dice durant entraînement

3. INFÉRENCE:
   - Utilisez sliding window pour volumes > 96³
   - Ajustez post-processing selon vos besoins
   - Sauvegardez cartes de probabilité pour analyse

📞 STRUCTURE DU PROJET
======================

SegFormer3D/
├── architectures/
│   ├── segformer3d.py              ← Architecture 3D
│   └── build_architecture.py       ← Factory
├── dataloaders/
│   ├── prostate_seg.py             ← NOUVEAU: ProstateSegDataset
│   ├── build_dataset.py            ← MODIFIÉ: Ajoute prostate_seg
│   └── (brats2021_seg.py, ...)
├── data/
│   ├── prostate_raw_data/
│   │   ├── prostate_preprocess.py  ← NOUVEAU: Preprocessing
│   │   └── create_prostate_splits.py ← NOUVEAU: CSV splits
│   └── (brats2017_seg/, ...)
├── experiments/
│   └── prostate_seg/
│       ├── config_prostate.yaml    ← NOUVEAU: Configuration
│       └── inference_prostate.py   ← NOUVEAU: Inférence
├── train_scripts/
│   ├── trainer_ddp.py              ← Entraînement (existant)
│   └── utils.py
├── losses/, metrics/, optimizers/, augmentations/
├── GUIDE_PROSTATE_COMPLETE_FR.md   ← NOUVEAU: Guide complet
└── README.md

🤖 PROCHAINES ÉTAPES
====================

1. □ Organiser données en prostate_raw_data/patient_XXX
2. □ Exécuter prostate_preprocess.py
3. □ Exécuter create_prostate_splits.py
4. □ Adapter config_prostate.yaml si besoin
5. □ Lancer entraînement avec trainer_ddp.py
6. □ Évaluer sur validation set
7. □ Inférence sur test set avec inference_prostate.py

✨ POINTS CLÉS À RETENIR
========================

✓ Format d'entrée: nii.gz (nibabel compatible)
✓ Sortie preprocessing: tenseurs PyTorch .pt
✓ Architecture: Inchangée (in_channels/num_classes flexibles)
✓ Configuration: Adaptée (config_prostate.yaml)
✓ Dataloader: Nouveau ProstateSegDataset
✓ Inférence: Classe ProstateInferencer avec post-traitement

═══════════════════════════════════════════════════════════════

Pour plus de détails, consultez:
📖 GUIDE_PROSTATE_COMPLETE_FR.md

Bon entraînement! 🚀

Dernière mise à jour: 2025-01-01
Version: 1.0
