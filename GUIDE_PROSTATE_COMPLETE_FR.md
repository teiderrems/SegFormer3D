# Guide Complet: Segmentation de Prostate avec SegFormer3D

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure des données](#structure-des-données)
3. [Installation des dépendances](#installation-des-dépendances)
4. [Pipeline complet](#pipeline-complet)
5. [Exemples pratiques](#exemples-pratiques)
6. [Configuration avancée](#configuration-avancée)
7. [Dépannage](#dépannage)

---

## 🎯 Vue d'ensemble

Ce guide explique comment adapter **SegFormer3D** pour la segmentation de la **prostate** à partir de fichiers **nii.gz** (NIfTI). 

### Différences par rapport à BraTS

| Aspect | BraTS | Prostate |
|--------|-------|----------|
| **Entrée** | 4 modalités (T1, T1CE, T2, FLAIR) | 2 modalités (T2, ADC) |
| **Classes** | 3 (fond, édème, nécrose) | 2 (fond, prostate) |
| **Format** | Tenseurs PyTorch .pt | Fichiers nii.gz |
| **Taille volume** | 128×128×128 | Variable (96×96×96 après resample) |
| **Nombre cas** | Milliers | Centaines à milliers |

---

## 📁 Structure des données

### Données brutes (avant prétraitement)

```
data/prostate_raw_data/
├── patient_001/
│   ├── T2.nii.gz              # IRM T2 (modalité 1)
│   ├── ADC.nii.gz             # IRM ADC (modalité 2)
│   └── segmentation.nii.gz    # Label (0: fond, 1: prostate)
├── patient_002/
│   ├── T2.nii.gz
│   ├── ADC.nii.gz
│   └── segmentation.nii.gz
└── ...
```

### Après prétraitement

```
data/prostate_data/
├── preprocessed/
│   ├── patient_001/
│   │   ├── patient_001_modalities.pt    # (2, 96, 96, 96)
│   │   └── patient_001_label.pt         # (1, 96, 96, 96)
│   ├── patient_002/
│   │   ├── patient_002_modalities.pt
│   │   └── patient_002_label.pt
│   └── ...
├── train.csv        # CSV avec splits train/val
└── validation.csv
```

### Format des CSV

**train.csv** et **validation.csv**:
```csv
data_path,case_name
./data/prostate_data/preprocessed/patient_001,patient_001
./data/prostate_data/preprocessed/patient_002,patient_002
./data/prostate_data/preprocessed/patient_003,patient_003
```

---

## 🔧 Installation des dépendances

### Dépendances supplémentaires pour prostate

```bash
# Fichiers NIfTI
pip install nibabel

# Traitement d'images
pip install scikit-image scipy

# Déjà installé généralement
pip install numpy pandas torch monai
```

Vérification:
```bash
python -c "import nibabel; import scipy; import skimage; print('✅ Toutes les dépendances OK')"
```

---

## 🔄 Pipeline complet

### Étape 1: Prétraitement des données brutes

Convertit les fichiers nii.gz en tenseurs PyTorch normalisés.

```bash
cd /workspaces/SegFormer3D

python data/prostate_raw_data/prostate_preprocess.py \
    --input_dir ./data/prostate_raw_data \
    --output_dir ./data/prostate_data/preprocessed \
    --target_size 96 \
    --normalize_method minmax \
    --skip_existing
```

**Paramètres:**
- `--input_dir`: Répertoire avec structure `patient_XXX/{T2,ADC,segmentation}.nii.gz`
- `--output_dir`: Où sauvegarder les tenseurs .pt
- `--target_size`: Taille de resample (96 par défaut)
- `--normalize_method`: "minmax" (0-1) ou "zscore" (gaussienne)
- `--skip_existing`: Saute les patients déjà prétraités

**Output:**
```
✅ Prétraitement de 100 patients...
✅ RÉSUMÉ: 100/100 patients prétraités avec succès
📁 Données prétraitées dans: ./data/prostate_data/preprocessed
```

### Étape 2: Génération des splits

Crée les fichiers train.csv et validation.csv.

```bash
# Simple train/val (80-20)
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --test_size 0.2
```

**Ou avec 5-fold cross-validation:**

```bash
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --kfold 5
```

**Output:**
```
📊 Génération des splits pour 100 patients
✅ Fichiers CSV créés:
   - train.csv (80 patients)
   - validation.csv (20 patients)
```

### Étape 3: Entraînement

Lance l'entraînement avec la configuration prostate.

```bash
# Single GPU
python train_scripts/trainer_ddp.py \
    --config experiments/prostate_seg/config_prostate.yaml

# Multi-GPU (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_scripts/trainer_ddp.py \
    --config experiments/prostate_seg/config_prostate.yaml
```

### Étape 4: Inférence

Prédit sur de nouvelles données.

```bash
python experiments/prostate_seg/inference_prostate.py \
    --model_path ./experiments/prostate_seg/checkpoints/best.pt \
    --input_dir ./test_data/raw \
    --output_dir ./test_data/predictions \
    --device cuda \
    --save_nifti true \
    --threshold 0.5
```

---

## 💡 Exemples pratiques

### Exemple 1: Workflow complet de 20 patients

```bash
#!/bin/bash
# Script complet pour tester sur 20 patients

# 1. Prétraitement
echo "🔄 Prétraitement..."
python data/prostate_raw_data/prostate_preprocess.py \
    --input_dir ./test_20_patients \
    --output_dir ./data/prostate_test \
    --target_size 96

# 2. Splits
echo "📊 Création des splits..."
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_test \
    --output_dir ./data/prostate_test \
    --test_size 0.2

# 3. Entraînement court (test)
echo "🚀 Entraînement (10 epochs pour test)..."
python train_scripts/trainer_ddp.py \
    --config experiments/prostate_seg/config_prostate.yaml \
    --num_epochs 10

echo "✅ Workflow complet terminé!"
```

### Exemple 2: Inférence sur un seul patient

```python
import torch
from experiments.prostate_seg.inference_prostate import ProstateInferencer
from pathlib import Path

# Initialise
inferencer = ProstateInferencer(
    model_path="./experiments/prostate_seg/checkpoints/best.pt",
    device="cuda"
)

# Charge les données
patient_dir = Path("./test_data/patient_001")
t2, t2_img = inferencer.load_nifti(str(patient_dir / "T2.nii.gz"))
adc, _ = inferencer.load_nifti(str(patient_dir / "ADC.nii.gz"))

# Prédit
prob_map = inferencer.predict(t2, adc)
segmentation = inferencer.post_process(prob_map, threshold=0.5)

# Sauvegarde
inferencer.save_nifti(
    segmentation,
    t2_img,
    "./results/patient_001_seg.nii.gz"
)

print(f"✅ Segmentation sauvegardée")
print(f"   Prostate voxels: {(segmentation > 0).sum()}")
```

### Exemple 3: Batch inference sur dossier

```bash
for patient_dir in test_data/raw/patient_*/; do
    patient_name=$(basename "$patient_dir")
    echo "Processing $patient_name..."
    
    python experiments/prostate_seg/inference_prostate.py \
        --model_path ./experiments/prostate_seg/checkpoints/best.pt \
        --input_dir "$patient_dir" \
        --output_dir "./predictions/$patient_name"
done
```

---

## ⚙️ Configuration avancée

### Modifier la configuration

Éditez `experiments/prostate_seg/config_prostate.yaml`:

```yaml
# Nombre d'épochescls
training:
  num_epochs: 300         # Augmenter pour plus de données

# Augmentations plus agressives
augmentation:
  augmentations:
    - type: "RandRotate90d"
      prob: 0.7          # Augmenter à 0.7
    - type: "RandAffined"
      prob: 0.5          # Augmenter à 0.5
      scale_range: [0.15, 0.15, 0.15]  # Plus agressif

# Optimiseur
training:
  optimizer: "adamw"
  lr: 0.002              # Augmenter learning rate
  weight_decay: 0.001    # Réduire régularization
```

### Utiliser une autre architecture

```yaml
model:
  name: "segformer3d"
  embed_dim: 128         # Augmenter pour plus grande capacité
  num_layers: 5          # Ajouter une couche
  num_heads: 8           # Plus de heads attention
```

### Loss personnalisée

```yaml
loss:
  loss_fn: "focal_loss"   # Si Dice ne converge pas
  aux_loss: "cross_entropy"
  loss_weight: 0.6
  
  # Poids des classes (prostate très minoritaire)
  class_weights:
    - 0.3   # background (moins important)
    - 2.0   # prostate (très important)
```

---

## 🐛 Dépannage

### Problème 1: "CUDA out of memory"

**Solution:** Réduire batch size ou utiliser gradient accumulation

```yaml
training:
  batch_size: 2          # Réduire de 4 à 2
  accumulation_steps: 2  # Compenser avec accumulation
```

### Problème 2: Prétraitement lent

**Solution:** Paralléliser le chargement

```bash
# Modifier prostate_preprocess.py:
# num_workers = 4 dans DataLoader
# ou utiliser multiprocessing.Pool
```

### Problème 3: Segmentation mauvaise (faible Dice)

**Vérifier:**
```python
# 1. Données correctes?
from dataloaders.prostate_seg import ProstateSegDataset
dataset = ProstateSegDataset("./data/prostate_data")
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")  # Devrait être (2, 96, 96, 96)
print(f"Label shape: {sample['label'].shape}")  # Devrait être (1, 96, 96, 96)
print(f"Label unique: {sample['label'].unique()}")  # Devrait être 0, 1

# 2. Augmentations correctes?
# Vérifier que les augmentations gardent les dimensions

# 3. Classes déséquilibrées?
# Augmenter class_weights pour prostate
```

### Problème 4: Prédictions bruitées

**Solution:** Augmenter le post-traitement

```python
segmentation = inferencer.post_process(
    prob_map,
    threshold=0.6,              # Augmenter le seuil
    remove_small_components=True,
    min_component_size=100      # Augmenter taille minimale
)
```

---

## 📊 Métriques d'évaluation

### Dice Score (principal)

```python
from metrics.segmentation_metrics import compute_dice

# Comparer prédiction vs ground truth
dice = compute_dice(segmentation_pred, segmentation_gt)
print(f"Dice Score: {dice:.3f}")  # Objectif: > 0.85
```

### Autres métriques

- **Hausdorff Distance**: Distance maximale entre contours
- **Surface Dice**: Dice des surfaces (robuste aux petits décalages)
- **Sensitivity/Specificity**: Pour classes déséquilibrées

---

## 🎓 Architecture SegFormer3D pour prostate

### Dimensions à travers le réseau

```
Input:  (Batch=1, C=2, D=96, H=96, W=96)     [T2, ADC]
  ↓
Encoder 1: (1, 64, 48, 48, 48)
  ↓
Encoder 2: (1, 128, 24, 24, 24)
  ↓
Encoder 3: (1, 256, 12, 12, 12)
  ↓
Encoder 4: (1, 512, 6, 6, 6)
  ↓
Decoder: Upsampling progressive
  ↓
Output: (1, 2, 96, 96, 96)                  [logits]
  ↓
Softmax: (1, 2, 96, 96, 96)                 [probas]
  ↓
Prediction: (1, 96, 96, 96)                 [0 ou 1]
```

### Paramètres par défaut

```python
config = {
    "in_channels": 2,        # T2, ADC
    "num_classes": 2,        # background, prostate
    "patch_size": 8,
    "embed_dim": 64,
    "num_layers": 4,
    "num_heads": 4,
    "mlp_ratio": 4,
    "drop_path_rate": 0.1,
    "use_checkpoint": False   # True pour économiser mémoire
}
```

---

## ✅ Checklist avant entraînement

- [ ] Données brutes organisées en `patient_XXX/{T2,ADC,segmentation}.nii.gz`
- [ ] Prétraitement complété sans erreurs
- [ ] Fichiers CSV train.csv et validation.csv créés
- [ ] Au moins 30-50 patients pour entraînement
- [ ] GPU disponible (NVIDIA avec CUDA)
- [ ] Dépendances installées: nibabel, torch, monai, scipy
- [ ] Config `config_prostate.yaml` adaptée à votre cas

---

## 📞 Support & Ressources

Pour plus d'informations:
- Documentation MONAI: https://docs.monai.io
- Documentation NIfTI: https://nifti.nimh.nih.gov
- Vision Transformers: https://arxiv.org/abs/2010.11929

---

**Dernière mise à jour:** 2025-01-01  
**Version:** 1.0  
**Auteur:** Guide SegFormer3D Prostate
