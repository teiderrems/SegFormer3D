# Guide: Segmentation Prostate + Bandelettes avec SegFormer3D

## 📋 Vue d'ensemble

Ce guide explique comment utiliser SegFormer3D pour segmenter **prostate ET bandelettes** à partir de fichiers NII.GZ.

### Architecture adaptée
- **Entrée**: 2 modalités (T2, ADC)
- **Sortie**: 3 classes
  - Classe 0: Fond (non-segmenté)
  - Classe 1: Prostate
  - Classe 2: Bandelettes

## 📁 Structure des données d'entrée

Vos données doivent être organisées avec les deux classes dans un **seul fichier NII.GZ**:

```
data/prostate_raw_data/
├── patient_001/
│   ├── T2.nii.gz                    # IRM T2
│   ├── ADC.nii.gz                   # IRM ADC
│   └── segmentation.nii.gz          # Multi-label: 0=fond, 1=prostate, 2=bandelettes
├── patient_002/
│   ├── T2.nii.gz
│   ├── ADC.nii.gz
│   └── segmentation.nii.gz
└── ...
```

**Important**: Le fichier `segmentation.nii.gz` doit contenir:
- **0** = Fond (voxels non-segmentés)
- **1** = Prostate
- **2** = Bandelettes

## 🔄 Étapes du pipeline

### Étape 1: Prétraitement

```bash
python data/prostate_raw_data/prostate_preprocess.py \
    --input_dir ./data/prostate_raw_data \
    --output_dir ./data/prostate_data/preprocessed \
    --target_size 96
```

**Ce que fait le script**:
1. Charge T2, ADC et segmentation multi-label
2. Resample à 96×96×96
3. Normalise les intensités
4. Sauvegarde en tenseurs PyTorch .pt
5. Affiche statistiques (voxels prostate + bandelettes)

**Output**:
```
data/prostate_data/preprocessed/
├── patient_001/
│   ├── patient_001_modalities.pt  (2, 96, 96, 96)
│   └── patient_001_label.pt       (1, 96, 96, 96) [labels 0, 1, 2]
└── ...
```

### Étape 2: Création des splits

```bash
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --test_size 0.2
```

Génère `train.csv` et `validation.csv`.

### Étape 3: Entraînement

```bash
python train_scripts/trainer_ddp.py \
    --config experiments/prostate_seg/config_prostate.yaml
```

**Configuration importante** (`config_prostate.yaml`):
```yaml
model:
  num_classes: 3       # ← 3 classes au lieu de 2

loss:
  class_weights:
    - 0.3   # Fond (moins important)
    - 1.5   # Prostate (important)
    - 1.2   # Bandelettes (important)
```

### Étape 4: Inférence

```bash
python experiments/prostate_seg/inference_prostate.py \
    --model_path ./experiments/prostate_seg/checkpoints/best.pt \
    --input_dir ./test_data/raw \
    --output_dir ./test_predictions \
    --threshold 0.5 \
    --threshold_bandelettes 0.5 \
    --save_separate_labels true
```

**Options d'output**:
- `--save_nifti true`: Sauvegarde segmentation multi-label (0, 1, 2)
- `--save_separate_labels true`: Sauvegarde prostate et bandelettes séparément
- `--save_prob_map true`: Sauvegarde cartes de probabilité

**Output**:
```
test_predictions/patient_XXX/
├── segmentation_pred.nii.gz         # Multi-label (0=fond, 1=prostate, 2=bandelettes)
├── prostate_pred.nii.gz             # Prostate seule (si --save_separate_labels)
├── bandelettes_pred.nii.gz          # Bandelettes seules (si --save_separate_labels)
├── prostate_probability.nii.gz      # Probas prostate (si --save_prob_map)
└── bandelettes_probability.nii.gz   # Probas bandelettes (si --save_prob_map)
```

## 🎯 Points clés

### Format d'entrée
✅ **Un seul fichier** `segmentation.nii.gz` avec labels 0, 1, 2
❌ Ne pas utiliser deux fichiers séparés (prostate.nii.gz + bandelettes.nii.gz)

### Architecture
- **in_channels**: 2 (T2, ADC)
- **num_classes**: 3 (fond, prostate, bandelettes)
- **Taille**: 96×96×96 après resampling

### Poids des classes (class_weights)
```yaml
class_weights:
  - 0.3   # Fond: moins pénalisé (classe dominante)
  - 1.5   # Prostate: fortement pénalisée (classe minoritaire)
  - 1.2   # Bandelettes: pénalisée (classe très minoritaire)
```

Ajustez ces valeurs selon:
- Augmentez le poids si la classe est mal prédite
- Diminuez si la classe domine trop

## 📊 Exemple avec vos données

Supposons que vous avez:
```
mon_data/
├── patient_001/
│   ├── t2.nii.gz
│   ├── adc.nii.gz
│   └── seg_multi_label.nii.gz  (0=fond, 1=prostate, 2=bandelettes)
├── patient_002/
│   └── ...
```

**Commandes**:
```bash
# 1. Organiser les données
mkdir -p data/prostate_raw_data
cp -r mon_data/patient_* data/prostate_raw_data/
for dir in data/prostate_raw_data/patient_*/; do
  mv "$dir/t2.nii.gz" "$dir/T2.nii.gz"
  mv "$dir/adc.nii.gz" "$dir/ADC.nii.gz"
  mv "$dir/seg_multi_label.nii.gz" "$dir/segmentation.nii.gz"
done

# 2. Prétraitement
python data/prostate_raw_data/prostate_preprocess.py

# 3. Splits
python data/prostate_raw_data/create_prostate_splits.py

# 4. Entraînement
python train_scripts/trainer_ddp.py --config experiments/prostate_seg/config_prostate.yaml

# 5. Inférence
python experiments/prostate_seg/inference_prostate.py \
    --model_path ./experiments/prostate_seg/checkpoints/best.pt \
    --input_dir ./test_data/raw \
    --output_dir ./test_predictions \
    --save_separate_labels true
```

## ✅ Vérification avant entraînement

- [ ] Fichiers nommés: `T2.nii.gz`, `ADC.nii.gz`, `segmentation.nii.gz`
- [ ] `segmentation.nii.gz` contient 3 valeurs: 0 (fond), 1 (prostate), 2 (bandelettes)
- [ ] Au moins 30-50 patients pour entraînement
- [ ] Config `config_prostate.yaml` a `num_classes: 3`
- [ ] Prétraitement complété sans erreurs

## 🐛 Dépannage

### Erreur: "segmentation manquante"
→ Vérifiez que chaque patient a un fichier `segmentation.nii.gz`
→ Le fichier doit être nommé exactement: `segmentation.nii.gz` (case-sensitive)

### Mauvaise segmentation
→ Vérifiez les valeurs dans `segmentation.nii.gz` (0, 1, 2)
→ Augmentez `num_epochs` dans config
→ Ajustez `class_weights` selon le déséquilibre

### Bandelettes non détectées
→ Augmentez `class_weights[2]` (poids bandelettes)
→ Baissez `--threshold_bandelettes` lors de l'inférence

## 📚 Fichiers modifiés

- ✅ `config_prostate.yaml`: `num_classes: 3`
- ✅ `prostate_preprocess.py`: Support multi-label dans un seul fichier
- ✅ `inference_prostate.py`: Post-processing pour 3 classes
- ✅ `dataloaders/build_dataset.py`: Support `prostate_seg`

---

**Dernière mise à jour**: 2025-01-01  
**Version**: 2.0 (Prostate + Bandelettes)
