# 🔧 Script de splits amélioré - Stratification multiclasse

## 📋 Mise à jour du script `create_prostate_splits.py`

Le script a été amélioré pour supporter **la stratification par classe** - cela garantit que chaque split train/val a une représentation équilibrée des classes.

---

## ✨ Nouvelles fonctionnalités

### 1. Stratification par classe
```bash
# Avec stratification (PAR DÉFAUT)
python create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --stratified true \
    --num_classes 3
```

**Effet**: 
- Classe 0 (fond): x patients → distribuée 80% train, 20% val
- Classe 1 (prostate): y patients → distribuée 80% train, 20% val  
- Classe 2 (bandelettes): z patients → distribuée 80% train, 20% val
- **Résultat**: Train et val ont les mêmes proportions de classes

### 2. Sans stratification (simple random)
```bash
# Sans stratification
python create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --stratified false
```

---

## 🎯 Paramètres disponibles

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `--input_dir` | str | `./data/prostate_preprocessed` | Répertoire des données prétraitées |
| `--output_dir` | str | `./data/prostate_data` | Répertoire de sortie (CSV) |
| `--test_size` | float | 0.2 | Proportion validation (0.2 = 80% train, 20% val) |
| `--kfold` | int | None | Nombre de folds (ex: 5 pour 5-fold CV) |
| `--random_state` | int | 42 | Seed (reproductibilité) |
| **`--stratified`** | bool | True | **NOUVEAU**: Stratifier par classe |
| **`--num_classes`** | int | 3 | **NOUVEAU**: Nombre de classes pour stratification |

---

## 💡 Cas d'utilisation

### Cas 1: Split train/val simple avec stratification
```bash
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --test_size 0.2 \
    --stratified true \
    --num_classes 3
```

**Génère**:
- `train.csv` (80% avec équilibre de classes)
- `validation.csv` (20% avec équilibre de classes)

---

### Cas 2: 5-fold cross-validation stratifiée
```bash
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --kfold 5 \
    --stratified true \
    --num_classes 3
```

**Génère**:
- `train_fold_1.csv`, `validation_fold_1.csv`
- `train_fold_2.csv`, `validation_fold_2.csv`
- `train_fold_3.csv`, `validation_fold_3.csv`
- `train_fold_4.csv`, `validation_fold_4.csv`
- `train_fold_5.csv`, `validation_fold_5.csv`

Chaque fold a une représentation équilibrée de toutes les classes.

---

### Cas 3: Cross-validation simple (sans stratification)
```bash
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --kfold 5 \
    --stratified false
```

---

## 🔍 Détails techniques

### Comment fonctionne la stratification?

1. **Détecte la classe dominante** pour chaque patient:
   ```python
   dominant_class = classe avec plus de voxels (sauf fond)
   ```

2. **Groupe par classe dominante**:
   - Groupe 0 (patients avec fond seul)
   - Groupe 1 (patients avec prostate dominante)
   - Groupe 2 (patients avec bandelettes dominantes)

3. **Split chaque groupe**:
   - Groupe 1: 80% → train, 20% → val
   - Groupe 2: 80% → train, 20% → val
   - Etc.

4. **Combine les splits**:
   - train.csv = Groupe1_train + Groupe2_train + ...
   - validation.csv = Groupe1_val + Groupe2_val + ...

**Résultat**: Distribution équilibrée de classes dans train et val

---

## 📊 Exemple de sortie

```
📊 Génération des splits pour 50 patients
   Répertoire source: ./data/prostate_data/preprocessed
   Répertoire de sortie: ./data/prostate_data
   Stratification: Activée (par classe dominante)
   Nombre de classes: 3

📊 Création d'une split train/val (80%/20%)...

✅ Fichiers CSV créés:
   - train.csv (40 patients)
   - validation.csv (10 patients)

📋 Aperçu train.csv:
    data_path                                     case_name
    ./data/prostate_data/preprocessed/patient_001  patient_001
    ./data/prostate_data/preprocessed/patient_002  patient_002
    ./data/prostate_data/preprocessed/patient_003  patient_003

✨ Prêt pour l'entraînement!
```

---

## ✅ Points clés

✅ **Stratification multiclasse** - Distribution équilibrée par classe  
✅ **Configurable** - Nombre de classes personnalisable  
✅ **Flexible** - Support train/val simple ET k-fold CV  
✅ **Reproductible** - Seed configurable pour reproductibilité  
✅ **Backward compatible** - Ancien code toujours fonctionnel  

---

## 🚀 Utilisation recommandée

Pour vos données (T2 seul, 3 classes):

```bash
# Après prétraitement
python data/prostate_raw_data/prostate_preprocess.py

# Créer les splits AVEC stratification
python data/prostate_raw_data/create_prostate_splits.py \
    --input_dir ./data/prostate_data/preprocessed \
    --output_dir ./data/prostate_data \
    --stratified true \
    --num_classes 3

# Puis entraîner
python train_scripts/trainer_ddp.py --config experiments/prostate_seg/config_prostate.yaml
```

---

**Version**: 2.2 (Stratification multiclasse)  
**Date**: 2025-01-01  
**Status**: ✅ PRÊT
