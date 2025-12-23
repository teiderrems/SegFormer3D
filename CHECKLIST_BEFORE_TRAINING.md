# ✅ CHECKLIST - Avant de commencer l'entraînement

## 📋 Préparation des données

- [ ] **Dossier créé**: `data/prostate_raw_data/`
- [ ] **Structure correcte**: `patient_*/` contient `T2.nii.gz`, `ADC.nii.gz`, `segmentation.nii.gz`
- [ ] **Noms exacts**: Vérifier la casse (T2, ADC, segmentation - case-sensitive)
- [ ] **Format correct**: Fichiers en NII.GZ compressés
- [ ] **Minimum patients**: Au moins 10 pour test, 50+ pour production
- [ ] **Labels corrects**: segmentation.nii.gz contient 0=fond, 1=prostate, 2=bandelettes

**Vérifier rapidement**:
```bash
ls -la data/prostate_raw_data/patient_001/
# Doit afficher: T2.nii.gz  ADC.nii.gz  segmentation.nii.gz
```

---

## 🧪 Tests de configuration

- [ ] **Python 3.9+**: `python --version`
- [ ] **PyTorch**: `python -c "import torch; print(torch.__version__)"`
- [ ] **MONAI**: `python -c "import monai; print(monai.__version__)"`
- [ ] **Tests passent**: `python test_prostate_3class.py`

**Résultat attendu**:
```
✅ PASS: Config
✅ PASS: Preprocessing
✅ PASS: Architecture
✅ PASS: Inference
✅ PASS: DataLoader

Total: 5/5 tests réussis 🎉
```

---

## 🔧 Configuration d'entraînement

- [ ] **Fichier config**: `experiments/prostate_seg/config_prostate.yaml`
- [ ] **num_classes: 3**: ✅ vérifié
- [ ] **class_weights**: `[0.3, 1.5, 1.2]` ✅ vérifié
- [ ] **in_channels: 2**: ✅ T2 + ADC
- [ ] **Taille**: 96×96×96 ✅ vérifié

**Vérifier**:
```bash
grep -A2 "num_classes" experiments/prostate_seg/config_prostate.yaml
# Doit afficher: num_classes: 3
```

---

## 💾 Espace disque

- [ ] **Espace disque libre**: ~50 GB minimum recommandé
  - ~5 GB: Données prétraitées (50 patients)
  - ~10 GB: Checkpoints d'entraînement
  - ~30 GB: Marge de sécurité

**Vérifier**:
```bash
df -h  # Voir espace disque
```

---

## 🖥️ Ressources GPU

- [ ] **GPU détecté**: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] **VRAM suffisant**: 12 GB minimum (24 GB recommandé)
- [ ] **Driver NVIDIA**: Version compatible

**Vérifier**:
```bash
nvidia-smi  # Voir GPU détails
```

---

## 📂 Prétraitement des données

- [ ] **Prétraitement lancé**: 
```bash
python data/prostate_raw_data/prostate_preprocess.py \
    --input_dir ./data/prostate_raw_data \
    --output_dir ./data/prostate_data/preprocessed
```

- [ ] **Output généré**: `data/prostate_data/preprocessed/patient_*/`
- [ ] **Fichiers créés**: `*_modalities.pt` et `*_label.pt`
- [ ] **Pas d'erreurs**: Tous les patients prétraités avec succès

**Vérifier**:
```bash
ls data/prostate_data/preprocessed/patient_001/
# Doit afficher: patient_001_modalities.pt  patient_001_label.pt
```

---

## 📊 Données d'entraînement/validation

- [ ] **Fichier train.csv créé**: `data/prostate_data/train.csv`
- [ ] **Fichier validation.csv créé**: `data/prostate_data/validation.csv`
- [ ] **Nombres cohérents**: train > validation (par ex. 80/20)

**Vérifier**:
```bash
head data/prostate_data/train.csv
# Doit afficher liste des patients
```

---

## 🏗️ Checkpoints et résultats

- [ ] **Dossier créé**: `experiments/prostate_seg/checkpoints/`
- [ ] **Droits d'écriture**: Vérifier permissions
- [ ] **Espace disponible**: Pour les checkpoints (~500 MB par epoch)

**Vérifier**:
```bash
mkdir -p experiments/prostate_seg/checkpoints
chmod 755 experiments/prostate_seg/checkpoints
```

---

## 🚀 Avant de lancer l'entraînement

- [ ] **Toutes les étapes complétées**: Jusqu'au prétraitement
- [ ] **Tests verts**: `python test_prostate_3class.py` → 5/5 ✅
- [ ] **Pas d'erreurs**: Revérifier les logs de prétraitement
- [ ] **Configuration finalisée**: Ajustements learning rate, epochs, batch_size

**Commande d'entraînement**:
```bash
python train_scripts/trainer_ddp.py \
    --config experiments/prostate_seg/config_prostate.yaml
```

---

## ⚠️ Points d'attention

### Si données insuffisantes
- [ ] Utiliser data augmentation (activé par défaut)
- [ ] Réduire taille modèle si nécessaire
- [ ] Augmenter class_weights pour classes minoritaires

### Si GPU limité
- [ ] Réduire batch_size dans config
- [ ] Réduire target_size de 96 à 64
- [ ] Utiliser CPU (plus lent, pour test seulement)

### Si overfitting
- [ ] Augmenter augmentation (rotations, flips)
- [ ] Augmenter dropout
- [ ] Réduire learning rate

---

## 📝 Logging et monitoring

- [ ] **TensorBoard**: Monitoring losses en temps réel (si configuré)
- [ ] **Checkpoints sauvegardés**: Chaque epoch
- [ ] **Best model sauvegardé**: Basé sur validation Dice

**Monitorer**:
```bash
# Voir GPU usage pendant entraînement
watch -n 1 nvidia-smi
```

---

## ✨ Après l'entraînement

- [ ] **Meilleur checkpoint trouvé**: `best_dice_*.pt` ou similaire
- [ ] **Logs d'entraînement examinés**: Pas d'anomalies
- [ ] **Validation Dice**: Prostate > 0.85, Bandelettes > 0.70
- [ ] **Model prêt pour inférence**: Chemin du checkpoint noté

**Test inférence**:
```bash
python experiments/prostate_seg/inference_prostate.py \
    --model_path ./experiments/prostate_seg/checkpoints/best.pt \
    --input_dir ./test_data \
    --output_dir ./predictions \
    --save_separate_labels true
```

---

## 🎯 Résumé pré-entraînement

```
✅ AVANT ENTRAÎNEMENT
├─ Données organisées correctement
├─ Tests configuration passés (5/5)
├─ Prétraitement complété
├─ Data splits générés
├─ GPU/ressources disponibles
├─ Checkpoints folder prêt
└─ Configuration finale validée

🚀 PRÊT À DÉMARRER!
```

---

## 📞 En cas de problème

**Erreur lors du prétraitement**?
→ Consulter logs, vérifier noms fichiers

**Tests échouent**?
→ Vérifier versions: PyTorch, MONAI, NumPy

**GPU non détecté**?
→ `nvidia-smi` pour vérifier driver NVIDIA

**Out of memory**?
→ Réduire batch_size ou target_size

---

**Date**: 2025-01-01  
**Version**: 2.0 (3 classes)  
**Status**: Prêt pour validation complète avant entraînement
