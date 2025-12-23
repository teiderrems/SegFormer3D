# SegFormer3D - Documentation Française Complète

## 📋 Vue d'ensemble

**SegFormer3D** est une implémentation de pointe d'un **Transformateur de Vision 3D** pour la segmentation sémantique d'images médicales volumétriques. Ce projet adapte le populaire modèle SegFormer (2D) en une architecture native 3D optimisée pour traiter des volumes médicaux complets (IRM, CT, etc.).

### 🎯 Objectifs principaux

- ✅ Segmentation sémantique multi-classe de volumes médicaux
- ✅ Efficacité mémoire via attention réduite spatialement
- ✅ Support d'entraînement distribué (DDP multi-GPU)
- ✅ Évaluation sur volumes complets via fenêtres glissantes
- ✅ Pré-entraînement et fine-tuning

### 📊 Benchmarks de performance

| Métrique | Valeur |
|----------|--------|
| Paramètres | ~26M (petit modèle) |
| Mémoire requise | ~8GB (batch=2, V100) |
| Débit d'inférence | 2-3 volumes/sec (GPU) |
| Taille d'entrée | 4 × 128 × 128 × 128 |

---

## 🏗️ Architecture

### Composants clés

```
                         INPUT (B, 4, D, H, W)
                               |
                         +-----+-----+
                         |
                         v
                  [Encodeur] MixVisionTransformer
                      |
        +-------+------+------+-------+
        |       |      |      |       |
        v       v      v      v       v
       c1      c2     c3     c4
    (1/4, 32) (1/8, 64) (1/8, 160) (1/8, 256)
        |       |      |      |
        +-------+------+------+
                       |
                       v
            [Décodeur] SegFormerDecoderHead
          (Fusion multi-échelle + MLP)
                       |
                       v
                OUTPUT (B, 3, D, H, W)
```

### Étapes de l'encodeur (Pyramide hiérarchique)

| Étape | Stride | Réduction | Dim. | Têtes | Blocs |
|-------|--------|-----------|------|-------|-------|
| 1 | 4 | 1/4 | 32 | 1 | 2 |
| 2 | 2 | 1/8 | 64 | 2 | 2 |
| 3 | 1 | 1/8 | 160 | 5 | 2 |
| 4 | 1 | 1/8 | 256 | 8 | 2 |

---

## 🚀 Démarrage Rapide

### Installation

```bash
# Clone le repository
git clone https://github.com/OSUPCVLab/SegFormer3D.git
cd SegFormer3D

# Installe les dépendances
pip install -r requirements.txt
```

### Entraînement rapide

```bash
# Édite la config
nano experiments/template_experiment/config.yaml

# Lance l'entraînement (single GPU)
python experiments/template_experiment/run_experiment.py \
  --config experiments/template_experiment/config.yaml \
  --device cuda:0

# Multi-GPU (DDP)
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 29500 \
  experiments/template_experiment/run_experiment.py \
  --config experiments/template_experiment/config.yaml
```

### Inférence

```python
import torch
from architectures.segformer3d import SegFormer3D

# Charge le modèle
model = SegFormer3D(
    in_channels=4,
    num_classes=3,
    embed_dims=[32, 64, 160, 256]
)

# Charge les poids
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Inférence
volume = torch.randn(1, 4, 128, 128, 128).cuda()
predictions = model(volume)  # (1, 3, 128, 128, 128)
```

---

## 📚 Structure du Projet Détaillée

```
SegFormer3D/
├── architectures/                      # Modèles
│   ├── __init__.py
│   ├── build_architecture.py           # Fabrique de modèles
│   └── segformer3d.py                  # Architecture complète
│       ├── build_segformer3d_model()   # Fonction de construction
│       ├── SegFormer3D                 # Modèle principal (encoder + decoder)
│       ├── PatchEmbedding              # Plongement de patchs 3D
│       ├── SelfAttention               # Attention multi-tête avec SR
│       ├── TransformerBlock            # Bloc Transformer complet
│       ├── MixVisionTransformer        # Encodeur pyramidal (4 étapes)
│       ├── _MLP                        # Couche MLP avec DWConv
│       ├── DWConv                      # Convolution dépendante 3D
│       ├── SegFormerDecoderHead        # Décodeur de fusion
│       └── cube_root()                 # Utilitaire mathématique
│
├── dataloaders/                        # Chargement de données
│   ├── __init__.py
│   ├── build_dataset.py                # Fabrique de datasets
│   │   ├── build_dataset()             # Sélectionne le dataset type
│   │   └── build_dataloader()          # Crée le DataLoader
│   ├── brats2021_seg.py                # Dataset BraTS 2021
│   │   └── Brats2021Task1Dataset       # Classe de dataset
│   └── brats2017_seg.py                # Dataset BraTS 2017
│       └── Brats2017Task1Dataset       # Classe de dataset
│
├── augmentations/                      # Augmentations MONAI
│   ├── __init__.py
│   └── augmentations.py                # Pipelines d'augmentation
│       └── build_augmentations()       # Crée les transforms MONAI
│           ├── Entraînement:
│           │   ├── RandSpatialCropSamplesd (4 crops)
│           │   ├── RandFlipd (30%)
│           │   ├── RandRotated (±20.6°)
│           │   ├── RandCoarseDropoutd (robustesse)
│           │   ├── GibbsNoised (artefacts MRI)
│           │   └── EnsureTyped
│           └── Validation: EnsureTyped seulement
│
├── losses/                             # Fonctions de perte
│   ├── __init__.py
│   └── losses.py                       # Implémentations
│       ├── CrossEntropyLoss            # CE standard
│       ├── BinaryCrossEntropyWithLogits # BCE binaire
│       ├── DiceLoss                    # Dice coefficient
│       └── FocalLoss                   # Focal (classes déséquilibrées)
│
├── metrics/                            # Évaluation
│   └── segmentation_metrics.py         # Métriques
│       └── SlidingWindowInference      # Inférence sur volumes complets
│
├── optimizers/                         # Optimisation
│   ├── __init__.py
│   ├── optimizers.py                   # Créateurs d'optimiseurs
│   │   ├── optim_adam()                # Adam
│   │   ├── optim_sgd()                 # SGD avec momentum
│   │   ├── optim_adamw()               # AdamW (weight decay découplé)
│   │   └── optim_lamb()                # LAMB (large batch)
│   └── schedulers.py                   # Planificateurs de LR
│       ├── warmup_lr_scheduler()       # Warmup linéaire
│       └── training_lr_scheduler()     # Scheduler principal (3 types)
│
├── train_scripts/                      # Entraînement
│   ├── __init__.py
│   ├── trainer_ddp.py                  # Boucle d'entraînement DDP
│   │   └── Segmentation_Trainer        # Classe maître
│   │       ├── fit()                   # Entraîne le modèle
│   │       ├── train_epoch()           # Une époche
│   │       ├── validate()              # Validation
│   │       ├── _create_ema_model()     # Créer modèle EMA
│   │       └── _save_checkpoint()      # Sauvegarder
│   └── utils.py                        # Utilitaires d'entraînement
│       ├── load_config()               # Charge YAML
│       ├── save_config()               # Sauvegarde YAML
│       ├── set_seed()                  # Reproduisibilité
│       └── initialize_wandb()          # Weights & Biases
│
├── experiments/                        # Configurations
│   ├── brats_2017/                     # Experimento BraTS 2017
│   │   ├── best_brats_2017_exp_dice_82.07/
│   │   │   ├── config.yaml             # Config d'entraînement
│   │   │   └── run_experiment.py       # Script de lancement
│   │   └── template_experiment/
│   └── [autres expériences]
│
├── data/                               # Données prétraitées
│   ├── brats2017_seg/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── brats2017_raw_data/
│   │       ├── brats2017_seg_preprocess.py
│   │       └── datameta_generator/    # Splits k-fold
│   └── brats2021_seg/
│       ├── train.csv
│       ├── validation.csv
│       └── brats2021_raw_data/
│           ├── brats2021_seg_preprocess.py
│           └── datameta_generator/
│
├── notebooks/
│   └── model_profiler.ipynb            # Analyse de complexité
│
├── docs/                               # Documentation HTML
│   ├── index.html
│   ├── style.css
│   └── assets/
│
├── requirements.txt                    # Dépendances
├── README.md                           # Documentation anglaise
├── DOCUMENTATION_FR.md                 # Documentation française (générale)
├── GUIDE_IMPLEMENTATION_FR.md          # Documentation française (implémentation)
├── LICENSE
└── .gitignore
```

---

## 🔑 Concepts Clés Expliqués

### 1. Attention Réduite Spatialement (Spatial Reduction Attention)

**Problème**: L'attention complète pour une séquence N coûte O(N²) en mémoire.

Pour un volume 3D de 128×128×128, N = 2M (2 millions de patchs) → O(4T) de mémoire !

**Solution**: Réduire spatialement les clés et valeurs par un facteur `sr_ratio`.

```python
# Avant (complet)
Q: (B, N, C)           # 2M patchs
K: (B, N, C)
V: (B, N, C)
Attention: O(N²)       # Coûteux !

# Après (avec sr_ratio=4)
Q: (B, N, C)           # 2M patchs complets
K: (B, N/4, C)         # Réduit 4x
V: (B, N/4, C)         # Réduit 4x
Attention: O(N²/4)     # 4x plus rapide et efficace !
```

**Impact progressif**:
- Étape 1: sr_ratio=4 → réduit 4x
- Étape 2: sr_ratio=2 → réduit 2x
- Étape 3-4: sr_ratio=1 → pas de réduction

### 2. Pyramide Hiérarchique d'Encodage

Capture les caractéristiques à différentes résolutions:

```
Input (128³)
     |
     v
[Étape 1] -> c1: 32³ (résolution maximale)
     |
     v
[Étape 2] -> c2: 64³
     |
     v
[Étape 3] -> c3: 64³ (même que c2, pas de réduction)
     |
     v
[Étape 4] -> c4: 64³ (même que c2, pas de réduction)
```

**Avantage**: Les caractéristiques de basse résolution (contexte global) s'ajoutent aux détails fins.

### 3. Fusion Multi-Échelle du Décodeur

Toutes les caractéristiques sont interpolées à la résolution maximale et fusionnées:

```
c1 (32³, dim=32)    ← résolution maximale
c2 (64³, dim=64)    ← interpolate à 32³
c3 (64³, dim=160)   ← interpolate à 32³
c4 (64³, dim=256)   ← interpolate à 32³
     |
     v
[Concatenate] -> (32³, 4*256=1024)
     v
[MLP Fusion] -> (32³, 256)
     v
[Linear Projection] -> (32³, 3)
     v
[Upsample 4x] -> Output (128³, 3)
```

### 4. Exponential Moving Average (EMA)

Maintient une copie lissée du modèle:

```python
# Après chaque batch
EMA_weight = decay * EMA_weight + (1 - decay) * current_weight

# Avec decay=0.999
# Le modèle EMA suit lentement le modèle actuel
# Meilleure généralisation et prédictions plus stables
```

**Utilisation**:
- Entraînement: Train sur le modèle actuel
- Validation: Valide avec le modèle EMA
- Meilleur de: Sauvegarder si EMA-Dice > best-EMA-Dice

---

## ⚙️ Configuration Détaillée

### Fichier `config.yaml` complet commenté

```yaml
##############################################################################
# IDENTITÉ DU MODÈLE
##############################################################################
model_name: "segformer3d"  # Doit être "segformer3d"

##############################################################################
# PARAMÈTRES DU MODÈLE ARCHITECTURE
##############################################################################
model_parameters:
  # Entrée
  in_channels: 4  # T1, T1CE, T2, FLAIR (modalités MRI)
  
  # Réduction spatiale de l'attention à chaque étape
  # sr_ratio=4 réduit les K,V par 4
  sr_ratios: [4, 2, 1, 1]
  
  # Dimension d'intégration à chaque étape (progressive)
  embed_dims: [32, 64, 160, 256]
  
  # Taille du noyau pour plongement de patchs
  patch_kernel_size: [7, 3, 3, 3]
  
  # Pas de convolution (détermine réduction spatiale)
  patch_stride: [4, 2, 2, 2]
  
  # Rembourrage
  patch_padding: [3, 1, 1, 1]
  
  # Ratio d'expansion du MLP (dim_mlp = dim * ratio)
  mlp_ratios: [4, 4, 4, 4]
  
  # Nombre de têtes d'attention à chaque étape
  num_heads: [1, 2, 5, 8]
  
  # Nombre de blocs Transformer par étape
  depths: [2, 2, 2, 2]
  
  # Dimension de la tête du décodeur
  decoder_head_embedding_dim: 256
  
  # Nombre de classes (BraTS: 3 = NCR, ED, ET)
  num_classes: 3
  
  # Dropout du décodeur
  decoder_dropout: 0.1

##############################################################################
# DONNÉES ET DATASET
##############################################################################
data:
  dataset_type: "brats2021_seg"  # ou "brats2017_seg"
  root_dir: "/data/BraTS2021_Training"
  fold_id: 1  # Pour k-fold cross-validation (1-5)

##############################################################################
# ENTRAÎNEMENT
##############################################################################
training_parameters:
  # Nombre d'epochs
  num_epochs: 100
  
  # Taille des batches
  batch_size: 2
  
  # Workers pour data loading
  num_workers: 4
  prefetch_factor: 2
  
  # Logging
  print_every: 10  # Print stats tous les 10 batches
  
  # Cutoff pour augmentations (peut réduire la variance tard)
  cutoff_epoch: 30
  
  # Calculer les métriques complètes (plus lent)
  calculate_metrics: true
  
  # Répertoire de sauvegarde
  checkpoint_save_dir: "./checkpoints/"

##############################################################################
# OPTIMISEUR
##############################################################################
optimizer:
  optimizer_type: "adamw"  # ou "adam", "sgd", "lamb"
  lr: 1e-4
  weight_decay: 0.01

##############################################################################
# SCHEDULER DE WARMUP (phase initiale)
##############################################################################
warmup_scheduler:
  enabled: true
  warmup_epochs: 5  # Augmente linéairement le LR pendant 5 epochs

##############################################################################
# SCHEDULER PRINCIPAL (après warmup)
##############################################################################
train_scheduler:
  # ReduceLROnPlateau: Réduit si plateau
  scheduler_type: "reducelronplateau"
  scheduler_args:
    mode: "max"           # "max" pour Dice, "min" pour Loss
    factor: 0.1           # Multiplie par 0.1 quand plateau
    patience: 10          # Patience en epochs
    threshold: 0.0001     # Seuil minimal d'amélioration

##############################################################################
# FONCTION DE PERTE
##############################################################################
loss:
  loss_type: "dice"  # ou "ce", "bce", "focal"

##############################################################################
# EXPONENTIAL MOVING AVERAGE (EMA)
##############################################################################
ema:
  enabled: true
  decay: 0.999        # Plus proche de 1 = lissage plus fort
  val_ema_every: 5    # Valide avec EMA tous les 5 epochs

##############################################################################
# SLIDING WINDOW INFERENCE (inférence sur volumes complets)
##############################################################################
sliding_window_inference:
  roi: [96, 96, 96]       # Taille des fenêtres
  sw_batch_size: 4        # Fenêtres traitées simultanément

##############################################################################
# LOGGING (Weights & Biases)
##############################################################################
logging:
  project_name: "segformer3d"
  entity_name: "your-wandb-entity"
  run_name: "brats2021_fold1_adamw_dice"
```

---

## 📈 Workflow d'Entraînement Typique

```
1. PRÉPARATION DES DONNÉES
   └─ Télécharge BraTS 2021
   └─ Prétraite (resize, normalise, format .pt)
   └─ Crée CSVs train/val k-fold

2. CONFIGURATION
   └─ Édite config.yaml (hyperparamètres)

3. INITIALISATION
   └─ Crée le modèle SegFormer3D
   └─ Crée l'optimiseur AdamW
   └─ Crée le scheduler (warmup + ReduceLROnPlateau)
   └─ Crée la perte (Dice)
   └─ Initialise W&B pour logging

4. ENTRAÎNEMENT (par epoch)
   ├─ train_epoch():
   │  ├─ Boucle sur les batches train
   │  ├─ Forward pass
   │  ├─ Calcul de la perte
   │  ├─ Backward pass
   │  ├─ Optimizer step
   │  └─ Update EMA
   │
   ├─ Validation (tous les N epochs):
   │  ├─ Mode eval
   │  ├─ Inférence avec SlidingWindowInference
   │  ├─ Calcul Dice, Loss
   │  ├─ Update EMA si meilleur
   │  └─ Checkpoint si meilleur
   │
   └─ Update Learning Rate (scheduler)

5. INFÉRENCE (post-entraînement)
   └─ Charge le meilleur checkpoint
   └─ Utilise SlidingWindowInference
   └─ Génère prédictions pour ensemble test
   └─ Compute métriques finales
```

---

## 🎓 Concepts de Machine Learning

### Normalisation par Couche (LayerNorm)

Utilisée dans les Transformers:

```python
# Avant
x = [10, 100, 1000]

# LayerNorm(x)
# 1. Calcule mean=370, std=395
# 2. Normalise: x_norm = (x - mean) / std
# 3. Scale + Shift: y = gamma * x_norm + beta

# Après
x_norm = [-0.93, -0.68, 1.61]  # Centrée, réduite
```

**Avantage**: Stabilise l'entraînement, permet des LR plus élevés

### Dropout

Désactive aléatoirement des neurones pendant l'entraînement:

```python
# Pendant l'entraînement: 10% des neurones désactivés
# Pendant l'inférence: Tous actifs, mais rescalés par (1-p)

# Effet
# - Régularisation (prévient l'overfitting)
# - Ensemble d'apprentissage implicite
```

### Batch Normalization vs Layer Normalization

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalise sur | Batch | Features |
| Dépend de | Taille batch | Non |
| Transformers | Non (post) | Oui (pré) |
| Convolutions | Oui | Non |

---

## 🔧 Dépannage Commun

### Problème: "CUDA out of memory"
**Solution**: Réduire `batch_size` ou `roi_size` dans config

### Problème: Loss = NaN
**Solution**: 
- Réduire le taux d'apprentissage (lr)
- Checker les données d'entrée (NaN/Inf?)
- Vérifier le preprocessing

### Problème: Métriques n'améliorent pas
**Solution**:
- Augmenter les epochs
- Ajuster les augmentations
- Vérifier les hyperparamètres
- Utiliser learning rate decay

### Problème: Entraînement très lent
**Solution**:
- Utiliser multi-GPU (DDP)
- Réduire `num_workers` (peut créer congestion)
- Checker la charge CPU
- Profiler avec `model_profiler.ipynb`

---

## 📝 Fichiers de Documentation

Ce repository inclut plusieurs fichiers de documentation:

1. **README.md**: Documentation en anglais (cette approche)
2. **DOCUMENTATION_FR.md**: Guide complet en français (concepts, architecture, usage)
3. **GUIDE_IMPLEMENTATION_FR.md**: Détails d'implémentation en français (fonctions, classes, config)
4. **README_FR.md**: Ce fichier - Ressource française principale

---

## 🤝 Contribution et Support

### Pour contribuer:
1. Fork le repository
2. Crée une branche (`git checkout -b feature/amazing-feature`)
3. Commit tes changements (`git commit -m 'Add amazing feature'`)
4. Push à la branche (`git push origin feature/amazing-feature`)
5. Ouvre une Pull Request

### Pour signaler des bugs:
- Ouvre une issue GitHub
- Décris le problème en détail
- Fournis un minimal reproducible example

---

## 📚 Références Académiques et Ressources

### Articles principaux:

1. **SegFormer** (ECCV 2022)
   - "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
   - Xie et al.
   - https://arxiv.org/abs/2105.15203

2. **Vision Transformer (ViT)** (ICLR 2021)
   - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - Dosovitskiy et al.
   - https://arxiv.org/abs/2010.11929

3. **BraTS Challenge**
   - Benchmark de segmentation de tumeurs au cerveau
   - https://www.med.upenn.edu/cbica/brats/

### Ressources recommandées:

- **PyTorch Documentation**: https://pytorch.org/docs/
- **MONAI (Medical Open Network for AI)**: https://monai.io/
- **Einops**: https://einops.readthedocs.io/ (opérations tenseurs)
- **Weights & Biases**: https://wandb.ai/ (logging d'expériences)

---

## 📄 Licence

Ce projet est sous licence [Consulte LICENSE].

---

## 👥 Auteurs et Remerciements

**Maintainers**: OSU PCVL Lab

**Basé sur**: Implémentations 2D de SegFormer, adaptées pour le 3D

---

## ⭐ Si ce projet vous a été utile

Pensez à mettre une star ⭐ sur le repository !

---

**Dernière mise à jour**: Décembre 2025
**Langue**: Français
**Couverture**: Architecture, Entraînement, Inférence, Configuration, Dépannage
