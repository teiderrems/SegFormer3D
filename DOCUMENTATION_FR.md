# Documentation SegFormer3D en Français

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Chargement des données](#chargement-des-données)
4. [Augmentations](#augmentations)
5. [Pertes et Métriques](#pertes-et-métriques)
6. [Optimisateurs et Planificateurs](#optimisateurs-et-planificateurs)
7. [Entraînement](#entraînement)
8. [Structure du Projet](#structure-du-projet)
9. [Guide d'Utilisation](#guide-dutilisation)

---

## Vue d'ensemble

**SegFormer3D** est une implémentation 3D du modèle SegFormer basée sur les Transformateurs pour la segmentation sémantique d'images médicales volumétriques. Ce projet est spécialement conçu pour traiter des volumes d'imagerie médicale (comme les images IRM du cerveau) avec une architecture efficace combinant:

- **Encodeur** : Transformer de vision mixte (MixVisionTransformer) avec plongement hiérarchique de patchs
- **Décodeur** : Tête de décodeur SegFormer avec fusion multi-échelle
- **Support 3D** : Opérations de convolution 3D pour traiter des volumes complets

### Cas d'usage
- Segmentation de tumeurs au cerveau (BraTS 2017, BraTS 2021)
- Segmentation sémantique 3D d'images médicales
- Tâches de segmentation multi-classe avec volumes volumétriques

---

## Architecture

### 1. Fichier Principal : `architectures/segformer3d.py`

Ce fichier contient l'implémentation complète de l'architecture SegFormer3D.

#### **Fonction `build_segformer3d_model(config)`**
Crée une instance du modèle SegFormer3D à partir d'un dictionnaire de configuration.

```python
# Récupère les paramètres depuis config et initialise le modèle
model = build_segformer3d_model(config)
```

**Paramètres attendus dans `config["model_parameters"]`:**
- `in_channels`: nombre de canaux d'entrée (généralement 4 pour T1, T1CE, T2, FLAIR)
- `sr_ratios`: taux de réduction spatiale pour l'attention (ex: [4, 2, 1, 1])
- `embed_dims`: dimensions de plongement à chaque étape (ex: [32, 64, 160, 256])
- `patch_kernel_size`: taille du noyau pour chaque étape de plongement de patchs
- `patch_stride`: pas de convolution pour chaque étape
- `patch_padding`: rembourrage pour chaque étape
- `mlp_ratios`: ratio d'expansion du MLP (ex: [4, 4, 4, 4])
- `num_heads`: nombre de têtes d'attention à chaque étape
- `depths`: nombre de blocs Transformer à chaque étape
- `decoder_head_embedding_dim`: dimension d'intégration de la tête du décodeur (256)
- `num_classes`: nombre de classes de sortie (ex: 3 pour segmentation multi-classe)
- `decoder_dropout`: taux de dropout dans le décodeur

---

### 2. Classe `SegFormer3D` (Modèle Principal)

Structure complète du modèle avec encodeur et décodeur.

```python
model = SegFormer3D(
    in_channels=4,
    embed_dims=[32, 64, 160, 256],
    num_classes=3,
    decoder_dropout=0.1
)

# Passage en avant
output = model(input_volume)  # Input: (B, 4, D, H, W) -> Output: (B, 3, D, H, W)
```

**Composants:**
- `segformer_encoder`: MixVisionTransformer encodeur
- `segformer_decoder`: SegFormerDecoderHead décodeur
- `_init_weights()`: Initialisation des poids (trunc_normal pour Linear, Kaiming pour Conv)

**Flux en avant:**
1. Passe l'entrée à l'encodeur
2. Récupère 4 niveaux de caractéristiques (c1, c2, c3, c4)
3. Passe au décodeur avec fusion multi-échelle
4. Retourne la segmentation prédite

---

### 3. Classe `PatchEmbedding`

Convertit les volumes d'entrée en patchs intégrés avec normalisation par couche.

```python
# Plongement de patchs avec convolution 3D
patch_embed = PatchEmbedding(
    in_channel=4,        # 4 modalités MRI
    embed_dim=32,        # Dimension de plongement
    kernel_size=7,       # Taille du noyau
    stride=4,            # Pas de réduction spatiale
    padding=3            # Rembourrage
)

patches = patch_embed(input_volume)  # (B, N, C) où N = nombre de patchs
```

**Process:**
1. Convolution 3D pour extraire les patchs
2. Aplatissement et transposition
3. Normalisation par couche

---

### 4. Classe `SelfAttention`

Implémente le mécanisme d'attention multi-tête avec réduction spatiale (Spatial Reduction Attention).

```python
attention = SelfAttention(
    embed_dim=64,
    num_heads=2,
    sr_ratio=2,          # Réduit les clés et valeurs
    qkv_bias=True
)

output = attention(patches)  # Applique l'auto-attention
```

**Caractéristiques:**
- **Réduction Spatiale (SR)**: Réduit les clés/valeurs par `sr_ratio` pour efficacité
- **Attention Scalée**: Utilise `scaled_dot_product_attention` (PyTorch 2.0+) ou calcul manuel
- **Dropout**: Dropout d'attention et de projection pour régularisation

**Formule d'attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d)·V
```

---

### 5. Classe `TransformerBlock`

Bloc Transformer complet avec attention et MLP.

```python
block = TransformerBlock(
    embed_dim=64,
    mlp_ratio=4,
    num_heads=2,
    sr_ratio=2
)

output = block(input_patches)  # (B, N, C) -> (B, N, C)
```

**Architecture:**
```
Input -> LayerNorm -> SelfAttention -> Skip Connection
       -> LayerNorm -> MLP -> Skip Connection -> Output
```

---

### 6. Classe `MixVisionTransformer` (Encodeur)

Transformateur hiérarchique multi-étapes pour extraction de caractéristiques.

```python
encoder = MixVisionTransformer(
    in_channels=4,
    embed_dims=[32, 64, 160, 256],
    depths=[2, 2, 2, 2],
    num_heads=[1, 2, 5, 8],
    sr_ratios=[4, 2, 1, 1]
)

features = encoder(input_volume)  # Retourne 4 niveaux: [c1, c2, c3, c4]
```

**Étapes (4 niveaux pyramidaux):**
1. **Étape 1**: Plongement rapide (stride=4), faible dimension (32)
2. **Étape 2**: Réduction 2x, dimension augmentée (64)
3. **Étape 3**: Pas de réduction spatiale, dimension (160)
4. **Étape 4**: Pas de réduction, dimension maximale (256)

**Output:** Liste de 4 tenseurs de caractéristiques avec réductions spatiales progressives

---

### 7. Classe `_MLP` (Couche MLP)

MLP avec convolution en profondeur pour modulation d'effectif.

```python
mlp = _MLP(in_feature=64, mlp_ratio=4, dropout=0.1)
output = mlp(input_patches)
```

**Architecture:**
```
Input -> FC (expand) -> DWConv3d -> GELU -> Dropout 
      -> FC (project) -> Dropout -> Output
```

---

### 8. Classe `DWConv` (Convolution en Profondeur)

Convolution 3D dépendante avec normalisation batch.

```python
dwconv = DWConv(dim=64)
output = dwconv(input_patches)  # Applique convolution spatiale
```

---

### 9. Classe `SegFormerDecoderHead` (Décodeur)

Tête de décodeur avec fusion multi-échelle et upsampling.

```python
decoder = SegFormerDecoderHead(
    input_feature_dims=[256, 160, 64, 32],  # Dimensions inversées
    decoder_head_embedding_dim=256,
    num_classes=3
)

output = decoder(c1, c2, c3, c4)  # Output: (B, 3, D, H, W)
```

**Process:**
1. Projette chaque niveau de caractéristiques avec MLP linéaire
2. Interpole tous les niveaux à la résolution du plus petit (c1)
3. Concatène les 4 niveaux fusionnés
4. Applique fusion convolutive
5. Projection linéaire finale
6. Upsampling x4 (retour à la résolution d'entrée)

---

### 10. Fonction Utilitaire `cube_root()`

Calcule efficacement la racine cubique pour remodeler les patchs.

```python
n = cube_root(N)  # Calcule D=H=W pour un volume cubique de N patchs
```

---

## Chargement des données

### Fichier : `dataloaders/build_dataset.py`

Usine pour construire les ensembles de données en fonction du type.

#### **Fonction `build_dataset(dataset_type, dataset_args)`**

```python
dataset = build_dataset(
    dataset_type="brats2021_seg",
    dataset_args={
        "root": "/path/to/BraTS2021",
        "train": True,
        "fold_id": 1
    }
)
```

**Types de datasets supportés:**
- `"brats2021_seg"`: BraTS 2021 Task 1 (Segmentation)
- `"brats2017_seg"`: BraTS 2017 Task 1 (Segmentation)

#### **Fonction `build_dataloader(dataset, dataloader_args, config, train)`**

Crée un DataLoader MONAI optimisé pour l'entraînement distribué.

```python
dataloader = build_dataloader(
    dataset=dataset,
    dataloader_args={
        "batch_size": 4,
        "num_workers": 4,
        "prefetch_factor": 2
    },
    config=config,
    train=True
)
```

---

### Classe `Brats2021Task1Dataset`

Charge les données BraTS 2021 prétraitées (patchs volumétriques 3D).

```python
dataset = Brats2021Task1Dataset(
    root_dir="/path/to/BraTS2021",
    is_train=True,
    fold_id=1,
    transform=augmentations
)

sample = dataset[0]
# {
#   'image': Tensor(4, D, H, W),    # 4 modalités MRI
#   'label': Tensor(1, D, H, W)     # Masques de segmentation
# }
```

**Structure CSV attendue:** `train_fold_1.csv`
```
data_path,case_name
/path/to/case1,BraTS2021_00000_transverse
/path/to/case2,BraTS2021_00001_transverse
```

**Chargement:**
- `{case_name}_modalities.pt`: Patchs 4 canaux
- `{case_name}_label.pt`: Masques de segmentation

---

### Classe `Brats2017Task1Dataset`

Implémentation similaire pour BraTS 2017.

---

## Augmentations

### Fichier : `augmentations/augmentations.py`

Pipeline d'augmentation de données pour entraînement efficace.

#### **Fonction `build_augmentations(train: bool)`**

```python
# Augmentations d'entraînement
train_augmentations = build_augmentations(train=True)

# Sans augmentations (validation)
val_augmentations = build_augmentations(train=False)
```

### Augmentations d'Entraînement

**1. Découpe Spatiale Aléatoire (RandSpatialCropSamplesd)**
```python
# 4 découpes par volume de taille 96x96x96
RandSpatialCropSamplesd(
    roi_size=(96, 96, 96),
    num_samples=4,
    random_center=True
)
```
- Génère 4 samples par volume d'entraînement
- Centre aléatoire pour diversité

**2. Retournement Horizontal Aléatoire (RandFlipd)**
```python
# 30% de probabilité
RandFlipd(keys=["image", "label"], prob=0.30, spatial_axis=1)
```

**3. Rotation Aléatoire (RandRotated)**
```python
# Rotation autour de l'axe X: ±20.6°
RandRotated(
    keys=["image", "label"],
    prob=0.50,
    range_x=0.36,  # radians
    range_y=0.0,
    range_z=0.0
)
```

**4. Dropout Grossier (RandCoarseDropoutd)**
```python
# Masque aléatoire pour robustesse
RandCoarseDropoutd(
    keys=["image", "label"],
    holes=20,
    spatial_size=(-1, 7, 7),
    fill_value=0,
    prob=0.5
)
```

**5. Bruit de Gibbs (GibbsNoised)**
```python
# Simule les artefacts de sonnerie MRI
GibbsNoised(keys=["image"])
```

**6. Assurance de Type (EnsureTyped)**
```python
# Conversion efficace en tenseurs PyTorch
EnsureTyped(keys=["image", "label"], track_meta=False)
```

### Augmentations de Validation

Minimal - conversion de type uniquement pour cohérence.

---

## Pertes et Métriques

### Fichier : `losses/losses.py`

Implémentations de fonctions de perte pour segmentation sémantique 3D.

#### **Classe `CrossEntropyLoss`**

Perte d'entropie croisée standard pour segmentation multi-classe.

```python
criterion = CrossEntropyLoss()
loss = criterion(predictions, targets)
# predictions: (B, C, D, H, W)
# targets: (B, C, D, H, W)
```

**Formule:**
```
CE = -Σ target_i * log(softmax(pred_i))
```

#### **Classe `BinaryCrossEntropyWithLogits`**

Pour tâches de segmentation binaire.

```python
criterion = BinaryCrossEntropyWithLogits()
loss = criterion(predictions, targets)
```

#### **Classe `DiceLoss`**

Coefficient Dice - métrique d'intersection sur union.

```python
criterion = DiceLoss(smooth=1e-5)
loss = criterion(predictions, targets)
```

**Formule Dice:**
```
Dice = 2|X∩Y| / (|X|+|Y|)
Loss = 1 - Dice
```

**Avantages:**
- Sensible aux classes déséquilibrées
- Pénalise les prédictions hors-cibles
- Métrique-agnostique de classe

#### **Classe `FocalLoss`**

Pour gérer les classes fortement déséquilibrées.

```python
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(predictions, targets)
```

**Formule:**
```
FL = -α * (1-p)^γ * log(p)
```

---

### Fichier : `metrics/segmentation_metrics.py`

Métriques d'évaluation post-entraînement.

#### **Classe `SlidingWindowInference`**

Inférence sur des volumes complets avec fenêtres glissantes (limite mémoire).

```python
swin_inference = SlidingWindowInference(
    roi_size=(96, 96, 96),
    sw_batch_size=4
)

predictions = swin_inference(model, volume)
```

**Process:**
1. Divise le volume en fenêtres 96x96x96 chevauchantes
2. Traite 4 fenêtres simultanément
3. Fusionne les prédictions avec chevauchement moyen

---

## Optimisateurs et Planificateurs

### Fichier : `optimizers/optimizers.py`

Optimisateurs configurables pour entraînement.

#### **Optimisateurs Supportés:**
- **Adam**: Optimiseur adaptatif standard
- **AdamW**: Adam avec weight decay découplé
- **SGD**: Descente de gradient stochastique
- **LAMB**: Large Batch Optimization for BERT

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

---

### Fichier : `optimizers/schedulers.py`

Planificateurs de taux d'apprentissage.

#### **Planificateurs Supportés:**

**1. Linear Warmup**
Échauffement linéaire suivi de décroissance linéaire.

```python
warmup_scheduler = LinearWarmupScheduler(
    optimizer=optimizer,
    warmup_epochs=5,
    total_epochs=100
)
```

**2. Cosine Annealing**
Décroissance en cosinus classique.

```python
scheduler = CosineAnnealingScheduler(
    optimizer=optimizer,
    T_max=100
)
```

**3. Step Decay**
Réduit le taux d'apprentissage par étapes.

```python
scheduler = StepLRScheduler(
    optimizer=optimizer,
    step_size=30,
    gamma=0.1
)
```

---

## Entraînement

### Fichier : `train_scripts/trainer_ddp.py`

Entraîneur principal avec support DDP (Distributed Data Parallel).

#### **Classe `Segmentation_Trainer`**

```python
trainer = Segmentation_Trainer(
    config=config,
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    warmup_scheduler=warmup_sched,
    training_scheduler=train_sched,
    accelerator=accelerator
)

trainer.fit()  # Lance l'entraînement
```

#### **Paramètres Config Clés:**

```yaml
training_parameters:
  num_epochs: 100
  print_every: 10
  cutoff_epoch: 30  # Quand arrêter les augmentations
  calculate_metrics: true
  checkpoint_save_dir: "./checkpoints/"

ema:  # Exponential Moving Average
  enabled: true
  decay: 0.999
  val_ema_every: 5

warmup_scheduler:
  enabled: true
  warmup_epochs: 5

sliding_window_inference:
  roi: [96, 96, 96]
  sw_batch_size: 4
```

#### **Méthodes Clés:**

**`fit()`** - Lance la boucle d'entraînement principal
```python
trainer.fit()
# - Boucle sur les épochs
# - Entraînement par batch
# - Validation toutes les N épochs
# - Sauvegarde des checkpoints meilleurs modèles
```

**`train_epoch()`** - Entraîne une époque
```python
train_loss = trainer.train_epoch()
```

**`validate()`** - Valide le modèle complet
```python
val_loss, val_dice = trainer.validate()
```

**`_save_checkpoint()`** - Sauvegarde l'état de l'entraînement
```python
trainer._save_checkpoint(epoch, is_best=True)
```

#### **Métriques Suivies:**

- **Loss d'entraînement**: Perte moyenne par batch
- **Loss de validation**: Perte moyenne sur validation
- **Dice Score**: Intersection sur union
- **EMA Dice**: Dice avec modèle EMA
- **Learning Rate**: Taux d'apprentissage actuel

#### **Checkpointing Automatique:**

Sauvegarde:
- Les meilleurs modèles (par Dice)
- Chaque N épochs (checkpoint périodique)
- État complet: poids, optimiseur, scheduler, métriques

---

### Fichier : `train_scripts/utils.py`

Fonctions utilitaires pour entraînement.

**Fonctions communes:**
- Chargement/sauvegarde de config
- Initialisation de seed
- Logging et tracking (Weights & Biases)
- Agrégation de métriques

---

## Structure du Projet

```
SegFormer3D/
├── architectures/              # Définitions de modèles
│   ├── __init__.py
│   ├── build_architecture.py   # Fabrique de modèles
│   └── segformer3d.py          # Architecture complète SegFormer3D
│
├── dataloaders/                # Chargement et prétraitement
│   ├── __init__.py
│   ├── build_dataset.py        # Fabrique de datasets
│   ├── brats2021_seg.py        # Dataset BraTS 2021
│   └── brats2017_seg.py        # Dataset BraTS 2017
│
├── augmentations/              # Augmentation de données
│   ├── __init__.py
│   └── augmentations.py        # Pipelines MONAI
│
├── losses/                     # Fonctions de perte
│   ├── __init__.py
│   └── losses.py               # CE, Dice, Focal, etc.
│
├── metrics/                    # Évaluation
│   └── segmentation_metrics.py # Sliding Window Inference
│
├── optimizers/                 # Optimiseurs et schedulers
│   ├── __init__.py
│   ├── optimizers.py           # Adam, AdamW, SGD, LAMB
│   └── schedulers.py           # Warmup, Cosine, StepLR
│
├── train_scripts/              # Entraînement
│   ├── __init__.py
│   ├── trainer_ddp.py          # Boucle d'entraînement DDP
│   └── utils.py                # Utilitaires
│
├── experiments/                # Configs et résultats
│   ├── brats_2017/
│   └── template_experiment/
│
├── data/                       # Données prétraitées
│   ├── brats2017_seg/
│   └── brats2021_seg/
│
├── notebooks/                  # Analyse et profiling
│   └── model_profiler.ipynb
│
├── docs/                       # Documentation HTML
│   ├── index.html
│   ├── style.css
│   └── assets/
│
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation anglaise
└── LICENSE                     # Licence
```

---

## Guide d'Utilisation

### 1. Installation

```bash
# Clone le repository
git clone https://github.com/OSUPCVLab/SegFormer3D.git
cd SegFormer3D

# Installe les dépendances
pip install -r requirements.txt
```

### 2. Préparation des Données

Télécharge BraTS 2021 ou BraTS 2017 et prétraite-les:

```bash
cd data/brats2021_seg/brats2021_raw_data
python brats2021_seg_preprocess.py --input_dir /path/to/raw --output_dir ./processed
```

Génère les CSVs d'entraînement/validation:

```bash
cd datameta_generator
python create_train_val_kfold_csv.py --data_dir ../processed
```

### 3. Configuration

Édite `experiments/template_experiment/config.yaml`:

```yaml
model_name: "segformer3d"
model_parameters:
  in_channels: 4
  embed_dims: [32, 64, 160, 256]
  num_heads: [1, 2, 5, 8]
  depths: [2, 2, 2, 2]
  num_classes: 3

training_parameters:
  num_epochs: 100
  batch_size: 2
  learning_rate: 1e-4
```

### 4. Lancement de l'Entraînement

**Single GPU:**
```bash
python experiments/template_experiment/run_experiment.py \
  --config config.yaml \
  --device cuda:0
```

**Multi-GPU (DDP):**
```bash
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 29500 \
  experiments/template_experiment/run_experiment.py \
  --config config.yaml
```

### 5. Monitoring

Utilise Weights & Biases pour suivre l'entraînement:

```bash
# Lance W&B
wandb login
# Configure le projet dans config.yaml
```

---

## Concepts Clés

### Attention Réduite Spatialement

L'attention complète est coûteuse en mémoire pour les volumes 3D. **Spatial Reduction Attention** réduit K et V:

```
Avant: Q(BxNxC) @ K(BxNxC)^T -> Complexité O(N²)
Après: Q(BxNxC) @ K_reduced(BxN/rxC)^T -> Complexité O(N²/r)
```

### Pyramide Hiérarchique

4 étapes d'encodage avec réductions spatiales progressives:

```
Étape 1: 1/4 résolution, 32 canaux
Étape 2: 1/8 résolution, 64 canaux  
Étape 3: 1/8 résolution, 160 canaux
Étape 4: 1/8 résolution, 256 canaux
```

Utilise `sr_ratio` pour réduire davantage à chaque étape.

### Fusion Multi-Échelle

Le décodeur fusionne tous les niveaux à la résolution maximale:

```
c1(1/4) ← interpolate ← c2(1/8) ← interpolate ← c3(1/8) ← interpolate ← c4(1/8)
         cat → MLP_fuse → linear_pred → upsample_x4
```

### EMA (Exponential Moving Average)

Maintient une copie lissée du modèle pour validation plus robuste:

```
EMA_weights = decay × EMA_weights + (1-decay) × current_weights
```

---

## Optimisation et Performance

### Techniques d'Optimisation Utilisées:

1. **Réduction Spatiale d'Attention**: O(N) au lieu de O(N²)
2. **Convolution Dépendante**: Réduit les paramètres vs convolution standard
3. **Dropout et Coarse Dropout**: Régularisation et robustesse
4. **Sliding Window Inference**: Traite volumes complets sans OOM
5. **Distributed Data Parallel**: Entraînement multi-GPU efficace

### Benchmarks Typiques:

- **Entrée**: 4 × 128 × 128 × 128 (BraTS)
- **Paramètres**: ~26M (petit modèle)
- **Mémoire**: ~8GB pour batch_size=2 (V100)
- **Speed**: ~2-3 volumes/sec (1 GPU)

---

## Références Académiques

- **SegFormer** (2D): "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (ECCV 2022)
- **MixVisionTransformer**: Pyramide hiérarchique avec attention efficace
- **BraTS Challenge**: Benchmark de segmentation de tumeurs au cerveau

---

## Licence

[Consulte le fichier LICENSE pour les détails]

---

**Documentation générée**: Décembre 2025
**Langue**: Français
**Auteur**: Basée sur le code source SegFormer3D
