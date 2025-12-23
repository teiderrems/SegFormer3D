# GUIDE DE DOCUMENTATION FRANÇAIS - SegFormer3D
## Fichiers d'implémentation détaillés

---

## 1. Fichier: `optimizers/optimizers.py`

### Fonction `optim_adam(model, optimizer_args)`
**Description**: Crée un optimiseur Adam.

```python
# Utilisation
optimizer = optim_adam(model, {"lr": 1e-4, "weight_decay": 0.01})
```

**Paramètres**:
- `model`: Modèle PyTorch
- `optimizer_args`: Dict avec "lr" (taux d'apprentissage) et "weight_decay" (décroissance)

**Retour**: Optimiseur Adam configuré

---

### Fonction `optim_sgd(model, optimizer_args)`
**Description**: Crée un optimiseur SGD avec momentum.

```python
optimizer = optim_sgd(model, {
    "lr": 0.01,
    "weight_decay": 0.0001,
    "momentum": 0.9
})
```

---

### Fonction `optim_adamw(model, optimizer_args)`
**Description**: Crée un optimiseur AdamW (Adam avec weight decay découplé).

AdamW améliore Adam en découplant la régularisation L2 de l'optimisation.

```python
optimizer = optim_adamw(model, {
    "lr": 1e-4,
    "weight_decay": 0.01  # Décroissance découplée
})
```

---

## 2. Fichier: `optimizers/schedulers.py`

### Fonction `warmup_lr_scheduler(config, optimizer)`
**Description**: Crée un planificateur de taux d'apprentissage avec échauffement linéaire.

**Processus**:
- Augmente linéairement le LR de 0 à LR initial pendant `warmup_epochs`
- Stabilise l'entraînement dans les premières épochs
- Typiquement suivi d'un autre scheduler pour la décroissance

```python
warmup_sched = warmup_lr_scheduler(config, optimizer)
# Exemple: warmup de 5 epochs avec LR initial = 1e-4
# Epoch 1: LR = 1e-4 * (1/5) = 2e-5
# Epoch 2: LR = 1e-4 * (2/5) = 4e-5
# Epoch 5: LR = 1e-4 * (5/5) = 1e-4
```

**Config attendue**:
```yaml
warmup_scheduler:
  enabled: true
  warmup_epochs: 5
```

---

### Fonction `training_lr_scheduler(config, optimizer)`
**Description**: Crée le scheduler de training principal (après warmup).

**Types supportés**:

#### 1. ReduceLROnPlateau
Réduit le LR si la métrique ne s'améliore pas.

```yaml
train_scheduler:
  scheduler_type: "reducelronplateau"
  scheduler_args:
    mode: "max"           # "max" pour Dice, "min" pour Loss
    factor: 0.1           # Multiplier par 0.1 si plateau
    patience: 10          # Patience (epochs)
    threshold: 0.0001     # Seuil minimal d'amélioration
    cooldown: 5           # Cooldown après réduction
```

#### 2. Cosine Annealing with Warm Restarts
Décroissance en cosinus avec redémarrages périodiques.

```yaml
train_scheduler:
  scheduler_type: "cosine_annealing_wr"
  scheduler_args:
    T_0: 10               # Période initiale
    T_mult: 2             # Multiplicateur de période
    eta_min: 1e-6         # LR minimum
```

#### 3. Polynomial LR Decay
Décroissance polynomiale du LR.

```yaml
train_scheduler:
  scheduler_type: "poly_lr"
  scheduler_args:
    total_epochs: 100
    power: 0.9
    lr_min: 1e-6
```

---

## 3. Fichier: `losses/losses.py`

### Classe `CrossEntropyLoss`
Perte d'entropie croisée standard pour segmentation multi-classe.

```python
criterion = CrossEntropyLoss()
loss = criterion(predictions, targets)
```

**Entrées**:
- `predictions`: (B, num_classes, D, H, W) - logits du réseau
- `targets`: (B, num_classes, D, H, W) - labels chauds (one-hot encoded)

**Sortie**: Scalar loss

---

### Classe `BinaryCrossEntropyWithLogits`
Perte binaire avec logits pour segmentation binaire.

```python
criterion = BinaryCrossEntropyWithLogits()
loss = criterion(predictions, targets)
```

---

### Classe `DiceLoss`
Perte Dice coefficient (1 - Dice).

Avantages:
- Sensible aux classes déséquilibrées
- Métrique direkte de IoU
- Pénalise les faux négatifs plus que CE

```python
criterion = DiceLoss(smooth=1e-5)
loss = criterion(predictions, targets)
```

**Formule**:
```
Dice = 2|X∩Y| / (|X| + |Y|)
Loss = 1 - Dice
```

---

### Classe `FocalLoss`
Perte Focal pour classes fortement déséquilibrées.

```python
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(predictions, targets)
```

**Formule**:
```
FL(pt) = -α_t * (1-pt)^γ * log(pt)
```

Paramètres:
- `alpha`: Poids de la classe positive (0.25)
- `gamma`: Paramètre de focus (2.0) - augmente la pénalité pour hard examples

---

## 4. Fichier: `metrics/segmentation_metrics.py`

### Classe `SlidingWindowInference`
Effectue l'inférence sur des volumes complets avec fenêtres glissantes.

**Problème**: Les volumes médicaux complets peuvent être trop grands pour la mémoire GPU.

**Solution**: Diviser en fenêtres chevauchantes, traiter par batch, et fusionner.

```python
swin_inference = SlidingWindowInference(
    roi_size=(96, 96, 96),      # Taille de chaque fenêtre
    sw_batch_size=4             # Fenêtres traitées simultanément
)

# Inférence
with torch.no_grad():
    predictions = swin_inference(model, volume)  # (1, num_classes, D, H, W)
```

**Processus**:
1. Divise le volume en fenêtres 96×96×96 chevauchantes
2. Traite 4 fenêtres à la fois (sw_batch_size=4)
3. Fusionne les prédictions avec moyenne du chevauchement
4. Retourne le volume de prédiction complète

---

## 5. Fichier: `dataloaders/brats2021_seg.py`

### Classe `Brats2021Task1Dataset`
Loader pour données BraTS 2021 prétraitées.

```python
from dataloaders.brats2021_seg import Brats2021Task1Dataset

dataset = Brats2021Task1Dataset(
    root_dir="/path/to/BraTS2021_Training",
    is_train=True,
    fold_id=1,  # 5-fold CV
    transform=augmentations
)

# Accès
sample = dataset[0]
# {
#   'image': Tensor(4, D, H, W),    # 4 modalités MRI
#   'label': Tensor(1, D, H, W)     # Masques (3 classes: NCR, ED, ET)
# }
```

**Structure de données attendue**:
```
BraTS2021_Training/
├── train_fold_1.csv          # CSV pour fold 1
├── validation_fold_1.csv
├── case1/
│   ├── BraTS2021_00000_modalities.pt
│   └── BraTS2021_00000_label.pt
├── case2/
│   ├── BraTS2021_00001_modalities.pt
│   └── BraTS2021_00001_label.pt
...
```

**Format CSV**:
```
data_path,case_name
/path/to/case1,BraTS2021_00000
/path/to/case2,BraTS2021_00001
```

---

### Classe `Brats2017Task1Dataset`
Identique à Brats2021Task1Dataset mais pour BraTS 2017.

---

## 6. Fichier: `augmentations/augmentations.py`

### Fonction `build_augmentations(train: bool)`

**Augmentations d'ENTRAÎNEMENT**:

#### 1. Découpe Spatiale Aléatoire (RandSpatialCropSamplesd)
```python
RandSpatialCropSamplesd(
    keys=["image", "label"],
    roi_size=(96, 96, 96),      # Taille des patchs
    num_samples=4,               # 4 patchs par volume
    random_center=True,
    random_size=False
)
```

**But**: Génère 4 samples par volume (augmentation multiplicative)

**Effet sur la shape**:
```
Input:  (1, 4, 128, 128, 128)
Output: (4, 4, 96, 96, 96)     # 4 patchs de 96×96×96
```

#### 2. Retournement Horizontal Aléatoire (RandFlipd)
```python
RandFlipd(
    keys=["image", "label"],
    prob=0.30,           # 30% de probabilité
    spatial_axis=1       # Axe Y (horizontal)
)
```

#### 3. Rotation Aléatoire Autour de l'Axe X (RandRotated)
```python
RandRotated(
    keys=["image", "label"],
    prob=0.50,           # 50% de probabilité
    range_x=0.36,        # ±20.6 degrés (0.36 radians)
    range_y=0.0,
    range_z=0.0
)
```

#### 4. Dropout Grossier (RandCoarseDropoutd)
```python
RandCoarseDropoutd(
    keys=["image", "label"],
    holes=20,                    # Nombre de trous
    spatial_size=(-1, 7, 7),     # Taille: (%, 7px, 7px)
    fill_value=0,                # Valeur de remplissage
    prob=0.5                     # 50% de probabilité
)
```

**But**: Rend le modèle robuste aux occlusions

#### 5. Bruit de Gibbs (GibbsNoised)
```python
GibbsNoised(keys=["image"])
```

**But**: Simule les artefacts de sonnerie MRI (phénomène de Gibbs)

#### 6. Assurance de Type (EnsureTyped)
```python
EnsureTyped(
    keys=["image", "label"],
    track_meta=False    # Plus rapide (pas de métadonnées)
)
```

**But**: Conversion en tenseurs PyTorch float32

---

**Augmentations de VALIDATION**:

Minimal - seulement conversion de type pour cohérence.

```python
EnsureTyped(keys=["image", "label"], track_meta=False)
```

Pas d'augmentations spatiales ou d'intensité (évaluation fidèle).

---

## 7. Fichier: `train_scripts/trainer_ddp.py`

### Classe `Segmentation_Trainer`

Classe maître pour orchestrer l'entraînement avec support DDP (Distributed Data Parallel).

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
    accelerator=accelerator  # Accelerate library
)

# Lance l'entraînement
trainer.fit()
```

#### Méthodes principales:

##### `fit()`
Boucle d'entraînement principal (multi-epoch).

```python
trainer.fit()
# - Boucle: pour chaque epoch
#   - train_epoch()
#   - validate() toutes les N epochs
#   - Checkpoint si meilleur
#   - Update schedulers
```

##### `train_epoch()`
Entraîne une seule époque.

```python
train_loss = trainer.train_epoch()
```

**Processus**:
1. Model en mode train
2. Boucle sur les batches
3. Forward pass
4. Calcul de la perte
5. Backward pass
6. Optimiseur step

##### `validate()`
Valide le modèle sur l'ensemble complet.

```python
val_loss, val_dice = trainer.validate()
```

**Processus**:
1. Model en mode eval
2. Pas de gradient
3. Prédictions avec SlidingWindowInference
4. Calcul des métriques (Dice, Loss)

##### `_save_checkpoint(epoch, is_best=False)`
Sauvegarde l'état du modèle.

```python
trainer._save_checkpoint(epoch, is_best=True)
```

**Sauvegarde**:
- Poids du modèle
- État de l'optimiseur
- État du scheduler
- Métriques actuelles
- Numéro d'époche

#### Métriques suivies:

- **epoch_train_loss**: Perte moyenne d'entraînement
- **epoch_val_loss**: Perte moyenne de validation
- **epoch_val_dice**: Score Dice de validation
- **best_val_dice**: Meilleur score Dice
- **epoch_val_ema_dice**: Dice du modèle EMA (Exponential Moving Average)
- **learning_rate**: Taux d'apprentissage actuel

#### EMA (Exponential Moving Average):

Maintient une copie lissée du modèle pour meilleure généralisation.

```python
if self.ema_enabled:
    self.ema_model = self._create_ema_model()
    
    # Update EMA après chaque batch
    # EMA_weight = decay * EMA_weight + (1-decay) * current_weight
```

**Avantages**:
- Meilleure généralisation
- Lisse les variations de poids
- Validation plus robuste

---

## 8. Fichier: `train_scripts/utils.py`

Fonctions utilitaires pour entraînement.

### Fonctions communes:

#### `load_config(config_path)`
Charge un fichier de configuration YAML.

```python
config = load_config("experiments/template_experiment/config.yaml")
```

#### `save_config(config, save_path)`
Sauvegarde la configuration actuelle.

```python
save_config(config, "experiments/results/config_used.yaml")
```

#### `set_seed(seed)`
Initialise tous les seeds (numpy, torch, random).

```python
set_seed(42)  # Reproduisibilité
```

#### `initialize_wandb(config, project_name)`
Initialise Weights & Biases pour logging.

```python
wandb_tracker = initialize_wandb(config, "segformer3d-brats")
```

---

## 9. Configuration Complète (config.yaml)

```yaml
# Identité du modèle
model_name: "segformer3d"

# Paramètres du modèle
model_parameters:
  in_channels: 4
  sr_ratios: [4, 2, 1, 1]
  embed_dims: [32, 64, 160, 256]
  patch_kernel_size: [7, 3, 3, 3]
  patch_stride: [4, 2, 2, 2]
  patch_padding: [3, 1, 1, 1]
  mlp_ratios: [4, 4, 4, 4]
  num_heads: [1, 2, 5, 8]
  depths: [2, 2, 2, 2]
  decoder_head_embedding_dim: 256
  num_classes: 3
  decoder_dropout: 0.1

# Données
data:
  dataset_type: "brats2021_seg"
  root_dir: "/path/to/BraTS2021"
  fold_id: 1  # Pour k-fold CV

# Paramètres d'entraînement
training_parameters:
  num_epochs: 100
  batch_size: 2
  num_workers: 4
  prefetch_factor: 2
  print_every: 10
  cutoff_epoch: 30  # Quand arrêter les augmentations
  calculate_metrics: true
  checkpoint_save_dir: "./checkpoints/"

# Optimiseur
optimizer:
  optimizer_type: "adamw"
  lr: 1e-4
  weight_decay: 0.01

# Scheduler de warmup
warmup_scheduler:
  enabled: true
  warmup_epochs: 5

# Scheduler principal
train_scheduler:
  scheduler_type: "reducelronplateau"
  scheduler_args:
    mode: "max"
    factor: 0.1
    patience: 10
    threshold: 0.0001

# Fonction de perte
loss:
  loss_type: "dice"  # ou "ce", "bce", "focal"

# EMA (Exponential Moving Average)
ema:
  enabled: true
  decay: 0.999
  val_ema_every: 5

# Sliding Window Inference
sliding_window_inference:
  roi: [96, 96, 96]
  sw_batch_size: 4

# Logging
logging:
  project_name: "segformer3d"
  entity_name: "your-wandb-entity"
  run_name: "brats2021_fold1"
```

---

**Documentation générée**: Décembre 2025
**Langue**: Français
**Couverture**: Tous les fichiers d'implémentation principal
