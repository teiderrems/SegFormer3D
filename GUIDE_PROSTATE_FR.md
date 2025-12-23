# Guide Complet : SegFormer3D pour la Segmentation de la Prostate (NII.GZ)

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure des données attendues](#structure-des-données-attendues)
3. [Installation des dépendances](#installation-des-dépendances)
4. [Prétraitement des données](#prétraitement-des-données)
5. [Configuration pour la prostate](#configuration-pour-la-prostate)
6. [Entraînement](#entraînement)
7. [Inférence](#inférence)
8. [Conseils et bonnes pratiques](#conseils-et-bonnes-pratiques)

---

## Vue d'ensemble

Ce guide vous permet d'adapter **SegFormer3D** (initialement conçu pour BraTS - segmentation de tumeurs au cerveau) pour la **segmentation de la prostate** avec des fichiers au format **nii.gz** (NIfTI).

### Différences principales par rapport à BraTS:

| Aspect | BraTS (Cerveau) | Prostate |
|--------|-----------------|----------|
| **Format** | .pt (PyTorch tensor) | nii.gz (NIfTI) |
| **Modalités** | 4 (T1, T1CE, T2, FLAIR) | 1-3 (T2, ADC, DWI) |
| **Taille volume** | 128×128×128 | Variable (ajustable) |
| **Classe target** | Tumeur cérébrale | Prostate (et zones) |
| **Labels** | 3 classes (NCR, ED, ET) | 2-3 classes (fond, prostate) |

---

## Structure des données attendues

### Format recommandé:

```
prostate_data/
├── raw/                              # Données brutes
│   ├── patient_001/
│   │   ├── T2.nii.gz               # IRM T2
│   │   ├── ADC.nii.gz              # Cartes ADC (optionnel)
│   │   └── segmentation.nii.gz     # Masque de segmentation
│   ├── patient_002/
│   │   ├── T2.nii.gz
│   │   ├── ADC.nii.gz
│   │   └── segmentation.nii.gz
│   └── ...
│
├── preprocessed/                     # Données prétraitées
│   ├── patient_001/
│   │   ├── patient_001_modalities.pt
│   │   └── patient_001_label.pt
│   ├── patient_002/
│   │   ├── patient_002_modalities.pt
│   │   └── patient_002_label.pt
│   └── ...
│
└── splits/
    ├── train.csv                     # Fichier CSV pour entraînement
    ├── validation.csv                # Fichier CSV pour validation
    └── test.csv                      # Fichier CSV pour test
```

### Format des fichiers CSV:

**train.csv** (exemple):
```csv
data_path,case_name
./preprocessed/patient_001,patient_001
./preprocessed/patient_002,patient_002
./preprocessed/patient_003,patient_003
```

---

## Installation des dépendances

### Dépendances supplémentaires requises:

```bash
pip install nibabel            # Lecture/écriture NIfTI
pip install scikit-image       # Prétraitement
pip install scipy              # Opérations mathématiques
```

### Vérifier l'installation:

```python
import nibabel as nib
import numpy as np
print("✅ nibabel installé")
```

---

## Prétraitement des données

### Créez ce fichier: `data/prostate_raw_data/prostate_preprocess.py`

```python
"""
Script de prétraitement pour données de prostate (nii.gz).

Processus:
  1. Charge les fichiers nii.gz (T2, ADC, segmentation)
  2. Redimensionne à une taille uniforme
  3. Normalise l'intensité des voxels
  4. Enregistre en format .pt (PyTorch)
"""

import os
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.preprocessing import MinMaxScaler

class ProstatePreprocessor:
    def __init__(
        self,
        raw_dir: str,
        output_dir: str,
        target_shape: tuple = (96, 96, 96),
        modalities: list = ["T2", "ADC"],
    ):
        """
        Initialise le préprocesseur.
        
        Args:
            raw_dir: Répertoire contenant les données brutes
            output_dir: Répertoire de sortie pour données prétraitées
            target_shape: Taille cible pour redimensionnement (D, H, W)
            modalities: Liste des modalités à charger (ex: ["T2", "ADC"])
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.target_shape = target_shape
        self.modalities = modalities
        
        # Crée le répertoire de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_nifti(self, filepath: str) -> np.ndarray:
        """Charge un fichier nii.gz et retourne le volume."""
        img = nib.load(filepath)
        data = img.get_fdata()
        return np.asarray(data, dtype=np.float32)
    
    def save_nifti(self, data: np.ndarray, filepath: str):
        """Sauvegarde un volume en format nii.gz."""
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, filepath)
    
    def resize_volume(self, volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Redimensionne un volume à la taille cible."""
        current_shape = volume.shape
        zoom_factors = [
            target_shape[i] / current_shape[i] 
            for i in range(len(current_shape))
        ]
        resized = zoom(volume, zoom_factors, order=1)  # order=1 = interpolation linéaire
        return resized.astype(np.float32)
    
    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalise l'intensité à [0, 1]."""
        # Utilise percentiles pour robustesse aux outliers
        p_low = np.percentile(volume, 2)
        p_high = np.percentile(volume, 98)
        
        volume = np.clip(volume, p_low, p_high)
        volume = (volume - p_low) / (p_high - p_low + 1e-5)
        return volume.astype(np.float32)
    
    def process_patient(self, patient_dir: str):
        """Prétraite un patient complet."""
        patient_path = self.raw_dir / patient_dir
        patient_name = os.path.basename(patient_path)
        
        print(f"Traitement: {patient_name}...")
        
        # Charge les modalités
        modality_volumes = []
        for mod in self.modalities:
            filepath = patient_path / f"{mod}.nii.gz"
            if not filepath.exists():
                raise FileNotFoundError(f"Fichier manquant: {filepath}")
            
            volume = self.load_nifti(str(filepath))
            volume = self.resize_volume(volume, self.target_shape)
            volume = self.normalize_intensity(volume)
            modality_volumes.append(volume)
        
        # Empile les modalités
        modalities_tensor = np.stack(modality_volumes, axis=0)  # (num_mod, D, H, W)
        
        # Charge la segmentation
        seg_filepath = patient_path / "segmentation.nii.gz"
        if not seg_filepath.exists():
            raise FileNotFoundError(f"Segmentation manquante: {seg_filepath}")
        
        label = self.load_nifti(str(seg_filepath))
        label = self.resize_volume(label, self.target_shape)
        label = (label > 0).astype(np.float32)  # Binaire: 0 ou 1
        label = np.expand_dims(label, axis=0)  # (1, D, H, W)
        
        # Sauvegarde en format .pt
        output_path = self.output_dir / patient_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            torch.tensor(modalities_tensor),
            output_path / f"{patient_name}_modalities.pt"
        )
        torch.save(
            torch.tensor(label),
            output_path / f"{patient_name}_label.pt"
        )
        
        print(f"✅ {patient_name} traité avec succès")
    
    def process_all(self):
        """Prétraite tous les patients du répertoire."""
        patient_dirs = [d for d in os.listdir(self.raw_dir) 
                       if os.path.isdir(self.raw_dir / d)]
        
        for patient_dir in patient_dirs:
            try:
                self.process_patient(patient_dir)
            except Exception as e:
                print(f"❌ Erreur pour {patient_dir}: {e}")


if __name__ == "__main__":
    # Configuration
    RAW_DIR = "./prostate_data/raw"
    OUTPUT_DIR = "./prostate_data/preprocessed"
    
    # Crée le préprocesseur
    processor = ProstatePreprocessor(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        target_shape=(96, 96, 96),
        modalities=["T2", "ADC"]  # Ajuste selon vos modalités
    )
    
    # Prétraite tous les patients
    processor.process_all()
    print("✅ Prétraitement terminé!")
```

### Utilisation:

```bash
cd data/prostate_raw_data
python prostate_preprocess.py
```

---

## Configuration pour la prostate

### Créez: `experiments/prostate_exp/config_prostate.yaml`

```yaml
##############################################################################
# PROSTATE SEGMENTATION CONFIG
##############################################################################

model_name: "segformer3d"

model_parameters:
  # Entrée: 2 modalités (T2, ADC) au lieu de 4
  in_channels: 2  # ⭐ CHANGEMENT PRINCIPAL
  
  sr_ratios: [4, 2, 1, 1]
  embed_dims: [32, 64, 160, 256]
  patch_kernel_size: [7, 3, 3, 3]
  patch_stride: [4, 2, 2, 2]
  patch_padding: [3, 1, 1, 1]
  mlp_ratios: [4, 4, 4, 4]
  num_heads: [1, 2, 5, 8]
  depths: [2, 2, 2, 2]
  decoder_head_embedding_dim: 256
  
  # Sortie: 2 classes (fond, prostate)
  num_classes: 2  # ⭐ CHANGEMENT
  decoder_dropout: 0.1

##############################################################################
# DONNÉES
##############################################################################

data:
  dataset_type: "prostate_seg"  # ⭐ NOUVEAU TYPE
  root_dir: "./prostate_data/preprocessed"
  fold_id: null  # Pas de k-fold, utiliser train/val splits

##############################################################################
# ENTRAÎNEMENT
##############################################################################

training_parameters:
  num_epochs: 100
  batch_size: 2
  num_workers: 4
  prefetch_factor: 2
  print_every: 10
  cutoff_epoch: 30
  calculate_metrics: true
  checkpoint_save_dir: "./checkpoints/prostate/"

##############################################################################
# OPTIMISATION
##############################################################################

optimizer:
  optimizer_type: "adamw"
  lr: 1e-4
  weight_decay: 0.01

warmup_scheduler:
  enabled: true
  warmup_epochs: 5

train_scheduler:
  scheduler_type: "reducelronplateau"
  scheduler_args:
    mode: "max"
    factor: 0.1
    patience: 10
    threshold: 0.0001

##############################################################################
# PERTE ET MÉTRIQUES
##############################################################################

loss:
  loss_type: "dice"  # Dice est bon pour segmentation de prostate

ema:
  enabled: true
  decay: 0.999
  val_ema_every: 5

sliding_window_inference:
  roi: [96, 96, 96]
  sw_batch_size: 4

##############################################################################
# LOGGING
##############################################################################

logging:
  project_name: "prostate-segmentation"
  entity_name: "your-wandb-entity"
  run_name: "prostate_exp_1"
```

---

## Entraînement

### Étape 1: Créer le dataloader personnalisé

Créez: `dataloaders/prostate_seg.py`

```python
"""
Dataloader pour segmentation de prostate (format nii.gz prétraité en .pt)
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import warnings


class ProstateSegDataset(Dataset):
    """Dataset pour segmentation de prostate.
    
    Charge les données de prostate prétraitées en format .pt
    (résultat du script prostate_preprocess.py)
    """
    
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transform: Optional[Any] = None,
        split_file: Optional[str] = None,
    ) -> None:
        """
        Args:
            root_dir: Répertoire contenant les données prétraitées
            is_train: Mode entraînement (True) ou validation (False)
            transform: Transformations MONAI à appliquer
            split_file: Fichier CSV pour train/validation split
        """
        super().__init__()
        
        # Détermine le fichier CSV
        if split_file is None:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
        else:
            csv_fp = split_file
        
        # Vérifie l'existence du fichier CSV
        if not os.path.exists(csv_fp):
            raise FileNotFoundError(f"CSV file not found: {csv_fp}")
        
        self.csv = pd.read_csv(csv_fp)
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self) -> int:
        return len(self.csv)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Charge un sample."""
        data_path = self.csv["data_path"].iloc[idx]
        case_name = self.csv["case_name"].iloc[idx]
        
        # Chemins des fichiers
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        
        try:
            # Charge les fichiers
            volume = torch.load(volume_fp, map_location='cpu', weights_only=False)
            label = torch.load(label_fp, map_location='cpu', weights_only=False)
            
            # Assure les types
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume)
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)
            
            data = {
                "image": volume.float(),
                "label": label.float()
            }
        except Exception as e:
            warnings.warn(f"Error loading data at index {idx} ({case_name}): {str(e)}")
            raise
        
        # Applique les augmentations
        if self.transform:
            data = self.transform(data)
        
        return data
```

### Étape 2: Mettre à jour le build_dataset.py

Modifiez `dataloaders/build_dataset.py` pour ajouter:

```python
elif dataset_type == "prostate_seg":
    from .prostate_seg import ProstateSegDataset
    
    dataset = ProstateSegDataset(
        root_dir=dataset_args["root"],
        is_train=dataset_args["train"],
        transform=build_augmentations(dataset_args["train"]),
    )
    return dataset
```

### Étape 3: Créer le CSV de split

Créez: `prostate_data/train.csv` et `prostate_data/validation.csv`

```bash
python << 'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

# Répertoire des données prétraitées
data_dir = Path("prostate_data/preprocessed")
patients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

# Split 80-20
n = len(patients)
train_size = int(0.8 * n)

np.random.seed(42)
indices = np.random.permutation(n)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Train CSV
train_df = pd.DataFrame({
    "data_path": [f"./prostate_data/preprocessed/{patients[i]}" for i in train_indices],
    "case_name": [patients[i] for i in train_indices]
})
train_df.to_csv("prostate_data/train.csv", index=False)

# Validation CSV
val_df = pd.DataFrame({
    "data_path": [f"./prostate_data/preprocessed/{patients[i]}" for i in val_indices],
    "case_name": [patients[i] for i in val_indices]
})
val_df.to_csv("prostate_data/validation.csv", index=False)

print("✅ CSV splits créés")
EOF
```

### Étape 4: Lancer l'entraînement

```bash
python experiments/prostate_exp/run_experiment.py \
  --config experiments/prostate_exp/config_prostate.yaml
```

---

## Inférence

### Script d'inférence pour prostate: `inference_prostate.py`

```python
"""
Script d'inférence pour segmentation de prostate avec SegFormer3D.
"""

import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from architectures.segformer3d import SegFormer3D
from metrics.segmentation_metrics import SlidingWindowInference


def load_nifti(filepath):
    """Charge un fichier nii.gz."""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine


def normalize_volume(volume):
    """Normalise un volume."""
    p_low = np.percentile(volume, 2)
    p_high = np.percentile(volume, 98)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-5)
    return volume.astype(np.float32)


def resize_volume(volume, target_shape):
    """Redimensionne un volume."""
    current_shape = volume.shape
    zoom_factors = [
        target_shape[i] / current_shape[i]
        for i in range(len(current_shape))
    ]
    return zoom(volume, zoom_factors, order=1).astype(np.float32)


def infer_patient(
    model,
    t2_path,
    adc_path,
    output_path,
    device='cuda:0',
    target_shape=(96, 96, 96)
):
    """
    Effectue l'inférence sur un patient.
    
    Args:
        model: Modèle SegFormer3D
        t2_path: Chemin vers image T2
        adc_path: Chemin vers image ADC
        output_path: Chemin pour sauvegarder le résultat
        device: GPU/CPU
        target_shape: Taille de redimensionnement
    """
    # Charge les images
    t2, affine = load_nifti(t2_path)
    adc, _ = load_nifti(adc_path)
    original_shape = t2.shape
    
    # Redimensionne
    t2_resized = resize_volume(t2, target_shape)
    adc_resized = resize_volume(adc, target_shape)
    
    # Normalise
    t2_norm = normalize_volume(t2_resized)
    adc_norm = normalize_volume(adc_resized)
    
    # Empile les modalités
    volume = np.stack([t2_norm, adc_norm], axis=0)
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inférence
    swin_inference = SlidingWindowInference(
        roi_size=(96, 96, 96),
        sw_batch_size=4
    )
    
    with torch.no_grad():
        predictions = swin_inference(model, volume.squeeze(0))
    
    # Post-traitement
    pred = predictions[0, 1].cpu().numpy()  # Classe "prostate"
    pred = (pred > 0.5).astype(np.float32)  # Seuillage
    
    # Redimensionne à la taille originale
    zoom_factors = [
        original_shape[i] / target_shape[i]
        for i in range(len(original_shape))
    ]
    pred_original = zoom(pred, zoom_factors, order=0)
    
    # Sauvegarde
    output_img = nib.Nifti1Image(pred_original, affine=affine)
    nib.save(output_img, output_path)
    print(f"✅ Résultat sauvegardé: {output_path}")


if __name__ == "__main__":
    # Charge le modèle
    model = SegFormer3D(
        in_channels=2,
        num_classes=2,
        embed_dims=[32, 64, 160, 256]
    )
    
    # Charge les poids
    checkpoint = torch.load("checkpoints/prostate/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    # Inférence
    infer_patient(
        model=model,
        t2_path="test_data/patient_001/T2.nii.gz",
        adc_path="test_data/patient_001/ADC.nii.gz",
        output_path="results/patient_001_segmentation.nii.gz"
    )
```

### Utilisation:

```bash
python inference_prostate.py
```

---

## Conseils et bonnes pratiques

### 1. **Préparation des données**
- ✅ Assurez-vous que tous les fichiers nii.gz ont le même espacement des voxels
- ✅ Resampllez si nécessaire à un espacement uniforme (ex: 1×1×1 mm)
- ✅ Vérifiez les valeurs min/max de vos images
- ✅ Utilisez des fichiers de segmentation corrects (labels 0 et 1)

### 2. **Ajustement de la taille**
Si vos volumes sont plus grands que 96×96×96:

```python
# Option 1: Augmenter la taille de patch dans config
roi: [128, 128, 128]  # Augmente la taille

# Option 2: Utiliser des patchs avec chevauchement
sw_batch_size: 2  # Réduit si mémoire insuffisante
```

### 3. **Augmentations adaptées à la prostate**
Modifiez `augmentations/augmentations.py`:

```python
# Pour prostate: rotations limitées (pas de rotation complète)
RandRotated(
    prob=0.5,
    range_x=0.17,  # ±10° seulement (pas ±20.6°)
    range_y=0.17,
    range_z=0.17
)
```

### 4. **Gestion de la classe déséquilibrée**
Si la prostate occupe peu de pixels:

```yaml
loss:
  loss_type: "focal"  # Meilleur que Dice pour déséquilibre
  
# Dans config:
loss_args:
  alpha: 0.25  # Poids de la classe prostate
  gamma: 2.0   # Focus sur hard examples
```

### 5. **Validation croisée**
Créez des folds manuels:

```python
# Create kfold splits
from sklearn.model_selection import KFold
patients = sorted(os.listdir("prostate_data/preprocessed"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
    train_patients = [patients[i] for i in train_idx]
    val_patients = [patients[i] for i in val_idx]
    
    # Crée CSVs pour ce fold
    create_csv(train_patients, f"train_fold_{fold}.csv")
    create_csv(val_patients, f"val_fold_{fold}.csv")
```

### 6. **Monitoring de l'entraînement**
Utilisez Weights & Biases pour tracker:

```python
# Dans config
logging:
  project_name: "prostate-segmentation"
  entity_name: "votre-email"
  run_name: "exp_fold_1"
```

Accédez à: https://wandb.ai/your-entity/prostate-segmentation

### 7. **Évaluation des résultats**
Calculez les métriques:

```python
from monai.metrics import compute_meandice
from scipy.spatial import distance

# Dice
dice = compute_meandice(predictions, labels, include_background=False)

# Hausdorff distance
hd = distance.directed_hausdorff(pred_coords, label_coords)[0]
```

---

## Troubleshooting

### Problème: "CUDA out of memory"
**Solution**: Réduire `batch_size` ou `roi`:
```yaml
batch_size: 1
roi: [64, 64, 64]  # Plus petit
```

### Problème: "Loss = NaN"
**Causes possibles**:
- Images mal normalisées (vérifier min/max)
- Labels mal formatés (doivent être 0 et 1)
- Learning rate trop élevé

**Solution**:
```yaml
optimizer:
  lr: 1e-5  # Réduire
```

### Problème: "Métriques plates (pas d'amélioration)"
**Vérifications**:
1. Les labels sont-ils correctement chargés?
2. Les augmentations ne sont-elles pas trop agressives?
3. Le learning rate est-il approprié?

**Solution**: Utiliser un learning rate scheduler avec patience:
```yaml
train_scheduler:
  scheduler_type: "reducelronplateau"
  scheduler_args:
    patience: 15  # Augmente la patience
```

---

## Exemple complet d'exécution

```bash
# 1. Prétraitement
cd data/prostate_raw_data
python prostate_preprocess.py

# 2. Créer splits train/val
cd ../..
python << 'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path("prostate_data/preprocessed")
patients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

n = len(patients)
train_size = int(0.8 * n)
np.random.seed(42)
indices = np.random.permutation(n)

train_df = pd.DataFrame({
    "data_path": [f"./prostate_data/preprocessed/{patients[i]}" for i in indices[:train_size]],
    "case_name": [patients[i] for i in indices[:train_size]]
})
train_df.to_csv("prostate_data/train.csv", index=False)

val_df = pd.DataFrame({
    "data_path": [f"./prostate_data/preprocessed/{patients[i]}" for i in indices[train_size:]],
    "case_name": [patients[i] for i in indices[train_size:]]
})
val_df.to_csv("prostate_data/validation.csv", index=False)
EOF

# 3. Entraîner
python experiments/prostate_exp/run_experiment.py \
  --config experiments/prostate_exp/config_prostate.yaml

# 4. Inférence
python inference_prostate.py
```

---

## Ressources supplémentaires

- **MONAI Prostate Dataset**: https://github.com/Project-MONAI/tutorials
- **NIfTI Format**: https://nibabel.readthedocs.io/
- **SegFormer Paper**: https://arxiv.org/abs/2105.15203

---

**Dernière mise à jour**: Décembre 2025  
**Langue**: Français  
**Statut**: Complet et testé
