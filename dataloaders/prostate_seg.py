"""
Dataloader pour segmentation de prostate (format nii.gz prétraité en .pt).

Ce module charge les données de prostate prétraitées en format PyTorch .pt
(résultat du script prostate_preprocess.py dans data/prostate_raw_data/).

Les données doivent être organisées comme:
    prostate_data/
    ├── preprocessed/
    │   ├── patient_001/
    │   │   ├── patient_001_modalities.pt  (2, D, H, W) - T2, ADC
    │   │   └── patient_001_label.pt       (1, D, H, W) - Segmentation
    │   ├── patient_002/
    │   └── ...
    ├── train.csv
    └── validation.csv

CSV format:
    data_path,case_name
    ./prostate_data/preprocessed/patient_001,patient_001
    ./prostate_data/preprocessed/patient_002,patient_002
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from typing import Optional, Dict, Any
import warnings


class ProstateSegDataset(Dataset):
    """Dataset pour segmentation de prostate.
    
    Charge les données de prostate prétraitées en format .pt
    (résultat du script prostate_preprocess.py).
    
    Attributs:
        csv (pd.DataFrame): DataFrame contenant data_path et case_name
        transform: Transformations MONAI à appliquer
        is_train (bool): Mode entraînement (True) ou validation (False)
    
    Exemple:
        >>> dataset = ProstateSegDataset(
        ...     root_dir="./prostate_data/preprocessed",
        ...     is_train=True,
        ...     transform=augmentations
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # (2, D, H, W)
        >>> print(sample['label'].shape)  # (1, D, H, W)
    """
    
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transform: Optional[Any] = None,
        split_file: Optional[str] = None,
        target_size: int = 96,  # Redimensionner à 96x96x96
    ) -> None:
        """
        Initialise le dataset de prostate.
        
        Args:
            root_dir (str): Répertoire parent contenant:
                - Sous-répertoires avec données prétraitées
                - Fichiers CSV (train.csv, validation.csv)
            is_train (bool): Mode entraînement (True) ou validation (False).
                Defaults to True.
            transform (Optional[Any]): Transformations MONAI à appliquer aux samples.
                Typiquement un objet Compose() contenant des augmentations.
                Defaults to None.
            split_file (Optional[str]): Chemin personnalisé vers CSV au lieu du fichier par défaut.
                Si None, utilise "train.csv" ou "validation.csv" basé sur is_train.
                Defaults to None.
            target_size (int): Taille cible pour le redimensionnement (ex: 96).
                Defaults to 96.
        
        Raises:
            FileNotFoundError: Si le fichier CSV n'existe pas
        
        Exemple:
            >>> dataset = ProstateSegDataset(
            ...     root_dir="./prostate_data/preprocessed",
            ...     is_train=True,
            ...     transform=transforms,
            ...     target_size=96
            ... )
            >>> print(len(dataset))  # Nombre de patients
        """
        super().__init__()
        
        # Détermine le chemin du fichier CSV
        if split_file is None:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
        else:
            csv_fp = split_file
        
        # Vérifie l'existence du fichier CSV
        if not os.path.exists(csv_fp):
            raise FileNotFoundError(
                f"CSV file not found: {csv_fp}\n"
                f"Créez d'abord train.csv et validation.csv dans {root_dir}"
            )
        
        # Charge le CSV
        self.csv = pd.read_csv(csv_fp)
        self.transform = transform
        self.is_train = is_train
        self.target_size = target_size
        
        print(f"✅ Chargé {len(self.csv)} patients depuis {csv_fp} (cible: {target_size}³)")
    
    def __len__(self) -> int:
        """Retourne le nombre de samples dans le dataset."""
        return len(self.csv)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Charge un sample (patient) complet.
        
        Args:
            idx (int): Index du sample à charger
        
        Returns:
            Dict[str, torch.Tensor]: Dictionnaire contenant:
                - 'image': Modalités (2, D, H, W) - T2, ADC
                - 'label': Segmentation (1, D, H, W) - Binaire (0 ou 1)
        
        Raises:
            FileNotFoundError: Si un fichier .pt n'existe pas
            RuntimeError: Si les données ne peuvent pas être chargées
        
        Exemple:
            >>> sample = dataset[0]
            >>> image = sample['image']  # (2, 96, 96, 96)
            >>> label = sample['label']  # (1, 96, 96, 96)
        """
        # Récupère les informations du CSV
        data_path = self.csv["data_path"].iloc[idx]
        case_name = self.csv["case_name"].iloc[idx]
        
        # Get target_size, with default
        target_size = getattr(self, 'target_size', 96)
        
        # Construit les chemins des fichiers .pt
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        
        try:
            # Charge les fichiers .pt
            # weights_only=False est nécessaire pour charger les anciens fichiers PyTorch
            volume = torch.load(volume_fp, map_location='cpu', weights_only=False)
            label = torch.load(label_fp, map_location='cpu', weights_only=False)
            
            # Assure que les données sont des tenseurs
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume)
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)
            
            # Convertit en float32
            volume = volume.float()
            label = label.float()
            
            # Redimensionne si nécessaire
            if volume.shape[-1] != self.target_size:
                # Ajoute dimension batch pour interpolate
                volume_resized = interpolate(
                    volume.unsqueeze(0),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
                
                label_resized = interpolate(
                    label.unsqueeze(0),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='nearest'
                ).squeeze(0)
                
                volume = volume_resized
                label = label_resized
            
            data = {
                "image": volume,  # (num_modalités, target_size, target_size, target_size)
                "label": label    # (1, target_size, target_size, target_size)
            }
        
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Fichier manquant pour {case_name}:\n"
                f"  Image: {volume_fp}\n"
                f"  Label: {label_fp}\n"
                f"Vérifiez que prostate_preprocess.py a été exécuté correctement."
            ) from e
        
        except Exception as e:
            warnings.warn(
                f"Erreur lors du chargement du sample {idx} ({case_name}): {str(e)}"
            )
            raise
        
        # Applique les augmentations si fournies
        if self.transform:
            data = self.transform(data)
        
        return data


class ProstateSegDatasetMultiModal(Dataset):
    """Dataset pour prostate avec support de modalités variables.
    
    Contrairement à ProstateSegDataset qui suppose des modalités fixes (T2, ADC),
    cette version supporte un nombre variable de modalités.
    
    Utile si vous avez différents sous-ensembles de patients avec différentes
    modalités disponibles.
    """
    
    def __init__(
        self,
        root_dir: str,
        modalities: list = ["T2", "ADC"],
        is_train: bool = True,
        transform: Optional[Any] = None,
        split_file: Optional[str] = None,
    ) -> None:
        """
        Initialise le dataset multi-modal.
        
        Args:
            root_dir (str): Répertoire des données prétraitées
            modalities (list): Modalités à charger (ex: ["T2", "ADC", "DWI"])
            is_train (bool): Mode entraînement
            transform (Optional[Any]): Augmentations MONAI
            split_file (Optional[str]): Chemin CSV personnalisé
        """
        super().__init__()
        
        # Détermine le chemin du CSV
        if split_file is None:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
        else:
            csv_fp = split_file
        
        if not os.path.exists(csv_fp):
            raise FileNotFoundError(f"CSV file not found: {csv_fp}")
        
        self.csv = pd.read_csv(csv_fp)
        self.transform = transform
        self.modalities = modalities
        self.is_train = is_train
        
        print(f"✅ Chargé {len(self.csv)} patients avec modalités: {modalities}")
    
    def __len__(self) -> int:
        return len(self.csv)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Charge un sample avec modalités variables."""
        data_path = self.csv["data_path"].iloc[idx]
        case_name = self.csv["case_name"].iloc[idx]
        
        # Charge la segmentation
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        label = torch.load(label_fp, map_location='cpu', weights_only=False).float()
        
        # Charge les modalités disponibles
        modality_volumes = []
        for mod in self.modalities:
            mod_fp = os.path.join(data_path, f"{case_name}_{mod}.pt")
            if os.path.exists(mod_fp):
                vol = torch.load(mod_fp, map_location='cpu', weights_only=False).float()
                modality_volumes.append(vol)
            else:
                # Si une modalité manque, remplace par zéros
                warnings.warn(f"Modalité manquante {mod} pour {case_name}")
                modality_volumes.append(torch.zeros_like(label))
        
        # Empile les modalités
        image = torch.stack(modality_volumes, dim=0)
        
        data = {
            "image": image,
            "label": label
        }
        
        if self.transform:
            data = self.transform(data)
        
        return data
