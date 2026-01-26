"""
Script de prétraitement pour données de prostate + bandelettes au format nii.gz.

Ce script:
1. Charge les fichiers nii.gz T2, ADC et segmentation multi-label
2. Resample à une taille uniforme (96x96x96)
3. Normalise les intensités
4. Convertit en format PyTorch .pt pour entraînement rapide

Structure d'entrée attendue:
    prostate_raw_data/
    ├── patient_001/
    │   ├── T2.nii.gz
    │   ├── ADC.nii.gz
    │   ├── prostate.nii.gz        (label 1)
    │   └── bandelettes.nii.gz     (label 2)
    ├── patient_002/
    │   ├── T2.nii.gz
    │   ├── ADC.nii.gz
    │   ├── prostate.nii.gz
    │   └── bandelettes.nii.gz
    └── ...

OU segmentation multi-label unique:
    prostate_raw_data/
    ├── patient_001/
    │   ├── T2.nii.gz
    │   ├── ADC.nii.gz
    │   └── segmentation.nii.gz   (0=fond, 1=prostate, 2=bandelettes)
    └── ...

Structure de sortie:
    prostate_preprocessed/
    ├── patient_001/
    │   ├── patient_001_modalities.pt  (2, 96, 96, 96) [T2, ADC] ou (1, 96, 96, 96) [T2 seul]
    │   └── patient_001_label.pt       (1, 96, 96, 96) [0/1/2]
    ├── patient_002/
    │   ├── patient_002_modalities.pt
    │   └── patient_002_label.pt
    └── ...

Utilisation:
    python prostate_preprocess.py \\
        --input_dir /path/to/prostate_raw_data \\
        --output_dir /path/to/prostate_preprocessed \\
        --target_size 96
"""

import os
import argparse
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm


class ProstatePreprocessor:
    """Préprocessor pour données de prostate nii.gz."""
    
    def __init__(
        self,
        target_size: int = 96,
        resample_mode: str = "linear",
        normalize_method: str = "minmax"
    ) -> None:
        """
        Initialise le préprocessor.
        
        Args:
            target_size (int): Taille cible pour resample (défaut: 96).
            resample_mode (str): Mode de resampling ("linear", "nearest").
            normalize_method (str): Méthode de normalisation ("minmax", "zscore").
        """
        self.target_size = target_size
        self.resample_mode = resample_mode
        self.normalize_method = normalize_method
    
    def load_nifti(self, filepath: str) -> np.ndarray:
        """Charge un fichier nifti (.nii.gz)."""
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement {filepath}: {str(e)}")
    
    def _load_segmentation(self, case_dir: str, case_name: str) -> np.ndarray:
        """
        Charge la segmentation multi-label.
        
        Supporte deux formats:
        1. Fichier unique: segmentation.nii.gz (0=fond, 1=prostate, 2=bandelettes)
        2. Fichiers séparés: prostate.nii.gz (label 1), bandelettes.nii.gz (label 2)
        
        Args:
            case_dir: Répertoire du patient
            case_name: Nom du patient
        
        Returns:
            Segmentation multi-label (D, H, W) avec valeurs 0, 1, 2
        """
        # Essaie d'abord le fichier unique multi-label
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        if os.path.exists(seg_path):
            seg = self.load_nifti(seg_path)
            # Pour segmentation binaire, convertir en 0/1 (tout >0 devient 1)
            seg = (seg > 0).astype(np.float32)
            return seg
        
        # Sinon, combine les fichiers séparés
        prostate_path = os.path.join(case_dir, "prostate.nii.gz")
        bandelettes_path = os.path.join(case_dir, "bandelettes.nii.gz")
        
        if not os.path.exists(prostate_path):
            raise FileNotFoundError(
                f"Segmentation manquante pour {case_name}\n"
                f"Attendu: segmentation.nii.gz OU prostate.nii.gz + bandelettes.nii.gz"
            )
        
        # Charge prostate
        prostate = self.load_nifti(prostate_path)
        prostate_mask = (prostate > 0.5).astype(np.uint8)
        
        # Charge bandelettes si disponible
        bandelettes_mask = np.zeros_like(prostate_mask)
        if os.path.exists(bandelettes_path):
            bandelettes = self.load_nifti(bandelettes_path)
            bandelettes_mask = (bandelettes > 0.5).astype(np.uint8)
        
        # Combine: 0=fond, 1=prostate (bandelettes ignorées pour segmentation binaire)
        seg_combined = np.zeros_like(prostate_mask)
        seg_combined[prostate_mask > 0] = 1
        # seg_combined[bandelettes_mask > 0] = 2  # Commenté pour segmentation binaire
        
        return seg_combined.astype(np.float32)
    
    def resample_volume(
        self,
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
        order: int = 1,
        mode: str = "constant"
    ) -> np.ndarray:
        """
        Resample un volume à la taille cible.
        
        Args:
            volume: Données d'entrée (D, H, W)
            target_shape: Taille cible (96, 96, 96)
            order: 1=bilinéaire, 0=nearest neighbor
            mode: Mode de remplissage ("constant", "reflect")
        
        Returns:
            Volume resamplé
        """
        if volume.shape == target_shape:
            return volume
        
        # Calcule les facteurs de zoom
        zoom_factors = [
            target_shape[i] / volume.shape[i]
            for i in range(3)
        ]
        
        # Resample
        if order == 0:  # Nearest neighbor pour segmentation
            resampled = zoom(volume, zoom_factors, order=0, mode=mode)
        else:  # Bilinéaire pour images
            resampled = zoom(volume, zoom_factors, order=order, mode=mode)
        
        # Ajuste la taille exacte (artefacts de zoom)
        if resampled.shape != target_shape:
            output = np.zeros(target_shape, dtype=resampled.dtype)
            slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
            output[slices] = resampled[slices]
            resampled = output
        
        return resampled
    
    def normalize_intensity(
        self,
        volume: np.ndarray,
        mask: np.ndarray = None,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalise les intensités d'un volume.
        
        Args:
            volume: Données d'entrée
            mask: Mask optionnel pour normaliser seulement la région d'intérêt
            method: "minmax" (0-1) ou "zscore" (gaussienne)
        
        Returns:
            Volume normalisé
        """
        # Utilise le mask si fourni, sinon tout le volume
        if mask is not None:
            data_to_normalize = volume[mask > 0]
        else:
            data_to_normalize = volume.flatten()
            data_to_normalize = data_to_normalize[data_to_normalize > 0]  # Exclut les zéros
        
        if len(data_to_normalize) == 0:
            warnings.warn("Pas de données non-zéro pour normaliser")
            return volume
        
        if method == "minmax":
            vmin = data_to_normalize.min()
            vmax = data_to_normalize.max()
            if vmax > vmin:
                normalized = (volume - vmin) / (vmax - vmin)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = volume
        
        elif method == "zscore":
            vmean = data_to_normalize.mean()
            vstd = data_to_normalize.std()
            if vstd > 0:
                normalized = (volume - vmean) / vstd
                normalized = np.clip(normalized, -3, 3)
                normalized = (normalized + 3) / 6  # Ramène à [0, 1]
            else:
                normalized = volume
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        return normalized.astype(np.float32)
    
    def preprocess_case(
        self,
        case_dir: str,
        case_name: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Prétraite un patient complet.
        
        Args:
            case_dir: Répertoire du patient (contient T2.nii.gz, ADC.nii.gz, seg.nii.gz)
            case_name: Nom du patient (ex: "patient_001")
            output_dir: Répertoire de sortie
        
        Returns:
            Dict avec statut et infos
        """
        result = {
            "case": case_name,
            "success": False,
            "message": "",
            "stats": {}
        }
        
        try:
            # Chemins des fichiers - support multi-label
            t2_path = os.path.join(case_dir, "T2.nii.gz")
            adc_path = os.path.join(case_dir, "ADC.nii.gz")
            
            # Vérifie l'existence du fichier T2
            if not os.path.exists(t2_path):
                raise FileNotFoundError(f"T2.nii.gz manquant dans {case_dir}")
            
            # Charge les données T2 et ADC
            t2 = self.load_nifti(t2_path)
            
            # Charge ADC si disponible
            adc = None
            if os.path.exists(adc_path):
                adc = self.load_nifti(adc_path)
            
            # Charge la segmentation (multi-label ou fichiers séparés)
            seg = self._load_segmentation(case_dir, case_name)
            
            # Resample à la taille cible
            target = (self.target_size, self.target_size, self.target_size)
            t2_resampled = self.resample_volume(t2, target, order=1)
            seg_resampled = self.resample_volume(seg, target, order=0)
            
            if adc is not None:
                adc_resampled = self.resample_volume(adc, target, order=1)
            
            # Préserve les labels multi-classe (0, 1, 2)
            seg_labels = np.round(seg_resampled).astype(np.uint8)
            seg_labels = np.clip(seg_labels, 0, 2)  # Assure labels 0, 1, 2
            
            # Crée un mask pour la normalisation (tout sauf fond = prostate + bandelettes)
            mask = (seg_labels > 0).astype(np.float32)
            
            # Normalise les intensités
            t2_norm = self.normalize_intensity(t2_resampled, mask, method=self.normalize_method)
            
            # Stack les modalités: T2 et ADC si disponible
            if adc is not None:
                adc_norm = self.normalize_intensity(adc_resampled, mask, method=self.normalize_method)
                modalities = np.stack([t2_norm, adc_norm], axis=0)  # (2, D, H, W)
            else:
                modalities = t2_norm[np.newaxis, :, :, :]  # (1, D, H, W) si pas d'ADC
            
            label = seg_labels[np.newaxis, :, :, :]  # (1, D, H, W) avec labels 0, 1, 2
            
            # Convertit en tenseurs PyTorch
            modalities_tensor = torch.from_numpy(modalities).float()
            label_tensor = torch.from_numpy(label).float()
            
            # Crée le répertoire de sortie
            output_patient_dir = os.path.join(output_dir, case_name)
            os.makedirs(output_patient_dir, exist_ok=True)
            
            # Sauvegarde les fichiers .pt
            modality_path = os.path.join(output_patient_dir, f"{case_name}_modalities.pt")
            label_path = os.path.join(output_patient_dir, f"{case_name}_label.pt")
            
            torch.save(modalities_tensor, modality_path)
            torch.save(label_tensor, label_path)
            
            # Collecte les stats
            result["success"] = True
            result["message"] = f" Prétraité avec succès"
            
            # Stats par classe
            prostate_count = int((seg_labels == 1).sum())
            bandelettes_count = int((seg_labels == 2).sum())
            
            result["stats"] = {
                "input_shape": t2.shape,
                "output_shape": tuple(modalities_tensor.shape),
                "t2_range": (float(t2_norm.min()), float(t2_norm.max())),
                "prostate_voxels": prostate_count,
                "bandelettes_voxels": bandelettes_count,
                "total_voxels": int(np.prod(seg_labels.shape)),
            }
        
        except Exception as e:
            result["success"] = False
            result["message"] = f" Erreur: {str(e)}"
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Prétraite les données de prostate nii.gz pour SegFormer3D"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/prostate_raw_data",
        help="Répertoire contenant les données brutes (défaut: ./data/prostate_raw_data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/prostate_preprocessed",
        help="Répertoire de sortie pour données prétraitées (défaut: ./data/prostate_preprocessed)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=96,
        help="Taille cible pour resample (défaut: 96)"
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="Méthode de normalisation (défaut: minmax)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Saute les patients déjà prétraités"
    )
    
    args = parser.parse_args()
    
    # Crée le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialise le préprocessor
    preprocessor = ProstatePreprocessor(
        target_size=args.target_size,
        normalize_method=args.normalize_method
    )
    
    # Trouve tous les répertoires de patients
    input_path = Path(args.input_dir)
    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print(f" Aucun répertoire de patient trouvé dans {args.input_dir}")
        print(f"Structure attendue: {args.input_dir}/patient_001/{{T2.nii.gz, ADC.nii.gz, segmentation.nii.gz}}")
        return
    
    print(f"\n Prétraitement de {len(patient_dirs)} patients...")
    print(f"   Entrée: {args.input_dir}")
    print(f"   Sortie: {args.output_dir}")
    print(f"   Taille cible: {args.target_size}x{args.target_size}x{args.target_size}")
    print(f"   Normalisation: {args.normalize_method}\n")
    
    # Prétraite chaque patient
    results = []
    for patient_dir in tqdm(patient_dirs, desc="Prétraitement"):
        case_name = patient_dir.name
        
        # Vérifie si déjà prétraité
        output_patient_dir = os.path.join(args.output_dir, case_name)
        if args.skip_existing and os.path.exists(output_patient_dir):
            print(f"⏭  {case_name}: déjà prétraité, skipper")
            continue
        
        result = preprocessor.preprocess_case(
            str(patient_dir),
            case_name,
            args.output_dir
        )
        results.append(result)
        
        # Affiche le statut
        status = "" if result["success"] else ""
        print(f"{status} {case_name}: {result['message']}")
    
    # Résumé final
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f" RÉSUMÉ: {success_count}/{len(results)} patients prétraités avec succès")
    
    if success_count > 0:
        print(f"\n Données prétraitées dans: {args.output_dir}")
        print(f"\n Prochaines étapes:")
        print(f"   1. Créer les fichiers CSV (train.csv, validation.csv)")
        print(f"   2. Configurer un fichier config_prostate.yaml")
        print(f"   3. Lancer l'entraînement avec trainer_ddp.py")
    
    if success_count < len(results):
        print(f"\n  {len(results) - success_count} erreurs - vérifiez les logs ci-dessus")


if __name__ == "__main__":
    main()
