"""
Script d'inférence pour segmentation de prostate + bandelettes avec SegFormer3D.

Charge un modèle entraîné et prédit sur de nouvelles données prostate en format nii.gz.
Génère des segmentations multi-label (0=fond, 1=prostate, 2=bandelettes).

Utilisation:
    python experiments/prostate_seg/inference_prostate.py \\
        --model_path ./experiments/prostate_seg/checkpoints/best.pt \\
        --input_dir ./test_data/raw \\
        --output_dir ./test_data/predictions \\
        --save_nifti true \\
        --save_separate_labels true \\
        --save_prob_map true
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, Dict
from scipy.ndimage import zoom, label, binary_closing, binary_opening
from tqdm import tqdm

# Ajoute le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from architectures.build_architecture import build_architecture


class ProstateInferencer:
    """Classe pour inférence sur données de prostate."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        target_size: int = 96,
        sw_batch_size: int = 2,
        sw_overlap: float = 0.5
    ) -> None:
        """
        Initialise l'inférence.
        
        Args:
            model_path: Chemin vers le checkpoint du modèle
            device: "cuda" ou "cpu"
            target_size: Taille d'entrée du modèle (défaut: 96)
            sw_batch_size: Batch size pour sliding window
            sw_overlap: Overlap entre les patches (0-1)
        """
        self.device = torch.device(device)
        self.target_size = target_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap
        
        # Charge le modèle
        print(f"📦 Chargement du modèle depuis {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✅ Modèle chargé sur {device}")
    
    def _load_model(self, model_path: str):
        """Charge le modèle PyTorch."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Crée l'architecture (prostate + bandelettes: 2 in, 3 out)
        model = build_architecture(
            {"name":"segformer3d",
            "in_channels":2,
            "num_classes":2,  # 2 classes: fond, prostate
            "patch_size":8,
            "embed_dim":64,
            "num_layers":4,
            "num_heads":4}
        )
        
        # Charge les weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle différents formats de checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Charge les weights
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        
        return model
    
    def load_nifti(self, filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Charge un fichier nifti et retourne les données et l'objet."""
        img = nib.load(filepath)
        data = img.get_fdata()
        return data, img
    
    def resample_to_target(
        self,
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
        order: int = 1
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample un volume à la taille cible.
        
        Retourne aussi les facteurs de zoom pour rescaler les prédictions.
        """
        if volume.shape == target_shape:
            return volume, (1.0, 1.0, 1.0)
        
        zoom_factors = tuple(
            target_shape[i] / volume.shape[i] for i in range(3)
        )
        
        resampled = zoom(volume, zoom_factors, order=order, mode='constant')
        
        # Ajuste la taille exacte
        if resampled.shape != target_shape:
            output = np.zeros(target_shape, dtype=resampled.dtype)
            slices = tuple(
                slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3)
            )
            output[slices] = resampled[slices]
            resampled = output
        
        return resampled, zoom_factors
    
    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalise les intensités en [0, 1]."""
        vmin = volume[volume > 0].min()
        vmax = volume[volume > 0].max()
        
        if vmax > vmin:
            normalized = (volume - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = volume
        
        return normalized.astype(np.float32)
    
    def sliding_window_inference(
        self,
        volume: torch.Tensor,
        roi_size: int = 96,
        overlap: float = 0.5
    ) -> torch.Tensor:
        """
        Inférence avec sliding window pour volumes larges.
        
        Args:
            volume: Tensor de forme (C, D, H, W)
            roi_size: Taille de la fenêtre
            overlap: Overlap fraction
        
        Returns:
            Prédiction (1, D, H, W)
        """
        C, D, H, W = volume.shape
        
        # Calcule le stride
        stride = int(roi_size * (1 - overlap))
        if stride == 0:
            stride = roi_size
        
        # Initialise l'accumulateur
        output = torch.zeros(1, D, H, W, device=self.device)
        count = torch.zeros(1, D, H, W, device=self.device)
        
        # Applique la fenêtre glissante
        d_idx = 0
        while d_idx + roi_size <= D:
            h_idx = 0
            while h_idx + roi_size <= H:
                w_idx = 0
                while w_idx + roi_size <= W:
                    # Extrait la région
                    patch = volume[
                        :,
                        d_idx:d_idx+roi_size,
                        h_idx:h_idx+roi_size,
                        w_idx:w_idx+roi_size
                    ].unsqueeze(0)  # (1, C, D, H, W)
                    
                    # Inférence
                    with torch.no_grad():
                        pred = self.model(patch)
                        pred = torch.softmax(pred, dim=1)
                        # Garde seulement la classe prostate (index 1)
                        pred = pred[:, 1:2, :, :, :]
                    
                    # Ajoute au résultat
                    output[:, d_idx:d_idx+roi_size, h_idx:h_idx+roi_size, w_idx:w_idx+roi_size] += pred[0]
                    count[:, d_idx:d_idx+roi_size, h_idx:h_idx+roi_size, w_idx:w_idx+roi_size] += 1
                    
                    w_idx += stride
                
                h_idx += stride
            
            d_idx += stride
        
        # Normalise par le nombre de fenêtres
        output = output / (count + 1e-6)
        
        return output
    
    @torch.no_grad()
    def predict(
        self,
        t2_volume: np.ndarray,
        adc_volume: np.ndarray,
        use_sliding_window: bool = True
    ) -> np.ndarray:
        """
        Prédit la segmentation de prostate.
        
        Args:
            t2_volume: Volume T2 (D, H, W)
            adc_volume: Volume ADC (D, H, W)
            use_sliding_window: Utiliser sliding window ou directement
        
        Returns:
            Probabilité de prostate (D, H, W) en [0, 1]
        """
        # Resample à la taille cible
        target = (self.target_size, self.target_size, self.target_size)
        t2_resampled, _ = self.resample_to_target(t2_volume, target, order=1)
        adc_resampled, _ = self.resample_to_target(adc_volume, target, order=1)
        
        # Normalise
        t2_norm = self.normalize_intensity(t2_resampled)
        adc_norm = self.normalize_intensity(adc_resampled)
        
        # Empile les modalités
        volume = np.stack([t2_norm, adc_norm], axis=0)  # (2, D, H, W)
        volume_tensor = torch.from_numpy(volume).float().to(self.device)
        
        # Inférence
        if use_sliding_window and max(volume.shape[1:]) > self.target_size:
            output = self.sliding_window_inference(volume_tensor)
        else:
            volume_tensor = volume_tensor.unsqueeze(0)  # (1, 2, D, H, W)
            output = self.model(volume_tensor)
            output = torch.softmax(output, dim=1)
            output = output[0]  # (3, D, H, W) - 3 classes
        
        return output.cpu().numpy()  # Retourne les probas pour les 3 classes
    
    def post_process_multiclass(
        self,
        probabilities: np.ndarray,
        threshold_prostate: float = 0.5,
        threshold_bandelettes: float = 0.5,
        remove_small_components: bool = True,
        min_component_size: int = 50
    ) -> np.ndarray:
        """
        Post-traitement pour segmentation multi-classe.
        
        Args:
            probabilities: Probabilités (3, D, H, W) ou (D, H, W)
                - Si 3D: Channel 0=fond, 1=prostate, 2=bandelettes
                - Si 2D: Prostate uniquement
            threshold_prostate: Seuil pour prostate
            threshold_bandelettes: Seuil pour bandelettes
            remove_small_components: Enlever les petites composantes
            min_component_size: Taille minimale d'une composante
        
        Returns:
            Segmentation multi-classe (D, H, W) avec labels 0, 1, 2
        """
        # Gère le cas 3D et 2D
        if probabilities.ndim == 3 and probabilities.shape[0] == 3:
            prob_prostate = probabilities[1]  # Class 1
            prob_bandelettes = probabilities[2]  # Class 2
        else:
            # Cas 2D ou probabilité unique
            prob_prostate = probabilities if probabilities.ndim == 2 else probabilities.squeeze()
            prob_bandelettes = np.zeros_like(prob_prostate)
        
        # Initialise la segmentation multi-label
        segmentation = np.zeros_like(prob_prostate, dtype=np.uint8)
        
        # Binarise prostate
        prostate_mask = (prob_prostate > threshold_prostate).astype(np.uint8)
        
        # Binarise bandelettes
        bandelettes_mask = (prob_bandelettes > threshold_bandelettes).astype(np.uint8)
        
        # Combine (bandelettes > prostate en cas de conflit)
        segmentation[prostate_mask > 0] = 1
        segmentation[bandelettes_mask > 0] = 2
        
        # Enlève les petites composantes
        if remove_small_components:
            for label_id in [1, 2]:
                labeled, num_components = label(segmentation == label_id)
                for comp_id in range(1, num_components + 1):
                    component = (labeled == comp_id)
                    if component.sum() < min_component_size:
                        segmentation[component] = 0
        
        # Opérations morphologiques
        for label_id in [1, 2]:
            mask = (segmentation == label_id)
            mask = binary_closing(binary_opening(mask, iterations=1), iterations=1)
            segmentation[~mask & (segmentation == label_id)] = 0
            segmentation[mask & (segmentation == 0)] = label_id
        
        return segmentation.astype(np.uint8)
    
    def post_process(
        self,
        segmentation: np.ndarray,
        threshold: float = 0.5,
        remove_small_components: bool = True,
        min_component_size: int = 50
    ) -> np.ndarray:
        """
        Post-traitement de la segmentation (compatibilité legacy).
        
        Args:
            segmentation: Probabilités (D, H, W)
            threshold: Seuil de binarisation
            remove_small_components: Enlever les petites composantes
            min_component_size: Taille minimale d'une composante
        
        Returns:
            Segmentation binaire (0 ou 1)
        """
        # Binarise
        binary = (segmentation > threshold).astype(np.uint8)
        
        # Nettoie les petites composantes
        if remove_small_components:
            labeled, num_components = label(binary)
            for comp_id in range(1, num_components + 1):
                component = (labeled == comp_id)
                if component.sum() < min_component_size:
                    binary[component] = 0
        
        # Opérations morphologiques
        binary = binary_closing(binary_opening(binary, iterations=1), iterations=1)
        
        return binary.astype(np.uint8)
    
    def save_nifti(
        self,
        data: np.ndarray,
        reference_img: nib.Nifti1Image,
        output_path: str
    ) -> None:
        """Sauvegarde un array en fichier nifti."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Utilise l'affine et le header de référence
        output_img = nib.Nifti1Image(data, affine=reference_img.affine, header=reference_img.header)
        nib.save(output_img, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Inférence pour segmentation de prostate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Chemin vers le checkpoint du modèle"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Répertoire contenant les données brutes (nii.gz)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de sortie pour les segmentations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device pour inférence"
    )
    parser.add_argument(
        "--save_nifti",
        type=bool,
        default=True,
        help="Sauvegarde les prédictions en nifti"
    )
    parser.add_argument(
        "--save_separate_labels",
        type=bool,
        default=False,
        help="Sauvegarde prostate et bandelettes dans des fichiers séparés"
    )
    parser.add_argument(
        "--save_prob_map",
        type=bool,
        default=False,
        help="Sauvegarde les cartes de probabilité"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil de binarisation prostate"
    )
    parser.add_argument(
        "--threshold_bandelettes",
        type=float,
        default=0.5,
        help="Seuil de binarisation bandelettes"
    )
    
    args = parser.parse_args()
    
    # Initialise l'inférence
    inferencer = ProstateInferencer(
        model_path=args.model_path,
        device=args.device
    )
    
    # Trouve les répertoires de patients
    input_path = Path(args.input_dir)
    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print(f"❌ Aucun répertoire de patient trouvé dans {args.input_dir}")
        return
    
    print(f"\n📊 Inférence sur {len(patient_dirs)} patients")
    print(f"   Modèle: {args.model_path}")
    print(f"   Sortie: {args.output_dir}")
    print(f"   Seuils: prostate={args.threshold}, bandelettes={args.threshold_bandelettes}\n")
    
    # Prédit pour chaque patient
    for patient_dir in tqdm(patient_dirs, desc="Inférence"):
        patient_name = patient_dir.name
        
        try:
            # Charge T2 et ADC
            t2_path = patient_dir / "T2.nii.gz"
            adc_path = patient_dir / "ADC.nii.gz"
            
            if not t2_path.exists() or not adc_path.exists():
                warnings.warn(f"Fichiers manquants pour {patient_name}")
                continue
            
            t2, t2_img = inferencer.load_nifti(str(t2_path))
            adc, _ = inferencer.load_nifti(str(adc_path))
            
            # Prédit - retourne probas (3, D, H, W)
            prob_maps = inferencer.predict(t2, adc, use_sliding_window=True)
            
            # Post-traitement multi-classe
            segmentation = inferencer.post_process_multiclass(
                prob_maps,
                threshold_prostate=args.threshold,
                threshold_bandelettes=args.threshold_bandelettes,
                remove_small_components=True,
                min_component_size=50
            )
            
            # Sauvegarde
            output_patient_dir = Path(args.output_dir) / patient_name
            
            if args.save_nifti:
                seg_path = output_patient_dir / "segmentation_pred.nii.gz"
                inferencer.save_nifti(segmentation, t2_img, str(seg_path))
            
            # Sauvegarde les labels séparés si demandé
            if args.save_separate_labels:
                prostate_mask = (segmentation == 1).astype(np.uint8)
                bandelettes_mask = (segmentation == 2).astype(np.uint8)
                
                prostate_path = output_patient_dir / "prostate_pred.nii.gz"
                bandelettes_path = output_patient_dir / "bandelettes_pred.nii.gz"
                
                inferencer.save_nifti(prostate_mask, t2_img, str(prostate_path))
                inferencer.save_nifti(bandelettes_mask, t2_img, str(bandelettes_path))
            
            if args.save_prob_map:
                # Sauvegarde cartes de probas
                prostate_prob_path = output_patient_dir / "prostate_probability.nii.gz"
                bandelettes_prob_path = output_patient_dir / "bandelettes_probability.nii.gz"
                
                if prob_maps.ndim == 3 and prob_maps.shape[0] == 3:
                    inferencer.save_nifti(prob_maps[1], t2_img, str(prostate_prob_path))
                    inferencer.save_nifti(prob_maps[2], t2_img, str(bandelettes_prob_path))
            
        except Exception as e:
            warnings.warn(f"Erreur pour {patient_name}: {str(e)}")
    
    print(f"\n✅ Inférence complétée. Résultats dans {args.output_dir}")


if __name__ == "__main__":
    main()
