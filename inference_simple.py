#!/usr/bin/env python3
"""
Script d'inférence simple pour SegFormer3D
"""

import os
import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import time

# Ajouter le répertoire parent au path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architectures.build_architecture import build_architecture
from train_scripts.logger import get_logger

# Logger d'inférence (configuré dans main)
inference_logger = None

def load_config(config_path):
    """Charge la configuration YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config):
    """Charge le modèle depuis le checkpoint"""
    global inference_logger
    # Créer le modèle
    model = build_architecture(config)
    target_size = config["dataset_parameters"]["train_dataset_args"]["target_size"]
    if inference_logger:
        inference_logger.info(f"Modèle créé: {config['model']['name']}")
        total_params = sum(p.numel() for p in model.parameters())
        inference_logger.debug(f"Paramètres du modèle: {total_params:,}")

    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        # Nouveau format enrichi (best_model.pth / final_model.pth / periodic)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if inference_logger:
                ep = checkpoint.get('epoch', '?')
                dice = checkpoint.get('val_dice', checkpoint.get('best_val_dice', '?'))
                inference_logger.info(
                    f"Checkpoint chargé (epoch {ep}, dice={dice}) depuis: {checkpoint_path}"
                )
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if inference_logger:
                inference_logger.info(f"Checkpoint chargé depuis: {checkpoint_path}")
        else:
            # Peut être un state_dict brut
            model.load_state_dict(checkpoint)
            if inference_logger:
                inference_logger.info(f"State dict chargé depuis: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        if inference_logger:
            inference_logger.info(f"Checkpoint chargé depuis: {checkpoint_path}")

    model.eval()
    return model

def load_pytorch_volume(volume_path):
    """Charge un volume PyTorch (.pt)"""
    data = torch.load(volume_path, map_location='cpu')
    return data

def load_patient_data(patient_dir):
    """Charge les données d'un patient (modalités et labels)"""
    patient_name = os.path.basename(patient_dir)

    # Charger les modalités
    modalities_path = os.path.join(patient_dir, f"{patient_name}_modalities.pt")
    modalities = load_pytorch_volume(modalities_path)  # Shape: (C, D, H, W)

    # Charger les labels (optionnel pour l'inférence)
    labels_path = os.path.join(patient_dir, f"{patient_name}_label.pt")
    if os.path.exists(labels_path):
        labels = load_pytorch_volume(labels_path)  # Shape: (D, H, W)
    else:
        labels = None

    return modalities, labels

def preprocess_volume(volume, target_size=(96, 96, 96)):
    """Prétraite le volume pour l'inférence"""
    # volume est déjà un tensor PyTorch (C, D, H, W)
    # Normalisation
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Redimensionnement si nécessaire
    current_size = volume.shape[1:]  # (D, H, W)
    if current_size != target_size:
        volume = volume.unsqueeze(0)  # (1, C, D, H, W)
        volume = torch.nn.functional.interpolate(
            volume,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        volume = volume.squeeze(0)  # (C, D, H, W)

    return volume

def predict_volume(model, volume, device='cpu'):
    """Fait la prédiction sur un volume"""
    model.to(device)
    volume = volume.to(device)

    with torch.no_grad():
        output = model(volume)
        prediction = torch.argmax(output, dim=1).squeeze().cpu()

    return prediction

def main():
    parser = argparse.ArgumentParser(description="Inférence SegFormer3D")
    parser.add_argument("--config", required=True, help="Chemin vers la configuration")
    parser.add_argument("--checkpoint", required=True, help="Chemin vers le checkpoint")
    parser.add_argument("--input_dir", required=True, help="Répertoire des données d'entrée")
    parser.add_argument("--output_dir", default="inference_results", help="Répertoire de sortie")
    parser.add_argument("--verbosity", choices=["quiet","normal","debug"], default="normal", help="Niveau de verbosité: quiet|normal|debug")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size pour l'inférence (défaut: 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda, défaut: cpu)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de confiance (défaut: 0.5)")
    parser.add_argument("--save_predictions", action="store_true", help="Sauvegarder les prédictions")
    parser.add_argument("--save_probabilities", action="store_true", help="Sauvegarder les probabilités")

    args = parser.parse_args()

    # Configurer le logger d'inférence
    global inference_logger
    level_map = {'quiet': 'WARNING', 'normal': 'INFO', 'debug': 'DEBUG'}
    inference_logger = get_logger(
        "inference",
        level=level_map.get(args.verbosity, 'INFO'),
    )

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger la configuration
    config = load_config(args.config)
    inference_logger.info(f"Configuration chargée: {config['model']['name']}")

    # Charger le modèle
    model = load_model(args.checkpoint, config)

    # Charger les données du patient
    modalities, labels = load_patient_data(args.input_dir)
    inference_logger.info(f"Modalités chargées: shape={modalities.shape}")
    if labels is not None:
        inference_logger.info(f"Labels chargés: shape={labels.shape}")
    inference_logger.debug(f"Modalités dtype={modalities.dtype}, min={modalities.min():.4f}, max={modalities.max():.4f}")
    if labels is not None:
        inference_logger.debug(f"Labels unique: {np.unique(labels)}")

    t0 = time.time()
    processed_volume = preprocess_volume(modalities, target_size=tuple(config['model']['input_size']))
    preprocessing_time = time.time() - t0
    inference_logger.info(f"Volume prétraité: shape={processed_volume.shape}")
    inference_logger.debug(f"Preprocessing: {preprocessing_time:.3f}s")

    # Faire la prédiction
    t0 = time.time()
    prediction = predict_volume(model, processed_volume.unsqueeze(0))  # Ajouter batch dim
    inference_time = time.time() - t0
    inference_logger.info(f"Prédiction faite: shape={prediction.shape} ({inference_time:.2f}s)")
    inference_logger.debug(f"Prediction unique values: {np.unique(prediction.numpy())}")

    # Sauvegarder la prédiction
    output_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(args.input_dir)}.pt")
    torch.save(prediction, output_path)
    inference_logger.info(f"Prédiction sauvegardée: {output_path}")
    try:
        size = os.path.getsize(output_path)
        inference_logger.debug(f"Taille du fichier: {size / 1024:.1f} KB")
    except Exception:
        pass

    # Statistiques
    unique, counts = np.unique(prediction.numpy(), return_counts=True)
    inference_logger.info("Statistiques de prédiction:")
    class_names = ['Background', 'Prostate', 'Bandelettes']
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(class_names):
            percentage = count / prediction.numel() * 100
            inference_logger.info(f"  {class_names[i]}: {percentage:.1f}%")

if __name__ == "__main__":
    main()