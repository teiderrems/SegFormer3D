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

from architectures.build_architecture import build_architecture
from torchsummary import summary

# Global verbosity level: 'quiet', 'normal', or 'debug' (set by CLI)
VERBOSITY = 'normal'   # default

def load_config(config_path):
    """Charge la configuration YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config):
    """Charge le modèle depuis le checkpoint"""
    # Créer le modèle
    model = build_architecture(config)
    target_size = config["dataset_parameters"]["train_dataset_args"]["target_size"]
    summary(model, input_size=(1, target_size, target_size, target_size))
    if VERBOSITY != 'quiet':
        print(f"Modèle créé: {config['model']['name']}")
    if VERBOSITY == 'debug':
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[debug] Model parameters: {total_params}")

    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    if VERBOSITY != 'quiet':
        print(f"Checkpoint chargé depuis: {checkpoint_path}")

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

    args = parser.parse_args()

    # Set global verbosity
    global VERBOSITY
    VERBOSITY = args.verbosity

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger la configuration
    config = load_config(args.config)
    if VERBOSITY != 'quiet':
        print(f"Configuration chargée: {config['model']['name']}")

    # Charger le modèle
    model = load_model(args.checkpoint, config)

    # Charger les données du patient
    modalities, labels = load_patient_data(args.input_dir)
    if VERBOSITY != 'quiet':
        print(f"Modalités chargées: shape={modalities.shape}")
        if labels is not None:
            print(f"Labels chargés: shape={labels.shape}")
    if VERBOSITY == 'debug':
        print(f"[debug] Modalités dtype={modalities.dtype}, min={modalities.min()}, max={modalities.max()}")
        if labels is not None:
            print(f"[debug] Labels unique: {np.unique(labels)}")

    t0 = time.time()
    processed_volume = preprocess_volume(modalities, target_size=tuple(config['model']['input_size']))
    preprocessing_time = time.time() - t0
    if VERBOSITY != 'quiet':
        print(f"Volume prétraité: shape={processed_volume.shape}")
    if VERBOSITY == 'debug':
        print(f"[debug] Preprocessing time: {preprocessing_time:.3f}s, processed shape={processed_volume.shape}, dtype={processed_volume.dtype}")

    # Faire la prédiction
    t0 = time.time()
    prediction = predict_volume(model, processed_volume.unsqueeze(0))  # Ajouter batch dim
    inference_time = time.time() - t0
    if VERBOSITY != 'quiet':
        print(f"Prédiction faite: shape={prediction.shape}")
    if VERBOSITY == 'debug':
        print(f"[debug] Inference time: {inference_time:.3f}s, prediction shape={prediction.shape}, unique={np.unique(prediction.numpy())}")

    # Sauvegarder la prédiction
    output_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(args.input_dir)}.pt")
    torch.save(prediction, output_path)
    if VERBOSITY != 'quiet':
        print(f"Prédiction sauvegardée: {output_path}")
    if VERBOSITY == 'debug':
        try:
            size = os.path.getsize(output_path)
            print(f"[debug] Saved prediction file size: {size} bytes")
        except Exception:
            pass

    # Statistiques
    unique, counts = np.unique(prediction.numpy(), return_counts=True)
    if VERBOSITY != 'quiet':
        print("\nStatistiques de prédiction:")
        class_names = ['Background', 'Prostate', 'Bandelettes']
        for i, (cls, count) in enumerate(zip(unique, counts)):
            if i < len(class_names):
                percentage = count / prediction.numel() * 100
                print(f"{class_names[i]}: {percentage:.1f}%")

if __name__ == "__main__":
    main()