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

# Ajouter le répertoire parent au path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.build_architecture import build_architecture

def load_config(config_path):
    """Charge la configuration YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config):
    """Charge le modèle depuis le checkpoint"""
    # Créer le modèle
    model = build_architecture(config)
    print(f"Modèle créé: {config['model']['name']}")

    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
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

    args = parser.parse_args()

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger la configuration
    config = load_config(args.config)
    print(f"Configuration chargée: {config['model']['name']}")

    # Charger le modèle
    model = load_model(args.checkpoint, config)

    # Charger les données du patient
    modalities, labels = load_patient_data(args.input_dir)
    print(f"Modalités chargées: shape={modalities.shape}")
    if labels is not None:
        print(f"Labels chargés: shape={labels.shape}")

    processed_volume = preprocess_volume(modalities, target_size=tuple(config['model']['input_size']))
    print(f"Volume prétraité: shape={processed_volume.shape}")

    # Faire la prédiction
    prediction = predict_volume(model, processed_volume.unsqueeze(0))  # Ajouter batch dim
    print(f"Prédiction faite: shape={prediction.shape}")

    # Sauvegarder la prédiction
    output_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(args.input_dir)}.pt")
    torch.save(prediction, output_path)
    print(f"Prédiction sauvegardée: {output_path}")

    # Statistiques
    unique, counts = np.unique(prediction.numpy(), return_counts=True)
    print("\nStatistiques de prédiction:")
    class_names = ['Background', 'Prostate', 'Bandelettes']
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(class_names):
            percentage = count / prediction.numel() * 100
            print(f"{class_names[i]}: {percentage:.1f}%")

if __name__ == "__main__":
    main()