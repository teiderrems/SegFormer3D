#!/usr/bin/env python3
"""
Script d'inférence simple pour SegFormer3D
"""

import os
try:
    import torch
except Exception:
    torch = None
import numpy as np
import argparse
import yaml
from pathlib import Path
import time

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

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


def load_metadata(patient_dir):
    """Charge les métadonnées du prétraitement (shape originale, affine NIfTI).
    
    Args:
        patient_dir: Répertoire du patient prétraité
    
    Returns:
        dict avec 'original_shape', 'original_affine', etc. ou None si non trouvé
    """
    patient_name = os.path.basename(patient_dir)
    metadata_path = os.path.join(patient_dir, f"{patient_name}_metadata.pt")
    if os.path.exists(metadata_path):
        return torch.load(metadata_path, map_location='cpu', weights_only=False)
    return None


def save_prediction_as_nifti(prediction, metadata, nifti_path):
    """Sauvegarde `prediction` au format NIfTI en utilisant `metadata['original_affine']`.

    - Accepte un `prediction` en `torch.Tensor` ou `numpy.ndarray`.
    - Retourne True si la sauvegarde a eu lieu, False sinon (manque nibabel ou metadata).
    """
    global inference_logger
    if not HAS_NIBABEL:
        if inference_logger:
            inference_logger.warning("nibabel non installé — impossible de sauvegarder en .nii.gz")
        return False

    if metadata is None or 'original_affine' not in metadata:
        if inference_logger:
            inference_logger.warning('Métadonnées originales absentes — impossible de sauvegarder NIfTI à la taille originale')
        return False

    # Convertir en numpy si nécessaire
    try:
        arr = prediction.numpy() if hasattr(prediction, 'numpy') else np.asarray(prediction)
    except Exception:
        try:
            arr = np.asarray(prediction)
        except Exception:
            if inference_logger:
                inference_logger.error('Impossible de convertir la prédiction en tableau numpy pour l’écriture NIfTI')
            return False

    # S'assurer d'un type entier pour les labels
    try:
        arr = arr.astype(np.uint8)
    except Exception:
        arr = arr.astype(np.int32)

    affine = metadata.get('original_affine')
    try:
        nifti_img = nib.Nifti1Image(arr, affine)
        nib.save(nifti_img, nifti_path)
        if inference_logger:
            inference_logger.info(f"Prédiction NIfTI sauvegardée: {nifti_path}")
        return True
    except Exception as e:
        if inference_logger:
            inference_logger.error(f"Échec sauvegarde NIfTI: {e}")
        return False

def resize_prediction_to_original(prediction, original_shape):
    """Redimensionne la prédiction à la taille originale du volume.
    
    Utilise l'interpolation nearest-neighbor pour préserver les labels entiers.
    
    Args:
        prediction: Tensor (D, H, W) avec les labels prédits
        original_shape: Tuple (D, H, W) de la taille originale
    
    Returns:
        Tensor (D, H, W) redimensionné à la taille originale
    """
    current_shape = tuple(prediction.shape)
    target_shape = tuple(original_shape[:3])  # (D, H, W)
    
    if current_shape == target_shape:
        return prediction
    
    # Ajouter les dimensions batch et channel pour interpolate: (1, 1, D, H, W)
    pred_5d = prediction.float().unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        pred_5d,
        size=target_shape,
        mode='nearest'
    )
    return resized.squeeze(0).squeeze(0).long()

def resolve_inference_params(args, config):
    """Resolve inference parameters with YAML priority (YAML > CLI), unless
    `--force-cli` was provided and the corresponding CLI option was explicitly
    passed (in that case CLI wins).

    Returns a dict with keys: verbosity, device, batch_size, save_predictions,
    save_probabilities, save_nifti, threshold.
    """
    import sys
    inf_cfg = (config or {}).get('inference_parameters', {}) if isinstance(config, dict) else {}

    def cli_provided(name):
        for a in sys.argv[1:]:
            if a == name or a.startswith(name + "="):
                return True
        return False

    force = getattr(args, 'force_cli', False)

    # verbosity
    if force and cli_provided('--verbosity'):
        verbosity = args.verbosity
    else:
        verbosity = inf_cfg.get('verbosity', args.verbosity)

    # device and batch_size
    if force and cli_provided('--device'):
        device = args.device
    else:
        device = inf_cfg.get('device', args.device)

    if force and cli_provided('--batch_size'):
        batch_size = args.batch_size
    else:
        batch_size = inf_cfg.get('batch_size', args.batch_size)

    # save flags: prefer YAML unless force+cli provided
    if force and cli_provided('--save_predictions'):
        # explicit CLI flag => True
        save_predictions = True
    elif 'save_predictions' in inf_cfg:
        save_predictions = bool(inf_cfg['save_predictions'])
    else:
        save_predictions = True if getattr(args, 'save_predictions', None) or True else True

    if force and cli_provided('--save_probabilities'):
        save_probabilities = True
    elif 'save_probabilities' in inf_cfg:
        save_probabilities = bool(inf_cfg['save_probabilities'])
    else:
        save_probabilities = bool(getattr(args, 'save_probabilities', False))

    # save NIfTI (default: True)
    if force and cli_provided('--save_nifti'):
        save_nifti = True
    elif 'save_nifti' in inf_cfg:
        save_nifti = bool(inf_cfg['save_nifti'])
    else:
        # default to True to provide NIfTI outputs alongside .pt
        save_nifti = True if getattr(args, 'save_nifti', None) or True else True

    # threshold
    if force and cli_provided('--threshold'):
        threshold = args.threshold
    else:
        threshold = inf_cfg.get('threshold', args.threshold)

    return {
        'verbosity': verbosity,
        'device': device,
        'batch_size': batch_size,
        'save_predictions': save_predictions,
        'save_probabilities': save_probabilities,
        'save_nifti': save_nifti,
        'threshold': threshold,
    }


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
    parser.add_argument("--save_predictions", action="store_true", help="Sauvegarder les prédictions", default=None)
    parser.add_argument("--save_probabilities", action="store_true", help="Sauvegarder les probabilités", default=None)
    parser.add_argument("--save_nifti", action="store_true", help="Sauvegarder la prédiction en NIfTI (.nii.gz)", default=None)
    parser.add_argument('--force-cli', action='store_true', help='Forcer les arguments CLI à remplacer les valeurs du YAML (par défaut: YAML > CLI)')

    args = parser.parse_args()

    # Charger la configuration
    config = load_config(args.config)

    # Résoudre les paramètres d'inférence (utilise la fonction top-level)
    resolved = resolve_inference_params(args, config)

    # Configurer le logger d'inférence (utilise la valeur effective après résolution YAML/CLI)
    global inference_logger
    level_map = {'quiet': 'WARNING', 'normal': 'INFO', 'debug': 'DEBUG'}
    inference_logger = get_logger(
        "inference",
        level=level_map.get(resolved['verbosity'], 'INFO'),
    )

    inference_logger.info(f"Configuration chargée: {config['model']['name']}")
    inference_logger.debug(f"Paramètres d'inférence effectifs: device={resolved['device']}, batch_size={resolved['batch_size']}, save_predictions={resolved['save_predictions']}, save_probabilities={resolved['save_probabilities']}, threshold={resolved['threshold']}")

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger le modèle
    model = load_model(args.checkpoint, config)

    # Charger les données du patient
    modalities, labels = load_patient_data(args.input_dir)
    inference_logger.info(f"Modalités chargées: shape={modalities.shape}")
    if labels is not None:
        inference_logger.info(f"Labels chargés: shape={labels.shape}")
    inference_logger.debug(f"modalities dtype={modalities.dtype}, min={modalities.min():.4f}, max={modalities.max():.4f}")
    if labels is not None:
        inference_logger.debug(f"Labels unique: {np.unique(labels)}")

    t0 = time.time()
    processed_volume = preprocess_volume(modalities, target_size=tuple(config['model']['input_size']))
    preprocessing_time = time.time() - t0
    inference_logger.info(f"Volume prétraité: shape={processed_volume.shape}")
    inference_logger.debug(f"Preprocessing: {preprocessing_time:.3f}s")

    # Faire la prédiction (utilise le device effectif)
    t0 = time.time()
    prediction = predict_volume(model, processed_volume.unsqueeze(0), device=resolved['device'])  # Ajouter batch dim
    inference_time = time.time() - t0
    inference_logger.info(f"Prédiction faite: shape={prediction.shape} ({inference_time:.2f}s)")
    inference_logger.debug(f"Prediction unique values: {np.unique(prediction.numpy())}")

    # Charger les métadonnées pour restaurer la taille originale
    metadata = load_metadata(args.input_dir)
    if metadata is not None and 'original_shape' in metadata:
        original_shape = metadata['original_shape']
        inference_logger.info(f"Taille originale trouvée: {original_shape}")
        prediction_original = resize_prediction_to_original(prediction, original_shape)
        inference_logger.info(f"Prédiction redimensionnée: {tuple(prediction.shape)} -> {tuple(prediction_original.shape)}")
    else:
        inference_logger.warning("Métadonnées non trouvées, la prédiction sera sauvegardée à la taille du modèle")
        prediction_original = prediction
        original_shape = None

    # Sauvegarder la prédiction (taille originale) — respect de la config YAML/CLI
    output_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(args.input_dir)}.pt")
    if resolved.get('save_predictions', True):
        torch.save(prediction_original, output_path)
        inference_logger.info(f"Prédiction sauvegardée: {output_path}")
        try:
            size = os.path.getsize(output_path)
            inference_logger.debug(f'Taille du fichier: {size / 1024:.1f} KB')
        except Exception:
            pass
    else:
        inference_logger.info("save_predictions=False (configured) — skipping saving prediction file")

    # save_probabilities is currently not implemented (placeholder)
    if resolved.get('save_probabilities', False):
        inference_logger.info('save_probabilities requested but not implemented in this script')

    # Sauvegarder la prédiction au format NIfTI (.nii.gz) **après** le retour à la
    # taille originale (nécessite `metadata['original_affine']`). Contrôlé par
    # `resolved['save_nifti']` et nécessite nibabel.
    nifti_requested = resolved.get('save_nifti', True)
    if nifti_requested:
        nifti_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(args.input_dir)}.nii.gz")
        saved = save_prediction_as_nifti(prediction_original, metadata, nifti_path)
        if not saved:
            inference_logger.debug('NIfTI non sauvegardé (métadonnées manquantes ou nibabel absent)')
    # Statistiques
    unique, counts = np.unique(prediction_original.numpy(), return_counts=True)
    inference_logger.info("Statistiques de prédiction:")
    class_names = ['Background', 'Prostate', 'Bandelettes']
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(class_names):
            percentage = count / prediction_original.numel() * 100
            inference_logger.info(f"  {class_names[i]}: {percentage:.1f}%")

if __name__ == "__main__":
    main()