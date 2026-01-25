#!/usr/bin/env python3
"""
Script de visualisation des résultats d'inférence SegFormer3D
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

def load_config(config_path):
    """Charge la configuration YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prediction_and_data(prediction_path, patient_dir):
    """Charge la prédiction et les données originales"""
    # Charger la prédiction
    prediction = torch.load(prediction_path, map_location='cpu').numpy()

    # Charger les données originales
    patient_name = os.path.basename(patient_dir)
    modalities_path = os.path.join(patient_dir, f"{patient_name}_modalities.pt")
    labels_path = os.path.join(patient_dir, f"{patient_name}_label.pt")

    modalities = torch.load(modalities_path, map_location='cpu').numpy()  # (C, D, H, W)
    labels = torch.load(labels_path, map_location='cpu').numpy()  # (1, D, H, W) ou (D, H, W)

    if labels.ndim == 4 and labels.shape[0] == 1:
        labels = labels.squeeze(0)

    return prediction, modalities, labels

def create_comparison_visualization(prediction, modalities, labels, output_dir, patient_name):
    """Crée des visualisations comparatives"""
    os.makedirs(output_dir, exist_ok=True)

    # Prendre la première modalité (T2)
    image = modalities[0]  # (D, H, W)

    # Normaliser l'image pour la visualisation
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_norm = np.clip(image_norm, 0, 1)

    # Coupes centrales
    d, h, w = image.shape
    slice_d = d // 2  # Coupe sagittale centrale
    slice_h = h // 2  # Coupe coronale centrale
    slice_w = w // 2  # Coupe axiale centrale

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Visualisation des résultats - {patient_name}', fontsize=16)

    # Titres des colonnes
    col_titles = ['Image T2', 'Vérité terrain', 'Prédiction', 'Overlay']

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    # Coupe axiale (slice_w)
    axes[0, 0].imshow(image_norm[slice_w], cmap='gray')
    axes[0, 0].set_ylabel('Axial', fontsize=12)

    axes[0, 1].imshow(labels[slice_w], cmap='tab10', vmin=0, vmax=1)
    axes[0, 2].imshow(prediction[slice_w], cmap='tab10', vmin=0, vmax=1)

    # Overlay pour axial
    overlay_ax = axes[0, 3].imshow(image_norm[slice_w], cmap='gray')
    pred_mask = prediction[slice_w] > 0  # Masque des prédictions non-fond
    axes[0, 3].imshow(pred_mask, cmap='Reds', alpha=0.3)

    # Coupe coronale (slice_h)
    axes[1, 0].imshow(image_norm[:, slice_h], cmap='gray')
    axes[1, 0].set_ylabel('Coronal', fontsize=12)

    axes[1, 1].imshow(labels[:, slice_h], cmap='tab10', vmin=0, vmax=1)
    axes[1, 2].imshow(prediction[:, slice_h], cmap='tab10', vmin=0, vmax=1)

    # Overlay pour coronal
    overlay_cor = axes[1, 3].imshow(image_norm[:, slice_h], cmap='gray')
    pred_mask = prediction[:, slice_h] > 0
    axes[1, 3].imshow(pred_mask, cmap='Reds', alpha=0.3)

    # Coupe sagittale (slice_d)
    axes[2, 0].imshow(image_norm[:, :, slice_d], cmap='gray')
    axes[2, 0].set_ylabel('Sagittal', fontsize=12)

    axes[2, 1].imshow(labels[:, :, slice_d], cmap='tab10', vmin=0, vmax=1)
    axes[2, 2].imshow(prediction[:, :, slice_d], cmap='tab10', vmin=0, vmax=1)

    # Overlay pour sagittal
    overlay_sag = axes[2, 3].imshow(image_norm[:, :, slice_d], cmap='gray')
    pred_mask = prediction[:, :, slice_d] > 0
    axes[2, 3].imshow(pred_mask, cmap='Reds', alpha=0.3)

    # Ajuster l'espacement
    plt.tight_layout()

    # Sauvegarder
    output_path = os.path.join(output_dir, f'{patient_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualisation sauvegardée: {output_path}")

def create_detailed_slices(prediction, modalities, labels, output_dir, patient_name):
    """Crée des visualisations détaillées de différentes coupes"""
    os.makedirs(output_dir, exist_ok=True)

    image = modalities[0]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_norm = np.clip(image_norm, 0, 1)

    d, h, w = image.shape

    # Créer plusieurs coupes axiales
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Coupes axiales détaillées - {patient_name}', fontsize=14)

    slice_positions = [w//6, w//3, w//2, 2*w//3, 5*w//6]

    for i, slice_pos in enumerate(slice_positions):
        # Image
        axes[0, i].imshow(image_norm[slice_pos], cmap='gray')
        axes[0, i].set_title(f'Image Z={slice_pos}')
        axes[0, i].axis('off')

        # Prédiction
        axes[1, i].imshow(prediction[slice_pos], cmap='tab10', vmin=0, vmax=1)
        axes[1, i].set_title(f'Prédiction Z={slice_pos}')
        axes[1, i].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{patient_name}_axial_slices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Coupes axiales sauvegardées: {output_path}")

def create_statistics_plot(prediction, labels, output_dir, patient_name):
    """Crée un graphique des statistiques de classification"""
    os.makedirs(output_dir, exist_ok=True)

    # Calculer les distributions
    unique_pred, counts_pred = np.unique(prediction, return_counts=True)
    unique_true, counts_true = np.unique(labels, return_counts=True)

    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Statistiques de classification - {patient_name}', fontsize=14)

    # Distribution prédiction
    class_names = ['Fond', 'Prostate']
    pred_counts = [counts_pred[unique_pred == i][0] if i in unique_pred else 0 for i in range(2)]
    true_counts = [counts_true[unique_true == i][0] if i in unique_true else 0 for i in range(2)]

    x = np.arange(len(class_names))
    width = 0.35

    ax1.bar(x - width/2, [c/prediction.size*100 for c in pred_counts], width, label='Prédiction', alpha=0.8)
    ax1.bar(x + width/2, [c/labels.size*100 for c in true_counts], width, label='Vérité terrain', alpha=0.8)
    ax1.set_ylabel('Pourcentage (%)')
    ax1.set_title('Distribution des classes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Métriques par classe
    dice_scores = []
    for class_idx in range(2):
        pred_class = (prediction == class_idx).astype(np.float32)
        true_class = (labels == class_idx).astype(np.float32)

        intersection = np.sum(pred_class * true_class)
        pred_sum = np.sum(pred_class)
        true_sum = np.sum(true_class)

        dice = (2. * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 0
        dice_scores.append(dice * 100)

    ax2.bar(class_names, dice_scores, color=['green', 'blue'], alpha=0.7)
    ax2.set_ylabel('Dice Score (%)')
    ax2.set_title('Performance par classe')
    ax2.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(dice_scores):
        ax2.text(i, v + 1, '.1f', ha='center', va='bottom')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{patient_name}_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Statistiques sauvegardées: {output_path}")

import json

def compute_segmentation_metrics(prediction, labels):
    """Calcule Dice, IoU, précision, rappel et RMSE entre prediction et labels pour chaque classe."""
    metrics = {}
    # Déterminer les classes à évaluer
    max_class = int(max(prediction.max(), labels.max()))
    num_classes = max_class + 1

    for c in range(num_classes):
        pred_c = (prediction == c).astype(np.int32)
        true_c = (labels == c).astype(np.int32)

        intersection = (pred_c * true_c).sum()
        union = ((pred_c + true_c) > 0).sum()
        pred_sum = pred_c.sum()
        true_sum = true_c.sum()

        dice = (2. * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 0.0
        iou = intersection / union if union > 0 else 0.0
        precision = intersection / pred_sum if pred_sum > 0 else 0.0
        recall = intersection / true_sum if true_sum > 0 else 0.0

        metrics[f'class_{c}'] = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'support': int(true_sum)
        }

    # RMSE global entre masques (utile si prédiction est probabiliste)
    rmse = np.sqrt(((prediction.astype(np.float32) - labels.astype(np.float32)) ** 2).mean())
    metrics['rmse'] = float(rmse)

    return metrics


# ------------------- Métriques avancées -------------------

def _surface_points(mask):
    """Retourne les coordonnées (N,3) des voxels de surface du masque (z,y,x)."""
    if mask.sum() == 0:
        return np.empty((0, 3), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    boundary = mask & (~eroded)
    coords = np.array(np.where(boundary)).T  # (z,y,x)
    return coords


def _to_physical(coords, spacing):
    """Convertit coords (N,3) z,y,x en coordonnées physiques (x,y,z) * spacing (sx,sy,sz).
    spacing attend en ordre (sx, sy, sz)."""
    if spacing is None:
        return coords.astype(np.float32)
    # coords are (z,y,x) -> convert to (x,y,z)
    xyz = coords[:, [2, 1, 0]].astype(np.float32)
    sx, sy, sz = spacing
    xyz[:, 0] *= sx
    xyz[:, 1] *= sy
    xyz[:, 2] *= sz
    return xyz


def hausdorff_distance(mask1, mask2, spacing=None):
    """Hausdorff symétrique entre deux masques binaires.
    Si spacing fourni (sx,sy,sz), distances sont en mm sinon en voxels."""
    pts1 = _surface_points(mask1)
    pts2 = _surface_points(mask2)
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return float('nan')

    p1 = _to_physical(pts1, spacing)
    p2 = _to_physical(pts2, spacing)

    # Use cKDTree for memory-efficient nearest neighbor distances
    tree_p2 = cKDTree(p2)
    tree_p1 = cKDTree(p1)
    d12_min, _ = tree_p2.query(p1)
    d21_min, _ = tree_p1.query(p2)
    hd = max(d12_min.max(), d21_min.max())
    return float(hd)


def hausdorff_95(mask1, mask2, spacing=None):
    """HD95: 95e percentile des distances minimales symétriques."""
    pts1 = _surface_points(mask1)
    pts2 = _surface_points(mask2)
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return float('nan')

    p1 = _to_physical(pts1, spacing)
    p2 = _to_physical(pts2, spacing)

    tree_p2 = cKDTree(p2)
    tree_p1 = cKDTree(p1)
    d12_min, _ = tree_p2.query(p1)
    d21_min, _ = tree_p1.query(p2)
    hd95 = max(np.percentile(d12_min, 95), np.percentile(d21_min, 95))
    return float(hd95)


def average_symmetric_surface_distance(mask1, mask2, spacing=None):
    """Average Symmetric Surface Distance (ASSD) en voxels (ou mm si spacing fourni)."""
    pts1 = _surface_points(mask1)
    pts2 = _surface_points(mask2)
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return float('nan')

    p1 = _to_physical(pts1, spacing)
    p2 = _to_physical(pts2, spacing)

    d12 = cdist(p1, p2)
    d21 = cdist(p2, p1)
    mean12 = d12.min(axis=1).mean()
    mean21 = d21.min(axis=1).mean()
    return float((mean12 + mean21) / 2.0)

def compute_ssim_3d(prediction, labels):
    """Calcule SSIM slice-wise en axial et renvoie la moyenne et la liste."""
    # Assumer forme (D,H,W) avec coupes axiales le long de l'axe 0 ou 2 suivant la convention
    # Ici nos prédictions sont indexées [z,y,x] dans la plupart des fonctions => boucle sur z
    z_dim = prediction.shape[0]
    ssim_list = []
    for z in range(z_dim):
        im_pred = prediction[z].astype(np.float32)
        im_true = labels[z].astype(np.float32)
        try:
            val = ssim(im_true, im_pred, data_range=im_true.max() - im_true.min())
        except Exception:
            # Fallback si toutes valeurs identiques
            val = 1.0 if np.array_equal(im_true, im_pred) else 0.0
        ssim_list.append(float(val))
    return float(np.mean(ssim_list)), ssim_list


def create_error_visualizations(prediction, modalities, labels, output_dir, patient_name, spacing=None):
    """Génère des fichiers d'erreurs : JSON de métriques, graphique par classe, et erreur slice-wise.
    spacing (sx,sy,sz) : optional voxel spacing in mm (x,y,z). If None distances are in voxels."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculer métriques
    metrics = compute_segmentation_metrics(prediction, labels)

    # Métriques avancées (Hausdorff, ASSD, SSIM)
    # Utiliser uniquement la classe d'intérêt (non-fond). Si plusieurs classes, calcule pour chaque classe >0 moyenné.
    advanced = {}
    classes = [int(k.split('_')[1]) for k in metrics.keys() if k.startswith('class_')]
    non_background = [c for c in classes if c != 0]

    # Calculer Hausdorff/ASSD par classe et moyenne
    hausdorffs = {}
    assds = {}
    ssim_means = {}

    for c in non_background:
        mask_pred = (prediction == c).astype(bool)
        mask_true = (labels == c).astype(bool)
        h_val = hausdorff_distance(mask_pred, mask_true, spacing=spacing)
        a_val = average_symmetric_surface_distance(mask_pred, mask_true, spacing=spacing)
        # HD95
        hd95 = hausdorff_95(mask_pred, mask_true, spacing=spacing)
        # distance distribution stats (in same units)
        pts1 = _surface_points(mask_pred)
        pts2 = _surface_points(mask_true)
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            dist_stats = {'mean': None, 'median': None, 'p95': None}
        else:
            p1 = _to_physical(pts1, spacing)
            p2 = _to_physical(pts2, spacing)
            tree_p2 = cKDTree(p2)
            tree_p1 = cKDTree(p1)
            d12_min, _ = tree_p2.query(p1)
            d21_min, _ = tree_p1.query(p2)
            mins = np.concatenate([d12_min, d21_min])
            dist_stats = {'mean': float(np.mean(mins)), 'median': float(np.median(mins)), 'p95': float(np.percentile(mins, 95))}
        ssim_mean, ssim_list = compute_ssim_3d(mask_pred.astype(np.uint8), mask_true.astype(np.uint8))
        ssim_means[f'class_{c}'] = {'ssim_mean': ssim_mean, 'ssim_slices': ssim_list}

        # Store additional metrics
        hausdorffs[f'class_{c}'] = {'hausdorff': h_val, 'hd95': hd95, 'dist_stats': dist_stats}
        assds[f'class_{c}'] = {'assd': a_val}

    # Moyennes (ignorer NaN) — extraire les valeurs numériques
    valid_haus_vals = [v['hausdorff'] for v in hausdorffs.values() if isinstance(v, dict) and v.get('hausdorff') is not None]
    valid_assd_vals = [v['assd'] for v in assds.values() if isinstance(v, dict) and v.get('assd') is not None]

    advanced['hausdorff'] = float(np.mean(valid_haus_vals)) if len(valid_haus_vals) > 0 else None
    advanced['assd'] = float(np.mean(valid_assd_vals)) if len(valid_assd_vals) > 0 else None
    advanced['hausdorff_per_class'] = hausdorffs
    advanced['assd_per_class'] = assds
    advanced['ssim_per_class'] = ssim_means

    metrics['advanced'] = advanced

    # Sauvegarder JSON
    json_path = os.path.join(output_dir, f"{patient_name}_errors.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Enregistrer un résumé lisible
    advanced_txt = os.path.join(output_dir, f"{patient_name}_advanced_errors.txt")
    with open(advanced_txt, 'w') as af:
        af.write('Advanced metrics summary\n')
        af.write(f"Hausdorff (mean across classes): {advanced['hausdorff']}\n")
        af.write(f"ASSD (mean across classes): {advanced['assd']}\n")
        for k, v in hausdorffs.items():
            af.write(f"{k} - Hausdorff: {v}, ASSD: {assds.get(k)}\n")

    print(f"Métriques sauvegardées: {json_path}")
    print(f"Résumé avancé sauvegardé: {advanced_txt}")
    # Graphique des métriques (Dice / IoU par classe)
    class_keys = [k for k in metrics.keys() if k.startswith('class_')]
    class_keys_sorted = sorted(class_keys, key=lambda x: int(x.split('_')[1]))

    dices = [metrics[k]['dice'] * 100 for k in class_keys_sorted]
    ious = [metrics[k]['iou'] * 100 for k in class_keys_sorted]
    supports = [metrics[k]['support'] for k in class_keys_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(class_keys_sorted))
    width = 0.35
    ax.bar(x - width/2, dices, width, label='Dice (%)')
    ax.bar(x + width/2, ious, width, label='IoU (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{int(k.split("_")[1])}' for k in class_keys_sorted])
    ax.set_ylabel('%')
    ax.set_title(f'Métriques de segmentation - {patient_name}')
    ax.legend()
    plt.tight_layout()
    metrics_png = os.path.join(output_dir, f"{patient_name}_errors.png")
    plt.savefig(metrics_png, dpi=300, bbox_inches='tight')
    plt.close()

    # Slice-wise MAE (axial)
    image = modalities[0]
    d, h, w = image.shape
    slice_errors = []
    for z in range(w):
        slice_error = np.mean(np.abs(prediction[z].astype(np.float32) - labels[z].astype(np.float32)))
        slice_errors.append(slice_error)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(slice_errors, marker='o')
    ax.set_xlabel('Slice Z')
    ax.set_ylabel('MAE')
    ax.set_title(f'Erreur slice-wise (MAE) - {patient_name}')
    plt.grid(True)
    plt.tight_layout()
    slice_png = os.path.join(output_dir, f"{patient_name}_slice_errors.png")
    plt.savefig(slice_png, dpi=300, bbox_inches='tight')
    plt.close()

    # Overlay d'erreur sur la coupe centrale
    slice_w = w // 2
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    abs_error = np.abs(prediction[slice_w].astype(np.float32) - labels[slice_w].astype(np.float32))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_norm[slice_w], cmap='gray')
    im = ax.imshow(abs_error, cmap='hot', alpha=0.6)
    ax.set_title(f'Erreur overlay - coupe Z={slice_w}')
    plt.colorbar(im, ax=ax)
    plt.axis('off')
    overlay_png = os.path.join(output_dir, f"{patient_name}_error_overlay.png")
    plt.tight_layout()
    plt.savefig(overlay_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Métriques sauvegardées: {json_path}")
    print(f"Graphiques d'erreurs sauvegardés: {metrics_png}, {slice_png}, {overlay_png}")


def create_volume_visualization(prediction, modalities, labels, output_dir, patient_name, interactive=False):
    """Crée une visualisation volumétrique 3D de la segmentation"""
    os.makedirs(output_dir, exist_ok=True)

    # Sous-échantillonnage pour la performance (optionnel, ajuster selon les besoins)
    step = 2  # Prendre tous les 2 voxels pour réduire la complexité
    pred_downsampled = prediction[::step, ::step, ::step]
    labels_downsampled = labels[::step, ::step, ::step]

    if interactive:
        # Utiliser Plotly pour une visualisation interactive
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Prédiction (Rouge)', 'Vérité terrain (Bleu)')
        )

        # Prédiction
        pred_coords = np.where(pred_downsampled > 0)
        if len(pred_coords[0]) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=pred_coords[2], y=pred_coords[1], z=pred_coords[0],
                    mode='markers',
                    marker=dict(size=2, color='red', opacity=0.6),
                    name='Prédiction'
                ),
                row=1, col=1
            )

        # Vérité terrain
        true_coords = np.where(labels_downsampled > 0)
        if len(true_coords[0]) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=true_coords[2], y=true_coords[1], z=true_coords[0],
                    mode='markers',
                    marker=dict(size=2, color='blue', opacity=0.6),
                    name='Vérité terrain'
                ),
                row=1, col=2
            )

        # Mise à jour des layouts
        fig.update_layout(
            title=f'Visualisation volumétrique interactive - {patient_name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # Sauvegarder en HTML
        output_path = os.path.join(output_dir, f'{patient_name}_volume_3d_interactive.html')
        fig.write_html(output_path)

    else:
        # Version statique avec matplotlib
        # Créer la figure 3D
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Visualisation volumétrique - {patient_name}', fontsize=16)

        # Sous-plot pour la prédiction
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Prédiction (Rouge)', fontsize=14)

        # Créer un masque pour les voxels à afficher (seulement les classes non-fond)
        pred_mask = pred_downsampled > 0

        # Utiliser voxels pour afficher le volume
        if np.any(pred_mask):
            ax1.voxels(pred_mask, facecolors='red', edgecolor='k', alpha=0.3)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Sous-plot pour la vérité terrain
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Vérité terrain (Bleu)', fontsize=14)

        true_mask = labels_downsampled > 0
        if np.any(true_mask):
            ax2.voxels(true_mask, facecolors='blue', edgecolor='k', alpha=0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{patient_name}_volume_3d.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualisation volumétrique sauvegardée: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualisation des résultats d'inférence SegFormer3D")
    parser.add_argument('--config', type=str, default='configs/config_segformer3d.yaml', help='Chemin vers le fichier de configuration')
    parser.add_argument('--prediction', type=str, required=True, help='Chemin vers le fichier de prédiction (.pt)')
    parser.add_argument('--input_dir', type=str, required=True, help='Répertoire du patient avec les données prétraitées')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--volume_vis', action='store_true', help='Générer la visualisation volumétrique 3D')
    parser.add_argument('--interactive', action='store_true', help='Générer une visualisation volumétrique interactive (HTML avec Plotly)')
    parser.add_argument('--compute_errors', action='store_true', help='Calculer et sauvegarder les erreurs de reconstruction / segmentation')
    parser.add_argument('--voxel_spacing', type=str, default=None, help='Voxel spacing en mm comme "sx,sy,sz" (x,y,z)')

    args = parser.parse_args()

    # Charger la configuration
    config = load_config(args.config)
    patient_name = os.path.basename(args.input_dir)

    print(" Génération des visualisations...")
    print(f"Patient: {patient_name}")
    print(f"Modèle: {config['model']['name']}")

    # Charger les données
    prediction, modalities, labels = load_prediction_and_data(args.prediction, args.input_dir)
    print(f"Données chargées - Prédiction: {prediction.shape}, Image: {modalities.shape}, Labels: {labels.shape}")

    # Parser le spacing si fourni
    voxel_spacing = None
    if args.voxel_spacing is not None:
        try:
            parts = [float(p) for p in args.voxel_spacing.split(',')]
            if len(parts) != 3:
                raise ValueError('voxel_spacing must have three comma-separated values: sx,sy,sz')
            voxel_spacing = (parts[0], parts[1], parts[2])
        except Exception as e:
            print(f"Warning: impossible de parser --voxel_spacing: {e}. Ignorer le spacing.")
            voxel_spacing = None

    # Créer les visualisations
    create_comparison_visualization(prediction, modalities, labels, args.output_dir, patient_name)
    create_detailed_slices(prediction, modalities, labels, args.output_dir, patient_name)
    create_statistics_plot(prediction, labels, args.output_dir, patient_name)
    if args.compute_errors:
        create_error_visualizations(prediction, modalities, labels, args.output_dir, patient_name, spacing=voxel_spacing)
    if args.volume_vis or args.interactive:
        create_volume_visualization(prediction, modalities, labels, args.output_dir, patient_name, interactive=args.interactive)

    print("\n Visualisations terminées !")
    print(f" Fichiers sauvegardés dans: {args.output_dir}/")

    # Lister les fichiers générés
    if os.path.exists(args.output_dir):
        files = os.listdir(args.output_dir)
        print(" Fichiers générés:")
        for file in sorted(files):
            print(f"   - {file}")

if __name__ == "__main__":
    main()