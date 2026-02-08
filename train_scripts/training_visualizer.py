"""
Module de visualisation des métriques d'entraînement.

Génère des courbes de loss et Dice score pendant l'entraînement et la validation,
sauvegardées sous forme d'images et de fichiers CSV pour analyse ultérieure.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveurs sans display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class TrainingVisualizer:
    """Visualise les métriques d'entraînement (loss, Dice) au cours des époques.
    
    Sauvegarde automatiquement :
    - Courbes de loss (train + validation)
    - Courbes de Dice score (validation)
    - Courbe combinée loss + Dice
    - Historique complet en CSV et JSON
    - Courbe du learning rate
    """

    def __init__(self, save_dir: str, experiment_name: str = "training") -> None:
        """Initialise le visualiseur.
        
        Args:
            save_dir: Répertoire de sauvegarde des graphiques
            experiment_name: Nom de l'expérience (utilisé dans les titres)
        """
        self.save_dir = Path(save_dir) / "plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Historique des métriques
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "learning_rate": [],
        }

        # Style matplotlib
        plt.style.use('default')
        self._colors = {
            "train_loss": "#2196F3",     # Bleu
            "val_loss": "#F44336",       # Rouge
            "val_dice": "#4CAF50",       # Vert
            "learning_rate": "#FF9800",  # Orange
            "best_marker": "#FFD700",    # Or
        }

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        learning_rate: float,
    ) -> None:
        """Enregistre les métriques d'une époque et met à jour les graphiques.
        
        Args:
            epoch: Numéro de l'époque (0-indexed)
            train_loss: Loss d'entraînement moyenne
            val_loss: Loss de validation moyenne
            val_dice: Score Dice de validation moyen (en %)
            learning_rate: Taux d'apprentissage courant
        """
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_dice"].append(val_dice)
        self.history["learning_rate"].append(learning_rate)

        # Générer les graphiques à intervalles réguliers ou à la fin
        self._plot_all()
        self._save_history()

    def _plot_all(self) -> None:
        """Génère tous les graphiques."""
        self._plot_loss()
        self._plot_dice()
        self._plot_combined()
        self._plot_learning_rate()

    def _plot_loss(self) -> None:
        """Trace les courbes de loss (train et validation)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history["epoch"]
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]

        ax.plot(epochs, train_loss, 
                color=self._colors["train_loss"], linewidth=2,
                label="Train Loss", marker='o', markersize=3, alpha=0.9)
        ax.plot(epochs, val_loss, 
                color=self._colors["val_loss"], linewidth=2,
                label="Val Loss", marker='s', markersize=3, alpha=0.9)

        # Marquer le meilleur val loss
        if val_loss:
            best_idx = int(np.argmin(val_loss))
            ax.scatter(epochs[best_idx], val_loss[best_idx], 
                      color=self._colors["best_marker"], s=150, zorder=5,
                      marker='*', edgecolors='black', linewidth=0.5,
                      label=f"Meilleur Val Loss: {val_loss[best_idx]:.5f} (ep. {epochs[best_idx]})")

        ax.set_xlabel("Époque", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Courbes de Loss", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Ajout d'une zone grisée montrant l'écart train/val
        ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')

        plt.tight_layout()
        plt.savefig(self.save_dir / "loss_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_dice(self) -> None:
        """Trace la courbe de Dice score (validation)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history["epoch"]
        val_dice = self.history["val_dice"]

        ax.plot(epochs, val_dice, 
                color=self._colors["val_dice"], linewidth=2.5,
                label="Val Dice Score", marker='D', markersize=4, alpha=0.9)

        # Marquer le meilleur Dice
        if val_dice:
            best_idx = int(np.argmax(val_dice))
            ax.scatter(epochs[best_idx], val_dice[best_idx],
                      color=self._colors["best_marker"], s=150, zorder=5,
                      marker='*', edgecolors='black', linewidth=0.5,
                      label=f"Meilleur Dice: {val_dice[best_idx]:.2f}% (ep. {epochs[best_idx]})")

        ax.set_xlabel("Époque", fontsize=12)
        ax.set_ylabel("Dice Score (%)", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Dice Score (Validation)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Zone de remplissage sous la courbe
        ax.fill_between(epochs, val_dice, alpha=0.15, color=self._colors["val_dice"])

        plt.tight_layout()
        plt.savefig(self.save_dir / "dice_curve.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_combined(self) -> None:
        """Trace un graphique combiné loss + Dice avec double axe Y."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        epochs = self.history["epoch"]
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]
        val_dice = self.history["val_dice"]

        # Axe gauche : Loss
        ln1 = ax1.plot(epochs, train_loss, 
                       color=self._colors["train_loss"], linewidth=2,
                       label="Train Loss", alpha=0.8)
        ln2 = ax1.plot(epochs, val_loss, 
                       color=self._colors["val_loss"], linewidth=2,
                       label="Val Loss", alpha=0.8)
        ax1.set_xlabel("Époque", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12, color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Axe droit : Dice
        ax2 = ax1.twinx()
        ln3 = ax2.plot(epochs, val_dice, 
                       color=self._colors["val_dice"], linewidth=2.5,
                       label="Val Dice (%)", linestyle='--', alpha=0.9)
        ax2.set_ylabel("Dice Score (%)", fontsize=12, color=self._colors["val_dice"])
        ax2.tick_params(axis='y', labelcolor=self._colors["val_dice"])
        ax2.set_ylim(bottom=0)

        # Légende combinée
        lines = ln1 + ln2 + ln3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=10)

        ax1.set_title(f"{self.experiment_name} — Loss & Dice Score", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.save_dir / "combined_metrics.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_learning_rate(self) -> None:
        """Trace la courbe du learning rate."""
        fig, ax = plt.subplots(figsize=(10, 4))

        epochs = self.history["epoch"]
        lr = self.history["learning_rate"]

        ax.plot(epochs, lr, 
                color=self._colors["learning_rate"], linewidth=2,
                marker='.', markersize=3)

        ax.set_xlabel("Époque", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Planning du Learning Rate", fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(self.save_dir / "learning_rate.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_history(self) -> None:
        """Sauvegarde l'historique en CSV et JSON."""
        # CSV
        csv_path = self.save_dir / "training_history.csv"
        with open(csv_path, 'w') as f:
            headers = list(self.history.keys())
            f.write(",".join(headers) + "\n")
            for i in range(len(self.history["epoch"])):
                row = [str(self.history[k][i]) for k in headers]
                f.write(",".join(row) + "\n")

        # JSON
        json_path = self.save_dir / "training_history.json"
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def finalize(self) -> str:
        """Génère les graphiques finaux et retourne un résumé.
        
        Returns:
            Résumé textuel des résultats d'entraînement
        """
        self._plot_all()
        self._save_history()

        # Résumé
        best_val_loss = min(self.history["val_loss"]) if self.history["val_loss"] else float('inf')
        best_dice = max(self.history["val_dice"]) if self.history["val_dice"] else 0.0
        best_dice_epoch = int(np.argmax(self.history["val_dice"])) + 1 if self.history["val_dice"] else 0
        best_loss_epoch = int(np.argmin(self.history["val_loss"])) + 1 if self.history["val_loss"] else 0

        summary = (
            f"\n{'='*60}\n"
            f"  RÉSUMÉ DE L'ENTRAÎNEMENT — {self.experiment_name}\n"
            f"{'='*60}\n"
            f"  Époques totales     : {len(self.history['epoch'])}\n"
            f"  Meilleur Val Loss   : {best_val_loss:.6f} (époque {best_loss_epoch})\n"
            f"  Meilleur Val Dice   : {best_dice:.2f}% (époque {best_dice_epoch})\n"
            f"  Dernière Train Loss : {self.history['train_loss'][-1]:.6f}\n"
            f"  Dernière Val Loss   : {self.history['val_loss'][-1]:.6f}\n"
            f"  Dernier Val Dice    : {self.history['val_dice'][-1]:.2f}%\n"
            f"  Graphiques sauvés   : {self.save_dir}\n"
            f"{'='*60}\n"
        )

        return summary
