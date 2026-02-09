"""
Module d'entraînement distribué (DDP) pour la segmentation 3D et l'auto-encodeur.

Ce module fournit deux classes de trainers :
- Segmentation_Trainer : entraînement supervisé pour la segmentation sémantique de
  volumes 3D (prostate / bandelettes). Supporte l'EMA (Exponential Moving Average),
  le warmup du learning rate, l'early stopping, les checkpoints périodiques, et
  le suivi WandB.
- AutoEncoder_Trainer : entraînement non supervisé pour un auto-encodeur 3D,
  utilisant la SSIM 3D comme métrique de qualité de reconstruction.

Les deux trainers reposent sur HuggingFace Accelerate pour le support multi-GPU
(DataParallel / DistributedDataParallel) et l'accumulation de gradient.

Flux d'exécution principal :
    1. main() parse les arguments, charge la config YAML, construit le modèle,
       les dataloaders, l'optimiseur, les schedulers et l'accélérateur.
    2. Le trainer approprié est instancié puis trainer.train() lance la boucle
       d'entraînement / validation.
    3. À chaque époque : train_step → val_step → update_metrics → save_and_print.
    4. En fin d'entraînement : sauvegarde du modèle final + génération des courbes.

Utilisation :
    python trainer_ddp.py --config configs/config_segformer3d.yaml
    python trainer_ddp.py --config configs/config_segformer3d.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import torch
try:
    import evaluate  # Bibliothèque HuggingFace Evaluate (optionnelle)
except Exception:
    evaluate = None
import yaml
import argparse
import warnings
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from termcolor import colored
from torch.utils.data import DataLoader
import monai  # MONAI : utilitaires médicaux (métriques, transforms, etc.)

# Supprimer les warnings de dépréciation MONAI/PyTorch qui polluent la sortie
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

# Ajouter le répertoire parent au PYTHONPATH pour les imports locaux
# (metrics/, architectures/, losses/, dataloaders/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.segmentation_metrics import SlidingWindowInference  # Inférence par fenêtre glissante + Dice
from train_scripts.training_visualizer import TrainingVisualizer  # Génération de courbes loss/dice
from train_scripts.logger import (
    get_logger, log_epoch_summary, log_training_start,
    log_training_end, log_section, KerasProgressBar,  # Logger unifié + barre de progression
)
import kornia  # Bibliothèque de vision par ordinateur (SSIM 3D pour l'auto-encodeur)


#################################################################################################
#                         TRAINER POUR LA SEGMENTATION 3D
#################################################################################################
class Segmentation_Trainer:
    """Trainer pour la segmentation sémantique de volumes médicaux 3D.

    Ce trainer gère la boucle complète d'entraînement / validation pour un réseau
    de segmentation (ex. SegFormer3D). Il intègre :

    - **Accumulation de gradient** via HuggingFace Accelerate, permettant de simuler
      un batch size plus grand que ce que la mémoire GPU autorise.
    - **EMA (Exponential Moving Average)** : maintient une copie lissée des poids du
      modèle pour une meilleure généralisation. Validée périodiquement.
    - **Warmup du learning rate** : augmente progressivement le LR en début
      d'entraînement pour stabiliser la convergence.
    - **Early stopping** : arrête l'entraînement si aucune amélioration n'est
      observée pendant `early_stopping_patience` époques consécutives.
    - **Checkpoints périodiques** avec rotation (conserve les N plus récents).
    - **Sliding Window Inference** pour le calcul du Dice score en validation,
      nécessaire quand les volumes sont trop grands pour tenir en mémoire d'un coup.
    - **Visualisation** des courbes de loss/dice via TrainingVisualizer.

    Flux d'une époque :
        1. ``_train_step()``  → passe forward, calcul de la loss, backward, mise à jour
        2. ``_val_step()``    → évaluation sans gradients, calcul du Dice
        3. ``_update_metrics()``  → mise à jour des meilleurs scores
        4. ``_save_and_print()``  → sauvegarde du meilleur modèle + log
        5. ``_save_periodic_checkpoint()``  → checkpoint toutes les N époques
    """

    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        """Initialise le trainer de segmentation.

        Args:
            config (Dict): Dictionnaire complet de configuration YAML contenant
                les sections 'training_parameters', 'ema', 'warmup_scheduler',
                'sliding_window_inference', 'model', etc.
            model (torch.nn.Module): Réseau de segmentation (ex. SegFormer3D).
                Doit produire un tenseur (B, C, D, H, W) en sortie.
            optimizer (torch.optim.Optimizer): Optimiseur (Adam, AdamW, SGD…).
            criterion (torch.nn.Module): Fonction de loss (DiceLoss, DiceCELoss…).
            train_dataloader (DataLoader): DataLoader d'entraînement. Chaque batch
                est un dict {'image': (B,C,D,H,W), 'label': (B,1,D,H,W)}.
            val_dataloader (DataLoader): DataLoader de validation (même format).
            warmup_scheduler (LRScheduler): Scheduler de warmup linéaire (ou None).
            training_scheduler (LRScheduler): Scheduler principal (CosineAnnealing…).
            accelerator: Objet HuggingFace Accelerate pour le support multi-GPU,
                l'accumulation de gradient et la précision mixte.
        """
        # ── Configuration générale ──
        # Stocke la config complète et extrait les variables utiles
        self.config = config
        self._configure_trainer()  # Initialise num_epochs, checkpoint_save_dir, etc.

        # ── Composants du modèle ──
        self.model = model      # Réseau de segmentation (déjà préparé par Accelerate)
        target_size = config["dataset_parameters"]["train_dataset_args"]["target_size"]
        self.optimizer = optimizer    # Optimiseur (Adam, AdamW, etc.)
        self.criterion = criterion   # Fonction de loss (DiceLoss, DiceCELoss, etc.)
        self.train_dataloader = train_dataloader  # DataLoader d'entraînement
        self.val_dataloader = val_dataloader      # DataLoader de validation

        # ── HuggingFace Accelerate ──
        # Gère automatiquement : distribution multi-GPU, précision mixte (fp16/bf16),
        # accumulation de gradient, et la synchronisation des gradients entre les GPU.
        self.accelerator = accelerator

        # ── Weights & Biases (optionnel) ──
        # Permet le suivi des métriques en ligne. Peut être désactivé sans impact.
        try:
            self.wandb_tracker = accelerator.get_tracker("wandb")
        except:
            self.wandb_tracker = None

        # ── Métriques d'entraînement ──
        # Suivies à chaque époque et comparées aux meilleurs scores historiques.
        self.start_epoch = 0            # Époque de départ (>0 si reprise d'entraînement)
        self.current_epoch = 0          # Numéro de l'époque courante
        self.epoch_train_loss = 0.0     # Loss moyenne d'entraînement de l'époque courante
        self.best_train_loss = 100.0    # Meilleure loss d'entraînement observée
        self.epoch_val_loss = 0.0       # Loss moyenne de validation de l'époque courante
        self.best_val_loss = 100.0      # Meilleure loss de validation observée
        self.epoch_val_dice = 0.0       # Dice score moyen de l'époque courante
        self.best_val_dice = 0.0        # Meilleur Dice score observé (critère de sauvegarde)
        
        # ── Early stopping ──
        # Arrête l'entraînement si le modèle ne s'améliore plus pendant
        # 'early_stopping_patience' époques consécutives. 0 = désactivé.
        self.early_stopping_patience = self.config.get("training_parameters", {}).get("early_stopping_patience", 0)
        self.early_stopping_counter = 0  # Compteur d'époques sans amélioration
        self.early_stop = False          # Flag pour interrompre la boucle

        # ── Inférence par fenêtre glissante (Sliding Window Inference) ──
        # Utilisée en validation pour calculer le Dice sur des volumes complets.
        # Le volume est découpé en patches (roi), inféré par morceaux (sw_batch_size),
        # puis les prédictions sont recomposées. Cela permet d'évaluer des volumes
        # plus grands que la mémoire GPU ne le permettrait en une seule passe.
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],       # Taille du patch (ex. [96,96,96])
            config["sliding_window_inference"]["sw_batch_size"],  # Nb de patches par batch
            num_classes=config.get("model", {}).get("num_classes", 2),  # Nb de classes
        )

        # ── Schedulers de learning rate ──
        # Deux phases possibles :
        #   1. Warmup (LinearLR) : augmentation progressive du LR pour stabiliser le début
        #   2. Training (CosineAnnealingLR) : décroissance cosinus du LR
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None  # Sera assigné dans _update_scheduler()

        # ── Modèle EMA (Exponential Moving Average) ──
        # L'EMA maintient une copie lissée des poids : θ_ema = α * θ_ema + (1-α) * θ_model
        # Cela produit un modèle plus stable et souvent plus performant que le modèle courant.
        # Validé périodiquement tous les `val_ema_every` époques.
        self.val_ema_model = None  # Copie temporaire pour la validation EMA
        self.ema_model = self._create_ema_model() if self.ema_enabled else None
        self.epoch_val_ema_dice = 0.0   # Dice du modèle EMA à l'époque courante
        self.best_val_ema_dice = 0.0    # Meilleur Dice EMA observé

        # ── Visualiseur de métriques d'entraînement ──
        # Génère automatiquement les courbes de loss et de Dice dans le répertoire
        # de checkpoints (training_history.png, training_history.csv).
        self.visualizer = TrainingVisualizer(
            save_dir=self.checkpoint_save_dir,
            experiment_name=config.get("model", {}).get("name", "SegFormer3D"),
        )

        # ── Logger unifié ──
        # Écrit à la fois dans la console et dans un fichier training.log.
        # Niveau DEBUG si verbosity > 1 dans la config, sinon INFO.
        log_file = os.path.join(self.checkpoint_save_dir, "training.log")
        self.logger = get_logger(
            "trainer",
            level="DEBUG" if config.get("advanced", {}).get("verbosity", 1) > 1 else "INFO",
            log_file=log_file,
        )

    def _configure_trainer(self) -> None:
        """Extrait les hyperparamètres clés de la config YAML en attributs d'instance.

        Variables extraites :
        - num_epochs : nombre total d'époques d'entraînement.
        - print_every : fréquence d'affichage des métriques (en époques).
        - ema_enabled : active/désactive le modèle EMA.
        - val_ema_every : fréquence de validation du modèle EMA (en époques).
        - warmup_enabled : active/désactive le warmup du learning rate.
        - warmup_epochs : durée du warmup en époques.
        - cutoff_epoch : époque après laquelle les checkpoints "post-cutoff" sont
          sauvegardés séparément (utile pour les expériences longues).
        - calculate_metrics : si True, calcule le Dice en validation (plus lent).
        - checkpoint_save_dir : répertoire pour les checkpoints.
        - checkpoint_save_freq : sauvegarde un checkpoint toutes les N époques (0 = désactivé).
        - checkpoint_keep_last : nombre max de checkpoints périodiques à conserver.
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.val_ema_every = self.config["ema"]["val_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]

        # ── Configuration des checkpoints périodiques ──
        # save_freq=0 désactive les checkpoints périodiques.
        # keep_last=5 signifie qu'on garde les 5 checkpoints les plus récents
        # et on supprime les plus anciens (rotation).
        ckpt_cfg = self.config.get("checkpoint", {})
        self.checkpoint_save_freq = ckpt_cfg.get("save_freq", 0)
        self.checkpoint_keep_last = ckpt_cfg.get("keep_last", 5)
        self._saved_checkpoints: list = []  # File de chemins pour la rotation FIFO

    def _load_checkpoint(self):
        """Charge un checkpoint existant pour reprendre l'entraînement.
        Non implémenté — utiliser le flag --checkpoint de main() à la place.
        """
        raise NotImplementedError

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Reprend l'entraînement exactement à l'époque où il a été interrompu.

        Charge l'état complet depuis un fichier .pth (best_model.pth ou checkpoint
        périodique) : poids du modèle, état de l'optimiseur, état du scheduler,
        métriques (best_val_dice, best_val_loss, etc.), et numéro d'époque.

        Contrairement à --checkpoint qui ne charge que les poids (fine-tuning),
        cette méthode restaure tout pour continuer exactement là où l'entraînement
        s'est arrêté.

        Args:
            checkpoint_path (str): Chemin vers le fichier .pth contenant l'état complet.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

        self.logger.info(f"Reprise de l'entraînement depuis: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # ── Restauration des poids du modèle ──
        if "model_state_dict" in checkpoint:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("Poids du modèle restaurés")
        else:
            self.logger.warning("Pas de model_state_dict dans le checkpoint")

        # ── Restauration de l'optimiseur ──
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info("État de l'optimiseur restauré")

        # ── Restauration du scheduler ──
        if "scheduler_state_dict" in checkpoint and self.training_scheduler is not None:
            self.training_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.logger.info("État du scheduler restauré")

        # ── Restauration de l'époque de départ ──
        self.start_epoch = checkpoint.get("epoch", 0)  # epoch sauvegardée = epoch+1
        self.logger.info(f"Reprise à l'époque {self.start_epoch}/{self.num_epochs}")

        # ── Restauration des meilleures métriques ──
        self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", 100.0)
        self.best_train_loss = checkpoint.get("train_loss", 100.0)
        self.logger.info(
            f"Métriques restaurées — best_dice={self.best_val_dice:.4f}, "
            f"best_val_loss={self.best_val_loss:.6f}"
        )

        # Réinitialiser le compteur d'early stopping (on repart propre)
        self.early_stopping_counter = 0
        self.early_stop = False

    def _create_ema_model(self) -> torch.nn.Module:
        """Crée un modèle EMA (Exponential Moving Average).

        L'EMA maintient une moyenne pondérée exponentiellement des poids du modèle :
            θ_ema ← decay × θ_ema + (1 - decay) × θ_model

        Cela produit un modèle plus lisse qui généralise mieux, au prix d'un léger
        délai d'adaptation. Le decay typique est 0.999 ou 0.9999.

        Utilise torch.optim.swa_utils.AveragedModel avec la fonction multi_avg_fn
        pour la mise à jour EMA (compatible avec torch >= 2.0).

        Returns:
            torch.nn.Module: Modèle EMA initialisé avec les mêmes poids que le modèle courant.
        """
        self.accelerator.print("Création du modèle EMA")
        ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            device=self.accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            ),
        )
        return ema_model

    def _train_step(self) -> float:
        """Exécute une époque complète d'entraînement.

        Pour chaque batch du DataLoader d'entraînement :
            1. Extraction des données (image) et labels (masque de segmentation)
            2. Remise à zéro des gradients (set_to_none=True pour efficacité mémoire)
            3. Passe forward : prédiction du modèle → tenseur (B, C, D, H, W)
            4. Calcul de la loss (ex. DiceCELoss entre prédiction et label)
            5. Rétropropagation (backward) via Accelerate
            6. Optionnel : clipping des gradients pour éviter l'explosion
            7. Mise à jour des poids (optimizer.step())
            8. Mise à jour du modèle EMA si activé

        L'accumulation de gradient est gérée automatiquement par le context manager
        ``self.accelerator.accumulate(self.model)`` : les gradients ne sont
        synchronisés et l'optimizer n'est mis à jour que toutes les N itérations.

        Returns:
            float: Loss moyenne sur l'ensemble d'entraînement pour cette époque.
        """
        # Initialisation de la loss cumulée pour cette époque
        epoch_avg_loss = 0.0

        # Passage du modèle en mode entraînement (active Dropout, BatchNorm en mode train)
        self.model.train()

        # Barre de progression tqdm affichée uniquement sur le processus principal
        # (évite la duplication en multi-GPU)
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=True,
            disable=not self.accelerator.is_main_process
        )

        # Note : set_epoch() sur le sampler serait nécessaire en DDP pur
        # pour assurer un shuffle différent à chaque époque. Ici Accelerate s'en charge.
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for index, raw_data in pbar:
            # Le context manager accumulate() gère l'accumulation de gradient :
            # les gradients sont accumulés sur N steps avant la synchronisation.
            with self.accelerator.accumulate(self.model):
                # ── 1. Extraction des données ──
                # raw_data est un dict {'image': (B,C,D,H,W), 'label': (B,1,D,H,W)}
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )

                # ── 2. Remise à zéro des gradients ──
                # set_to_none=True est plus performant que zero_grad() classique
                # car il évite une allocation mémoire pour les tenseurs de gradients nuls.
                self.optimizer.zero_grad(set_to_none=True)

                # ── 3. Passe forward ──
                # Le modèle produit un tenseur de logits (B, num_classes, D, H, W)
                predicted = self.model.forward(data)

                # ── 4. Calcul de la loss ──
                # La loss combine typiquement Dice + Cross-Entropy pour
                # gérer le déséquilibre de classes (fond >> prostate)
                loss = self.criterion(predicted, labels)

                # ── 5. Rétropropagation ──
                # accelerator.backward() gère le scaling en précision mixte (fp16)
                # et l'accumulation de gradient si configurée.
                self.accelerator.backward(loss)

                # ── 6. Gradient clipping (optionnel) ──
                # Limite la norme des gradients pour éviter les explosions de gradient,
                # particulièrement utile pour les Transformers avec attention.
                if self.config.get("clip_gradients", {}).get("enabled", False):
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["clip_gradients"]["clip_gradients_value"]
                        )

                # ── 7. Mise à jour des poids ──
                self.optimizer.step()

                # ── 8. Mise à jour EMA ──
                # Mise à jour des poids EMA uniquement sur le processus principal
                # pour éviter les doublons en multi-GPU.
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model)

                # Accumulation de la loss (detach pour libérer le graphe de calcul)
                epoch_avg_loss += loss.detach().item()

                # Mise à jour de la barre de progression avec la loss et le LR courants
                avg_loss = epoch_avg_loss / (index + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })

        # Loss moyenne de l'époque (moyenne sur tous les batches)
        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """Exécute une époque complète de validation.

        Évalue le modèle (ou sa version EMA) sur l'ensemble de validation, sans
        calculer ni stocker de gradients (torch.no_grad()). Pour chaque batch :
            1. Passe forward → prédiction
            2. Calcul de la loss
            3. Calcul optionnel du Dice score via Sliding Window Inference

        Le Dice est calculé par _calc_dice_metric() qui utilise l'inférence par
        fenêtre glissante pour gérer les volumes trop grands pour la mémoire GPU.

        Args:
            use_ema (bool): Si True, utilise le modèle EMA au lieu du modèle courant
                pour la prédiction. Permet de comparer les performances EMA vs standard.

        Returns:
            float: Loss moyenne de validation pour cette époque.
        """
        # Initialisation des compteurs
        epoch_avg_loss = 0.0  # Loss cumulée
        total_dice = 0.0      # Dice cumulé (pour calcul de la moyenne)

        # Passage en mode évaluation (désactive Dropout, BatchNorm en mode inférence)
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # Contexte sans gradients pour économiser la mémoire GPU et accélérer l'inférence
        with torch.no_grad():
            # Barre de progression style Keras pour la validation
            pbar = KerasProgressBar(
                total=len(self.val_dataloader),
                epoch=self.current_epoch,
                num_epochs=self.num_epochs,
                prefix="Val",
                enabled=self.accelerator.is_main_process,
            )
            for index, (raw_data) in enumerate(self.val_dataloader):
                # Extraction des données et labels
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # Passe forward avec le modèle approprié (standard ou EMA)
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # Calcul de la loss (detach pour éviter l'accumulation mémoire)
                loss = self.criterion(predicted, labels)

                # Calcul du Dice score si activé dans la config
                # Le Dice mesure le chevauchement entre prédiction et label :
                #   Dice = 2 * |P ∩ L| / (|P| + |L|)
                #   Valeur entre 0 (aucun chevauchement) et 1 (parfait)
                if self.calculate_metrics:
                    mean_dice = self._calc_dice_metric(data, labels, use_ema)
                    total_dice += mean_dice

                # Accumulation de la loss
                epoch_avg_loss += loss.detach().item()

                # Mise à jour de la barre de progression
                avg_val_loss = epoch_avg_loss / (index + 1)
                val_metrics = {'val_loss': avg_val_loss}
                if self.calculate_metrics and total_dice > 0:
                    val_metrics['dice'] = total_dice / (index + 1)
                pbar.update(index + 1, val_metrics)

                # Libération périodique du cache CUDA pour éviter la fragmentation mémoire
                if index % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if use_ema:
            self.epoch_val_ema_dice = total_dice / float(index + 1)
        else:
            self.epoch_val_dice = total_dice / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        # Finaliser la barre de validation
        final_metrics = {'val_loss': epoch_avg_loss}
        if self.calculate_metrics:
            dice_val = self.epoch_val_ema_dice if use_ema else self.epoch_val_dice
            final_metrics['dice'] = dice_val
        pbar.finish(final_metrics)

        return epoch_avg_loss

    def _calc_dice_metric(self, data, labels, use_ema: bool) -> float:
        """Calcule le Dice score moyen par Sliding Window Inference.

        Le volume d'entrée est découpé en patches (ROI) qui sont inférés
        individuellement puis recombinés. Cela permet d'évaluer des volumes
        de résolution arbitraire sans dépasser la mémoire GPU.

        Le Dice coefficient mesure la similarité entre deux ensembles :
            Dice = 2 × TP / (2 × TP + FP + FN)
        où TP=vrais positifs, FP=faux positifs, FN=faux négatifs.

        Args:
            data (Tensor): Image d'entrée (B, C, D, H, W).
            labels (Tensor): Masque de segmentation ground truth (B, 1, D, H, W).
            use_ema (bool): Si True, utilise le modèle EMA pour l'inférence.

        Returns:
            float: Dice score moyen sur toutes les classes et tous les exemples du batch.
        """
        if use_ema:
            avg_dice_score = self.sliding_window_inference(
                data,
                labels,
                self.ema_model,
            )
        else:
            avg_dice_score = self.sliding_window_inference(
                data,
                labels,
                self.model,
            )
        return avg_dice_score

    def _run_train_val(self) -> None:
        """Boucle principale d'entraînement et de validation.

        Orchestre l'ensemble du processus d'entraînement :

        Pour chaque époque :
            1. Mise à jour du scheduler (warmup → training)
            2. ``_train_step()``  → entraînement sur tous les batches
            3. ``_val_step()``    → validation sans gradient
            4. ``_val_ema_model()`` → validation EMA périodique
            5. ``_update_metrics()`` → mise à jour des meilleurs scores
            6. ``_log_metrics()``   → envoi vers WandB
            7. ``_save_and_print()`` → sauvegarde du meilleur modèle + affichage
            8. ``_save_periodic_checkpoint()`` → checkpoint périodique avec rotation
            9. Mise à jour du visualiseur (courbes loss/dice)
            10. scheduler.step() → décroissance du learning rate
            11. Vérification early stopping

        En fin de boucle :
            - Sauvegarde du modèle final
            - Génération des graphiques récapitulatifs
            - Affichage du résumé final
        """
        # Connexion WandB pour le suivi en temps réel des poids et gradients
        if self.accelerator.is_main_process:
            try:
                if hasattr(self.wandb_tracker, 'run'):
                    self.wandb_tracker.run.watch(
                        self.model, self.criterion, log="all", log_freq=10, log_graph=True
                    )
            except Exception as e:
                self.logger.warning(f"Impossible de connecter wandb: {e}")

        # Afficher bannière de démarrage
        if self.accelerator.is_main_process:
            num_params = sum(p.numel() for p in self.model.parameters())
            log_training_start(
                self.logger, self.config,
                model_name=self.config.get("model", {}).get("name", "?"),
                num_params=num_params,
            )

        # Information de reprise si applicable
        if self.start_epoch > 0:
            self.logger.info(
                f"Reprise de l'entraînement à l'époque {self.start_epoch}/{self.num_epochs} "
                f"(best_dice={self.best_val_dice:.4f}, best_loss={self.best_val_loss:.6f})"
            )

        # Run Training and Validation
        for epoch in range(self.start_epoch, self.num_epochs):
            # update epoch
            self.current_epoch = epoch
            self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            self.epoch_train_loss = train_loss

            # run a single validation step
            val_loss = self._val_step(use_ema=False)
            self.epoch_val_loss = val_loss

            # if enabled run ema every x steps
            self._val_ema_model()

            # update metrics
            self._update_metrics()

            # log metrics
            self._log_metrics()

            # save and print
            self._save_and_print()

            # ── Sauvegarde périodique du checkpoint ──
            self._save_periodic_checkpoint()

            # ── Mise à jour de la visualisation ──
            if self.accelerator.is_main_process:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                self.visualizer.update(
                    epoch=epoch,
                    train_loss=self.epoch_train_loss,
                    val_loss=self.epoch_val_loss,
                    val_dice=self.epoch_val_dice,
                    learning_rate=current_lr,
                )

            # update schduler
            self.scheduler.step()
            
            # Clear CUDA cache periodically to avoid memory fragmentation
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Early stopping check
            if self.early_stop:
                self.logger.warning(
                    f"Early stopping déclenché après {self.current_epoch + 1} époques"
                )
                break
        
        # Sauvegarde finale du modèle à la fin de l'entraînement
        self._save_final_model()

        # ── Graphiques et résumé final ──
        if self.accelerator.is_main_process:
            summary_text = self.visualizer.finalize()
            self.logger.info(summary_text)
            log_training_end(
                self.logger,
                total_epochs=self.current_epoch + 1,
                best_val_dice=self.best_val_dice,
                best_val_loss=self.best_val_loss,
                checkpoint_dir=self.checkpoint_save_dir,
            )

    def _update_scheduler(self) -> None:
        """Gère la transition entre les deux phases de planification du learning rate.

        Phase 1 (Warmup) — époques [0, warmup_epochs) :
            Le learning rate augmente linéairement de start_factor × lr → lr.
            Cela stabilise les premières mises à jour lorsque les poids sont
            aléatoires et que les gradients sont bruités.

        Phase 2 (Training) — époques [warmup_epochs, num_epochs) :
            Le learning rate décroît selon un cosinus (CosineAnnealing) de lr → eta_min.
            La décroissance douce évite les chutes brutales qui perturbent la convergence.

        Si le warmup est désactivé, seul le scheduler de décroissance est utilisé.
        """
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.logger.info("Démarrage phase de warmup du learning rate")
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.logger.info("Transition vers le scheduler de décroissance du LR")
                self.scheduler = self.training_scheduler
        elif self.current_epoch == 0:
            self.logger.info("Activation du scheduler de décroissance du LR")
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """Met à jour les records de métriques (meilleure loss, meilleur Dice).

        Ces "meilleurs" scores sont utilisés pour :
        - Déterminer si le modèle courant doit être sauvegardé (is_best)
        - Le résumé final d'entraînement
        - Les métadonnées des checkpoints
        """
        # Mise à jour du meilleur train loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # Mise à jour de la meilleure validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        # Mise à jour du meilleur Dice score (critère principal de sauvegarde)
        if self.calculate_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

    def _log_metrics(self) -> None:
        """Envoie les métriques de l'époque courante vers WandB.

        Les métriques loguées sont :
        - epoch : numéro de l'époque (0-indexed)
        - train_loss : loss moyenne d'entraînement
        - val_loss : loss moyenne de validation
        - mean_dice : Dice score moyen de validation

        Si WandB n'est pas configuré, l'appel est silencieusement ignoré.
        """
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
        }
        try:
            if self.wandb_tracker is not None:
                self.accelerator.log(log_data)
        except Exception as e:
            pass  # Le logging WandB n'est pas critique

    def _save_and_print(self) -> None:
        """Sauvegarde du meilleur modèle, gestion de l'early stopping, et affichage.

        Critère de sélection du meilleur modèle :
        - Si le Dice est calculé et > 0 : meilleur Dice score
        - Sinon : meilleure (plus basse) loss de validation

        Si le modèle actuel est le meilleur :
        - Le compteur d'early stopping est remis à zéro
        - Un checkpoint complet (Accelerate state) est sauvegardé
        - Le meilleur modèle est sauvegardé en format simple (.pth)
        - Si on dépasse le cutoff_epoch, le checkpoint va dans un sous-dossier séparé

        Sinon :
        - Le compteur d'early stopping est incrémenté
        - Si le compteur atteint la patience, le flag early_stop est activé
        """
        is_best = False
        
        # Détermine si l'époque courante est la meilleure
        if self.calculate_metrics and self.epoch_val_dice > 0:
            is_best = self.epoch_val_dice >= self.best_val_dice
        else:
            is_best = self.epoch_val_loss <= self.best_val_loss
        
        if is_best:
            # Réinitialisation du compteur d'early stopping car amélioration détectée
            self.early_stopping_counter = 0
            
            # Le cutoff_epoch permet de séparer les checkpoints de début et fin
            # d'entraînement (utile pour les entraînements très longs)
            if self.current_epoch <= self.cutoff_epoch:
                save_path = self.checkpoint_save_dir
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_dice_model_post_cutoff",
                )

            # Sauvegarde du checkpoint complet (modèle + optimiseur + scheduler)
            # via Accelerate (gère automatiquement le unwrap en multi-GPU)
            self._save_checkpoint(save_path)
            
            # Sauvegarde additionnelle en format simple .pth (plus facile à charger
            # pour l'inférence sans Accelerate)
            self._save_best_model()

        else:
            # Pas d'amélioration : incrémenter le compteur d'early stopping
            self.early_stopping_counter += 1
            
            # Si la patience est dépassée, activer le flag d'arrêt
            if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stop = True

        # ── Affichage unifié via le logger ──
        if self.accelerator.is_main_process:
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
            log_epoch_summary(
                self.logger,
                epoch=self.current_epoch,
                train_loss=self.epoch_train_loss,
                val_loss=self.epoch_val_loss,
                val_dice=self.epoch_val_dice,
                lr=current_lr,
                is_best=is_best,
                early_stop_counter=self.early_stopping_counter,
                early_stop_patience=self.early_stopping_patience,
            )
    
    def _save_best_model(self) -> None:
        """Sauvegarde le meilleur modèle en format .pth simple.

        Contrairement à _save_checkpoint() qui utilise Accelerate (sauvegarde
        l'état complet dans un dossier avec shards), cette méthode produit un
        fichier .pth unique contenant :
        - model_state_dict : poids du modèle (unwrapped)
        - optimizer_state_dict : état de l'optimiseur
        - scheduler_state_dict : état du scheduler
        - métriques courantes et meilleures métriques
        - config complète

        Ce format est directement utilisable pour l'inférence sans Accelerate.
        Un fichier texte lisible (best_model_info.txt) est aussi généré.
        """
        if self.accelerator.is_main_process:
            best_model_path = os.path.join(self.checkpoint_save_dir, "best_model.pth")
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = {
                "epoch": self.current_epoch + 1,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.epoch_train_loss,
                "val_loss": self.epoch_val_loss,
                "val_dice": self.epoch_val_dice,
                "best_val_dice": self.best_val_dice,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            }
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(checkpoint, best_model_path)

            # Sauvegarder aussi les infos lisibles
            info_path = os.path.join(self.checkpoint_save_dir, "best_model_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Epoch: {self.current_epoch + 1}\n")
                f.write(f"Train Loss: {self.epoch_train_loss:.6f}\n")
                f.write(f"Val Loss: {self.epoch_val_loss:.6f}\n")
                f.write(f"Val Dice: {self.epoch_val_dice:.6f}\n")
                f.write(f"Best Dice: {self.best_val_dice:.6f}\n")

            self.logger.info(
                f"Meilleur modèle sauvegardé → {best_model_path} "
                f"(epoch {self.current_epoch + 1}, dice={self.epoch_val_dice:.2f}%)"
            )

    def _save_periodic_checkpoint(self) -> None:
        """Sauvegarde un checkpoint complet toutes les `save_freq` époques.

        Les checkpoints sont numérotés et une rotation est appliquée pour
        ne conserver que les `keep_last` plus récents.
        """
        if self.checkpoint_save_freq <= 0:
            return  # fonctionnalité désactivée

        # Sauvegarder uniquement aux époques multiples de save_freq
        epoch_1based = self.current_epoch + 1
        if epoch_1based % self.checkpoint_save_freq != 0:
            return

        if not self.accelerator.is_main_process:
            return

        ckpt_dir = os.path.join(self.checkpoint_save_dir, "periodic_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_name = f"checkpoint_epoch_{epoch_1based:04d}.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint = {
            "epoch": epoch_1based,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "val_dice": self.epoch_val_dice,
            "best_val_dice": self.best_val_dice,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)
        self._saved_checkpoints.append(ckpt_path)

        self.logger.info(
            f"Checkpoint périodique sauvegardé → {ckpt_name}"
        )

        # ── Rotation : supprimer les checkpoints les plus anciens ──
        if self.checkpoint_keep_last > 0:
            while len(self._saved_checkpoints) > self.checkpoint_keep_last:
                old_path = self._saved_checkpoints.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    self.logger.debug(
                        f"Ancien checkpoint supprimé: {os.path.basename(old_path)}"
                    )
    
    def _save_final_model(self) -> None:
        """Sauvegarde le modèle à la dernière époque (final_model.pth).

        Ce modèle n'est pas nécessairement le meilleur : c'est simplement l'état
        du réseau en fin d'entraînement. Utile pour comparer les performances
        entre le meilleur modèle (best_model.pth) et le modèle final.
        """
        if self.accelerator.is_main_process:
            final_model_path = os.path.join(self.checkpoint_save_dir, "final_model.pth")
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = {
                "epoch": self.current_epoch + 1,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.epoch_train_loss,
                "val_loss": self.epoch_val_loss,
                "val_dice": self.epoch_val_dice,
                "best_val_dice": self.best_val_dice,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            }
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(checkpoint, final_model_path)

            self.logger.info(f"Modèle final sauvegardé: {final_model_path}")
            self.logger.info(
                f"Époques: {self.current_epoch + 1} | "
                f"Meilleur Dice: {self.best_val_dice:.2f}% | "
                f"Meilleure Val Loss: {self.best_val_loss:.6f}"
            )

    def _save_checkpoint(self, filename: str) -> None:
        """Sauvegarde l'état complet d'entraînement via HuggingFace Accelerate.

        Accelerate sauvegarde dans un dossier :
        - Le modèle (unwrapped si DDP)
        - L'optimiseur
        - Le scheduler
        - L'état de l'accélérateur (random states, etc.)

        Cela permet de reprendre l'entraînement exactement là où il s'est arrêté.
        safe_serialization=False utilise le format PyTorch classique (.bin)
        au lieu de safetensors.

        Args:
            filename (str): Chemin du dossier de destination pour le checkpoint.
        """
        # Note : la sauvegarde du modèle EMA est désactivée (non testée)
        # if self.ema_enabled and self.val_ema_model:
        #     checkpoint = {
        #         "state_dict": self.val_ema_model.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #     }
        #     torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")

        # Sauvegarde complète de l'état d'Accelerate
        self.accelerator.save_state(filename, safe_serialization=False)

    def _val_ema_model(self):
        """Valide périodiquement le modèle EMA et sauvegarde le meilleur.

        Tous les `val_ema_every` époques :
        1. Met à jour les statistiques BatchNorm du modèle EMA (nécessaire car
           l'EMA ne passe jamais par la forward pass d'entraînement)
        2. Lance une validation complète avec le modèle EMA
        3. Si le Dice EMA est le meilleur observé, sauvegarde le modèle

        Le modèle EMA ayant des poids plus lisses, il peut surpasser le modèle
        standard en termes de généralisation.
        """
        if self.ema_enabled and (self.current_epoch % self.val_ema_every == 0):
            # Recalcule les stats BatchNorm (running_mean, running_var) car
            # l'EMA ne les a jamais calculées lors de l'entraînement
            self.val_ema_model = self._update_ema_bn(duplicate_model=False)
            _ = self._val_step(use_ema=True)
            self.logger.info(
                f"EMA val dice: {self.epoch_val_ema_dice:.2f}% "
                f"(device: {self.accelerator.device})"
            )

        # Sauvegarde du meilleur modèle EMA
        if self.epoch_val_ema_dice > self.best_val_ema_dice:
            torch.save(self.val_ema_model.module, "best_ema_model_ckpt.pth")
            self.best_val_ema_dice = self.epoch_val_ema_dice

    def _update_ema_bn(self, duplicate_model: bool = True):
        """Recalcule les statistiques BatchNorm (running_mean / running_var) pour le modèle EMA.

        Le modèle EMA ne passe jamais par la forward pass d'entraînement, donc ses
        couches BatchNorm ont des stats incorrectes (initialisées ou obsolètes).
        Cette méthode fait passer l'ensemble du DataLoader d'entraînement dans le
        modèle EMA en mode évaluation pour recalculer les stats de normalisation.

        Args:
            duplicate_model (bool): Si True, travaille sur une copie profonde du
                modèle EMA (pour la validation intermédiaire sans modifier l'EMA
                original). Si False, modifie directement le modèle EMA (pour la
                sauvegarde finale).

        Returns:
            torch.nn.Module ou None: La copie mise à jour si duplicate_model=True,
                None sinon (le modèle EMA original est modifié in-place).
        """
        self.logger.info("Mise à jour des stats BatchNorm pour le modèle EMA")
        if duplicate_model:
            # Copie profonde pour ne pas perturber l'EMA original
            temp_ema_model = deepcopy(self.ema_model).to(
                self.accelerator.device
            )
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                temp_ema_model,
                device=self.accelerator.device,
            )
            return temp_ema_model
        else:
            # Mise à jour in-place de l'EMA original
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                self.ema_model,
                device=self.accelerator.device,
            )
            return None

    def train(self) -> None:
        """Point d'entrée principal : lance la boucle complète d'entraînement/validation.

        Appelle _run_train_val() puis signale à Accelerate que l'entraînement
        est terminé (libération des ressources, finalisation WandB, etc.).
        """
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        """Évaluation sur un dataset de test (non implémenté).

        Pour l'inférence, utiliser inference_simple.py ou le pipeline.
        """
        raise NotImplementedError("evaluate function is not implemented yet")


#################################################################################################
#                          TRAINER POUR L'AUTO-ENCODEUR 3D
#################################################################################################
class AutoEncoder_Trainer:
    """Trainer pour l'entraînement d'un auto-encodeur 3D.

    L'auto-encodeur apprend à reconstruire l'image d'entrée (T2) à travers un
    goulot d'étranglement (bottleneck), capturant ainsi des représentations latentes
    des volumes médicaux. La qualité de la reconstruction est mesurée par la SSIM 3D
    (Structural Similarity Index Measure).

    Différences clés avec Segmentation_Trainer :
    - La cible (target) est l'image d'entrée elle-même (apprentissage non supervisé)
    - Seule la première modalité (T2) est utilisée : data[:, 0, :, :, :]
    - La métrique de qualité est la SSIM 3D (via kornia) au lieu du Dice
    - Pas d'early stopping ni de checkpoints périodiques
    """

    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        """Initialise le trainer d'auto-encodeur.

        Args:
            config (Dict): Configuration YAML complète.
            model (torch.nn.Module): Réseau auto-encodeur 3D. Entrée et sortie ont la
                même forme (B, 1, D, H, W).
            optimizer (torch.optim.Optimizer): Optimiseur (Adam, AdamW, etc.).
            criterion (torch.nn.Module): Fonction de loss de reconstruction (MSE, L1, etc.).
            train_dataloader (DataLoader): DataLoader d'entraînement.
            val_dataloader (DataLoader): DataLoader de validation.
            warmup_scheduler (LRScheduler): Scheduler de warmup linéaire (ou None).
            training_scheduler (LRScheduler): Scheduler principal (CosineAnnealing).
            accelerator: Objet HuggingFace Accelerate pour le support multi-GPU.
        """
        # ── Configuration générale ──
        self.config = config
        self._configure_trainer()  # Extrait les hyperparamètres de la config

        # ── Composants du modèle ──
        self.model = model        # Auto-encodeur 3D
        self.optimizer = optimizer
        self.criterion = criterion  # Loss de reconstruction (MSE, L1, etc.)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # ── HuggingFace Accelerate ──
        self.accelerator = accelerator

        # ── Weights & Biases (obligatoire pour AutoEncoder_Trainer) ──
        self.wandb_tracker = accelerator.get_tracker("wandb")

        # ── Métriques d'entraînement ──
        self.start_epoch = 0            # Époque de départ (>0 si reprise)
        self.current_epoch = 0          # Époque courante
        self.epoch_train_loss = 0.0     # Loss de reconstruction (train)
        self.best_train_loss = 100.0    # Meilleure loss d'entraînement
        self.epoch_val_loss = 0.0       # Loss de reconstruction (validation)
        self.best_val_loss = 100.0      # Meilleure loss de validation
        self.epoch_val_iou = 0.0        # SSIM 3D de l'époque courante
        self.best_val_iou = 0.0         # Meilleur SSIM 3D observé
        self.ema_val_acc = 0.0          # SSIM du modèle EMA (non utilisé)

        # ── Schedulers de learning rate ──
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None  # Assigné dans _update_scheduler()

        # ── Modèle EMA (non utilisé activement dans l'auto-encodeur) ──
        self.val_ema_model = None

    def _configure_trainer(self) -> None:
        """Extrait les hyperparamètres de la config YAML en attributs d'instance.

        Identique à Segmentation_Trainer._configure_trainer(), mais utilise
        print_ema_every au lieu de val_ema_every.
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.print_ema_every = self.config["ema"]["print_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]

    def _load_checkpoint(self):
        """Non implémenté — utiliser --checkpoint de main()."""
        raise NotImplementedError

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Reprend l'entraînement exactement à l'époque d'interruption.

        Restaure : poids du modèle, optimiseur, scheduler, métriques, époque.

        Args:
            checkpoint_path (str): Chemin vers le fichier .pth.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

        self.accelerator.print(f"Reprise de l'entraînement depuis: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.print("Poids du modèle restaurés")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.accelerator.print("État de l'optimiseur restauré")

        if "scheduler_state_dict" in checkpoint and self.training_scheduler is not None:
            self.training_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.accelerator.print("État du scheduler restauré")

        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_val_iou = checkpoint.get("best_val_iou", checkpoint.get("best_val_dice", 0.0))
        self.best_val_loss = checkpoint.get("best_val_loss", 100.0)
        self.best_train_loss = checkpoint.get("train_loss", 100.0)
        self.accelerator.print(
            f"Reprise à l'époque {self.start_epoch}/{self.num_epochs} "
            f"(best_ssim={self.best_val_iou:.4f}, best_loss={self.best_val_loss:.6f})"
        )

    def _create_ema_model(self, gpu_id: int) -> torch.nn.Module:
        """Crée un modèle EMA sur le GPU spécifié.

        Args:
            gpu_id (int): Identifiant du GPU cible.

        Returns:
            torch.nn.Module: Modèle EMA initialisé.
        """
        self.accelerator.print(f"[info] -- creating ema model")
        ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            device=gpu_id,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            ),
        )
        return ema_model

    def _train_step(self) -> float:
        """Exécute une époque d'entraînement pour l'auto-encodeur.

        Différence clé avec Segmentation_Trainer._train_step() :
        - Seule la première modalité (canal 0 = T2) est utilisée en entrée
        - La cible est l'image d'entrée elle-même (reconstruction)
        - La loss mesure la différence entre l'entrée et la sortie reconstruite

        Returns:
            float: Loss de reconstruction moyenne pour cette époque.
        """
        # Initialisation de la loss cumulée
        epoch_avg_loss = 0.0

        # Mode entraînement
        self.model.train()

        # Barre de progression style Keras
        pbar = KerasProgressBar(
            total=len(self.train_dataloader),
            epoch=self.current_epoch,
            num_epochs=self.num_epochs,
            enabled=self.accelerator.is_main_process,
        )
        pbar.epoch_header()

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for index, raw_data in enumerate(self.train_dataloader):
            # Accumulation de gradient via Accelerate
            with self.accelerator.accumulate(self.model):
                # Extraction de la première modalité (T2) uniquement
                # data[:, 0, :, :, :].unsqueeze(1) : (B, C, D, H, W) → (B, 1, D, H, W)
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],  # Le label est ignoré (non supervisé)
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # Remise à zéro des gradients
                self.optimizer.zero_grad()

                # Passe forward : le modèle reconstruit l'entrée
                predicted = self.model.forward(data)

                # Loss de reconstruction : predicted vs data (pas vs label !)
                loss = self.criterion(predicted, data)

                # Rétropropagation
                self.accelerator.backward(loss)

                # Mise à jour des poids
                self.optimizer.step()

                # Mise à jour EMA si activé
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model.module)

                # Accumulation de la loss
                epoch_avg_loss += loss.item()

                # Mise à jour de la barre de progression
                avg_loss = epoch_avg_loss / (index + 1)
                pbar.update(index + 1, {
                    'loss': avg_loss,
                    'lr': self.scheduler.get_last_lr()[0],
                })

        epoch_avg_loss = epoch_avg_loss / (index + 1)
        pbar.finish({'loss': epoch_avg_loss})

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """Exécute une époque de validation pour l'auto-encodeur.

        Évalue la qualité de reconstruction en comparant l'entrée et la sortie.
        La métrique SSIM 3D (Structural Similarity) est utilisée à la place
        du Dice car on mesure la similarité structurelle (luminosité, contraste,
        structure) plutôt que le chevauchement de classes.

        Args:
            use_ema (bool): Si True, utilise le modèle EMA pour l'inférence.

        Returns:
            float: Loss de reconstruction moyenne de validation.
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_iou = 0.0

        # set model to train mode
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # set epoch to shift data order each epoch
        # self.val_dataloader.sampler.set_epoch(self.current_epoch)
        with torch.no_grad():
            pbar = KerasProgressBar(
                total=len(self.val_dataloader),
                epoch=self.current_epoch,
                num_epochs=self.num_epochs,
                prefix="Val",
                enabled=self.accelerator.is_main_process,
            )
            for index, (raw_data) in enumerate(self.val_dataloader):
                # Extraction de la première modalité (T2) uniquement
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],  # Le label est ignoré
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # Passe forward : reconstruction
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # Loss de reconstruction (predicted vs data)
                loss = self.criterion(predicted, data)

                # Calcul de la SSIM 3D si les métriques sont activées
                if self.calculate_metrics:
                    mean_iou = self._calc_mean_ssim(predicted, data)
                    total_iou += mean_iou

                # Accumulation de la loss
                epoch_avg_loss += loss.item()

                # Mise à jour barre de validation
                avg_val_loss = epoch_avg_loss / (index + 1)
                val_metrics = {'val_loss': avg_val_loss}
                if self.calculate_metrics and total_iou > 0:
                    val_metrics['ssim'] = total_iou / (index + 1)
                pbar.update(index + 1, val_metrics)

        if use_ema:
            self.epoch_val_iou = total_iou / float(index + 1)
        else:
            self.epoch_val_iou = total_iou / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        # Finaliser la barre de validation
        final_metrics = {'val_loss': epoch_avg_loss}
        if self.calculate_metrics:
            final_metrics['ssim'] = self.epoch_val_iou
        pbar.finish(final_metrics)

        return epoch_avg_loss

    def _calc_mean_ssim(self, predicted, ground_truth) -> float:
        """Calcule la SSIM 3D moyenne (Structural Similarity Index Measure).

        La SSIM compare deux images en termes de :
        - Luminosité : comparaison des moyennes
        - Contraste : comparaison des variances
        - Structure : comparaison de la corrélation

        SSIM = 1 signifie des images identiques, SSIM = 0 signifie aucune similarité.
        Utilise kornia.metrics.ssim3d avec une fenêtre de 5×5×5.

        Args:
            predicted (Tensor): Volume reconstruit (B, 1, D, H, W).
            ground_truth (Tensor): Volume original (B, 1, D, H, W).

        Returns:
            float: SSIM 3D moyen sur le batch.
        """
        # gather_for_metrics rassemble les résultats de tous les GPU en multi-GPU
        predictions, ground_truth = self.accelerator.gather_for_metrics(
            (predicted, ground_truth)
        )
        ssim_map = kornia.metrics.ssim3d(predictions, ground_truth, window_size=5)
        ssim_map = ssim_map.mean()

        return ssim_map.item()

    def _run_train_val(self) -> None:
        """Boucle principale d'entraînement/validation de l'auto-encodeur.

        Pour chaque époque : train_step → val_step → update_metrics → log → save.
        Pas d'early stopping ni de checkpoints périodiques (version simplifiée).
        """
        # Connexion WandB pour suivre les poids et gradients
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        # Run Training and Validation
        for epoch in range(self.start_epoch, self.num_epochs):
            # update epoch
            self.current_epoch = epoch
            if self.warmup_enabled or self.current_epoch == 0:
                self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            self.epoch_train_loss = train_loss

            # run a single validation step
            val_loss = self._val_step(use_ema=False)
            self.epoch_val_loss = val_loss

            # update metrics
            self._update_metrics()

            # log metrics
            self._log_metrics()

            # save and print
            self._save_and_print()

            # update schduler
            self.scheduler.step()

    def _update_scheduler(self) -> None:
        """Gère la transition warmup → scheduler de décroissance du LR.

        Même logique que Segmentation_Trainer._update_scheduler().
        """
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
        else:
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """Met à jour les records de métriques (meilleure loss, meilleur SSIM)."""
        # Meilleure loss d'entraînement
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # Meilleure loss de validation
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        # Meilleur SSIM de validation
        if self.calculate_metrics:
            if self.epoch_val_iou >= self.best_val_iou:
                self.best_val_iou = self.epoch_val_iou

    def _log_metrics(self) -> None:
        """Envoie les métriques de l'époque courante vers WandB."""
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_iou": self.epoch_val_iou,  # SSIM stockée sous le nom 'mean_iou'
        }
        self.accelerator.log(log_data)

    def _save_and_print(self) -> None:
        """Sauvegarde le meilleur modèle et affiche les métriques dans la console.

        Le critère de sauvegarde est le meilleur SSIM de validation (epoch_val_iou).
        Utilise des couleurs (termcolor) pour distinguer visuellement les époques
        où le modèle s'améliore (vert) des autres.
        """
        # print only on the first gpu
        if self.epoch_val_iou >= self.best_val_iou:
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state",
                )
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state_post_cutoff.pth",
                )

            # save checkpoint and log
            self._save_checkpoint(save_path)

            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"val mean_ssim -- {colored(f'{self.best_val_iou:.5f}', color='green')} -- saved"
            )
        else:
            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {self.epoch_train_loss:.5f} || "
                f"val loss -- {self.epoch_val_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"val mean_ssim -- {self.epoch_val_iou:.5f}"
            )

    def _save_checkpoint(self, filename: str) -> None:
        """Sauvegarde l'état complet d'entraînement via Accelerate.

        Inclut la sauvegarde du modèle EMA si disponible, dans un fichier
        séparé (ema_model_ckpt.pth).

        Args:
            filename (str): Chemin du dossier de destination.
        """
        # saves the ema model checkpoint if availabale
        # TODO: ema saving untested
        if self.ema_enabled and self.val_ema_model:
            checkpoint = {
                "state_dict": self.val_ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")
            self.val_ema_model = (
                None  # set ema model to None to avoid duplicate model saving
            )

        # standard model checkpoint
        self.accelerator.save_state(filename, safe_serialization=False)

    def _update_ema_bn(self, duplicate_model: bool = True):
        """Recalcule les stats BatchNorm pour le modèle EMA.

        Même logique que Segmentation_Trainer._update_ema_bn().

        Args:
            duplicate_model (bool): Si True, copie profonde ; si False, modification in-place.

        Returns:
            torch.nn.Module ou None: Copie mise à jour ou None.
        """
        # TODO: tester la fonctionnalité EMA complète
        self.accelerator.print(colored(f"[info] -- updating ema batch norm stats", color="red"))
        if duplicate_model:
            temp_ema_model = deepcopy(self.ema_model).to(self.accelerator.device)
            torch.optim.swa_utils.update_bn(
                self.train_dataloader, temp_ema_model, device=self.accelerator.device
            )
            return temp_ema_model
        else:
            torch.optim.swa_utils.update_bn(
                self.train_dataloader, self.ema_model, device=self.accelerator.device
            )
            return None

    def train(self) -> None:
        """Lance la boucle complète d'entraînement puis libère les ressources."""
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        """Évaluation sur un dataset de test (non implémenté)."""
        pass

#################################################################################################
#                             POINT D'ENTRÉE PRINCIPAL
#################################################################################################
def main():
    """Point d'entrée pour l'entraînement via ligne de commande.

    Orchestre l'ensemble du processus :
    1. Parse les arguments CLI (--config, --checkpoint, --local_rank)
    2. Charge la configuration YAML
    3. Détecte et configure le matériel (GPU, mémoire, déterminisme)
    4. Initialise l'accélérateur HuggingFace Accelerate
    5. Construit le modèle, l'optimiseur, la loss, les dataloaders et les schedulers
    6. Optionnel : charge un checkpoint pour le fine-tuning
    7. Prépare tous les composants avec Accelerate (dispatch multi-GPU)
    8. Instancie le trainer (Segmentation ou AutoEncoder selon la config)
    9. Lance l'entraînement

    Utilisation :
        python trainer_ddp.py --config configs/config_segformer3d.yaml
        python trainer_ddp.py --config configs/config_segformer3d.yaml --checkpoint checkpoints/best_model.pth
        python trainer_ddp.py --config configs/config_segformer3d.yaml --resume checkpoints/best_model.pth
    """
    import argparse
    import yaml
    from accelerate import Accelerator  # Multi-GPU / précision mixte
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
    
    # Logger principal pour le processus de setup
    logger = get_logger("main", level="INFO")
    logger.debug("main() appelé")
    
    # ============ ARGUMENTS EN LIGNE DE COMMANDE ============
    parser = argparse.ArgumentParser(description="Train SegFormer3D model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration YAML de l'architecture"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Rang local pour l'entraînement distribué (géré automatiquement par torchrun)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers un checkpoint (.pth) pour le fine-tuning (charge les poids, reset optimiseur)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint (.pth) pour reprendre l'entraînement à l'époque d'interruption (restaure tout : modèle, optimiseur, scheduler, métriques)"
    )
    args = parser.parse_args()
    
    # Vérifier qu'on n'utilise pas --checkpoint et --resume en même temps
    if args.checkpoint and args.resume:
        raise ValueError("Impossible d'utiliser --checkpoint (fine-tuning) et --resume (reprise) en même temps. "
                         "Utilisez --resume pour reprendre l'entraînement, --checkpoint pour du fine-tuning.")
    
    # ============ CHARGEMENT DE LA CONFIGURATION ============
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file {args.config} is empty or invalid")
    
    # ============ CONFIGURATION MATÉRIELLE ============
    # La section 'hardware' de la config YAML permet de contrôler :
    # - Le device (auto, cuda, cpu, mps)
    # - Les GPU spécifiques à utiliser (gpu_ids)
    # - Les optimisations mémoire (TF32, cuDNN benchmark)
    # - Le mode déterministe (reproductibilité)
    # - La seed aléatoire
    hardware_cfg = config.get("hardware", {})
    
    # Détection automatique du meilleur device disponible
    device_type = hardware_cfg.get("device", "auto")
    if device_type == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"  # GPU NVIDIA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"   # GPU Apple Silicon
        else:
            device_type = "cpu"   # Pas de GPU disponible
    
    # Sélection de GPU spécifiques (pour les systèmes multi-GPU)
    # Par ex. gpu_ids: [0, 1] pour utiliser les 2 premiers GPU
    gpu_ids = hardware_cfg.get("gpu_ids", None)
    if gpu_ids is not None and device_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Optimisations mémoire CUDA
    memory_efficient = hardware_cfg.get("memory_efficient", False)
    if memory_efficient and device_type == "cuda":
        # cuDNN benchmark : teste plusieurs algorithmes de convolution et garde le plus rapide.
        # Améliore les performances quand la taille des inputs est fixe.
        torch.backends.cudnn.benchmark = hardware_cfg.get("cudnn_benchmark", True)
        # TF32 (TensorFloat-32) : utilise une précision réduite (19 bits au lieu de 32)
        # pour les multiplications matricielles. ~2x plus rapide sur Ampere+ (RTX 30xx).
        torch.backends.cuda.matmul.allow_tf32 = hardware_cfg.get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = hardware_cfg.get("allow_tf32", True)
    
    # Mode déterministe pour la reproductibilité exacte
    # Attention : ralentit l'entraînement car désactive les optimisations non-déterministes
    deterministic = hardware_cfg.get("deterministic", False)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    # Seed aléatoire pour la reproductibilité
    # Fixe les générateurs Python, NumPy, et PyTorch
    seed = hardware_cfg.get("seed", None)
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(seed)  # Toutes les GPUs
    
    # ============ INITIALISATION DE L'ACCÉLÉRATEUR ============
    # HuggingFace Accelerate gère automatiquement :
    # - DataParallel / DistributedDataParallel selon le nombre de GPU
    # - L'accumulation de gradient (simule un batch size plus grand)
    # - La précision mixte (fp16 / bf16) pour réduire la VRAM et accélérer
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training_parameters", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=config.get("training_parameters", {}).get("mixed_precision", "no"),
        log_with=None,  # WandB désactivé par défaut pour éviter les blocages
        cpu=(device_type == "cpu"),
    )
    
    # Affichage des informations matérielles
    log_section(logger, "CONFIGURATION MATÉRIELLE")
    logger.info(f"Device: {device_type}")
    if device_type == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ============ CONSTRUCTION DU MODÈLE ============
    from architectures.build_architecture import build_architecture
    model = build_architecture(config)  # Construit SegFormer3D depuis la config
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modèle construit ({num_params:,} paramètres)")
    
    # ============ FINE-TUNING (optionnel) ============
    # Charge les poids d'un checkpoint pré-entraîné pour le fine-tuning.
    # Seuls les poids sont chargés, l'optimiseur et le scheduler repartent de zéro.
    # Note: --resume est géré après la création du trainer (car il restaure aussi
    # l'optimiseur et le scheduler)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")
        log_section(logger, "FINE-TUNING")
        logger.info(f"Chargement du checkpoint: {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            logger.info(f"Poids du modèle chargés (epoch {checkpoint_data.get('epoch', '?')}, "
                        f"dice={checkpoint_data.get('val_dice', checkpoint_data.get('best_val_dice', '?'))})")
        else:
            # Cas où le fichier contient directement le state_dict
            model.load_state_dict(checkpoint_data)
            logger.info("Poids du modèle chargés (state_dict direct)")
        logger.info("Fine-tuning: l'optimiseur et le scheduler repartent de zéro")
    
    # ============ CONSTRUCTION DES DATALOADERS ============
    # build_dataloaders() construit les datasets + dataloaders depuis la config YAML :
    # - Charge les fichiers .pt (modalities + labels) prétraités
    # - Applique les augmentations de données (si activées) sur le train uniquement
    # - Configure le DataLoader avec batch_size, num_workers, etc.
    from dataloaders.build_dataset import build_dataloaders
    train_dataloader, val_dataloader = build_dataloaders(config)
    logger.info(f"Dataloaders construits (train: {len(train_dataloader)} batches, val: {len(val_dataloader)} batches)")
    
    # ============ CONSTRUCTION DE L'OPTIMISEUR ============
    # Supporte : Adam, AdamW, SGD, etc. selon la config
    from optimizers.optimizers import build_optimizer
    optimizer = build_optimizer(model, config)
    logger.info(f"Optimiseur construit: {config.get('optimizer', {}).get('type', '?')}")
    
    # ============ CONSTRUCTION DE LA LOSS ============
    # Supporte : DiceLoss, DiceCELoss, etc.
    from losses.losses import build_loss
    criterion = build_loss(config)
    logger.info(f"Fonction de loss construite")
    
    # ============ CONSTRUCTION DES SCHEDULERS ============
    # Deux schedulers possibles :
    # 1. Warmup (LinearLR) : LR augmente de start_factor*lr → lr sur warmup_epochs
    # 2. Training (CosineAnnealingLR) : LR décroît de lr → eta_min sur T_max époques
    training_cfg = config.get("training_parameters", config.get("training", {}))
    
    warmup_scheduler = None
    training_scheduler = None
    
    num_epochs = training_cfg.get("num_epochs", config.get("training", {}).get("num_epochs", 100))
    
    warmup_cfg = config.get("warmup_scheduler", {})
    if warmup_cfg.get("enabled", False):
        warmup_epochs = warmup_cfg.get("warmup_epochs", config.get("training", {}).get("warmup_epochs", 10))
        warmup_steps = warmup_epochs * len(train_dataloader)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_cfg.get("start_factor", 1e-3),
            total_iters=warmup_steps
        )
    
    training_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=training_cfg.get("min_lr", 1e-6)
    )
    
    logger.info("Schedulers construits")
    
    # ============ PRÉPARATION AVEC ACCELERATE ============
    # accelerator.prepare() distribue automatiquement les composants :
    # - Le modèle est wrapé en DDP (DistributedDataParallel) si multi-GPU
    # - L'optimiseur est adapté pour la précision mixte (gradient scaling)
    # - Les dataloaders sont dispatchés entre les GPUs (chaque GPU reçoit un sous-ensemble)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    logger.info("Modèle préparé pour l'entraînement distribué")
    
    # ============ SÉLECTION DU TRAINER ============
    # 'segmentation' : entraînement supervisé avec labels (Dice + CE)
    # 'autoencoder' : entraînement non supervisé de reconstruction (MSE / SSIM)
    trainer_type = training_cfg.get("trainer_type", "segmentation")
    
    if trainer_type == "segmentation":
        trainer = Segmentation_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            warmup_scheduler=warmup_scheduler,
            training_scheduler=training_scheduler,
            accelerator=accelerator,
        )
    elif trainer_type == "autoencoder":
        trainer = AutoEncoder_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            warmup_scheduler=warmup_scheduler,
            training_scheduler=training_scheduler,
            accelerator=accelerator,
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    logger.info(f"Trainer initialisé (type: {trainer_type})")
    
    # ============ REPRISE D'ENTRAÎNEMENT (optionnel) ============
    # --resume restaure l'état complet (poids, optimiseur, scheduler, métriques, époque)
    # pour continuer exactement là où l'entraînement a été interrompu.
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint de reprise introuvable: {resume_path}")
        log_section(logger, "REPRISE D'ENTRAÎNEMENT")
        trainer.resume_from_checkpoint(str(resume_path))
        logger.info(f"Reprise depuis l'époque {trainer.start_epoch}/{num_epochs}")
    else:
        logger.info(f"Démarrage de l'entraînement pour {num_epochs} époques")
    
    # Start training
    trainer.train()
    
    logger.info("Entraînement terminé avec succès")


if __name__ == "__main__":
    main()