import os
import sys
import torch
try:
    import evaluate
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
import monai

# Supprimer les warnings de dépréciation MONAI/PyTorch
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.segmentation_metrics import SlidingWindowInference
from train_scripts.training_visualizer import TrainingVisualizer
from train_scripts.logger import (
    get_logger, log_epoch_summary, log_training_start,
    log_training_end, log_section, KerasProgressBar,
)
import kornia


#################################################################################################
class Segmentation_Trainer:
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
        """classification trainer class init function

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        self._configure_trainer()

        # model components
        self.model = model
        target_size = config["dataset_parameters"]["train_dataset_args"]["target_size"]
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # accelerate object
        self.accelerator = accelerator

        # get wandb object (optional, with error handling)
        try:
            self.wandb_tracker = accelerator.get_tracker("wandb")
        except:
            self.wandb_tracker = None

        # metrics
        self.current_epoch = 0  # current epoch
        self.epoch_train_loss = 0.0  # epoch train loss
        self.best_train_loss = 100.0  # best train loss
        self.epoch_val_loss = 0.0  # epoch validation loss
        self.best_val_loss = 100.0  # best validation loss
        self.epoch_val_dice = 0.0  # epoch validation accuracy
        self.best_val_dice = 0.0  # best validation accuracy
        
        # Early stopping
        self.early_stopping_patience = self.config.get("training_parameters", {}).get("early_stopping_patience", 0)
        self.early_stopping_counter = 0
        self.early_stop = False

        # external metric functions we can add
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
            num_classes=config.get("model", {}).get("num_classes", 2),
        )

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # temp ema model copy
        self.val_ema_model = None
        self.ema_model = self._create_ema_model() if self.ema_enabled else None
        self.epoch_val_ema_dice = 0.0
        self.best_val_ema_dice = 0.0

        # ── Visualiseur de métriques d'entraînement ──
        self.visualizer = TrainingVisualizer(
            save_dir=self.checkpoint_save_dir,
            experiment_name=config.get("model", {}).get("name", "SegFormer3D"),
        )

        # ── Logger unifié ──
        log_file = os.path.join(self.checkpoint_save_dir, "training.log")
        self.logger = get_logger(
            "trainer",
            level="DEBUG" if config.get("advanced", {}).get("verbosity", 1) > 1 else "INFO",
            log_file=log_file,
        )

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
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
        ckpt_cfg = self.config.get("checkpoint", {})
        self.checkpoint_save_freq = ckpt_cfg.get("save_freq", 0)  # 0 = désactivé
        self.checkpoint_keep_last = ckpt_cfg.get("keep_last", 5)
        self._saved_checkpoints: list = []  # liste des chemins pour la rotation

    def _load_checkpoint(self):
        raise NotImplementedError

    def _create_ema_model(self) -> torch.nn.Module:
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
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # Barre de progression avec tqdm
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=True,
            disable=not self.accelerator.is_main_process
        )

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for index, raw_data in pbar:
            # add in gradient accumulation
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # print("data ", data.shape, "label ", labels.shape)

                # zero out existing gradients (set_to_none=True for better performance)
                self.optimizer.zero_grad(set_to_none=True)

                # forward pass
                predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, labels)

                # backward pass
                self.accelerator.backward(loss)

                # gradient clipping if enabled
                if self.config.get("clip_gradients", {}).get("enabled", False):
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["clip_gradients"]["clip_gradients_value"]
                        )

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model)

                # update loss (detach to free computation graph)
                epoch_avg_loss += loss.detach().item()

                # Mise à jour de la barre de progression
                avg_loss = epoch_avg_loss / (index + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })

        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """Run validation step.

        Args:
            use_ema (bool, optional): if use_ema runs validation with ema_model. Defaults to False.

        Returns:
            float: average validation loss
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_dice = 0.0

        # set model to eval mode
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # set epoch to shift data order each epoch
        # self.val_dataloader.sampler.set_epoch(self.current_epoch)
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
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # calculate loss (detach to avoid memory accumulation)
                loss = self.criterion(predicted, labels)

                # calculate metrics
                if self.calculate_metrics:
                    mean_dice = self._calc_dice_metric(data, labels, use_ema)
                    # keep track of number of total correct
                    total_dice += mean_dice

                # update loss for the current batch
                epoch_avg_loss += loss.detach().item()

                # Mise à jour barre de validation
                avg_val_loss = epoch_avg_loss / (index + 1)
                val_metrics = {'val_loss': avg_val_loss}
                if self.calculate_metrics and total_dice > 0:
                    val_metrics['dice'] = total_dice / (index + 1)
                pbar.update(index + 1, val_metrics)

                # Free up memory periodically during validation
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
        """_summary_

        Args:
            predicted (_type_): _description_
            labels (_type_): _description_

        Returns:
            float: _description_
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
        """Run full training and validation loop with memory optimization."""
        # Tell wandb to watch the model and optimizer values
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

        # Run Training and Validation
        for epoch in range(self.num_epochs):
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
        """Met à jour le scheduler (warmup -> training)."""
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
        """Met à jour les meilleures métriques enregistrées."""
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

    def _log_metrics(self) -> None:
        """Log les métriques vers wandb et le logger unifié."""
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
        }
        # log the data (only if tracker is available)
        try:
            if self.wandb_tracker is not None:
                self.accelerator.log(log_data)
        except Exception as e:
            pass  # Logging not critical

    def _save_and_print(self) -> None:
        """Sauvegarde le meilleur modèle et affiche les métriques."""
        is_best = False
        
        # Vérifier si c'est le meilleur modèle (basé sur dice ou val_loss si dice=0)
        if self.calculate_metrics and self.epoch_val_dice > 0:
            is_best = self.epoch_val_dice >= self.best_val_dice
        else:
            # Si pas de dice, utiliser la val_loss
            is_best = self.epoch_val_loss <= self.best_val_loss
        
        if is_best:
            # Reset early stopping counter
            self.early_stopping_counter = 0
            
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                save_path = self.checkpoint_save_dir
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_dice_model_post_cutoff",
                )

            # save checkpoint and log
            self._save_checkpoint(save_path)
            
            # Sauvegarder aussi le modèle seul en format simple
            self._save_best_model()

        else:
            # Increment early stopping counter
            self.early_stopping_counter += 1
            
            # Check early stopping
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
        """Sauvegarde le meilleur modèle avec métadonnées complètes."""
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
        """Sauvegarde le modèle final avec métadonnées complètes."""
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
        """_summary_

        Args:
            filename (str): _description_
        """
        # saves the ema model checkpoint if availabale
        # TODO: ema saving untested (deprecated)
        # if self.ema_enabled and self.val_ema_model:
        #     checkpoint = {
        #         "state_dict": self.val_ema_model.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #     }
        #     torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")
        #     self.val_ema_model = (
        #         None  # set ema model to None to avoid duplicate model saving
        #     )

        # standard model checkpoint
        self.accelerator.save_state(filename, safe_serialization=False)

    def _val_ema_model(self):
        if self.ema_enabled and (self.current_epoch % self.val_ema_every == 0):
            self.val_ema_model = self._update_ema_bn(duplicate_model=False)
            _ = self._val_step(use_ema=True)
            self.logger.info(
                f"EMA val dice: {self.epoch_val_ema_dice:.2f}% "
                f"(device: {self.accelerator.device})"
            )

        if self.epoch_val_ema_dice > self.best_val_ema_dice:
            torch.save(self.val_ema_model.module, "best_ema_model_ckpt.pth")
            self.best_val_ema_dice = self.epoch_val_ema_dice

    def _update_ema_bn(self, duplicate_model: bool = True):
        """
        updates the batch norm stats for the ema model
        if duplicate_model is true, a copy of the model is made and
        the batch norm stats are updated for the copy. This is used
        for intermediate ema model saving and validation purpose
        if duplicate model is false, then the original ema model is used
        for the batch norm updates and will be saved as the final
        ema model.
        Args:
            duplicate_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # update batch norm stats for ema model after training
        self.logger.info("Mise à jour des stats BatchNorm pour le modèle EMA")
        if duplicate_model:
            temp_ema_model = deepcopy(self.ema_model).to(
                self.accelerator.device
            )  # make temp copy
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                temp_ema_model,
                device=self.accelerator.device,
            )
            return temp_ema_model
        else:
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                self.ema_model,
                device=self.accelerator.device,
            )
            return None

    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        raise NotImplementedError("evaluate function is not implemented yet")


#################################################################################################
class AutoEncoder_Trainer:
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
        """classification trainer class init function

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        self._configure_trainer()

        # model components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # accelerate object
        self.accelerator = accelerator

        # get wandb object
        self.wandb_tracker = accelerator.get_tracker("wandb")

        # metrics
        self.current_epoch = 0  # current epoch
        self.epoch_train_loss = 0.0  # epoch train loss
        self.best_train_loss = 100.0  # best train loss
        self.epoch_val_loss = 0.0  # epoch validation loss
        self.best_val_loss = 100.0  # best validation loss
        self.epoch_val_iou = 0.0  # epoch validation accuracy
        self.best_val_iou = 0.0  # best validation accuracy
        self.ema_val_acc = 0.0  # best ema validation accuracy

        # external metric functions we can add
        # self.metric = evaluate.load("mean_iou")
        # self.metric = compute_iou()

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # temp ema model copy
        self.val_ema_model = None

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
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
        raise NotImplementedError

    def _create_ema_model(self, gpu_id: int) -> torch.nn.Module:
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
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
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
            # add in gradient accumulation
            # TODO: test gradient accumulation
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, _)
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, data)

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model.module)

                # update loss
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
        """_summary_

        Args:
            use_ema (bool, optional): if use_ema runs validation with ema_model. Defaults to False.

        Returns:
            float: _description_
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
                # get data ex: (data, _)
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, data)

                if self.calculate_metrics:
                    mean_iou = self._calc_mean_ssim(predicted, data)
                    # keep track of number of total correct
                    total_iou += mean_iou

                # update loss for the current batch
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
        predictions, ground_truth = self.accelerator.gather_for_metrics(
            (predicted, ground_truth)
        )
        ssim_map = kornia.metrics.ssim3d(predictions, ground_truth, window_size=5)
        ssim_map = ssim_map.mean()

        return ssim_map.item()

    def _run_train_val(self) -> None:
        """_summary_"""
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        # Run Training and Validation
        for epoch in range(self.num_epochs):
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
        """_summary_"""
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
        """_summary_"""
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_iou >= self.best_val_iou:
                self.best_val_iou = self.epoch_val_iou

    def _log_metrics(self) -> None:
        """_summary_"""
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_iou": self.epoch_val_iou,
        }
        # log the data
        self.accelerator.log(log_data)

    def _save_and_print(self) -> None:
        """_summary_"""
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
        """_summary_

        Args:
            filename (str): _description_
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
        """
        updates the batch norm stats for the ema model
        if duplicate_model is true, a copy of the model is made and
        the batch norm stats are updated for the copy. This is used
        for intermediate ema model saving and validation purpose
        if duplicate model is false, then the original ema model is used
        for the batch norm updates and will be saved as the final
        ema model.
        Args:
            duplicate_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # update batch norm stats for ema model after training
        # TODO: test ema functionality
        self.accelerator.print(colored(f"[info] -- updating ema batch norm stats", color="red"))
        if duplicate_model:
            temp_ema_model = deepcopy(self.ema_model).to(self.accelerator.device)  # make temp copy
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
        """
        Runs a full training and validation of the dataset.
        """
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        pass

#################################################################################################
def main():
    """Main function to run training with configuration file."""
    import argparse
    import yaml
    from accelerate import Accelerator
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
    
    # Logger principal
    logger = get_logger("main", level="INFO")
    logger.debug("main() appelé")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train SegFormer3D model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file {args.config} is empty or invalid")
    
    # ============ HARDWARE CONFIGURATION ============
    hardware_cfg = config.get("hardware", {})
    
    # Device selection
    device_type = hardware_cfg.get("device", "auto")  # auto, cuda, cpu, mps
    if device_type == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    # GPU selection (for multi-GPU systems)
    gpu_ids = hardware_cfg.get("gpu_ids", None)  # e.g., [0, 1] or [0]
    if gpu_ids is not None and device_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Memory optimization options
    memory_efficient = hardware_cfg.get("memory_efficient", False)
    if memory_efficient and device_type == "cuda":
        # Enable memory efficient options
        torch.backends.cudnn.benchmark = hardware_cfg.get("cudnn_benchmark", True)
        torch.backends.cuda.matmul.allow_tf32 = hardware_cfg.get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = hardware_cfg.get("allow_tf32", True)
    
    # Deterministic mode for reproducibility
    deterministic = hardware_cfg.get("deterministic", False)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    # Random seed
    seed = hardware_cfg.get("seed", None)
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(seed)
    
    # Initialize accelerator with hardware settings
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training_parameters", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=config.get("training_parameters", {}).get("mixed_precision", "no"),
        log_with=None,  # Disable logging by default - no init_trackers
        cpu=(device_type == "cpu"),
    )
    
    # Print hardware info
    log_section(logger, "CONFIGURATION MATÉRIELLE")
    logger.info(f"Device: {device_type}")
    if device_type == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # WandB disabled - skip init_trackers to avoid blocking
    # Trainers will check for wandb_tracker = None
    
    # Build model
    from architectures.build_architecture import build_architecture
    model = build_architecture(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modèle construit ({num_params:,} paramètres)")
    
    # Build datasets and dataloaders
    from dataloaders.build_dataset import build_dataloaders
    train_dataloader, val_dataloader = build_dataloaders(config)
    logger.info(f"Dataloaders construits (train: {len(train_dataloader)} batches, val: {len(val_dataloader)} batches)")
    
    # Build optimizer
    from optimizers.optimizers import build_optimizer
    optimizer = build_optimizer(model, config)
    logger.info(f"Optimiseur construit: {config.get('optimizer', {}).get('type', '?')}")
    
    # Build criterion
    from losses.losses import build_loss
    criterion = build_loss(config)
    logger.info(f"Fonction de loss construite")
    
    # Get training parameters from config (support both old and new formats)
    training_cfg = config.get("training_parameters", config.get("training", {}))
    
    # Build schedulers
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
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    logger.info("Modèle préparé pour l'entraînement distribué")
    
    # Determine trainer type
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
    logger.info(f"Démarrage de l'entraînement pour {num_epochs} époques")
    
    # Start training
    trainer.train()
    
    logger.info("Entraînement terminé avec succès")


if __name__ == "__main__":
    main()