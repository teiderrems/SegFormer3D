"""
Système de logging unifié pour le projet SegFormer3D.

Fournit un logger configurable avec formatage coloré, niveaux de verbosité,
et écriture optionnelle dans un fichier. Remplace les print() dispersés
par un système cohérent à travers tout le projet.
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


# ─── Codes couleur ANSI ───────────────────────────────────────────────────────
class _Colors:
    """Codes ANSI pour la coloration du terminal."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Couleurs
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Fond
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"

    @staticmethod
    def supports_color() -> bool:
        """Vérifie si le terminal supporte les couleurs."""
        if os.environ.get("NO_COLOR"):
            return False
        if not hasattr(sys.stdout, "isatty"):
            return False
        return sys.stdout.isatty()


# ─── Formateur coloré ─────────────────────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    """Formateur de logs avec coloration ANSI par niveau."""

    _LEVEL_COLORS = {
        logging.DEBUG:    _Colors.DIM + _Colors.CYAN,
        logging.INFO:     _Colors.GREEN,
        logging.WARNING:  _Colors.YELLOW,
        logging.ERROR:    _Colors.RED,
        logging.CRITICAL: _Colors.BOLD + _Colors.BG_RED + _Colors.WHITE,
    }

    _LEVEL_ICONS = {
        logging.DEBUG:    "🔍",
        logging.INFO:     "ℹ️ ",
        logging.WARNING:  "⚠️ ",
        logging.ERROR:    "❌",
        logging.CRITICAL: "🚨",
    }

    def __init__(self, use_color: bool = True, show_time: bool = True):
        self.use_color = use_color and _Colors.supports_color()
        self.show_time = show_time
        fmt = "%(message)s"
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self._LEVEL_COLORS.get(record.levelno, _Colors.RESET)
            icon = self._LEVEL_ICONS.get(record.levelno, "")
            level = f"{color}{record.levelname:<8}{_Colors.RESET}"
            
            if self.show_time:
                timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
                prefix = f"{_Colors.DIM}{timestamp}{_Colors.RESET} {level} {icon} "
            else:
                prefix = f"{level} {icon} "

            # Colorer le message pour les niveaux warning+
            if record.levelno >= logging.WARNING:
                msg = f"{color}{record.getMessage()}{_Colors.RESET}"
            else:
                msg = record.getMessage()

            record.msg = prefix + msg
            record.args = None
            return super().format(record)
        else:
            if self.show_time:
                timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
                record.msg = f"[{timestamp}] [{record.levelname}] {record.getMessage()}"
            else:
                record.msg = f"[{record.levelname}] {record.getMessage()}"
            record.args = None
            return super().format(record)


# ─── Formateur fichier (sans couleur) ─────────────────────────────────────────
class FileFormatter(logging.Formatter):
    """Formateur de logs pour fichier (sans couleur ANSI)."""

    def __init__(self):
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)


# ─── Fonction utilitaire de création ──────────────────────────────────────────
def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_time: bool = True,
) -> logging.Logger:
    """Crée ou récupère un logger unifié pour le projet.
    
    Args:
        name: Nom du logger (ex: 'trainer', 'pipeline', 'inference')
        level: Niveau de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Chemin optionnel vers un fichier de log
        show_time: Afficher le timestamp dans les logs console
        
    Returns:
        Logger configuré
    
    Example:
        >>> logger = get_logger("trainer", level="INFO", log_file="./logs/train.log")
        >>> logger.info("Début de l'entraînement")
        >>> logger.warning("Mémoire GPU faible")
    """
    logger = logging.getLogger(f"segformer3d.{name}")
    
    # Éviter les doublons de handlers si le logger existe déjà
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Handler console avec couleurs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(use_color=True, show_time=show_time))
    logger.addHandler(console_handler)

    # Handler fichier (optionnel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    return logger


# ─── Fonctions pratiques pour le training ─────────────────────────────────────
def log_epoch_header(logger: logging.Logger, epoch: int, total_epochs: int) -> None:
    """Affiche un en-tête d'époque formaté."""
    width = 50
    header = f" Époque {epoch + 1}/{total_epochs} "
    logger.info("─" * width)
    logger.info(f"{header:─^{width}}")
    logger.info("─" * width)


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_dice: float,
    lr: float,
    is_best: bool = False,
    early_stop_counter: int = 0,
    early_stop_patience: int = 0,
) -> None:
    """Affiche un résumé complet de l'époque."""
    # Indicateurs visuels
    if is_best:
        star = " ⭐ MEILLEUR"
    else:
        star = ""

    logger.info(
        f"Ép. {epoch + 1:>4d} │ "
        f"Train Loss: {train_loss:.5f} │ "
        f"Val Loss: {val_loss:.5f} │ "
        f"Dice: {val_dice:.2f}% │ "
        f"LR: {lr:.2e}"
        f"{star}"
    )

    # Indicateur d'early stopping
    if early_stop_patience > 0 and early_stop_counter > 0:
        remaining = early_stop_patience - early_stop_counter
        bar_len = 10
        filled = int(bar_len * early_stop_counter / early_stop_patience)
        bar = "█" * filled + "░" * (bar_len - filled)
        logger.info(
            f"         │ Early stopping: [{bar}] "
            f"{early_stop_counter}/{early_stop_patience} "
            f"(encore {remaining} époque(s))"
        )


def log_training_start(
    logger: logging.Logger,
    config: dict,
    model_name: str = "",
    num_params: int = 0,
) -> None:
    """Affiche un bannière de démarrage de l'entraînement."""
    logger.info("=" * 60)
    logger.info("  DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info("=" * 60)
    if model_name:
        logger.info(f"  Modèle         : {model_name}")
    if num_params > 0:
        logger.info(f"  Paramètres     : {num_params:,}")
    
    training_cfg = config.get("training_parameters", {})
    logger.info(f"  Époques        : {training_cfg.get('num_epochs', '?')}")
    logger.info(f"  Batch size     : {config.get('dataloader', {}).get('batch_size', '?')}")
    
    optimizer_cfg = config.get("optimizer", {})
    logger.info(f"  Optimiseur     : {optimizer_cfg.get('type', '?')}")
    logger.info(f"  Learning rate  : {optimizer_cfg.get('lr', '?')}")
    
    loss_cfg = config.get("loss", config.get("loss_fn", {}))
    logger.info(f"  Loss           : {loss_cfg.get('type', loss_cfg.get('loss_type', '?'))}")
    logger.info("=" * 60)


def log_training_end(
    logger: logging.Logger,
    total_epochs: int,
    best_val_dice: float,
    best_val_loss: float,
    checkpoint_dir: str = "",
) -> None:
    """Affiche un résumé de fin d'entraînement."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ENTRAÎNEMENT TERMINÉ")
    logger.info("=" * 60)
    logger.info(f"  Époques complétées : {total_epochs}")
    logger.info(f"  Meilleur Val Loss  : {best_val_loss:.6f}")
    logger.info(f"  Meilleur Val Dice  : {best_val_dice:.2f}%")
    if checkpoint_dir:
        logger.info(f"  Checkpoints        : {checkpoint_dir}")
    logger.info("=" * 60)


def log_section(logger: logging.Logger, title: str) -> None:
    """Affiche un séparateur de section."""
    logger.info("")
    logger.info(f"{'─'*20} {title} {'─'*20}")


def log_pipeline_step(logger: logging.Logger, step_num: int, total_steps: int, description: str) -> None:
    """Affiche l'avancement d'une étape de la pipeline."""
    logger.info(f"[{step_num}/{total_steps}] {description}")


# ─── Barre de progression style TensorFlow/Keras ─────────────────────────────
class KerasProgressBar:
    """Barre de progression style TensorFlow/Keras.

    Affiche une barre de la forme :
        Epoch 3/40
        122/122 [==============================] - 45s 368ms/step - loss: 0.5234 - lr: 1.00e-04

    Usage:
        pbar = KerasProgressBar(total=len(dataloader), epoch=1, num_epochs=40)
        for i, batch in enumerate(dataloader):
            ...
            pbar.update(i + 1, {"loss": loss_val, "lr": lr_val})
        pbar.finish(extra_metrics={"val_loss": 0.45, "dice": 52.1})
    """

    BAR_LENGTH = 30  # Nombre de caractères de la barre

    def __init__(
        self,
        total: int,
        epoch: int = 0,
        num_epochs: int = 1,
        prefix: str = "",
        enabled: bool = True,
    ):
        """
        Args:
            total: Nombre total de steps (batches) dans l'époque.
            epoch: Numéro de l'époque (0-indexed).
            num_epochs: Nombre total d'époques.
            prefix: Préfixe optionnel (ex: "Train", "Val").
            enabled: Si False, n'affiche rien (pour les processus non-main).
        """
        self.total = max(total, 1)
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.prefix = prefix
        self.enabled = enabled
        self._start_time = time.time()
        self._last_print_len = 0

    def _format_time(self, seconds: float) -> str:
        """Formate un temps en string lisible."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, remainder = divmod(int(seconds), 3600)
            m, s = divmod(remainder, 60)
            return f"{h}h {m}m {s}s"

    def update(self, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """Met à jour la barre de progression.

        Args:
            step: Step courant (1-indexed).
            metrics: Dictionnaire de métriques à afficher.
        """
        if not self.enabled:
            return

        elapsed = time.time() - self._start_time
        step = min(step, self.total)

        # Construction de la barre
        ratio = step / self.total
        filled = int(self.BAR_LENGTH * ratio)
        if step < self.total:
            bar = "=" * max(filled - 1, 0) + ">" + "." * (self.BAR_LENGTH - filled)
        else:
            bar = "=" * self.BAR_LENGTH

        # Temps par step
        time_per_step = elapsed / step if step > 0 else 0
        if step < self.total:
            eta = time_per_step * (self.total - step)
            time_info = f"ETA: {self._format_time(eta)}"
        else:
            time_info = f"{self._format_time(elapsed)}"

        time_per_step_str = self._format_time(time_per_step) + "/step"

        # Métriques
        metrics_str = ""
        if metrics:
            parts = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    if abs(v) < 0.01 or abs(v) >= 1000:
                        parts.append(f"{k}: {v:.2e}")
                    else:
                        parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            metrics_str = " - " + " - ".join(parts)

        # Assemblage
        line = (
            f"\r{step:>{len(str(self.total))}}/{self.total} "
            f"[{bar}] - {time_info} - {time_per_step_str}"
            f"{metrics_str}"
        )

        # Effacer les caractères restants de la ligne précédente
        pad = max(self._last_print_len - len(line), 0)
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
        self._last_print_len = len(line)

    def finish(self, extra_metrics: Optional[Dict[str, float]] = None) -> None:
        """Finalise la barre et passe à la ligne.

        Args:
            extra_metrics: Métriques supplémentaires de fin d'époque
                           (ex: val_loss, dice) à afficher sur la même ligne.
        """
        if not self.enabled:
            return

        elapsed = time.time() - self._start_time
        time_per_step = elapsed / self.total if self.total > 0 else 0
        bar = "=" * self.BAR_LENGTH

        metrics_str = ""
        if extra_metrics:
            parts = []
            for k, v in extra_metrics.items():
                if isinstance(v, float):
                    if abs(v) < 0.01 or abs(v) >= 1000:
                        parts.append(f"{k}: {v:.2e}")
                    else:
                        parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            metrics_str = " - " + " - ".join(parts)

        line = (
            f"\r{self.total}/{self.total} "
            f"[{bar}] - {self._format_time(elapsed)} "
            f"- {self._format_time(time_per_step)}/step"
            f"{metrics_str}"
        )
        pad = max(self._last_print_len - len(line), 0)
        sys.stdout.write(line + " " * pad + "\n")
        sys.stdout.flush()

    def epoch_header(self) -> None:
        """Affiche l'en-tête d'époque style Keras."""
        if not self.enabled:
            return
        header = f"Epoch {self.epoch + 1}/{self.num_epochs}"
        if self.prefix:
            header = f"{self.prefix} - {header}"
        sys.stdout.write(f"{header}\n")
        sys.stdout.flush()
