import torch
import monai
import torch.nn as nn
from typing import Dict, Optional
from monai import losses


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss wrapper for semantic segmentation."""
    
    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, C, D, H, W)
            
        Returns:
            Scalar loss tensor
        """
        return self._loss(predictions, targets)


###########################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    """Binary cross-entropy with logits for binary segmentation tasks."""
    
    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE with logits loss.
        
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, C, D, H, W)
            
        Returns:
            Scalar loss tensor
        """
        return self._loss(predictions, targets)


###########################################################################
class DiceLoss(nn.Module):
    """Dice loss for volumetric segmentation."""
    
    def __init__(self) -> None:
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            predicted: Model predictions (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W)
            
        Returns:
            Scalar loss tensor
        """
        return self._loss(predicted, target)


###########################################################################
class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss for robust segmentation."""
    
    def __init__(self) -> None:
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice-CE loss.
        
        Args:
            predicted: Model predictions (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W)
            
        Returns:
            Scalar loss tensor
        """
        return self._loss(predicted, target)


###########################################################################
def build_loss_fn(loss_type: str, loss_args: Optional[Dict] = None) -> nn.Module:
    """Factory function to build loss functions.
    
    Args:
        loss_type: Type of loss function ('crossentropy', 'binarycrossentropy', 'dice', 'diceCE')
        loss_args: Additional arguments for loss function (currently unused)
        
    Returns:
        Instantiated loss module
        
    Raises:
        ValueError: If loss_type is not supported
    """
    loss_registry = {
        "crossentropy": CrossEntropyLoss,
        "binarycrossentropy": BinaryCrossEntropyWithLogits,
        "dice": DiceLoss,
        "diceCE": DiceCELoss,
    }
    
    if loss_type not in loss_registry:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. "
            f"Supported types: {list(loss_registry.keys())}"
        )
    
    return loss_registry[loss_type]()

def build_loss(config: Dict) -> nn.Module:
    """Build loss function from config dictionary.
    
    Supports both old and new config formats:
    - Old: loss section with type and args
    - New: loss_fn section with loss_type and loss_args
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Instantiated loss module
    """
    # Handle new format: loss_fn block
    if "loss_fn" in config:
        loss_config = config["loss_fn"]
        loss_type = loss_config.get("loss_type", "dice").lower()
        loss_args = loss_config.get("loss_args", {})
    else:
        # Handle old format: loss block
        loss_config = config.get("loss", {})
        loss_type = loss_config.get("type", "diceCE").lower()
        loss_args = loss_config.get("args", {})
    
    # Normalize loss type names
    loss_type_map = {
        "dice": "dice",
        "dicece": "diceCE",
        "crossentropy": "crossentropy",
        "binarycrossentropy": "binarycrossentropy",
    }
    
    normalized_type = loss_type_map.get(loss_type.lower(), loss_type)
    
    return build_loss_fn(normalized_type, loss_args)