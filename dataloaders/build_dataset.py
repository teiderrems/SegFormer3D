import sys

sys.path.append("../")

from typing import Dict
# Import build_augmentations but allow import to succeed even if MONAI is not installed
try:
    from augmentations.augmentations import build_augmentations  # type: ignore
except Exception:
    # Fallback no-op transform builder used in unit tests when MONAI is not available
    def build_augmentations(train: bool = True):
        return None


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    """Construit un dataset pour la segmentation.
    
    Args:
        dataset_type: Type de dataset ("prostate_seg")
        dataset_args: Arguments du dataset
        
    Returns:
        Dataset configuré
    """
    if dataset_type == "prostate_seg":
        from .prostate_seg import ProstateSegDataset

        dataset = ProstateSegDataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            # Appliquer les augmentations uniquement si le flag 'augmentations' est True (par défaut True pour conserver le comportement historique)
            transform=(build_augmentations(train=dataset_args["train"]) if dataset_args.get("augmentations", True) and dataset_args["train"] else None),
            split_file=dataset_args.get("split_file", None),
            target_size=dataset_args.get("target_size", 96),
            debug_augment=dataset_args.get("debug_augment", False),
        )

        # Journaliser l'état des augmentations pour faciliter le debug / reproductibilité
        aug_flag = bool(dataset_args.get("augmentations", True)) and bool(dataset_args.get("train", False))
        mode = "train" if dataset_args.get("train", False) else "val"
        print(f"[INFO] Dataset '{mode}' @ {dataset_args.get('root')} - augmentations: {'ENABLED' if aug_flag else 'DISABLED'}")

        return dataset
    else:
        raise ValueError(
            f"Dataset '{dataset_type}' non supporté. Utilisez: prostate_seg"
        )


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
):
    """Builds the dataloader for given dataset.

    This function imports MONAI's DataLoader lazily so the module can be imported
    even if MONAI is not installed (useful for unit tests). If MONAI is not
    available, it will fall back to PyTorch's DataLoader when possible.
    """
    # Import lazily to avoid hard dependency at module import time
    try:
        from monai.data import DataLoader
    except Exception:
        try:
            from torch.utils.data import DataLoader  # type: ignore
        except Exception as e:
            raise ImportError("MONAI or torch is required to build dataloaders: " + str(e))

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=False,  # Disabled to avoid memory issues
    )
    return dataloader

######################################################################
def build_dataloaders(config: Dict):
    """Build both train and validation dataloaders from config
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Handle both old and new config formats
    if "data" in config:
        # New format
        dataset_config = config.get("data", {})
        dataset_type = dataset_config.get("dataset_type", "prostate_seg")
        dataset_args_base = dataset_config.get("dataset_args", {})
        train_args = dataset_args_base.copy()
        train_args["train"] = True
        val_args = dataset_args_base.copy()
        val_args["train"] = False
    else:
        # Old format with dataset_parameters
        dataset_config = config.get("dataset_parameters", {})
        dataset_type = dataset_config.get("dataset_type", "prostate_seg")
        train_args = dataset_config.get("train_dataset_args", {})
        val_args = dataset_config.get("val_dataset_args", {})
    
    dataloader_config = config.get("dataloader", {})
    
    # Build datasets
    train_dataset = build_dataset(dataset_type, train_args)
    val_dataset = build_dataset(dataset_type, val_args)
    
    # Build dataloaders
    train_dataloader = build_dataloader(
        train_dataset,
        dataloader_config.get("train_loader", dataloader_config),
        config,
        train=True
    )
    
    val_dataloader = build_dataloader(
        val_dataset,
        dataloader_config.get("val_loader", dataloader_config),
        config,
        train=False
    )
    
    return train_dataloader, val_dataloader