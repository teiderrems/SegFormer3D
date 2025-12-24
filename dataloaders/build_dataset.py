import sys

sys.path.append("../")

from typing import Dict
from monai.data import DataLoader
from augmentations.augmentations import build_augmentations


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    if dataset_type == "brats2021_seg":
        from .brats2021_seg import Brats2021Task1Dataset

        dataset = Brats2021Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    elif dataset_type == "brats2017_seg":
        from .brats2017_seg import Brats2017Task1Dataset

        dataset = Brats2017Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    elif dataset_type == "prostate_seg":
        from .prostate_seg import ProstateSegDataset

        dataset = ProstateSegDataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            split_file=dataset_args.get("split_file", None),
            target_size=dataset_args.get("target_size", 96),
        )
        return dataset
    else:
        raise ValueError(
            "Formats supportés: brats2021_seg, brats2017_seg, prostate_seg"
        )


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
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