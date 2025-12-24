"""
Fabrique d'architectures pour construire les modèles de segmentation basés sur le fichier de configuration.

Ce module permet de sélectionner l'architecture en fonction d'un fichier de configuration YAML.
Pour ajouter une nouvelle architecture, importez-la dans ce fichier et ajoutez un bloc conditionnel.

Architectures supportées:
- segformer3d: Transformateur Vision Mixte 3D avec décodeur SegFormer

Exemple d'utilisation:
    config = load_config("config.yaml")
    model = build_architecture(config)
"""

######################################################################
def build_architecture(config):
    """Crée une architecture de modèle basée sur le fichier de configuration.
    
    Args:
        config (dict): Dictionnaire de configuration contenant au minimum:
            - config["model"]["name"]: Nom du modèle (ex: "segformer3d")
            - config["model"]: Paramètres du modèle
    
    Returns:
        torch.nn.Module: Instance du modèle configuré
    
    Raises:
        ValueError: Si le model_name n'est pas supporté
    
    Exemple:
        >>> config = {"model": {"name": "segformer3d", ...}}
        >>> model = build_architecture(config)
    """
    # Support both old and new config formats
    model_name = config.get("model", {}).get("name") or config.get("model_name")
    
    if model_name == "segformer3d":
        from .segformer3d import build_segformer3d_model

        model = build_segformer3d_model(config)

        return model
    else:
        raise ValueError(
            "specified model not supported, edit build_architecture.py file"
        )
