"""
Fabrique d'architectures pour SegFormer3D.

Ce module permet de construire les modèles SegFormer3D basés sur le fichier de configuration.

Architectures supportées:
- segformer3d: Transformateur Vision Mixte 3D avec décodeur SegFormer

Exemple d'utilisation:
    config = load_config("config.yaml")
    model = build_architecture(config)
"""

######################################################################
def build_architecture(config):
    """Crée une architecture SegFormer3D basée sur le fichier de configuration.
    
    Args:
        config (dict): Dictionnaire de configuration contenant au minimum:
            - config["model"]["name"]: Nom du modèle (doit être "segformer3d")
            - config["model"]: Paramètres du modèle
    
    Returns:
        torch.nn.Module: Instance du modèle SegFormer3D configuré
    
    Raises:
        ValueError: Si le model_name n'est pas "segformer3d"
    
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
            f"Model '{model_name}' not supported. This project only supports: segformer3d"
        )
