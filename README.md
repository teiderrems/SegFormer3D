#  Segmentation 3D de la Prostate - Pipeline Automatisée



Projet complet pour la segmentation 3D de la prostate utilisant l'architecture **SegFormer3D**.



##  Vue d'ensemble



Ce projet fournit une **pipeline entièrement automatisée** allant du prétraitement des données NIfTI brutes à l'inférence finale, avec génération intelligente des splits d'entraînement/validation.



###  Fonctionnalités principales



- ** Pipeline end-to-end** : Prétraitement → Entraînement → Inférence → Visualisation

- ** Stratification intelligente** : Équilibre automatique des classes dominantes

- ** Cross-validation k-fold** : Validation rigoureuse avec stratification préservée

- ** Métriques complètes** : Dice, IoU, précision, rappel pour chaque classe

- ** Visualisations avancées** : Comparaisons, coupes 3D, statistiques détaillées

- ** Optimisation GPU** : Support multi-GPU avec DDP

- ** Suivi d'entraînement** : Logs détaillés et checkpoints automatiques

- ** Reprise d'entraînement** : Continuer exactement à l'époque d'interruption (--resume)

- ** Crop prostate** : Suppression optionnelle des slices sans prostate avant le resampling



##  Architectures disponibles



### 1. SegFormer3D

- **Type** : Architecture purement Transformer

- **Points forts** : Efficacité computationnelle, contexte global

- **Usage** : Recommandé pour datasets volumineux











## Télécharger les données brutes

Les données NIfTI non prétraitées sont disponibles ici :

**[Télécharger les données (Google Drive)](https://drive.google.com/drive/folders/1YqhVbcLjrRdp5iyTsD68H_rYCfl8S7B0?usp=sharing)**

Placez les dossiers patients téléchargés dans `data/raw_prostate/`.

## Télécharger le meilleur modèle

Le meilleur checkpoint pré-entraîné est disponible ici :

**[Télécharger le modèle (Google Drive)](https://drive.google.com/drive/folders/1ljtUTyMSudLxjsUtNhPVjZgYvwvlvlUz?usp=sharing)**

Placez les fichiers téléchargés dans le répertoire `checkpoints/`.

##  Prérequis



- **Python** : 3.8+

- **PyTorch** : 2.0+

- **CUDA** : 11.8+ (recommandé)

- **RAM** : 32GB+ pour gros volumes

- **GPU** : 8GB+ VRAM minimum



##  Installation rapide



```bash

# Cloner le repository

git clone <repository-url>

cd SegFormer3D



# Installer les dépendances

pip install -r requirements.txt

```



##  Utilisation rapide



### Pipeline complète automatisée



```bash

# Utilisation simple avec configuration par défaut (recommandé)

python pipeline.py

# Configuration haute résolution (256x256x256)

python pipeline.py --config pipeline_config_high_res.yaml

# Personnalisation via ligne de commande

python pipeline.py --target_size 256 --architectures SegFormer3D --split_type kfold

# Configuration avancée avec arguments

python pipeline.py \

    --raw_data_dir ./data/raw_prostate \

    --architectures SegFormer3D \

    --target_size 256 \

    --split_type fixed \

    --train_ratio 0.8 \

    --val_ratio 0.1 \

    --test_ratio 0.1

```

### Fine-tuning et reprise d'entraînement

```bash
# Fine-tuning à partir d'un checkpoint (reset optimiseur, repart de l'époque 0)
python pipeline.py --skip_preprocess --finetune_checkpoint checkpoints/best_model.pth

# Reprendre un entraînement interrompu (restaure modèle + optimiseur + scheduler + époque)
python pipeline.py --skip_preprocess --resume_checkpoint checkpoints/best_model.pth

# Fine-tuning en arrière-plan (sessions longues)
nohup python pipeline.py --config pipeline_config.yaml --skip_preprocess \
    --finetune_checkpoint checkpoints/best_model.pth > finetune.log 2>&1 &

# Suivre les logs en temps réel
tail -f finetune.log
```

> **Attention** : `--finetune_checkpoint` et `--resume_checkpoint` sont **mutuellement exclusifs**.
> - `--finetune_checkpoint` : charge les poids, remet optimiseur/scheduler à zéro, repart de l'époque 0.
> - `--resume_checkpoint` : restaure l'état complet (modèle, optimiseur, scheduler, métriques, époque).

> **Piège fréquent** : ne pas confondre `--checkpoints` (choisir quels modèles **inférer**) et `--finetune_checkpoint` (réentraîner depuis un checkpoint).

### Augmentations et prétraitement

```bash
# Désactiver toutes les augmentations (utile pour tests / comparaisons)
python pipeline.py --disable_augmentations

# Cropping prostate (supprimer les slices sans prostate avant resampling)
python pipeline.py --crop_to_prostate --crop_margin 3

# Skip preprocessing (données déjà prétraitées)
python pipeline.py --skip_preprocess
```

> Astuce : la pipeline détecte automatiquement `test.csv` dans `paths.preprocessed_data_dir` et sautera le prétraitement si ce fichier existe — pratique pour relancer uniquement l'inférence/visualisation sur un dataset déjà préparé.

### Inférence et visualisation

```bash
# Inférence avec des checkpoints spécifiques
python pipeline.py --skip_preprocess --checkpoints best_model final_model

# Inférence + visualisations automatiques
python pipeline.py --skip_preprocess --checkpoints best_model --visualize

# Visualisations sans volumétrique 3D (plus rapide)
python pipeline.py --skip_preprocess --checkpoints best_model --visualize --skip_volume

# Exécuter seulement l'inférence/visualisation (sauter l'entraînement)
python pipeline.py --skip_training --checkpoints best_model --visualize
```

### Priorité YAML vs CLI et option `--force-cli`

- Par défaut, les valeurs définies explicitement dans les fichiers YAML prennent **priorité** sur les arguments en ligne de commande (cela garantit des runs reproductibles et centralisés).  
- Si vous avez besoin d'overrider une valeur du YAML pour un run ponctuel (ou en CI), passez l'option CLI **et** ajoutez `--force-cli`. Exemple :

```bash
# YAML définit device=cuda, mais on force CLI -> device=cpu
python inference_simple.py --config configs/config_segformer3d.yaml --device cpu --force-cli

# YAML définit test dataset, mais on force CLI -> use the CLI dataset
python scripts/run_visualizations_all.py --test_data_dir /path/to/other --force-cli
```

> Note : `--force-cli` est disponible sur `pipeline.py`, `inference_simple.py`, `visualize_results.py` et les scripts batch (`scripts/*`).

### Comportements pratiques récents

- Prétraitement automatique : la pipeline **saute automatiquement** le prétraitement si un fichier `test.csv` est présent dans `paths.preprocessed_data_dir` (utile lorsqu'on exécute l'inférence sur un jeu déjà prétraité).  
- Sélection de checkpoint : la pipeline préfère un fichier binaire de checkpoint (`.pth`, `.pt`, `.ckpt`, `.tar`) — les fichiers d'information textuels (`best_model_info.txt`) sont ignorés pour l'inférence.  
- Nouvelle clé YAML `inference_parameters` (ex. `device`, `batch_size`, `save_predictions`, `threshold`, `verbosity`) et `visualization` (ex. `volume_vis`, `compute_errors`, `output_dir`) — configurez l'inférence/visualisation entièrement depuis le YAML si souhaité.

Pour des exemples d'exécution sur cluster (SLURM / OAR) et un script d'exemple, consultez `README_PIPELINE.md` ou utilisez `scripts/submit_oar_example.sh`.

### Référence complète des arguments `pipeline.py`

#### Configuration générale

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--config` | `str` | `./pipeline_config.yaml` | Fichier de configuration YAML |
| `--architectures` | `str` (liste) | config | Architectures à traiter |
| `--arch_config` | `str` | `None` | Config d'architecture personnalisée |

#### Chemins

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--raw_data_dir` | `str` | config | Données brutes NIfTI |
| `--preprocessed_data_dir` | `str` | config | Données prétraitées |
| `--config_dir` | `str` | config | Configurations des modèles |
| `--checkpoint_dir` | `str` | config | Checkpoints |
| `--results_dir` | `str` | config | Résultats d'inférence |
| `--test_data_dir` | `str` | config | Données de test |

#### Prétraitement

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--skip_preprocess` | flag | `false` | Sauter le prétraitement |
| `--target_size` | `int` | config | Taille des volumes (256, 256, 256) |
| `--crop_to_prostate` | flag | `false` | Supprimer les slices sans prostate |
| `--crop_margin` | `int` | `2` | Marge de slices autour de la prostate |

#### Splits

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--split_type` | `fixed`/`kfold` | config | Type de split |
| `--train_ratio` | `float` | config | Ratio entraînement (split fixed) |
| `--val_ratio` | `float` | config | Ratio validation (split fixed) |
| `--test_ratio` | `float` | config | Ratio test (split fixed) |
| `--k_folds` | `int` | config | Nombre de folds (split kfold) |
| `--random_seed` | `int` | config | Seed reproductibilité |

#### Entraînement

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--finetune_checkpoint` | `str` | `None` | Fine-tuning (reset optimiseur, époque 0) |
| `--resume_checkpoint` | `str` | `None` | Reprise exacte (restaure tout) |
| `--disable_augmentations` | flag | `false` | Désactiver les augmentations |
| `--skip_training` | flag | `false` | **Sauter l'entraînement** — n'exécute que l'inférence et les visualisations (utile pour évaluer des checkpoints existants) |

#### Inférence et visualisation

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--checkpoints` | `str` (liste) | `best_model final_model` | Checkpoints à inférer |
| `--visualize` | flag | `false` | Générer les visualisations après inférence |
| `--skip_volume` | flag | `false` | Ignorer visualisations 3D |
| `--vis_timeout` | `int` | `600` | Timeout par patient (secondes) || `--save_nifti` | flag | `true` (via YAML) | Sauvegarder la prédiction au format `prediction_*.nii.gz` (nécessite `metadata['original_affine']` et `nibabel`) |
### Utilisation manuelle par architecture



```bash

# 1. Prétraitement avec taille personnalisée

cd data/prostate_raw_data

python prostate_preprocess.py --input_dir ../ --output_dir ../prostate_preprocessed --target_size 256



# 2. Génération des splits

python create_prostate_splits.py --input_dir ../prostate_preprocessed --stratified true



# 3. Entraînement
cd ../..
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml

# Fine-tuning depuis un checkpoint (charge les poids, reset optimiseur)
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --checkpoint checkpoints/best_model.pth

# Reprendre un entraînement interrompu (restaure tout)
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --resume checkpoints/best_model.pth

# Entraînement local non-DDP
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --local_rank 0
```

**Correspondance des arguments pipeline / trainer :**

| `pipeline.py` | `trainer_ddp.py` | Effet |
|----------------|-------------------|-------|
| `--finetune_checkpoint` | `--checkpoint` | Fine-tuning (reset optimiseur, époque 0) |
| `--resume_checkpoint` | `--resume` | Reprise exacte (restaure tout) |

```bash

# 4. Inférence

- Inference manuelle (par patient)

```bash
python inference_simple.py --checkpoint ./checkpoints/best_model.pth --input_dir data/preprocessed_data_240_240_240/patient_002 --output_dir results/SegFormer3D/patient_002
```

- Inference batch (auto détecte `test.csv`, config et checkpoint si présents) :

```bash
python scripts/run_inference_all.py --verbosity normal
```

- Via la pipeline (exécute les checkpoints demandés et place les résultats dans `results/SegFormer3D/<checkpoint>/`):

```bash
python pipeline.py --checkpoints best_model final_model
```

# 5. Visualisations batch (métriques + images)

Après avoir généré les prédictions, vous pouvez lancer toutes les visualisations et calculer les métriques pour l'ensemble du `test.csv` grâce au script batch :

```bash
# Visualisations pour les prédictions présentes dans results/SegFormer3D (par défaut cherche le CSV de test)
python scripts/run_visualizations_all.py --verbosity normal
```

Options utiles :

- `--results_subdir <name>` : utiliser un sous-dossier sous `results/SegFormer3D` (ex: `best_model`, `final_model`) pour rechercher les prédictions
- `--vis_tag <name>` : nommer le dossier de visualisations généré sous `visualizations/SegFormer3D/<name>/`
- `--skip_volume` : ignorer la visualisation volumétrique 3D pour un traitement plus rapide
- `--timeout <s>` : timeout en secondes par patient (0 = pas de timeout)

```

## Inférence sur les datasets de test avec les checkpoints best_model et final_model

Pour effectuer l'inférence sur les datasets de test en utilisant les checkpoints `best_model` et `final_model` :

```bash
# Inférence avec le checkpoint best_model
python pipeline.py --checkpoints best_model

# Inférence avec le checkpoint final_model
python pipeline.py --checkpoints final_model

# Inférence avec les deux checkpoints
python pipeline.py --checkpoints best_model final_model
```

Les résultats seront sauvegardés dans `results/SegFormer3D/best_model/` et `results/SegFormer3D/final_model/` respectivement.

## Visualisation des résultats de l'inférence avec calcul des métriques

Après l'inférence, pour visualiser les résultats et calculer les métriques :

```bash
# Visualisation pour best_model
python scripts/run_visualizations_all.py --results_subdir best_model --vis_tag best_model --verbosity normal

# Visualisation pour final_model
python scripts/run_visualizations_all.py --results_subdir final_model --vis_tag final_model --verbosity normal
```

Les visualisations incluront les métriques de segmentation (Dice, IoU, précision, rappel) et des images comparatives pour chaque patient du dataset de test.

```



##  Configuration



### Fichier de configuration principal



Le fichier `pipeline_config.yaml` définit tous les paramètres configurables :



```yaml

preprocessing:

  target_size: 256          # Taille des volumes (64, 256, 256, 256)

  normalize_method: "minmax"

  skip_existing: true

  crop_to_prostate: false  # Supprimer les slices sans prostate avant resampling

  crop_margin: 2           # Nombre de slices de marge autour de la prostate



splits:

  split_type: "fixed"      # 'fixed' ou 'kfold'

  train_ratio: 0.8

  val_ratio: 0.2

  k_folds: 5

  stratified: true         # Stratification par classe dominante

  random_seed: 42



architectures:

  enabled: ["SegFormer3D"]


# Contrôle global des augmentations
augmentations:

  enabled: true            # Activer/désactiver les augmentations globalement (peut être surchargé par config d'architecture)

Note: Vous pouvez désactiver temporairement toutes les augmentations lors de l'exécution de la pipeline via l'argument CLI `--disable_augmentations`. Les augmentations appliquées sont définies par chaque configuration d'architecture (`dataset_parameters.*.augmentations`).

Vous pouvez spécifier un répertoire dédié pour les données de test via `paths.test_data_dir` dans `pipeline_config.yaml` ou en passant `--test_data_dir /chemin/vers/test` à `pipeline.py`. Si `test_data_dir` n'est pas fourni, la pipeline utilise `paths.preprocessed_data_dir` par défaut.

```



### Configurations prédéfinies



- `pipeline_config.yaml` : Configuration standard (256×256×256)

- `pipeline_config_high_res.yaml` : Haute résolution (256×256×256) avec cross-validation



### Personnalisation



```bash

# Modifier la taille des volumes

python pipeline.py --target_size 256



# Changer le type de split

python pipeline.py --split_type kfold --k_folds 3



# Utiliser une configuration personnalisée

python pipeline.py --config ./my_config.yaml

```



##  Structure du projet



```

.

├── pipeline.py                    #  Pipeline principale automatisée

├── configs/                       #  Configurations YAML commentées

├── run_pipeline.*                 #  Scripts d'exécution cross-platform

├── preprocessed_data/             #  Données prétraitées (auto-généré)

├── results/                       #  Résultats d'inférence par architecture

├── visualizations/                #  Visualisations comparatives

├── README_PIPELINE.md             #  Documentation détaillée pipeline

├── architectures/                 #  Code source des architectures (SegFormer3D)

│   ├── visualize_results.py       #  Script de visualisation

│   ├── inference_simple.py        #  Script d'inférence

│   ├── create_prostate_splits.py  #  Génération splits stratifiés

│   ├── prostate_preprocess.py     #  Prétraitement NIfTI

│   └── ...

```



##  Données et formats



### Format d'entrée

```

data/raw_prostate/

├── patient_001/

│   ├── T2.nii.gz          # Séquence T2

│   ├── ADC.nii.gz         # Carte ADC

│   └── segmentation.nii.gz # Labels (0=fond, 1=prostate, 2=bandelettes)

```



### Format de sortie

- **Prétraitées** : Tensors PyTorch (.pt) 256×256×256

- **Splits** : CSV stratifiés avec équilibrage des classes

- **Résultats** : Prédictions + métriques détaillées



##  Métriques et évaluation



### Métriques calculées

- **Dice Score** : Similarité volumétrique (0-1)

- **IoU (Jaccard)** : Intersection sur union (0-1)

- **Précision/Rappel** : Par classe et global

- **F1-Score** : Moyenne harmonique précision/rappel



### Classes évaluées

- **Classe 0** : Fond (background)

- **Classe 1** : Prostate

- **Classe 2** : Bandelettes urétrales



##  Visualisations et analyse des résultats



### Types de visualisations disponibles



Le projet inclut des outils avancés de visualisation pour analyser les performances des modèles :



#### 1. Comparaisons visuelles (`*_comparison.png`)

- **Overlay prédictions** : Superposition des prédictions sur les images originales

- **Labels ground truth** : Comparaison directe avec les annotations réelles

- **Différences colorées** : Mise en évidence des erreurs de segmentation



#### 2. Coupes axiales (`*_axial_slices.png`)

- **Multiples tranches** : Visualisation de différentes profondeurs du volume 3D

- **Comparaisons côte à côte** : Prédiction vs Ground truth vs Image originale

- **Navigation volumétrique** : Exploration interactive des résultats



#### 3. Statistiques détaillées (`*_statistics.png`)

- **Métriques par classe** : Dice, IoU, précision, rappel

- **Distributions** : Analyse des classes prédites vs réelles



#### 4. Visualisation volumétrique 3D (`*_volume_3d.png`)

- **Représentation 3D** : Visualisation des volumes segmentés en 3D

- **Comparaison prédiction/ground truth** : Côté à côté des volumes

- **Sous-échantillonnage** : Optimisé pour la performance



#### 5. Visualisation volumétrique interactive (`*_volume_3d_interactive.html`)

- **Navigation interactive** : Zoom, rotation, panoramique dans le navigateur

- **Plotly** : Graphiques 3D interactifs avec tooltips

- **Comparaison côte à côte** : Prédiction vs Vérité terrain

- **Ouverture facile** : Double-clic sur le fichier HTML



#### 6. Erreurs de reconstruction / segmentation

- **Métriques détaillées** : Dice, IoU, précision, rappel, RMSE

- **Métriques avancées** : Hausdorff, HD95, ASSD, SSIM (par classe) — distances en voxels par défaut

- **Graphiques** : Dice/IoU par classe, erreur slice-wise (MAE), overlay d'erreurs

- **Utilisation** : `--compute_errors` pour générer `*_errors.json`, `*_errors.png`, `*_slice_errors.png`, `*_error_overlay.png`

- **Spacing optionnel** : `--voxel_spacing sx,sy,sz` pour obtenir les distances en mm (x,y,z)



### Utilisation des visualisations



```bash



# Visualisation pour SegFormer3D

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_256_256_256/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001



# Avec visualisation volumétrique 3D (optionnel)

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_256_256_256/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001 \

    --volume_vis



# Avec visualisation volumétrique interactive (HTML)

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_256_256_256/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001 \

    --interactive

```

```



### Structure des visualisations générées



```

visualizations/

├── SegFormer3D/

│   ├── patient_001/

│   │   ├── patient_001_comparison.png     # Comparaisons overlay

│   │   ├── patient_001_axial_slices.png   # Coupes axiales

│   │   └── patient_001_statistics.png     # Métriques détaillées

│   └── patient_002/

│       └── ...



```



##  Fonctionnalités avancées



### Stratification intelligente

- Analyse automatique des classes dominantes par patient

- Équilibrage train/validation pour éviter les biais

- Support cross-validation avec stratification préservée



### Optimisations

- **Mixed Precision** : Entraînement FP16 pour 2x plus rapide

- **Gradient Accumulation** : Batch virtuel pour gros modèles

- **Early Stopping** : Arrêt automatique sur plateau

- **Multi-GPU** : DDP pour entraînement distribué

- **Reprise d'entraînement** : `--resume` pour continuer à l'époque exacte d'interruption (restaure modèle, optimiseur, scheduler, métriques)



##  Résultats typiques



| Architecture | Dice Prostate | Dice Bandelettes | Temps entraînement |

|-------------|---------------|------------------|-------------------|

| SegFormer3D | 0.89 ± 0.03  | 0.76 ± 0.05     | ~4h              |





*Résultats sur dataset interne, 5-fold cross-validation*



## 🆕 Dernières améliorations



### Version actuelle : Pipeline professionnelle complète avec visualisation



-  **Système de configuration YAML avancé**

  - Fichier `pipeline_config.yaml` pour tous les paramètres

  - Configuration haute résolution `pipeline_config_high_res.yaml`

  - Paramètres remplaçables via ligne de commande

  - Taille des volumes configurable (64, 256, 256, 256 voxels)



-  **Génération automatique des splits CSV** avec stratification par classe dominante

-  **Support cross-validation k-fold** intégré dans la pipeline

-  **Scripts d'inférence unifiés** pour toutes les architectures

-  **Outils de visualisation avancés** pour analyse des résultats

  - Comparaisons visuelles prédiction vs ground truth

  - Coupes axiales interactives

  - Statistiques détaillées par patient

-  **Configurations YAML entièrement commentées**

-  **Code nettoyé professionnellement** (suppression de tous les emojis)

-  **Documentation enrichie** dans tous les README

-  **Pipeline end-to-end** : du prétraitement NIfTI à la visualisation finale



### Améliorations récentes



- **Configuration flexible** : YAML + arguments ligne de commande

- **Tailles de volumes configurables** : de 64×64×64 à 256×256×256

- **Stratification intelligente** : Analyse des tensors PyTorch pour équilibrer les classes

- **Flexibilité maximale** : Support splits fixes et cross-validation

- **Reproductibilité garantie** : Seeds configurables pour résultats cohérents

- **Reprise d'entraînement** : `--resume` restaure modèle, optimiseur, scheduler et métriques pour continuer à l'époque exacte d'interruption

- **Crop prostate** : `--crop_to_prostate` supprime les slices axiales sans prostate avant le resampling, concentrant l'information utile

- **Commentaires détaillés** : Docstrings complètes dans le code d'entraînement (`trainer_ddp.py`)



##  Contribution



1. Fork le projet

2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)

3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)

4. Pushez vers la branche (`git push origin feature/AmazingFeature`)

5. Ouvrez une Pull Request



##  Citation



Si vous utilisez ce code dans vos recherches, veuillez citer :



```bibtex

@software{segformer3d_pipeline,

  title = {Pipeline automatisée de segmentation 3D prostate},

  author = {Votre Nom},

  year = {2024},

  url = {https://github.com/username/SegFormer3D}

}

```



##  Support



- **Issues** : [GitHub Issues](https://github.com/username/SegFormer3D/issues)

- **Documentation** : Voir `README_PIPELINE.md` et les README des architectures

- **Exemples** : Notebooks dans `notebooks/`



---



** Pour lancer les tests** : `pip install -r requirements-dev.txt && pytest -q`

** Prêt à segmenter !** Lancez `python pipeline.py --help` pour commencer.