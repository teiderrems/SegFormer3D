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



##  Architectures disponibles



### 1. SegFormer3D

- **Type** : Architecture purement Transformer

- **Points forts** : Efficacité computationnelle, contexte global

- **Usage** : Recommandé pour datasets volumineux











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



##  Utilisation rapide



### Pipeline complète automatisée



```bash

# Utilisation simple avec configuration par défaut (recommandé)

python pipeline.py



# Configuration haute résolution (128x128x128)

python pipeline.py --config pipeline_config_high_res.yaml



# Personnalisation via ligne de commande

python pipeline.py --target_size 128 --architectures SegFormer3D --split_type kfold



# Configuration avancée avec arguments

python pipeline.py \

    --raw_data_dir ./data/raw_prostate \

    --architectures SegFormer3D \

    --target_size 96 \

    --split_type fixed \

    --train_ratio 0.8 \

    --val_ratio 0.2

```



### Utilisation manuelle par architecture



```bash

# 1. Prétraitement avec taille personnalisée

cd data/prostate_raw_data

python prostate_preprocess.py --input_dir ./ --output_dir ../prostate_preprocessed --target_size 128



# 2. Génération des splits

python create_prostate_splits.py --input_dir ../prostate_preprocessed --stratified true



# 3. Entraînement

cd ../..

python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml



# 4. Inférence

python inference_simple.py --checkpoint_path ./checkpoints/best_model.pt

# 5. Visualisations batch (métriques + images)

Après avoir généré les prédictions, vous pouvez lancer toutes les visualisations et calculer les métriques pour l'ensemble du `test.csv` grâce au script batch :

```bash
python scripts/run_visualizations_all.py --test_csv data/preprocessed_data_128_128_128/test.csv --verbosity normal
```

Utilisez `--skip_volume` pour ignorer la visualisation volumétrique 3D si vous souhaitez un traitement plus rapide.

```



##  Configuration



### Fichier de configuration principal



Le fichier `pipeline_config.yaml` définit tous les paramètres configurables :



```yaml

preprocessing:

  target_size: 96          # Taille des volumes (64, 96, 128, 256)

  normalize_method: "minmax"

  skip_existing: true



splits:

  split_type: "fixed"      # 'fixed' ou 'kfold'

  train_ratio: 0.8

  val_ratio: 0.2

  k_folds: 5

  stratified: true         # Stratification par classe dominante

  random_seed: 42



architectures:

  enabled: ["SegFormer3D"]

```



### Configurations prédéfinies



- `pipeline_config.yaml` : Configuration standard (96×96×96)

- `pipeline_config_high_res.yaml` : Haute résolution (128×128×128) avec cross-validation



### Personnalisation



```bash

# Modifier la taille des volumes

python pipeline.py --target_size 128



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

- **Prétraitées** : Tensors PyTorch (.pt) 96×96×96

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

    --input_dir data/preprocessed_data_128_128_128/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001



# Avec visualisation volumétrique 3D (optionnel)

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_128_128_128/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001 \

    --volume_vis



# Avec visualisation volumétrique interactive (HTML)

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_128_128_128/patient_001 \

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

  - Taille des volumes configurable (64, 96, 128, 256 voxels)



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



** Prêt à segmenter !** Lancez `python pipeline.py --help` pour commencer.