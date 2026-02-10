# Pipeline Automatisée d'Entraînement et Test



Ce script automatise la pipeline complète allant du prétraitement des données à l'inférence sur les données de test pour l'architecture SegFormer3D.



## Structure du Projet



```

.

├── pipeline.py                    # Script principal de la pipeline

├── configs/                       # Configurations pour chaque architecture

│   ├── config_segformer3d.yaml

├── architectures/                 # Code source des architectures (SegFormer3D)

└── train_scripts/                 # Scripts d'entraînement centralisés

```



## Prérequis



- Python 3.8+

- PyTorch

- Les dépendances de chaque architecture (voir requirements.txt dans chaque dossier)

- Données brutes au format NIfTI (.nii.gz)



## Utilisation



### 1. Préparation des Données



Placez vos données brutes dans un répertoire avec la structure suivante :



```

raw_data/

├── patient_001/

│   ├── T2.nii.gz

│   └── segmentation.nii.gz  # ou prostate.nii.gz + bandelettes.nii.gz

├── patient_002/

│   └── ...

```



### 2. Lancement de la Pipeline



```bash

# Pipeline complète (split train/val stratifié recommandé)
python pipeline.py \
    --config ./pipeline_config.yaml \
    --raw_data_dir /path/to/raw_data \
    --architectures SegFormer3D \
    --preprocessed_data_dir ./preprocessed_data \
    --config_dir ./configs \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --split_type fixed \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --test_ratio 0.0 \
    --random_seed 42 \
    --target_size 96





# Cross-validation 5-fold (stratifié)
python pipeline.py \
    --config ./pipeline_config_high_res.yaml \
    --raw_data_dir /path/to/raw_data \
    --architectures SegFormer3D \
    --preprocessed_data_dir ./preprocessed_data_128_128_128 \
    --config_dir ./configs \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --split_type kfold \
    --k_folds 5 \
    --random_seed 42 \
    --target_size 128





# Pipeline rapide (skip preprocessing si déjà fait)

python pipeline.py \

    --raw_data_dir /path/to/raw_data \

    --skip_preprocess \

    --split_type fixed


# Inférence et visualisations (sans prétraitement)

python pipeline.py \
    --skip_preprocess \
    --checkpoints best_model final_model \
    --visualize \
    --skip_volume

# Fine-tuning à partir d'un checkpoint existant

python pipeline.py \
    --finetune_checkpoint ./checkpoints/SegFormer3D/best_model.pth \
    --target_size 128

# Reprendre un entraînement interrompu (à l'époque exacte d'interruption)
# Restaure : modèle, optimiseur, scheduler, métriques, numéro d'époque
python pipeline.py \
    --skip_preprocess \
    --resume_checkpoint ./checkpoints/best_model.pth

# Prétraitement avec cropping prostate (supprime les slices sans prostate)
python pipeline.py \
    --crop_to_prostate \
    --crop_margin 3
```

### Utilisation via Makefile

Le dépôt fournit un `Makefile` pratique pour automatiser les tâches courantes. Deux cibles utiles pour les visualisations :

- `visualize-config` : exécute `scripts/run_visualizations_all.py` en utilisant explicitement un fichier de configuration YAML.

  Exemple :

  ```bash
  make visualize-config CONFIG=configs/config_segformer3d.yaml RESULTS_SUBDIR=best_model VIS_TAG=best_model
  ```

- `visualize-test` : exécute les visualisations pour un répertoire prétraité donné (doit contenir `test.csv`).

  Exemple :

  ```bash
  make visualize-test TEST_DATA_DIR=/path/to/preprocessed_data_240_240_240 RESULTS_SUBDIR=best_model VIS_TAG=best_model
  ```

Ces cibles facilitent les runs reproductibles et l'intégration dans des scripts CI. Si tu veux, je peux ajouter une cible `make ci-visualize` qui exécute une validation rapide de visualisation dans un environnement CI (nécessite un dataset de test ou un mock).


### Comparaison : Exécution avec / sans augmentations

Voici un exemple simple pour comparer deux runs identiques, l’un avec augmentations (comportement par défaut) et l’autre sans augmentations. L’astuce est d’écrire les résultats dans des sous-répertoires distincts pour pouvoir comparer facilement les visualisations et métriques.

1) Run avec augmentations (par défaut) :

```bash
# Entraînement + inférence (résultats -> results/with_aug)
python pipeline.py --config pipeline_config.yaml --results_dir ./results/with_aug
```

2) Run sans augmentations :

```bash
# Désactiver globalement les augmentations et écrire les résultats séparément
python pipeline.py --config pipeline_config.yaml --disable_augmentations --results_dir ./results/no_aug
```

3) Générer les visualisations pour chaque jeu de résultats et comparer :

```bash
# Visualisations pour le run avec augmentations
python scripts/run_visualizations_all.py --test_data_dir ./data/preprocessed_data_240_240_240 --results_subdir with_aug --vis_tag with_aug

# Visualisations pour le run sans augmentations
python scripts/run_visualizations_all.py --test_data_dir ./data/preprocessed_data_240_240_240 --results_subdir no_aug --vis_tag no_aug
```

4) Comparer les dossiers `visualizations/SegFormer3D/with_aug/` et `visualizations/SegFormer3D/no_aug/` (graphs, `summary_metrics.json`, images comparatives) pour analyser l’effet des augmentations.


### Référence complète des arguments de `pipeline.py`

Tous les arguments CLI de `pipeline.py`. Chaque argument surcharge la valeur correspondante dans le fichier de configuration YAML.

#### Configuration générale

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--config` | `str` | `./pipeline_config.yaml` | Fichier de configuration YAML de la pipeline |
| `--architectures` | `str` (liste) | config | Liste des architectures à traiter (ex: `--architectures SegFormer3D`) |
| `--arch_config` | `str` | `None` | Fichier de configuration d'architecture à utiliser au lieu de `configs/config_<arch>.yaml` |

#### Chemins des répertoires

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--raw_data_dir` | `str` | config | Répertoire des données brutes NIfTI |
| `--preprocessed_data_dir` | `str` | config | Répertoire pour les données prétraitées |
| `--config_dir` | `str` | config | Répertoire des fichiers de configuration des modèles |
| `--checkpoint_dir` | `str` | config | Répertoire pour sauvegarder les checkpoints |
| `--results_dir` | `str` | config | Répertoire des résultats d'inférence |
| `--test_data_dir` | `str` | config | Répertoire des données de test (si différent des données prétraitées) |

#### Prétraitement

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--skip_preprocess` | flag | `false` | Sauter l'étape de prétraitement (si déjà fait) |
| `--target_size` | `int` | config | Taille cible pour le resampling des volumes (ex: 96, 128, 256) |
| `--crop_to_prostate` | flag | `false` | Supprimer les slices axiales sans prostate avant le resampling |
| `--crop_margin` | `int` | `2` | Nombre de slices de marge autour de la prostate lors du cropping |

#### Splits d'entraînement

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--split_type` | `fixed` ou `kfold` | config | Type de split des données |
| `--train_ratio` | `float` | config (0.6) | Ratio pour l'ensemble d'entraînement (seulement `split_type=fixed`) |
| `--val_ratio` | `float` | config (0.2) | Ratio pour l'ensemble de validation (seulement `split_type=fixed`) |
| `--test_ratio` | `float` | config (0.2) | Ratio pour l'ensemble de test (seulement `split_type=fixed`) |
| `--k_folds` | `int` | config (5) | Nombre de folds pour la cross-validation (seulement `split_type=kfold`) |
| `--random_seed` | `int` | config (42) | Graine aléatoire pour la reproductibilité |

#### Entraînement et checkpoints

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--finetune_checkpoint` | `str` | `None` | Checkpoint pour le **fine-tuning** : charge les poids du modèle, remet l'optimiseur et le scheduler à zéro, repart de l'époque 0 |
| `--resume_checkpoint` | `str` | `None` | Checkpoint pour la **reprise d'entraînement** : restaure modèle + optimiseur + scheduler + métriques + numéro d'époque |
| `--disable_augmentations` | flag | `false` | Désactiver les augmentations de données pendant l'entraînement (surcharge `augmentations.enabled` dans la config) |

> **Attention** : `--finetune_checkpoint` et `--resume_checkpoint` sont **mutuellement exclusifs**. Utilisez `--finetune_checkpoint` pour réentraîner un modèle depuis un autre jeu de données, et `--resume_checkpoint` pour continuer un entraînement interrompu.

#### Inférence

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--checkpoints` | `str` (liste) | `best_model final_model` | Liste des noms de checkpoints pour lesquels lancer l'inférence (ex: `--checkpoints best_model final_model`). La pipeline cherche les fichiers `.pth` correspondants dans `checkpoint_dir` |

> **Piège fréquent** : `--checkpoints` sert uniquement à choisir quels modèles **inférer**, ce n'est PAS l'option pour le fine-tuning. Pour du fine-tuning, utilisez `--finetune_checkpoint`.

#### Visualisation

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--visualize` | flag | `false` | Générer les visualisations et métriques automatiquement après chaque inférence |
| `--skip_volume` | flag | `false` | Ignorer les visualisations volumétriques 3D (plus rapide) |
| `--vis_timeout` | `int` | `600` | Timeout (en secondes) pour la visualisation de chaque patient |

---

### Référence des arguments de `train_scripts/trainer_ddp.py`

Le script d'entraînement peut aussi être appelé directement (hors pipeline) :

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--config` | `str` | **requis** | Chemin vers le fichier de configuration YAML de l'architecture |
| `--local_rank` | `int` | `-1` | Rang local pour l'entraînement distribué (géré automatiquement par `torchrun`) |
| `--checkpoint` | `str` | `None` | Checkpoint pour le fine-tuning (charge les poids, reset optimiseur) |
| `--resume` | `str` | `None` | Checkpoint pour reprendre l'entraînement (restaure tout) |

> **Note** : `--checkpoint` et `--resume` sont **mutuellement exclusifs**, comme leurs équivalents dans `pipeline.py`.

**Correspondance pipeline / trainer :**

| `pipeline.py` | `trainer_ddp.py` | Effet |
|----------------|-------------------|-------|
| `--finetune_checkpoint` | `--checkpoint` | Fine-tuning (reset optimiseur, époque 0) |
| `--resume_checkpoint` | `--resume` | Reprise exacte (restaure tout) |

---

### Exécution directe de l'entraînement

```bash
# Entraînement standard
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml

# Entraînement local non-DDP (passer --local_rank 0)
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --local_rank 0

# Fine-tuning depuis un checkpoint (charge les poids, reset optimiseur)
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --checkpoint checkpoints/best_model.pth

# Reprendre un entraînement interrompu (restaure modèle + optimiseur + scheduler + époque)
python train_scripts/trainer_ddp.py --config configs/config_segformer3d.yaml --resume checkpoints/best_model.pth
```

### Exécution en arrière-plan (nohup)

Pour les entraînements longs, utilisez `nohup` pour détacher le processus du terminal :

```bash
# Fine-tuning en arrière-plan
nohup python pipeline.py --config pipeline_config.yaml --skip_preprocess \
    --finetune_checkpoint checkpoints/best_model.pth > finetune.log 2>&1 &

# Reprise d'entraînement en arrière-plan
nohup python pipeline.py --config pipeline_config.yaml --skip_preprocess \
    --resume_checkpoint checkpoints/best_model.pth > resume.log 2>&1 &

# Pipeline complète en arrière-plan
nohup python pipeline.py --config pipeline_config.yaml > pipeline.log 2>&1 &

# Suivre les logs en temps réel
tail -f finetune.log
```

### Utilisation du Makefile

Le `Makefile` fournit des cibles pratiques pour les tâches courantes :

```bash
make help                    # Afficher toutes les cibles disponibles

# Installation
make install                 # Installer les dépendances de production
make install-dev             # Installer les dépendances de développement

# Pipeline complète
make run-pipeline            # Pipeline standard (pipeline_config.yaml)
make run-pipeline-highres    # Pipeline haute résolution (pipeline_config_high_res.yaml)

# Étapes individuelles
make preprocess              # Prétraiter les données brutes
make splits                  # Générer les splits CSV stratifiés
make train                   # Entraînement DDP
make train-local             # Entraînement local (non-DDP, --local_rank 0)

# Inférence
make infer CHECKPOINT=checkpoints/best_model.pth   # Inférence avec un checkpoint spécifique
make infer-all               # Inférence batch (scripts/run_inference_all.py)

# Visualisations
make visualize               # Visualisations par défaut
make visualize-test TEST_DATA_DIR=/path/to/data RESULTS_SUBDIR=best_model VIS_TAG=best_model
make visualize-config CONFIG=configs/config_segformer3d.yaml RESULTS_SUBDIR=best_model VIS_TAG=best_model

# Tests et qualité
make test                    # Suite de tests complète (pytest)
make test-fast               # Tests rapides
make lint                    # Linting (ruff)
make format                  # Formatage du code (ruff)
make clean                   # Nettoyer les caches
```

Variables configurables du Makefile :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `PY` | `python` | Interpréteur Python |
| `CONFIG` | `configs/config_segformer3d.yaml` | Config d'architecture |
| `ARCH` | `SegFormer3D` | Architecture active |
| `PREP_INPUT` | `$(PWD)/data/raw_prostate` | Répertoire des données brutes |
| `PREP_OUTPUT` | `$(PWD)/data/prostate_preprocessed` | Répertoire de sortie prétraitement |
| `CHECKPOINT_DIR` | `$(PWD)/checkpoints` | Répertoire des checkpoints |
| `RESULTS_DIR` | `$(PWD)/results` | Répertoire des résultats |
| `TARGET_SIZE` | `96` | Taille cible pour le resampling |

---

**Remarque importante** : installez d'abord PyTorch et les dépendances (voir `requirements.txt`) adaptées à votre configuration CUDA/Python :

```bash
pip install -r requirements.txt
# ou installez torch explicitement avec la bonne roue CUDA, par ex.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- `--test_data_dir` peut aussi être passé à `scripts/run_visualizations_all.py` (ou `--test_dir`/`--test_csv`) pour contrôler quelles données sont utilisées pour les visualisations.
- `--config` sur `scripts/run_visualizations_all.py` : permet de fournir explicitement le fichier YAML d'architecture à utiliser pour détecter `test_dataset_args` (ex: `--config configs/config_segformer3d.yaml`).



## Configuration de la Pipeline



### Fichier de configuration YAML



La pipeline utilise un fichier de configuration YAML (`pipeline_config.yaml`) pour définir tous les paramètres configurables :



```yaml

# Configuration de la pipeline automatisée

preprocessing:

  target_size: 96          # Taille des volumes (64, 96, 128, 256)

  normalize_method: "minmax"  # 'minmax' ou 'zscore'

  skip_existing: true      # Sauter les patients déjà prétraités

  crop_to_prostate: false  # Supprimer les slices sans prostate avant resampling

  crop_margin: 2           # Nombre de slices de marge autour de la prostate



splits:

  split_type: "fixed"      # 'fixed' ou 'kfold'

  train_ratio: 0.8         # Pour split_type='fixed'

  val_ratio: 0.2           # Pour split_type='fixed'

  k_folds: 5               # Pour split_type='kfold'

  stratified: true         # Stratification par classe dominante

  random_seed: 42          # Pour reproductibilité



architectures:

  enabled: ["SegFormer3D"]



paths:

  raw_data_dir: "./data/raw_prostate"

  preprocessed_data_dir: "./preprocessed_data"

  config_dir: "./configs"

  checkpoint_dir: "./checkpoints"

  results_dir: "./results"

```



### Utilisation avec configuration



```bash

# Utiliser la configuration par défaut

python pipeline.py



# Spécifier un fichier de configuration personnalisé

python pipeline.py --config ./my_config.yaml



# Remplacer certains paramètres via ligne de commande

python pipeline.py \
    --config ./my_config.yaml \
    --raw_data_dir /path/to/raw_data \
    --preprocessed_data_dir ./preprocessed_data \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --split_type fixed \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --random_seed 42 \
    --target_size 128 \
    --architectures SegFormer3D

```



## Étapes de la Pipeline



### 1. Prétraitement

- Conversion NIfTI → PyTorch tensors

- Suppression optionnelle des slices sans prostate (`crop_to_prostate`)

- Resampling à 96x96x96

- Normalisation des intensités

- Sauvegarde au format .pt



### 2. Génération automatique des splits CSV

La pipeline génère automatiquement les fichiers train.csv et validation.csv (ou les fichiers de cross-validation) en utilisant une **stratification intelligente par classe dominante** :



- **Analyse des données** : Le script examine les tensors PyTorch pour déterminer la classe la plus représentée dans chaque patient

- **Équilibrage automatique** : Les splits sont créés pour maintenir un équilibre entre les classes dans train et validation

- **Support cross-validation** : Option pour générer k folds avec stratification préservée



**Avantages de la stratification :**

- Évite les déséquilibres de classe entre train/validation

- Améliore la stabilité de l'entraînement

- Donne des métriques plus représentatives



### 3. Entraînement

- Chargement de la configuration

- Construction du modèle

- Entraînement avec les paramètres spécifiés

- Sauvegarde des checkpoints

- **Augmentations** : Par défaut, les augmentations de données sont **activées** pour l'entraînement et **désactivées** pour la validation (contrôlées via `dataset_parameters.*.augmentations` dans les fichiers de config d'architecture). Vous pouvez **désactiver globalement** toutes les augmentations lors d'un run de la pipeline avec l'argument CLI `--disable_augmentations`.



### 4. Inférence

- Chargement du meilleur checkpoint

- Prédiction sur les données de test

- Sauvegarde des résultats

- Calcul des métriques


### Format de sortie des visualisations et métriques

Après exécution des scripts d'inférence et de visualisation, les fichiers suivants sont générés :

- `results/SegFormer3D/<checkpoint>/<patient>/prediction_<patient>.pt` : fichier de prédiction PyTorch par patient (entrée du script de visualisation).
- `visualizations/SegFormer3D/<tag>/<patient>/*` : dossier de visualisations par patient contenant :
  - `<patient>_comparison.png` : comparaison Image / GT / Prédiction / Overlay
  - `<patient>_axial_slices.png` : coupes axiales détaillées
  - `<patient>_statistics.png` : distribution des classes et Dice par classe
  - `<patient>_errors.json` : métriques détaillées (dice, iou, precision, recall, support)
  - `<patient>_errors.png` : graphique résumé des métriques
  - `<patient>_slice_errors.png` : erreur slice-wise (MAE)
  - `<patient>_error_overlay.png` : overlay d'erreur sur la coupe centrale
  - `<patient>_advanced_errors.txt` : résumé indiquant si les métriques avancées (Hausdorff/ASSD/SSIM) ont été calculées ou désactivées

- `visualizations/SegFormer3D/<tag>/summary_metrics.json` : agrégation des métriques par classe (moyennes, médianes, support total) pour tous les patients traités.

Notes :
- Par défaut, les métriques avancées (Hausdorff, ASSD, SSIM) sont désactivées dans `visualize_results.py` pour éviter des problèmes mémoire sur grands volumes (240^3). Elles peuvent être réactivées en modifiant le code si nécessaire.
- Utilisez `scripts/run_visualizations_all.py` avec les options `--results_subdir <name>` et `--vis_tag <name>` pour contrôler d'où proviennent les prédictions et où seront stockées les visualisations.
- Pour accélérer le traitement, passez `--skip_volume` pour ignorer la visualisation volumétrique 3D et `--timeout <s>` pour définir un timeout par patient.



## Script de génération des splits CSV



Le script `create_prostate_splits.py` (présent dans chaque architecture) offre une génération avancée des splits avec **stratification par classe dominante** :



### Fonctionnalités principales



- **Analyse intelligente des classes** : Examine les tensors PyTorch pour identifier la classe la plus représentée par patient

- **Stratification automatique** : Équilibre la distribution des classes dominantes entre train/validation

- **Support cross-validation** : Génération de k folds avec préservation de la stratification

- **Reproductibilité** : Seed configurable pour obtenir les mêmes splits



### Utilisation manuelle



```bash

# Split train/val simple avec stratification (recommandé)

python create_prostate_splits.py \
    --input_dir ./preprocessed_data \
    --output_dir ./data \
    --test_size 0.2 \
    --test_ratio 0.0 \
    --stratified True \
    --random_state 42



# Cross-validation 5-fold (stratifié)
python create_prostate_splits.py \
    --input_dir ./preprocessed_data \
    --output_dir ./data \
    --kfold 5 \
    --stratified True \
    --random_state 42



# Split simple sans stratification
python create_prostate_splits.py \
    --input_dir ./preprocessed_data \
    --output_dir ./data \
    --test_size 0.2 \
    --stratified False

```



### Avantages de la stratification



- **Équilibre des classes** : Évite les biais dus aux déséquilibres de classe

- **Métriques représentatives** : Validation plus fiable des performances

- **Stabilité d'entraînement** : Réduction de l'overfitting sur certaines classes

- **Reproductibilité** : Résultats cohérents entre différentes exécutions



## Sortie



La pipeline génère :

- `data/preprocessed_data_{size}/patient_xxx/` : Données prétraitées au format PyTorch (par patient)

- `data/preprocessed_data_{size}/train.csv` : Liste des patients pour l'entraînement

- `data/preprocessed_data_{size}/validation.csv` : Liste des patients pour la validation

- `data/preprocessed_data_{size}/train_fold_{1-5}.csv` : Fichiers pour cross-validation (si kfold)

- `data/preprocessed_data_{size}/validation_fold_{1-5}.csv` : Fichiers pour cross-validation (si kfold)

- `checkpoints/{architecture}/` : Checkpoints du modèle sauvegardés automatiquement

- `results/{architecture}/` : Prédictions sur les données de test avec métriques

- `visualizations/{architecture}/` : Visualisations comparatives des résultats



## Visualisation des résultats



### Génération des visualisations



Après l'inférence, vous pouvez générer des visualisations détaillées pour analyser les performances :



```bash

# Pour SegFormer3D

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_128_128_128/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001







# Avec visualisation volumétrique 3D (optionnel, plus lent)

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



### Types de visualisations



1. **Comparaisons overlay** (`*_comparison.png`) : Superposition des prédictions sur les images originales

2. **Coupes axiales** (`*_axial_slices.png`) : Visualisation de tranches multiples du volume 3D

3. **Statistiques détaillées** (`*_statistics.png`) : Métriques Dice, IoU, précision par classe

4. **Visualisation volumétrique 3D** (`*_volume_3d.png`) : Représentation 3D des volumes segmentés (optionnel avec --volume_vis)

5. **Visualisation volumétrique interactive** (`*_volume_3d_interactive.html`) : Version interactive avec Plotly (optionnel avec --interactive)



### Erreurs de reconstruction / segmentation



Vous pouvez calculer et sauvegarder des métriques détaillées et des graphiques d'erreur par patient avec l'option `--compute_errors`. Cela génère :



- `*_errors.json` : métriques (Dice, IoU, précision, rappel, RMSE) incluant désormais les métriques avancées (Hausdorff, HD95, ASSD, SSIM per class)

- `*_errors.png` : graphique comparatif Dice / IoU par classe

- `*_slice_errors.png` : erreur slice-wise (MAE) en axial

- `*_error_overlay.png` : overlay d'erreur sur la coupe centrale



Notes:

- Les distances (Hausdorff, HD95, ASSD) sont calculées en *voxels* par défaut.

- Vous pouvez fournir le spacing voxel en mm via `--voxel_spacing sx,sy,sz` (x,y,z) pour obtenir les distances en mm.

- HD95 et statistiques de distance (mean/median/95th percentile) sont maintenant incluses par classe.



Exemple :



```bash

python visualize_results.py --config configs/config_segformer3d.yaml \

    --prediction results/SegFormer3D/patient_001/prediction_patient_001.pt \

    --input_dir data/preprocessed_data_128_128_128/patient_001 \

    --output_dir visualizations/SegFormer3D/patient_001 \

    --compute_errors

```



### Analyse des résultats



Les visualisations permettent de :

- **Évaluer qualitativement** les performances de segmentation

- **Identifier les zones d'erreur** communes

- **Comparer les architectures** visuellement

- **Valider les métriques quantitatives** avec l'analyse visuelle



## Tests & CI

- Installer les dépendances de développement :

```bash
pip install -r requirements-dev.txt
```

- Lancer la suite de tests :

```bash
pytest -q
```

- Suggestion : ajouter un job GitHub Actions simple pour exécuter les tests sur push/PR. Exemple minimal :

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest -q
```

## Dépannage



- Vérifiez que les chemins vers les données sont corrects

- Assurez-vous que les dépendances sont installées dans chaque architecture

- Vérifiez les logs pour les erreurs spécifiques à chaque étape



## Personnalisation



Pour modifier les paramètres :

1. Éditez les fichiers `configs/config_{architecture}.yaml`

2. Ajustez les chemins dans `pipeline.py` si nécessaire

3. Modifiez les scripts d'inférence si besoin



## Dernières améliorations



### Version actuelle : Pipeline complet automatisé avec visualisation



-  **Génération automatique des splits CSV** avec stratification par classe dominante

-  **Support cross-validation k-fold** intégré dans la pipeline

-  **Scripts d'inférence unifiés** pour toutes les architectures

-  **Outils de visualisation avancés** pour analyse qualitative des résultats

-  **Configurations YAML commentées** pour faciliter la compréhension

-  **Code nettoyé** : suppression de tous les emojis pour un style professionnel

-  **Documentation enrichie** dans tous les README

-  **Pipeline end-to-end** : du prétraitement NIfTI à la visualisation finale
-  **Reprise d'entraînement** : `--resume` / `--resume_checkpoint` pour continuer à l'époque exacte d'interruption
-  **Crop prostate** : `--crop_to_prostate` pour supprimer les slices sans prostate avant resampling



### Améliorations récentes



- **Stratification intelligente** : Analyse des tensors PyTorch pour équilibrer les classes

- **Flexibilité maximale** : Support splits fixes et cross-validation

- **Reproductibilité garantie** : Seeds configurables pour résultats cohérents

- **Maintenance facilitée** : Code propre et bien documenté

- **Automatisation complète** : Pipeline zero-touch pour l'expérimentation