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

```



### Arguments



- `--raw_data_dir` : Répertoire contenant les données brutes NIfTI

- `--architectures` : Liste des architectures à traiter (défaut : toutes)

- `--preprocessed_data_dir` : Répertoire pour les données prétraitées

- `--config_dir` : Répertoire des fichiers de configuration

- `--checkpoint_dir` : Répertoire pour sauvegarder les checkpoints

- `--results_dir` : Répertoire pour les résultats d'inférence

- `--skip_preprocess` : Sauter l'étape de prétraitement si déjà fait

- `--split_type` : Type de split ('fixed' pour train/val fixe, 'kfold' pour cross-validation)

- `--train_ratio` : Proportion pour l'entraînement (défaut: 0.7, seulement pour split_type='fixed')

- `--val_ratio` : Proportion pour la validation (défaut: 0.2, seulement pour split_type='fixed')

- `--test_ratio` : Proportion pour le test (défaut: 0.1, seulement pour split_type='fixed')

- `--k_folds` : Nombre de folds pour cross-validation (défaut: 5, seulement pour split_type='kfold')

- `--random_seed` : Seed pour reproductibilité (défaut: 42)

- `--target_size` : Taille cible pour le resampling des volumes (remplace la config)

- `--checkpoints` : Liste de checkpoints à inférer (ex: `--checkpoints best_model final_model`). Si non fournie, la pipeline essaie par défaut `best_model` puis `final_model`.

- `--visualize` : Générer visualisations et métriques après inférence (utilise `scripts/run_visualizations_all.py`).

- `--skip_volume` : Ignorer la visualisation volumétrique 3D pour accélérer la génération des visualisations.

- `--vis_timeout` : Timeout par patient (en secondes) pour la génération de visualisations (par défaut 600 s).



## Configuration de la Pipeline



### Fichier de configuration YAML



La pipeline utilise un fichier de configuration YAML (`pipeline_config.yaml`) pour définir tous les paramètres configurables :



```yaml

# Configuration de la pipeline automatisée

preprocessing:

  target_size: 96          # Taille des volumes (64, 96, 128, 256)

  normalize_method: "minmax"  # 'minmax' ou 'zscore'

  skip_existing: true      # Sauter les patients déjà prétraités



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



### Améliorations récentes



- **Stratification intelligente** : Analyse des tensors PyTorch pour équilibrer les classes

- **Flexibilité maximale** : Support splits fixes et cross-validation

- **Reproductibilité garantie** : Seeds configurables pour résultats cohérents

- **Maintenance facilitée** : Code propre et bien documenté

- **Automatisation complète** : Pipeline zero-touch pour l'expérimentation