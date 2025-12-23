#!/bin/bash
# Quick-Start: SegFormer3D Prostate + Bandelettes (3 Classes)
# 
# Ce script guide les étapes essentielles pour entraîner et faire
# l'inférence sur la segmentation prostate + bandelettes

set -e

echo "=========================================="
echo "SegFormer3D - Prostate + Bandelettes"
echo "Quick-Start Configuration"
echo "=========================================="

# Configuration
INPUT_DIR="${1:-.}"
PROSTATE_RAW="data/prostate_raw_data"
PREPROCESSED_DIR="data/prostate_data/preprocessed"
SPLITS_DIR="data/prostate_data"
CHECKPOINT_DIR="experiments/prostate_seg/checkpoints"
PREDICTIONS_DIR="predictions"

echo ""
echo "📁 Répertoires:"
echo "   Input:        $INPUT_DIR"
echo "   Raw data:     $PROSTATE_RAW"
echo "   Preprocessed: $PREPROCESSED_DIR"
echo "   Checkpoints:  $CHECKPOINT_DIR"
echo ""

# ============================================================================
# ÉTAPE 1: VÉRIFICATION DES DONNÉES
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ÉTAPE 1: Vérification des données d'entrée"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_patient_files() {
    local patient_dir="$1"
    local patient_name=$(basename "$patient_dir")
    
    # Vérifie T2.nii.gz
    if [[ ! -f "$patient_dir/T2.nii.gz" ]]; then
        echo "❌ $patient_name: T2.nii.gz manquant"
        return 1
    fi
    
    # Vérifie ADC.nii.gz
    if [[ ! -f "$patient_dir/ADC.nii.gz" ]]; then
        echo "❌ $patient_name: ADC.nii.gz manquant"
        return 1
    fi
    
    # Vérifie segmentation.nii.gz
    if [[ ! -f "$patient_dir/segmentation.nii.gz" ]]; then
        echo "❌ $patient_name: segmentation.nii.gz manquant"
        return 1
    fi
    
    echo "✅ $patient_name: OK (T2, ADC, segmentation)"
    return 0
}

if [[ ! -d "$PROSTATE_RAW" ]]; then
    echo "❌ Répertoire $PROSTATE_RAW non trouvé"
    echo ""
    echo "📌 Structure attendue:"
    echo "   $PROSTATE_RAW/"
    echo "   ├── patient_001/"
    echo "   │   ├── T2.nii.gz"
    echo "   │   ├── ADC.nii.gz"
    echo "   │   └── segmentation.nii.gz"
    echo "   ├── patient_002/"
    echo "   └── ..."
    echo ""
    echo "💡 Créez la structure et lancez à nouveau"
    exit 1
fi

patient_count=0
valid_count=0
for patient_dir in "$PROSTATE_RAW"/patient_*; do
    if [[ -d "$patient_dir" ]]; then
        ((patient_count++))
        if check_patient_files "$patient_dir"; then
            ((valid_count++))
        fi
    fi
done

echo ""
echo "📊 Résumé: $valid_count/$patient_count patients valides"

if [[ $valid_count -lt 10 ]]; then
    echo ""
    echo "⚠️  Minimum 10 patients recommandé pour entraînement (vous en avez $valid_count)"
fi

# ============================================================================
# ÉTAPE 2: TESTS DE CONFIGURATION
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ÉTAPE 2: Tests de configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python non trouvé"
    exit 1
fi

echo "🧪 Lancement des tests..."
if $PYTHON_CMD test_prostate_3class.py; then
    echo ""
    echo "✅ Tous les tests passés!"
else
    echo ""
    echo "⚠️  Certains tests ont échoué (vérifiez les dépendances)"
fi

# ============================================================================
# ÉTAPE 3: PRÉTRAITEMENT
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ÉTAPE 3: Prétraitement des données"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ -d "$PREPROCESSED_DIR" ]] && [[ -n "$(ls -A "$PREPROCESSED_DIR")" ]]; then
    echo "✅ Données déjà prétraitées dans $PREPROCESSED_DIR"
    read -p "Voulez-vous les re-prétraiter? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏭️  Passage à l'étape suivante"
    else
        rm -rf "$PREPROCESSED_DIR"
        echo "Lancement du prétraitement..."
        $PYTHON_CMD data/prostate_raw_data/prostate_preprocess.py \
            --input_dir "$PROSTATE_RAW" \
            --output_dir "$PREPROCESSED_DIR"
    fi
else
    echo "Lancement du prétraitement (peut prendre plusieurs minutes)..."
    mkdir -p "$PREPROCESSED_DIR"
    $PYTHON_CMD data/prostate_raw_data/prostate_preprocess.py \
        --input_dir "$PROSTATE_RAW" \
        --output_dir "$PREPROCESSED_DIR"
fi

# ============================================================================
# ÉTAPE 4: RÉSUMÉ ET PROCHAINES ÉTAPES
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CONFIGURATION COMPLÈTE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📌 Prochaines étapes:"
echo ""
echo "1️⃣  Entraînement (optionnel):"
echo "   python train_scripts/trainer_ddp.py \\"
echo "       --config experiments/prostate_seg/config_prostate.yaml"
echo ""
echo "2️⃣  Inférence (sur données de test):"
echo "   python experiments/prostate_seg/inference_prostate.py \\"
echo "       --model_path ./experiments/prostate_seg/checkpoints/best.pt \\"
echo "       --input_dir ./test_data \\"
echo "       --output_dir ./predictions \\"
echo "       --save_separate_labels true"
echo ""
echo "3️⃣  Documentation:"
echo "   - GUIDE_PROSTATE_BANDELETTES_FR.md    (guide complet)"
echo "   - README_PROSTATE_BANDELETTES.md     (configuration)"
echo ""
echo "📚 Configuration 3 classes:"
echo "   - num_classes: 3 (fond, prostate, bandelettes)"
echo "   - class_weights: [0.3, 1.5, 1.2]"
echo "   - in_channels: 2 (T2 + ADC)"
echo "   - Taille: 96×96×96"
echo ""
echo "✨ Système prêt pour entraînement!"
echo ""
