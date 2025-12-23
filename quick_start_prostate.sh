#!/bin/bash
###############################################################################
# SCRIPT DE DÉMARRAGE RAPIDE - Segmentation de Prostate avec SegFormer3D
#
# Ce script automatise tout le pipeline de prostate:
# 1. Prétraitement (nii.gz → PyTorch .pt)
# 2. Génération des CSV
# 3. Entraînement
# 4. Inférence
#
# Utilisation:
#   chmod +x quick_start_prostate.sh
#   ./quick_start_prostate.sh /chemin/vers/data/prostate_raw_data
###############################################################################

set -e  # Exit on error

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paramètres
PROJECT_ROOT="/workspaces/SegFormer3D"
INPUT_DIR="${1:-.}"
PREP_OUTPUT="${PROJECT_ROOT}/data/prostate_data/preprocessed"
DATA_OUTPUT="${PROJECT_ROOT}/data/prostate_data"
CHECKPOINT_DIR="${PROJECT_ROOT}/experiments/prostate_seg/checkpoints"
CONFIG_FILE="${PROJECT_ROOT}/experiments/prostate_seg/config_prostate.yaml"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  SegFormer3D - Segmentation de Prostate - Quick Start         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifie que les données d'entrée existent
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ Erreur: Répertoire non trouvé: $INPUT_DIR${NC}"
    echo -e "${YELLOW}Usage: ./quick_start_prostate.sh /chemin/vers/prostate_raw_data${NC}"
    exit 1
fi

# Compte les patients
patient_count=$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$patient_count" -eq 0 ]; then
    echo -e "${RED}❌ Erreur: Aucun répertoire de patient trouvé dans $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}📊 Trouvé $patient_count patients${NC}"
echo ""

# ============================================================================
# ÉTAPE 1: PRÉTRAITEMENT
# ============================================================================
echo -e "${BLUE}═══ ÉTAPE 1: PRÉTRAITEMENT (nii.gz → PyTorch) ═══${NC}"
echo ""

if [ -d "$PREP_OUTPUT" ] && [ "$(ls -A "$PREP_OUTPUT" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Données déjà prétraitées trouvées.${NC}"
    read -p "   Continuer ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Utilisation des données prétraitées existantes...${NC}"
        skip_preprocessing=true
    else
        echo -e "${YELLOW}Suppression des données prétraitées précédentes...${NC}"
        rm -rf "$PREP_OUTPUT"
        skip_preprocessing=false
    fi
else
    skip_preprocessing=false
fi

if [ "$skip_preprocessing" != "true" ]; then
    echo -e "${GREEN}🔄 Prétraitement en cours...${NC}"
    python "$PROJECT_ROOT/data/prostate_raw_data/prostate_preprocess.py" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$PREP_OUTPUT" \
        --target_size 96 \
        --normalize_method minmax
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erreur lors du prétraitement${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Prétraitement complété${NC}"
fi

echo ""

# ============================================================================
# ÉTAPE 2: GÉNÉRATION DES CSV
# ============================================================================
echo -e "${BLUE}═══ ÉTAPE 2: GÉNÉRATION DES SPLITS (train/val CSV) ═══${NC}"
echo ""

if [ -f "$DATA_OUTPUT/train.csv" ] && [ -f "$DATA_OUTPUT/validation.csv" ]; then
    echo -e "${YELLOW}⚠️  Fichiers CSV déjà existants${NC}"
    read -p "   Régénérer ? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Utilisation des CSV existants...${NC}"
        skip_splits=true
    else
        skip_splits=false
    fi
else
    skip_splits=false
fi

if [ "$skip_splits" != "true" ]; then
    echo -e "${GREEN}📊 Création des splits (80-20)...${NC}"
    python "$PROJECT_ROOT/data/prostate_raw_data/create_prostate_splits.py" \
        --input_dir "$PREP_OUTPUT" \
        --output_dir "$DATA_OUTPUT" \
        --test_size 0.2
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erreur lors de la création des splits${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Splits créés${NC}"
fi

echo ""

# ============================================================================
# ÉTAPE 3: ENTRAÎNEMENT (OPTIONNEL)
# ============================================================================
echo -e "${BLUE}═══ ÉTAPE 3: ENTRAÎNEMENT (OPTIONNEL) ═══${NC}"
echo ""
echo -e "${YELLOW}Lancer l'entraînement ? (y/n)${NC}"
read -p "" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}🚀 Lancement de l'entraînement...${NC}"
    echo -e "${YELLOW}Configuration: $CONFIG_FILE${NC}"
    echo ""
    
    # Vérifie le GPU
    python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    echo ""
    
    cd "$PROJECT_ROOT"
    python train_scripts/trainer_ddp.py --config "$CONFIG_FILE"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erreur lors de l'entraînement${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Entraînement complété${NC}"
else
    echo -e "${YELLOW}⏭️  Entraînement skippé${NC}"
fi

echo ""

# ============================================================================
# ÉTAPE 4: INFÉRENCE (OPTIONNEL)
# ============================================================================
echo -e "${BLUE}═══ ÉTAPE 4: INFÉRENCE (OPTIONNEL) ═══${NC}"
echo ""
echo -e "${YELLOW}Lancer l'inférence sur test set ? (y/n)${NC}"
read -p "" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Cherche le meilleur checkpoint
    best_checkpoint=$(find "$CHECKPOINT_DIR" -name "best*.pt" 2>/dev/null | head -1)
    
    if [ -z "$best_checkpoint" ]; then
        # Cherche le dernier checkpoint
        best_checkpoint=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$best_checkpoint" ]; then
        echo -e "${RED}❌ Aucun checkpoint trouvé dans $CHECKPOINT_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎯 Utilisation du checkpoint: $best_checkpoint${NC}"
    echo ""
    
    # Demande le répertoire d'entrée pour inférence
    echo -e "${YELLOW}Chemin des données test (défaut: $INPUT_DIR):${NC}"
    read -p "" test_input
    test_input=${test_input:-$INPUT_DIR}
    
    test_output="$PROJECT_ROOT/test_predictions"
    
    echo -e "${GREEN}📊 Inférence en cours...${NC}"
    python "$PROJECT_ROOT/experiments/prostate_seg/inference_prostate.py" \
        --model_path "$best_checkpoint" \
        --input_dir "$test_input" \
        --output_dir "$test_output" \
        --save_nifti true \
        --save_prob_map false
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erreur lors de l'inférence${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Inférence complétée${NC}"
    echo -e "${BLUE}Résultats sauvegardés dans: $test_output${NC}"
else
    echo -e "${YELLOW}⏭️  Inférence skippée${NC}"
fi

echo ""

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}✅ WORKFLOW PROSTATE TERMINÉ${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}📁 Répertoires créés:${NC}"
echo "   • Données prétraitées: $PREP_OUTPUT"
echo "   • CSV splits: $DATA_OUTPUT"
echo "   • Checkpoints: $CHECKPOINT_DIR"
echo ""

echo -e "${YELLOW}📝 Prochaines étapes:${NC}"
echo "   1. Inspectez les CSV dans $DATA_OUTPUT"
echo "   2. Modifiez config_prostate.yaml si besoin"
echo "   3. Lancez l'entraînement: python train_scripts/trainer_ddp.py --config $CONFIG_FILE"
echo "   4. Évaluez sur validation set"
echo "   5. Inférence sur test set: python experiments/prostate_seg/inference_prostate.py"
echo ""

echo -e "${BLUE}📚 Documentation complète: GUIDE_PROSTATE_COMPLETE_FR.md${NC}"
echo ""

echo -e "${GREEN}🎉 Bon entraînement!${NC}"
