#!/bin/bash
# Script de lancement rapide de la pipeline

# Chemins par défaut
RAW_DATA_DIR=${1:-"./raw_data"}
ARCHITECTURES=${2:-"SegFormer3D"}

echo "Lancement de la pipeline automatisée..."
echo "Données brutes: $RAW_DATA_DIR"
echo "Architectures: $ARCHITECTURES"

python pipeline.py \
    --raw_data_dir "$RAW_DATA_DIR" \
    --architectures $ARCHITECTURES \
    --preprocessed_data_dir "./data/preprocessed_data_128_128_128" \
    --config_dir "./configs" \
    --checkpoint_dir "./checkpoints" \
    --results_dir "./results"

echo "Pipeline terminée!"