#!/usr/bin/env bash
# Exemple de wrapper de soumission OAR pour calcululco / clusters universitaires.
# Éditez les variables ci-dessous (venv, chemins de données) avant d'exécuter.
# Usage interactif (adaptez la syntaxe au cluster si nécessaire) :
#   oarsub -I -p "gpu>0" -l host=1/gpu=1,walltime=04:30:00
# Exemple de soumission batch :
#   oarsub -S /bin/bash -p "gpu>0" -l host=1/gpu=1,walltime=04:30:00 ./scripts/submit_oar_example.sh --skip_training --checkpoints best_model

# --- CONFIG (à adapter) ---
VENV_ACTIVATE="/path/to/venv/bin/activate"
RAW_DATA_DIR="/data/raw_prostate"
PREPROCESSED_DIR="/scratch/preprocessed"
CHECKPOINT_DIR="/scratch/checkpoints"
RESULTS_DIR="/scratch/results"
CONFIG_FILE="pipeline_config.yaml"
# ---------------------------

set -euo pipefail
mkdir -p logs

if [ -n "${VENV_ACTIVATE}" ] && [ -f "${VENV_ACTIVATE}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
fi

cd "${OAR_WORKDIR:-$PWD}"

echo "[segformer3D] host=$(hostname) pwd=$(pwd)"
echo "[segformer3D] config=${CONFIG_FILE}"

python pipeline.py \
  --config "${CONFIG_FILE}" \
  --raw_data_dir "${RAW_DATA_DIR}" \
  --preprocessed_data_dir "${PREPROCESSED_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --architectures SegFormer3D \
  --target_size 128 \
  "$@"

# Exemple d'utilisation pour inference-only :
# oarsub -S /bin/bash -p "gpu>0" -l host=1/gpu=1,walltime=04:30:00 ./scripts/submit_oar_example.sh --skip_training --checkpoints best_model --visualize --force-cli

# REMARQUE:
# - Adaptez VENV_ACTIVATE et les chemins /scratch avant exécution.
# - Rendre exécutable si besoin: chmod +x scripts/submit_oar_example.sh
