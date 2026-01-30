# Makefile pour la pipeline SegFormer3D
# Usage: make <target>

PY ?= python
PIP ?= pip
CONFIG ?= configs/config_segformer3d.yaml
ARCH ?= SegFormer3D
PREP_INPUT ?= $(PWD)/data/raw_prostate
PREP_OUTPUT ?= $(PWD)/data/prostate_preprocessed
CHECKPOINT_DIR ?= $(PWD)/checkpoints
RESULTS_DIR ?= $(PWD)/results
TARGET_SIZE ?= 96

.PHONY: help install install-dev test test-fast lint format ci ci-local run-pipeline run-pipeline-highres preprocess splits train train-local infer infer-all visualize clean

help:
	@echo "Makefile - cibles disponibles:"
	@echo "  install           - installer les dépendances de production (requirements.txt)"
	@echo "  install-dev       - installer les dépendances de développement (requirements-dev.txt)"
	@echo "  test              - exécuter la suite de tests (pytest)"
	@echo "  test-fast         - exécuter les tests rapides (ex: tests/test_build_dataset.py)"
	@echo "  lint              - linting (si ruff/flake8 installés)"
	@echo "  format            - formater le code (si black/ruff installés)"
	@echo "  run-pipeline      - exécuter la pipeline complète (par défaut ./pipeline_config.yaml)"
	@echo "  run-pipeline-highres - exécuter la pipeline avec pipeline_config_high_res.yaml"
	@echo "  preprocess        - prétraiter les données brutes"
	@echo "  splits            - générer les CSV de splits à partir des données prétraitées"
	@echo "  train             - lancer l'entraînement (DDP via train_scripts/trainer_ddp.py)"
	@echo "  train-local       - lancer l'entraînement local sans DDP"
	@echo "  infer             - lancer l'inférence pour un checkpoint (args: CHECKPOINT=<path>)"
	@echo "  infer-all         - lancer les inférences batch (scripts/run_inference_all.py)"
	@echo "  visualize         - lancer les visualisations (scripts/run_visualizations_all.py)"
	@echo "  clean             - nettoyer fichiers temporaires et caches"
	@echo "  ci                - exécuter l'ensemble CI (install-dev + test)"
	@echo "  ci-local          - exécuter CI local (install-dev + pytest --cov)"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	pytest -q

test-fast:
	pytest tests/test_build_dataset.py -q

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check . || echo "ruff not found; install ruff or run 'pip install ruff'"

format:
	@command -v ruff >/dev/null 2>&1 && ruff format . || echo "ruff not found; install ruff or run 'pip install ruff'"

run-pipeline:
	$(PY) pipeline.py --config pipeline_config.yaml

run-pipeline-highres:
	$(PY) pipeline.py --config pipeline_config_high_res.yaml

preprocess:
	$(PY) data/prostate_raw_data/prostate_preprocess.py --input_dir $(PREP_INPUT) --output_dir $(PREP_OUTPUT) --target_size $(TARGET_SIZE)

splits:
	$(PY) data/prostate_raw_data/create_prostate_splits.py --input_dir $(PREP_OUTPUT) --output_dir $(PREP_OUTPUT) --stratified True

train:
	@echo "Lancement DDP training (vérifier la commande dans train_scripts/trainer_ddp.py)"
	$(PY) train_scripts/trainer_ddp.py --config $(CONFIG)

train-local:
	@echo "Lancement training local (non-DDP)"
	$(PY) train_scripts/trainer_ddp.py --config $(CONFIG) --local_rank 0

infer:
	@if [ -z "$(CHECKPOINT)" ]; then echo "ERROR: passez CHECKPOINT=<path>"; exit 1; fi
	$(PY) inference_simple.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --input_dir $(PREP_OUTPUT) --output_dir $(RESULTS_DIR)/$(ARCH)

infer-all:
	$(PY) scripts/run_inference_all.py --verbosity normal

visualize:
	$(PY) scripts/run_visualizations_all.py --verbosity normal

visualize-test:
	@echo "Run visualizations for a specific test dataset directory: make visualize-test TEST_DATA_DIR=/path/to/preprocessed_data_240_240_240 RESULTS_SUBDIR=best_model VIS_TAG=best_model"
	@if [ -z "$(TEST_DATA_DIR)" ]; then echo "ERROR: pass TEST_DATA_DIR=/path/to/preprocessed_data"; exit 1; fi
	$(PY) scripts/run_visualizations_all.py --verbosity normal --test_data_dir "$(TEST_DATA_DIR)" $(if $(RESULTS_SUBDIR),--results_subdir "$(RESULTS_SUBDIR)",) $(if $(VIS_TAG),--vis_tag "$(VIS_TAG)",)

visualize-config:
	@echo "Run visualizations using an explicit config file: make visualize-config CONFIG=configs/config_segformer3d.yaml RESULTS_SUBDIR=best_model VIS_TAG=best_model"
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: pass CONFIG=path/to/config.yaml"; exit 1; fi
	$(PY) scripts/run_visualizations_all.py --verbosity normal --config "$(CONFIG)" $(if $(RESULTS_SUBDIR),--results_subdir "$(RESULTS_SUBDIR)",) $(if $(VIS_TAG),--vis_tag "$(VIS_TAG)",)
ci: install-dev test

ci-local:
	install-dev
	pytest --cov -q

clean:
	@echo "Suppression des caches et fichiers temporaires..."
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	rm -rf .pytest_cache || true
	find . -name "*.pyc" -delete || true
	find . -name "*.pyo" -delete || true
	@echo "Nettoyé."