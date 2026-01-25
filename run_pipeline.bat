@echo off
REM Script de lancement rapide de la pipeline pour Windows

REM Chemins par défaut
set RAW_DATA_DIR=%1
if "%RAW_DATA_DIR%"=="" set RAW_DATA_DIR=./raw_data

set ARCHITECTURES=%2
if "%ARCHITECTURES%"=="" set ARCHITECTURES=SegFormer3D

echo Lancement de la pipeline automatisée...
echo Données brutes: %RAW_DATA_DIR%
echo Architectures: %ARCHITECTURES%

python pipeline.py ^
    --raw_data_dir "%RAW_DATA_DIR%" ^
    --architectures %ARCHITECTURES% ^
    --preprocessed_data_dir "./data/preprocessed_data_128_128_128" ^
    --config_dir "./configs" ^
    --checkpoint_dir "./checkpoints" ^
    --results_dir "./results"

echo Pipeline terminée!
pause