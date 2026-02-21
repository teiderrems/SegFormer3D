#!/usr/bin/env python3
"""Batch inference helper: reads test.csv and runs inference_simple.py per patient."""
import csv
import argparse
from tqdm import tqdm
import subprocess
import sys
import time
import os
from pathlib import Path
import concurrent.futures

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Auto-detect test CSV in any prostate_preprocessed_* folder (prefer higher resolution like 350 if available)
csv_candidates = list((ROOT / 'data').glob('prostate_preprocessed_*' '/test.csv'))
if not csv_candidates:
    # fallback: look for test.csv anywhere under data/
    csv_candidates = list((ROOT / 'data').rglob('test.csv'))
if csv_candidates:
    # prefer one containing '350' if present, else pick the most recently modified
    preferred = None
    for p in csv_candidates:
        if '350' in str(p):
            preferred = p
            break
    CSV = preferred.resolve() if preferred else max(csv_candidates, key=lambda p: p.stat().st_mtime).resolve()
else:
    CSV = (ROOT / 'data' /'prostate_preprocessed'/ 'prostate_preprocessed_350_350_350' / 'test.csv').resolve()
# Detect checkpoint candidates (prefer repo-level checkpoints/ then SegFormer3D subfolder, try final if best missing)
candidates = [
    ROOT / 'checkpoints' / 'best_model.pth',
    ROOT / 'checkpoints' / 'final_model.pth',
    ROOT / 'checkpoints' / 'best_model.pth',
    ROOT / 'checkpoints' / 'final_model.pth',
    ROOT / '..' / 'checkpoints'/ 'best_model.pth'
]
CHECKPOINT = None
for p in candidates:
    if p.exists():
        CHECKPOINT = p.resolve()
        break

CONFIG = (ROOT / 'configs' / 'config_segformer3d.yaml').resolve()
DEFAULT_RESULTS_DIR = (ROOT / 'results').resolve()
INFER_SCRIPT = (ROOT / 'inference_simple.py').resolve()


def process_inference(row, VERBOSITY, ROOT, INFER_SCRIPT, CONFIG, CHECKPOINT, RESULTS_DIR, batch_size, device, save_predictions, save_probabilities, threshold, force_cli=False):
    """Execute inference for a single CSV row and return case name if failed.

    `force_cli` controls whether the `--force-cli` flag is propagated to the
    child `inference_simple.py` process.
    """
    data_path = row['data_path']
    case = row['case_name']
    # Normalize data_path: if relative to project root, convert
    input_dir = Path(data_path)
    if not input_dir.is_absolute():
        input_dir = ROOT / data_path
    out = RESULTS_DIR / case
    out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(INFER_SCRIPT), '--config', str(CONFIG), '--checkpoint', str(CHECKPOINT), '--input_dir', str(input_dir), '--output_dir', str(out)]
    # Always pass verbosity to child script so behavior is consistent
    cmd.extend(['--verbosity', VERBOSITY])
    # Ajouter les paramètres d'inférence
    cmd.extend(['--batch_size', str(batch_size), '--device', device, '--threshold', str(threshold)])
    if save_predictions:
        cmd.append('--save_predictions')
    if save_probabilities:
        cmd.append('--save_probabilities')
    # Support save_nifti from YAML (enabled by default in configs)
    save_nifti = inference_params.get('save_nifti', True)
    if save_nifti:
        cmd.append('--save_nifti')
    if force_cli:
        cmd.append('--force-cli')

    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - t0
        if VERBOSITY == 'debug':
            print(f"[debug] Inference {case} duration: {duration:.3f}s")
        if res.returncode != 0:
            print(f"Inference failed for {case}: returncode {res.returncode}")
            if VERBOSITY != 'quiet':
                print(res.stderr)
            return case  # failed
        return None
    except Exception as e:
        print(f"Inference failed for {case}: {e}")
        return case

def main():
    # CLI options: verbosity, checkpoint override and tag for results subfolder
    parser = argparse.ArgumentParser(description='Batch inference runner')
    parser.add_argument('--verbosity', choices=['quiet','normal','debug'], default='normal', help='Niveau de verbosité: quiet|normal|debug')
    parser.add_argument('--checkpoint', type=str, default=None, help='Chemin vers un checkpoint à utiliser (remplace la détection automatique)')
    parser.add_argument('--tag', type=str, default=None, help='Suffixe pour le dossier de résultats (ex: best_model, final_model)')
    parser.add_argument('--force-cli', action='store_true', help='Forcer les arguments CLI à remplacer les valeurs du YAML (par défaut: YAML > CLI)')
    args = parser.parse_args()
    VERBOSITY = args.verbosity

    # If user passed a checkpoint, prefer it; else fall back to auto-detection and raise if none found
    if args.checkpoint:
        global CHECKPOINT
        CHECKPOINT = Path(args.checkpoint).resolve()
        if not CHECKPOINT.exists():
            # do not raise: allow network-mounted paths; warn and continue
            print(f"Warning: checkpoint file not found locally at {CHECKPOINT}; the inference command will still be attempted.")
    if CHECKPOINT is None:
        raise FileNotFoundError(f"No checkpoint found in expected locations: {candidates} (or via --checkpoint)")

    # Determine results dir (optionally namespaced by tag)
    if args.tag:
        results_dir = DEFAULT_RESULTS_DIR / args.tag
    else:
        results_dir = DEFAULT_RESULTS_DIR

    import yaml

    # Charger la configuration pour récupérer les paramètres d'inférence
    with open(CONFIG, 'r') as f:
        config = yaml.safe_load(f) or {}

    inference_params = config.get('inference_parameters', {})
    batch_size = inference_params.get('batch_size', 1)
    device = inference_params.get('device', 'cuda')
    save_predictions = inference_params.get('save_predictions', True)
    save_probabilities = inference_params.get('save_probabilities', False)
    threshold = inference_params.get('threshold', 0.5)

    if VERBOSITY != 'quiet':
        print(f"Paramètres d'inférence: device={device}, batch_size={batch_size}, save_predictions={save_predictions}, save_probabilities={save_probabilities}, threshold={threshold}")

    if not CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {CSV}")
    # Do not abort if checkpoint cannot be stat-checked (may be on mounted path); warn and continue
    if not CHECKPOINT.exists():
        print(f"Warning: checkpoint file not found locally at {CHECKPOINT}; the inference command will still be attempted.")
    if not INFER_SCRIPT.exists():
        raise FileNotFoundError(f"Inference script not found: {INFER_SCRIPT}")

    failed = []
    with open(CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if VERBOSITY != 'quiet':
        print(f"Found {len(rows)} cases in {CSV}")

    # Parallel processing
    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_inference, row, VERBOSITY, ROOT, INFER_SCRIPT, CONFIG, CHECKPOINT, results_dir, batch_size, device, save_predictions, save_probabilities, threshold, args.force_cli): row for row in rows}
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows), desc="Inference", unit="case"):
            failed_case = future.result()
            if failed_case:
                failed.append(failed_case)

    print('\nBatch inference completed')
    if failed:
        print('Failed cases:', failed)
    else:
        print('All cases processed successfully')


if __name__ == '__main__':
    main()
