#!/usr/bin/env python3
"""Batch inference helper: reads test.csv and runs inference_simple.py per patient."""
import csv
import argparse
from tqdm import tqdm
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Auto-detect test CSV in any preprocessed_data_* folder (prefer higher resolution like 240 if available)
csv_candidates = list((ROOT / 'data').glob('preprocessed_data_*' '/test.csv'))
if not csv_candidates:
    # fallback: look for test.csv anywhere under data/
    csv_candidates = list((ROOT / 'data').rglob('test.csv'))
if csv_candidates:
    # prefer one containing '240' if present, else pick the most recently modified
    preferred = None
    for p in csv_candidates:
        if '240' in str(p):
            preferred = p
            break
    CSV = preferred.resolve() if preferred else max(csv_candidates, key=lambda p: p.stat().st_mtime).resolve()
else:
    CSV = (ROOT / 'data' / 'preprocessed_data_128_128_128' / 'test.csv').resolve()
# Detect checkpoint candidates (prefer repo-level checkpoints/ then SegFormer3D subfolder, try final if best missing)
candidates = [
    ROOT / 'checkpoints' / 'best_model.pth',
    ROOT / 'checkpoints' / 'final_model.pth',
    ROOT / 'checkpoints' / 'SegFormer3D' / 'best_model.pth',
    ROOT / 'checkpoints' / 'SegFormer3D' / 'final_model.pth',
    ROOT / '..' / 'checkpoints' / 'SegFormer3D' / 'best_model.pth'
]
CHECKPOINT = None
for p in candidates:
    if p.exists():
        CHECKPOINT = p.resolve()
        break
if CHECKPOINT is None:
    raise FileNotFoundError(f"No checkpoint found in expected locations: {candidates}")

CONFIG = (ROOT / 'configs' / 'config_segformer3d.yaml').resolve()
RESULTS_DIR = (ROOT / 'results' / 'SegFormer3D').resolve()
INFER_SCRIPT = (ROOT / 'inference_simple.py').resolve()

# CLI: verbosity level
parser = argparse.ArgumentParser(description='Batch inference runner')
parser.add_argument('--verbosity', choices=['quiet','normal','debug'], default='normal', help='Niveau de verbosité: quiet|normal|debug')
args = parser.parse_args()
VERBOSITY = args.verbosity

if VERBOSITY != 'quiet':
    print(f"Using CSV: {CSV}")
    print(f"Using checkpoint: {CHECKPOINT}")
    print(f"Using config: {CONFIG}")

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
for row in tqdm(rows, desc="Inference", unit="case"):
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
    # tqdm affiche déjà le contexte, on réduit les prints pour limiter le bruit
    try:
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
        duration = time.time() - t0
        if VERBOSITY == 'debug':
            tqdm.write(f"[debug] Inference {case} duration: {duration:.3f}s")
        if p.stdout and VERBOSITY != 'quiet':
            tqdm.write(p.stdout)
        if p.stderr:
            tqdm.write('stderr: ' + p.stderr)
    except subprocess.CalledProcessError as e:
        tqdm.write(f"Inference failed for {case}: returncode {e.returncode}")
        tqdm.write('stdout: ' + (e.stdout or ''))
        tqdm.write('stderr: ' + (e.stderr or ''))
        failed.append(case)

print('\nBatch inference completed')
if failed:
    print('Failed cases:', failed)
else:
    print('All cases processed successfully')
