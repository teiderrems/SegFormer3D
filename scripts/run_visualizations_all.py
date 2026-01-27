#!/usr/bin/env python3
"""Batch visualizations for all patients listed in test.csv
Generates: comparison PNG, axial slices PNG, statistics PNG, errors JSON + PNG + slice errors + error overlay + 3D static PNG
Skips patients already having <patient>_errors.json in the visualizations directory.
"""
import csv
import argparse
from tqdm import tqdm
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'config_segformer3d.yaml'
# Auto-detect test CSV in any preprocessed_data_* folder (prefer '240' if available)
csv_candidates = list((ROOT / 'data').glob('preprocessed_data_*' '/test.csv'))
if not csv_candidates:
    csv_candidates = list((ROOT / 'data').rglob('test.csv'))
if csv_candidates:
    preferred = None
    for p in csv_candidates:
        if '240' in str(p):
            preferred = p
            break
    TEST_CSV = preferred if preferred else max(csv_candidates, key=lambda p: p.stat().st_mtime)
else:
    TEST_CSV = ROOT / 'data' / 'preprocessed_data_128_128_128' / 'test.csv'

RESULTS_DIR = ROOT / 'results' / 'SegFormer3D'
VIS_DIR = ROOT / 'visualizations' / 'SegFormer3D'
PYTHON = 'python'


def read_test_csv(csv_path):
    """Read test CSV robustly: support CSVs with header (data_path/case_name) or plain list of paths/ids.
    Returns list of patient folder names (e.g., patient_002)
    """
    patients = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Try DictReader first (handles headered CSVs)
        dict_reader = csv.DictReader(f)
        if dict_reader.fieldnames and any(h in dict_reader.fieldnames for h in ["case_name", "data_path"]):
            for r in dict_reader:
                if 'case_name' in r and r['case_name']:
                    patients.append(Path(r['case_name']).name)
                elif 'data_path' in r and r['data_path']:
                    patients.append(Path(r['data_path']).name)
            return patients
        # Fallback to plain reader (no header)
        f.seek(0)
        reader = csv.reader(f)
        for r in reader:
            if len(r) == 0:
                continue
            val = r[0].strip()
            if not val:
                continue
            if val.lower() == 'data_path' or val.lower() == 'case_name':
                continue
            patients.append(Path(val).name)
    return patients


def find_prediction_for_patient(patient):
    pdir = RESULTS_DIR / patient
    if not pdir.exists():
        return None
    for f in pdir.iterdir():
        if f.name.startswith('prediction_') and f.suffix == '.pt':
            return f
    return None


def has_been_processed(patient):
    outdir = VIS_DIR / patient
    return (outdir / f"{patient}_errors.json").exists()


def main():
    # CLI: verbosity level
    parser = argparse.ArgumentParser(description='Batch visualizations runner')
    parser.add_argument('--verbosity', choices=['quiet','normal','debug'], default='normal', help='Niveau de verbosité: quiet|normal|debug')
    args = parser.parse_args()
    verbosity = args.verbosity

    os.makedirs(VIS_DIR, exist_ok=True)
    patients = read_test_csv(TEST_CSV)
    if verbosity != 'quiet':
        print(f"Found {len(patients)} patients in {TEST_CSV}")

    for patient in tqdm(patients, desc="Visualizations", unit="patient"):
        if verbosity != 'quiet':
            tqdm.write('---')
            tqdm.write(f"Processing {patient}")
        pred = find_prediction_for_patient(patient)
        if pred is None:
            if verbosity != 'quiet':
                print(f"  No prediction found for {patient} in {RESULTS_DIR / patient}. Skipping.")
            continue
        outdir = VIS_DIR / patient
        if has_been_processed(patient):
            if verbosity != 'quiet':
                print(f"  Already processed (found {patient}_errors.json). Skipping.")
            continue

        # Use the same preprocessed folder where TEST_CSV resides (handles 240/128 variants)
        input_dir = TEST_CSV.parent / patient
        outdir.mkdir(parents=True, exist_ok=True)
        if not input_dir.exists():
            # fallback: try previous hardcoded path to not break older setups
            input_dir = ROOT / 'data' / 'preprocessed_data_128_128_128' / patient
            if not input_dir.exists():
                print(f"  Input data not found for {patient} (checked {TEST_CSV.parent} and preprocessed_data_128_128_128). Skipping.")
                continue

        cmd = [PYTHON, str(ROOT / 'visualize_results.py'),
               '--config', str(CONFIG),
               '--prediction', str(pred),
               '--input_dir', str(input_dir),
               '--output_dir', str(outdir),
               '--compute_errors',
               '--volume_vis']
        # Always pass verbosity to child script so behavior is consistent
        cmd.extend(['--verbosity', verbosity])

        if verbosity != 'quiet':
            print('  Running:', ' '.join(cmd))
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - t0
        if verbosity == 'debug':
            print(f"[debug] Visualization {patient} duration: {duration:.3f}s")
        if res.returncode != 0:
            if verbosity != 'quiet':
                print('  Error running visualization:', res.returncode)
                print(res.stderr)
            else:
                # always show errors even in quiet mode
                print(f"Error for {patient}: {res.returncode}\n{res.stderr}")
        else:
            if verbosity != 'quiet':
                print(f"  Visualizations saved to {outdir}")



if __name__ == '__main__':
    main()
