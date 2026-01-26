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
TEST_CSV = ROOT / 'data' / 'preprocessed_data_128_128_128' / 'test.csv'
RESULTS_DIR = ROOT / 'results' / 'SegFormer3D'
VIS_DIR = ROOT / 'visualizations' / 'SegFormer3D'
PYTHON = 'python'


def read_test_csv(csv_path):
    patients = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) == 0:
                continue
            # Expect either full path or folder name like patient_002
            val = r[0].strip()
            if val:
                # Normalize: if path contains patient_ prefix, take basename
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

        input_dir = ROOT / 'data' / 'preprocessed_data_128_128_128' / patient
        outdir.mkdir(parents=True, exist_ok=True)

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
