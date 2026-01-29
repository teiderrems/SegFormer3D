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
    parser = argparse.ArgumentParser(description="Générateur de visualisations en batch pour tous les patients")
    parser.add_argument('--verbosity', choices=['quiet','normal','debug'], default='normal', help='Niveau de verbosité : quiet | normal | debug')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout (secondes) pour chaque sous-processus de visualisation ; 0 = pas de timeout')
    parser.add_argument('--skip_volume', action='store_true', help='Ignorer les visualisations volumétriques 3D (option `--volume_vis`) pour réduire la durée d\'exécution')
    parser.add_argument('--test_csv', type=str, default=None, help='Chemin vers un fichier `test.csv` (prend le pas sur la détection automatique)')
    parser.add_argument('--test_dir', type=str, default=None, help='Chemin vers le répertoire contenant le dataset prétraité (utilise `<dir>/test.csv`)')
    parser.add_argument('--results_subdir', type=str, default=None, help='Sous-dossier sous results/SegFormer3D contenant les prédictions (ex: best_model, final_model)')
    parser.add_argument('--vis_tag', type=str, default=None, help='Suffixe pour nommer le dossier de visualisations (ex: best_model, final_model)')
    args = parser.parse_args()
    verbosity = args.verbosity
    timeout = args.timeout
    skip_volume = args.skip_volume
    failed = []  # Liste pour collecter les patients ayant échoué

    # Configure RESULTS_DIR and VIS_DIR according to optional tags/subdirs
    if args.results_subdir:
        RESULTS_DIR = ROOT / 'results' / 'SegFormer3D' / args.results_subdir
    else:
        RESULTS_DIR = ROOT / 'results' / 'SegFormer3D'

    if args.vis_tag:
        VIS_DIR = ROOT / 'visualizations' / 'SegFormer3D' / args.vis_tag
    else:
        VIS_DIR = ROOT / 'visualizations' / 'SegFormer3D'

    os.makedirs(VIS_DIR, exist_ok=True)

    # Determine which test CSV to use (priority: --test_csv > --test_dir > auto-detected TEST_CSV)
    if args.test_csv:
        test_csv = Path(args.test_csv)
        if not test_csv.exists():
            print(f"Provided --test_csv does not exist: {test_csv}")
            return
    elif args.test_dir:
        test_csv = Path(args.test_dir) / 'test.csv'
        if not test_csv.exists():
            print(f"No test.csv found in provided --test_dir: {args.test_dir}")
            return
    else:
        test_csv = TEST_CSV

    patients = read_test_csv(test_csv)
    if verbosity != 'quiet':
        print(f"Found {len(patients)} patients in {test_csv}")

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

        # Use the same preprocessed folder where the selected test CSV resides (handles 240/128 variants)
        input_dir = test_csv.parent / patient
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
               '--compute_errors']
        # Add volume visualization unless user asked to skip it
        if not skip_volume:
            cmd.append('--volume_vis')
        # Always pass verbosity to child script so behavior is consistent
        cmd.extend(['--verbosity', verbosity])

        if verbosity != 'quiet':
            print('  Running:', ' '.join(cmd))
        t0 = time.time()
        try:
            if timeout and timeout > 0:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            else:
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
                failed.append(patient)
            else:
                if verbosity != 'quiet':
                    print(f"  Visualizations saved to {outdir}")
        except subprocess.TimeoutExpired as e:
            print(f"Visualization for {patient} timed out after {timeout} seconds; skipping.")
            failed.append(patient)
        except Exception as e:
            print(f"Unexpected error during visualization for {patient}: {e}")
            failed.append(patient)

    # After processing all patients, aggregate per-patient metrics into a summary JSON
    try:
        import json
        from statistics import mean, median

        summary = {}
        class_metrics = {}
        patient_count = 0
        errors_files = list(VIS_DIR.glob('*/*_errors.json'))
        for ef in errors_files:
            try:
                with open(ef, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
            patient_count += 1
            for k, v in data.items():
                if not k.startswith('class_'):
                    continue
                if k not in class_metrics:
                    class_metrics[k] = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'support': []}
                class_metrics[k]['dice'].append(v.get('dice', 0.0))
                class_metrics[k]['iou'].append(v.get('iou', 0.0))
                class_metrics[k]['precision'].append(v.get('precision', 0.0))
                class_metrics[k]['recall'].append(v.get('recall', 0.0))
                class_metrics[k]['support'].append(v.get('support', 0))

        # Compute aggregates
        summary['n_patients'] = patient_count
        summary['classes'] = {}
        for k, vals in class_metrics.items():
            summary['classes'][k] = {
                'dice_mean': mean(vals['dice']) if vals['dice'] else None,
                'dice_median': median(vals['dice']) if vals['dice'] else None,
                'iou_mean': mean(vals['iou']) if vals['iou'] else None,
                'precision_mean': mean(vals['precision']) if vals['precision'] else None,
                'recall_mean': mean(vals['recall']) if vals['recall'] else None,
                'support_total': sum(vals['support']) if vals['support'] else 0
            }

        summary_path = VIS_DIR / 'summary_metrics.json'
        with open(summary_path, 'w') as sf:
            json.dump(summary, sf, indent=2)
        if verbosity != 'quiet':
            print(f"Summary metrics written to {summary_path}")
    except Exception as e:
        print(f"Warning: failed to aggregate metrics: {e}")

    if failed:
        print('\nSome patients failed during visualization:')
        print(failed)
    else:
        if verbosity != 'quiet':
            print('\nAll patients processed successfully')



if __name__ == '__main__':
    main()
