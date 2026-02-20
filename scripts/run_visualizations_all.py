#!/usr/bin/env python3
"""Batch visualizations for all patients listed in test.csv
Generates: comparison PNG, axial slices PNG, statistics PNG, errors JSON + PNG + slice errors + error overlay + 3D static PNG
Skips patients already having <patient>_errors.json in the visualizations directory.
"""
import csv
import argparse
import sys
from tqdm import tqdm
import os
import subprocess
import time
from pathlib import Path
import concurrent.futures

ROOT = Path(os.path.realpath(os.getcwd()))
CONFIG = ROOT / 'configs' / 'config_segformer3d.yaml'
# Auto-detect test CSV in any prostate_preprocessed_* folder (prefer '128' if available)
csv_candidates = list((ROOT / 'data'/'prostate_preprocessed').glob('preprocessed_data_*' '/test.csv'))
if not csv_candidates:
    csv_candidates = list((ROOT / 'data').rglob('test.csv'))
if csv_candidates:
    preferred = None
    for p in csv_candidates:
        if '128' in str(p):
            preferred = p
            break
    TEST_CSV = preferred if preferred else max(csv_candidates, key=lambda p: p.stat().st_mtime)
else:
    TEST_CSV = ROOT / 'data'/'prostate_preprocessed'/ 'preprocessed_data_128_128_128' / 'test.csv'

RESULTS_DIR = ROOT / 'results'
VIS_DIR = ROOT / 'visualizations'
PYTHON = sys.executable


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


# Helper function to determine the test CSV path given CLI args and config
def get_test_csv_from_args_and_config(args, config_path=CONFIG, default=TEST_CSV):
    """Return a Path to the test.csv to use.

    Behavior:
      - If the user explicitly provided a config file via `--config`, YAML takes
        precedence over CLI (YAML > CLI).
      - If no explicit config was provided (using the script default), CLI
        options keep precedence over the default YAML (CLI > default YAML).

    This preserves backward-compatibility while giving explicit user configs
    higher authority.
    """
    # If user explicitly requested CLI to win, check CLI first
    if getattr(args, 'force_cli', False):
        if getattr(args, 'test_csv', None):
            p = Path(args.test_csv)
            return p if p.exists() else None
        if getattr(args, 'test_dir', None):
            p = Path(args.test_dir) / 'test.csv'
            return p if p.exists() else None
        if getattr(args, 'test_data_dir', None):
            p = Path(args.test_data_dir) / 'test.csv'
            return p if p.exists() else None

    use_yaml_first = getattr(args, 'config', None) is not None or (config_path is not None and str(config_path) != str(CONFIG))

    def try_config_csv():
        try:
            import yaml
            cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8')) or {}
            ds = cfg.get('dataset_parameters', {}).get('test_dataset_args', {})
            if 'split_file' in ds and ds['split_file']:
                p = Path(ds['split_file'])
                if p.exists():
                    return p
            if 'root' in ds and ds['root']:
                p = Path(ds['root']) / 'test.csv'
                if p.exists():
                    return p
        except Exception:
            pass
        return None

    # When explicit config provided, check YAML first
    if use_yaml_first:
        p = try_config_csv()
        if p:
            return p

    # CLI precedence (or fallback when explicit config had no dataset info)
    if getattr(args, 'test_csv', None):
        p = Path(args.test_csv)
        return p if p.exists() else None
    if getattr(args, 'test_dir', None):
        p = Path(args.test_dir) / 'test.csv'
        return p if p.exists() else None
    if getattr(args, 'test_data_dir', None):
        p = Path(args.test_data_dir) / 'test.csv'
        return p if p.exists() else None

    # If not found yet, try YAML (covers the case of default CONFIG or explicit config lacking the keys)
    p = try_config_csv()
    if p:
        return p

    # Last resort: auto-detection
    return default if default.exists() else None


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


def process_patient(patient, verbosity, skip_volume, timeout, PYTHON, ROOT, CONFIG, RESULTS_DIR, VIS_DIR, test_csv):
    """Process visualization for a single patient"""
    failed = []
    pred = find_prediction_for_patient(patient)
    if pred is None:
        if verbosity != 'quiet':
            print(f"  No prediction found for {patient} in {RESULTS_DIR / patient}. Skipping.")
        return None  # not failed, just skipped

    outdir = VIS_DIR / patient
    if has_been_processed(patient):
        if verbosity != 'quiet':
            print(f"  Already processed (found {patient}_errors.json). Skipping.")
        return None

    # Use the same preprocessed folder where the selected test CSV resides (handles 240/128 variants)
    input_dir = test_csv.parent / patient
    outdir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        # fallback: try previous hardcoded path to not break older setups
        input_dir = ROOT / 'data' / 'prostate_preprocessed_128_128_128' / patient
        if not input_dir.exists():
            print(f"  Input data not found for {patient} (checked {test_csv.parent} and prostate_preprocessed_128_128_128). Skipping.")
            return None

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
            return patient  # failed
        else:
            if verbosity != 'quiet':
                print(f"  Visualizations saved to {outdir}")
            return None
    except subprocess.TimeoutExpired as e:
        print(f"Visualization for {patient} timed out after {timeout} seconds; skipping.")
        return patient
    except Exception as e:
        print(f"Unexpected error during visualization for {patient}: {e}")
        return patient


def main():
    global RESULTS_DIR, VIS_DIR
    # CLI: verbosity level
    parser = argparse.ArgumentParser(description="Générateur de visualisations en batch pour tous les patients")
    parser.add_argument('--verbosity', choices=['quiet','normal','debug'], default='normal', help='Niveau de verbosité : quiet | normal | debug')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout (secondes) pour chaque sous-processus de visualisation ; 0 = pas de timeout')
    parser.add_argument('--skip_volume', action='store_true', help='Ignorer les visualisations volumétriques 3D (option `--volume_vis`) pour réduire la durée d\'exécution')
    parser.add_argument('--test_csv', type=str, default=None, help='Chemin vers un fichier `test.csv` (prend le pas sur la détection automatique)')
    parser.add_argument('--test_dir', type=str, default=None, help='Chemin vers le répertoire contenant le dataset prétraité (utilise `<dir>/test.csv`)')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Répertoire prétraité contenant `test.csv` (prend le pas sur la détection automatique et peut être défini dans la config)')
    parser.add_argument('--config', type=str, default=None, help='Fichier de configuration YAML à utiliser pour détecter le dataset de test (remplace la valeur par défaut du script)')
    parser.add_argument('--results_subdir', type=str, default=None, help='Sous-dossier sous results contenant les prédictions (ex: best_model, final_model)')
    parser.add_argument('--vis_tag', type=str, default=None, help='Suffixe pour nommer le dossier de visualisations (ex: best_model, final_model)')
    parser.add_argument('--force-cli', action='store_true', help='Forcer les arguments CLI à remplacer les valeurs du YAML (par défaut: YAML > CLI)')
    args = parser.parse_args()
    verbosity = args.verbosity
    timeout = args.timeout
    skip_volume = args.skip_volume
    failed = []  # Liste pour collecter les patients ayant échoué

    # Configure RESULTS_DIR and VIS_DIR according to optional tags/subdirs
    if args.results_subdir:
        RESULTS_DIR = ROOT / 'results' / args.results_subdir
    else:
        RESULTS_DIR = ROOT / 'results'

    if args.vis_tag:
        VIS_DIR = ROOT / 'visualizations' / args.vis_tag
    else:
        VIS_DIR = ROOT / 'visualizations'
    # Respect explicit --config if provided
    config_path = Path(args.config) if args.config else CONFIG
    test_csv = get_test_csv_from_args_and_config(args, config_path=config_path, default=TEST_CSV)
    if test_csv is None:
        print("No valid test.csv could be determined. Provide --test_csv / --test_dir / --test_data_dir or update the config with test_dataset_args.root or split_file, or pass --config <file>.")
        return

    patients = read_test_csv(test_csv)
    if verbosity != 'quiet':
        print(f"Found {len(patients)} patients in {test_csv}")

    # Parallel processing
    max_workers = os.cpu_count() or 4  # Use all available CPUs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_patient = {
            executor.submit(process_patient, patient, verbosity, skip_volume, timeout, PYTHON, ROOT, CONFIG, RESULTS_DIR, VIS_DIR, test_csv): patient
            for patient in patients
        }
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patients), desc="Visualizations", unit="patient"):
            patient = future_to_patient[future]
            try:
                failed_patient = future.result()
                if failed_patient:
                    failed.append(failed_patient)
            except Exception as exc:
                print(f"Patient {patient} generated an exception: {exc}")
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
