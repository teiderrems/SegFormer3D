#!/usr/bin/env python
"""Resize preprocessed prostate data to 96x96x96 for training"""
import os
import torch
import glob
from pathlib import Path
from tqdm import tqdm

input_dir = "/workspaces/SegFormer3D/data/prostate_data/preprocessed"
output_dir = "/workspaces/SegFormer3D/data/prostate_data/preprocessed_96"

os.makedirs(output_dir, exist_ok=True)

# Get all patient directories
patient_dirs = sorted([d for d in glob.glob(f"{input_dir}/*") if os.path.isdir(d)])

print(f"Found {len(patient_dirs)} patients")
print(f"Resizing from 128x128x128 to 96x96x96...")

from torch.nn.functional import interpolate

for patient_dir in tqdm(patient_dirs):
    patient_name = os.path.basename(patient_dir)
    output_patient_dir = os.path.join(output_dir, patient_name)
    os.makedirs(output_patient_dir, exist_ok=True)
    
    # Load modalities
    modalities_path = os.path.join(patient_dir, f"{patient_name}_modalities.pt")
    label_path = os.path.join(patient_dir, f"{patient_name}_label.pt")
    
    if os.path.exists(modalities_path) and os.path.exists(label_path):
        # Load
        modalities = torch.load(modalities_path)  # (C, H, W, D)
        labels = torch.load(label_path)  # (C, H, W, D)
        
        # Resize (add batch dimension for interpolate)
        modalities_resized = interpolate(
            modalities.unsqueeze(0).float(),
            size=(96, 96, 96),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        labels_resized = interpolate(
            labels.unsqueeze(0).float(),
            size=(96, 96, 96),
            mode='nearest'
        ).squeeze(0).long()
        
        # Save
        torch.save(modalities_resized, os.path.join(output_patient_dir, f"{patient_name}_modalities.pt"))
        torch.save(labels_resized, os.path.join(output_patient_dir, f"{patient_name}_label.pt"))

print(f"Resized data saved to {output_dir}")

# Create new CSV files pointing to resized data
import csv

def create_csv(csv_path, patient_list, new_root):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['data_path', 'case_name'])
        for patient_name in patient_list:
            writer.writerow([f"{new_root}/{patient_name}", patient_name])

# Read original CSVs
train_patients = []
val_patients = []

with open("/workspaces/SegFormer3D/data/prostate_data/train.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_patients.append(row['case_name'])

with open("/workspaces/SegFormer3D/data/prostate_data/validation.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_patients.append(row['case_name'])

# Create new CSVs
create_csv(
    "/workspaces/SegFormer3D/data/prostate_data/train_96.csv",
    train_patients,
    "./data/prostate_data/preprocessed_96"
)

create_csv(
    "/workspaces/SegFormer3D/data/prostate_data/validation_96.csv",
    val_patients,
    "./data/prostate_data/preprocessed_96"
)

print("CSV files created for 96x96x96 data")
