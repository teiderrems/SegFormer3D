#!/usr/bin/env python
"""Minimal training test script"""
import sys
import os
sys.path.insert(0, '/workspaces/SegFormer3D')

import torch
import yaml
from tqdm import tqdm

# Load config
with open('/workspaces/SegFormer3D/experiments/prostate_seg/config_prostate.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("[1/5] Loading config...")

# Build model
from architectures.build_architecture import build_architecture
model = build_architecture(config)
print("[2/5] Model built successfully")

# Build datasets and dataloaders
from dataloaders.build_dataset import build_dataloaders
train_dataloader, val_dataloader = build_dataloaders(config)
print(f"[3/5] Dataloaders built - Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches")

# Build optimizer
from optimizers.optimizers import build_optimizer
optimizer = build_optimizer(model, config)
print("[4/5] Optimizer built successfully")

# Build criterion
from losses.losses import build_loss
criterion = build_loss(config)
print("[5/5] Loss criterion built successfully")

print("\n=== Starting simple training loop ===")

# Simple training without Accelerator
device = "cpu"
model.to(device)
model.train()

num_epochs = 2
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        if batch_idx >= 3:  # Only process first 3 batches for testing
            break
            
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")
    
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

print("\n=== Training completed successfully! ===")
