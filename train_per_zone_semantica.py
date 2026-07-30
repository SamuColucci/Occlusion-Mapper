# Script di Addestramento Variante 2: BCE + Penalizzazione Zona Semantica (train_per_zone_semantica.py)
# Addestra il modello PerZoneModel con WeightedBCESemanticLoss_penalizzazione_zona_semantica (10 Epoche)
# Salva il checkpoint dei pesi addestrati nel file per_zone_checkpoint_semantica.pth

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from per_zone_model import PerZoneModel
from dataset_generator_per_zone import OcclusionDatasetPerZone
from loss_functions import WeightedBCESemanticLoss_penalizzazione_zona_semantica

def train_bce_semantica_model(epochs=10, batch_size=32, lr=1e-3, checkpoint_path="per_zone_checkpoint_semantica.pth"):
    print("\n" + "=" * 75)
    print("   ADDESTRAMENTO VARIANTE: BCE LOSS + PENALIZZAZIONE ZONA SEMANTICA (10 EPOCHE)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di Calcolo Selezionato: {device}")

    dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Campioni Ombra Estratti: {len(dataset)} | Num Batches (size={batch_size}): {len(dataloader)}")

    model = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    criterion = WeightedBCESemanticLoss_penalizzazione_zona_semantica().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\nInizio Addestramento BCE + Penalizzazione Semantica per {epochs} Epoche...\n")
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for patch_batch, scalar_batch, target_batch in dataloader:
            patch_batch = patch_batch.to(device, dtype=torch.float32)
            scalar_batch = scalar_batch.to(device, dtype=torch.float32)
            target_batch = target_batch.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            logits = model(patch_batch, scalar_batch)
            
            loss = criterion(logits, target_batch, scalar_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"  [Epoca {epoch+1:2d}/{epochs:2d}] Loss Media BCE Semantica: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nAddestramento BCE + Penalizzazione Semantica completato in {total_time:.2f} secondi.")

    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, checkpoint_path)
    print(f"Pesi della Variante BCE Semantica salvati in: {os.path.abspath(checkpoint_path)}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    train_bce_semantica_model(epochs=10, batch_size=32, lr=1e-3)
