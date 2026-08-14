# Script di Addestramento Focal Loss Neurosimbolica Completa (train_per_zone_focal_completa.py)
# Addestra il modello PerZoneModel con FocalLoss_neurosimbolica_completa (20 Epoche con Cosine Scheduler)
# Salva il checkpoint dei pesi addestrati nel file per_zone_checkpoint_focal_completa.pth

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from architettura_neurale.per_zone_model import PerZoneModel
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetPerZone
from architettura_neurale.loss_functions import FocalLoss_neurosimbolica_completa

def train_focal_completa_model(epochs=20, batch_size=32, lr=1e-3, checkpoint_path=os.path.join("pesi_modelli", "per_zone_checkpoint_focal_completa.pth")):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print("\n" + "=" * 85)
    print("   ADDESTRAMENTO VARIANTE: FOCAL LOSS NEUROSIMBOLICA COMPLETA (INIBIZIONE + PROMOZIONE VRU - 20 EPOCHE)")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di Calcolo Selezionato: {device}")

    dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Campioni Ombra Estratti: {len(dataset)} | Num Batches (size={batch_size}): {len(dataloader)}")

    model = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    criterion = FocalLoss_neurosimbolica_completa(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nInizio Addestramento Focal Loss Completa per {epochs} Epoche...\n")
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
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(dataloader)
        print(f"  [Epoca {epoch+1:2d}/{epochs:2d}] Loss Media Focal Completa: {avg_loss:.4f} | LR: {current_lr:.6f}")

    total_time = time.time() - start_time
    print(f"\nAddestramento Focal Loss Completa terminato in {total_time:.2f} secondi.")

    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, checkpoint_path)
    print(f"Pesi salvati in: {os.path.abspath(checkpoint_path)}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    train_focal_completa_model(epochs=20, batch_size=32, lr=1e-3)
