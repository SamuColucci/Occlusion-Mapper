# Script di Addestramento variante Asymmetric Loss Standard (train_per_zone_asl_standard.py)
# Addestra il modello PerZoneModel utilizzando la Asymmetric Loss (ASL - Ridnik et al., CVPR 2021) 
# standard SENZA la Penalizzazione della Zona Semantica per lo studio di ablazione.
# Salva il checkpoint dei pesi addestrati nel file per_zone_checkpoint_asl_standard.pth

# Import delle librerie di sistema per la manipolazione dei percorsi di ambiente
import os
import sys
# Import di time per il tracciamento dei tempi di esecuzione ed addestramento
import time
# Import di torch per il calcolo ed i modelli di Deep Learning PyTorch
import torch
# Import di DataLoader per il caricamento a batch del dataset
from torch.utils.data import DataLoader

# Aggiunge la cartella radice del progetto al sys.path per consentire l'importazione di moduli accessori
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'architettura neurale PerZoneModel dal pacchetto dedicato
from architettura_neurale.per_zone_model import PerZoneModel
# Import del caricatore del dataset OcclusionDatasetPerZone dal pacchetto adapter
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetPerZone
# Import della funzione di perdita Asymmetric Loss standard
from architettura_neurale.loss_functions import AsymmetricLoss

def train_asl_standard_model(epochs=20, batch_size=32, lr=1e-3, checkpoint_path=os.path.join("pesi_modelli", "per_zone_checkpoint_asl_standard.pth")):
    # Assicura la presenza della cartella pesi_modelli su disco
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print("\n" + "=" * 80)
    print("   ADDESTRAMENTO VARIANTE: ASYMMETRIC LOSS (CVPR 2021) STANDARD (SENZA REGOLE - 20 EPOCHE)")
    print("=" * 80)

    # Selezione automatica del dispositivo di calcolo (CUDA GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di Calcolo Selezionato: {device}")

    # Caricamento del Dataset Per-Zone
    dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Campioni Ombra Estratti: {len(dataset)} | Num Batches (size={batch_size}): {len(dataloader)}")

    # Inizializzazione Rete Neurale Ibrida PerZoneModel
    model = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    # Inizializza la Asymmetric Loss con Margin Shift (gamma_neg=4.0, clip=0.05)
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Ciclo di Addestramento per Epoche
    print(f"\nInizio Addestramento Asymmetric Loss Standard per {epochs} Epoche...\n")
    start_time = time.time()
    
    # Imposta il modello in modalità addestramento
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Scorre ciascun mini-batch fornito dal DataLoader
        for patch_batch, scalar_batch, target_batch in dataloader:
            patch_batch = patch_batch.to(device, dtype=torch.float32)
            scalar_batch = scalar_batch.to(device, dtype=torch.float32)
            target_batch = target_batch.to(device, dtype=torch.float32)
            
            # Azzeramento dei gradienti
            optimizer.zero_grad()
            logits = model(patch_batch, scalar_batch)
            
            # Calcolo della Loss
            loss = criterion(logits, target_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Aggiornamento dello scheduler di Learning Rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(dataloader)
        print(f"  [Epoca {epoch+1:2d}/{epochs:2d}] Loss Media ASL Standard: {avg_loss:.4f} | LR: {current_lr:.6f}")

    total_time = time.time() - start_time
    print(f"\nAddestramento Asymmetric Loss Standard completato in {total_time:.2f} secondi.")

    # Salvataggio del Checkpoint con i pesi addestrati (.pth)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, checkpoint_path)
    print(f"Pesi della Variante Asymmetric Loss Standard salvati in: {os.path.abspath(checkpoint_path)}")
    print("=" * 80 + "\n")

# Blocco principale di esecuzione se avviato da riga di comando
if __name__ == "__main__":
    train_asl_standard_model(epochs=20, batch_size=32, lr=1e-3)
