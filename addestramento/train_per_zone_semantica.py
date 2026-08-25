# Script di Addestramento Variante 2: BCE + Penalizzazione Zona Semantica (train_per_zone_semantica.py)
# Addestra il modello PerZoneModel con WeightedBCESemanticLoss_penalizzazione_zona_semantica (10 Epoche)
# Salva il checkpoint dei pesi addestrati nel file per_zone_checkpoint_semantica.pth

# Import delle librerie di sistema per la manipolazione dei percorsi di ambiente
import os
import sys
# Import di time per il tracciamento dei tempi di esecuzione ed addestramento
import time
# Import di torch per il calcolo ed i modelli di Deep Learning PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import di DataLoader per il caricamento a batch del dataset
from torch.utils.data import DataLoader

# Aggiunge la cartella radice del progetto al sys.path per consentire l'importazione di moduli accessori
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'architettura neurale PerZoneModel dal pacchetto dedicato
from architettura_neurale.per_zone_model import PerZoneModel
# Import del caricatore del dataset OcclusionDatasetPerZone dal pacchetto adapter
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetPerZone
# Import della funzione di perdita BCE con penalizzazione semantica
from architettura_neurale.loss_functions import WeightedBCESemanticLoss_penalizzazione_zona_semantica

def train_bce_semantica_model(epochs=10, batch_size=32, lr=1e-3, checkpoint_path=os.path.join("pesi_modelli", "per_zone_checkpoint_semantica.pth")):
    # Assicura la presenza della cartella pesi_modelli su disco
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print("\n" + "=" * 75)
    print("   ADDESTRAMENTO VARIANTE: BCE LOSS + PENALIZZAZIONE ZONA SEMANTICA (10 EPOCHE)")
    print("=" * 75)

    # Selezione automatica del dispositivo di calcolo (CUDA GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di Calcolo Selezionato: {device}")

    # Caricamento del Dataset Per-Zone
    dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Campioni Ombra Estratti: {len(dataset)} | Num Batches (size={batch_size}): {len(dataloader)}")

    # Inizializzazione Rete Neurale Ibrida PerZoneModel
    model = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    # Funzione di Perdita BCE con Penalizzazione Semantica ed Ottimizzatore AdamW
    criterion = WeightedBCESemanticLoss_penalizzazione_zona_semantica().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Ciclo di Addestramento per Epoche
    print(f"\nInizio Addestramento BCE Semantica per {epochs} Epoche...\n")
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
            loss = criterion(logits, target_batch, scalar_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"  [Epoca {epoch+1:2d}/{epochs:2d}] Loss Media BCE Semantica: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nAddestramento BCE Semantica completato in {total_time:.2f} secondi.")

    # Salvataggio del Checkpoint con i pesi addestrati (.pth)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, checkpoint_path)
    print(f"Pesi della Variante BCE Semantica salvati in: {os.path.abspath(checkpoint_path)}")
    print("=" * 75 + "\n")

# Blocco principale di esecuzione se avviato da riga di comando
if __name__ == "__main__":
    train_bce_semantica_model(epochs=10, batch_size=32, lr=1e-3)
