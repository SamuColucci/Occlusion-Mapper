"""
Script di Addestramento per la UNet multi-classe con Logging Dettagliato.

Questo script allena la rete UNet fornendo feedback in tempo reale sul terminale
per monitorare l'avvio del dataset e l'avanzamento dei batch durante ogni epoca.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_generator import OcclusionDataset
from unet_model import UNetBEV

def train(epochs=5, batch_size=8, lr=1e-3):
    start_time = time.time()
    
    # 1. Rilevamento hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"[1/4] LOG HARDWARE: Dispositivo rilevato -> {device.type.upper()}")
    print("="*60)
    
    # 2. Caricamento del Dataset (Fase lenta legata all'SDK NuScenes)
    print("\n[2/4] CARICAMENTO DATASET...")
    print("  -> Inizializzazione SDK NuScenes (scansione metadati, attendere circa 10-15 secondi)...")
    dataset_start = time.time()
    try:
        dataset = OcclusionDataset()
        num_samples = len(dataset)
        if num_samples == 0:
            print("[ERROR] Nessun file JSON geometrico trovato in 'extracted_occlusions'!")
            return
        print(f"  -> Dataset caricato con successo in {time.time() - dataset_start:.1f} secondi.")
        print(f"  -> Campioni totali pronti per l'addestramento: {num_samples}")
    except Exception as e:
        print(f"[ERROR] Impossibile caricare il dataset: {e}")
        return

    # Creazione Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)
    
    # 3. Istanziazione Modello e Ottimizzatori
    print("\n[3/4] INIZIALIZZAZIONE RETE NEURALE UNet...")
    model = UNetBEV(in_channels=4, out_channels=3)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("  -> Rete creata (4 canali input, 3 canali output semantici).")
    print("  -> Ottimizzatore Adam configurato.")
    
    # 4. Loop di addestramento con logging per batch
    print("\n[4/4] AVVIO CICLO DI ADDESTRAMENTO (LOOP)...")
    print(f"  -> Totale Epoche: {epochs} | Batch per epoca: {num_batches} (Batch Size: {batch_size})")
    print("-"*60)
    
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        print(f"\n[EPOCA {epoch+1}/{epochs}]")
        
        for batch_idx, (x, y) in enumerate(dataloader):
            batch_start = time.time()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Log ogni 5 batch per non intasare lo schermo, ma mostrando l'avanzamento reale
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - batch_start
                print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {current_loss:.4f} | Tempo batch: {elapsed:.2f}s")
                
        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / num_batches
        print(f"-> Fine Epoca {epoch+1} | Loss Media: {mean_loss:.4f} | Tempo Epoca: {epoch_time:.1f} secondi")
        print("-"*60)
        
    # 5. Salvataggio finale
    print("\nSALVATAGGIO MODELLO...")
    torch.save(model.state_dict(), "best_model.pth")
    
    total_time = time.time() - start_time
    print(f"[SUCCESS] Modello salvato in 'best_model.pth'!")
    print(f"Tempo totale complessivo di esecuzione: {total_time/60:.2f} minuti.")
    print("="*60 + "\n")

if __name__ == "__main__":
    train(epochs=5, batch_size=8, lr=1e-3)
