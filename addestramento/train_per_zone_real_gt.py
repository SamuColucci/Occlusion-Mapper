# Script di Addestramento di Baseline: Modello AttentionPerZoneModel su Ground Truth REALE nuScenes (train_per_zone_real_gt.py)
# Addestra l'architettura AttentionPerZoneModel (11 canali BEV, 9 scalari, 6 classi di pericolo)
# utilizzando la funzione Asymmetric Loss (ASL) e la supervisione diretta dei box 3D annotati da operatori umani (GT Reale).

# Import dei moduli di sistema per percorsi, filesystem e misurazione tempi
import os
import sys
import time
import json
# Import di PyTorch e NumPy per il calcolo tensoriale e su GPU
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Aggiunge la directory radice del progetto al sys.path per importare i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'Adapter Factory per la lettura universale dei frame del dataset
from dataset_adapter.factory_dataset import create_adapter
# Import delle funzioni di estrazione e rasterizzazione della Ground Truth
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
# Import dell'architettura AttentionPerZoneModel con Channel Attention (SE) e FiLM
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel
# Import della funzione di costo Asymmetric Loss (CVPR 2021)
from architettura_neurale.loss_functions import AsymmetricLoss


def train_real_gt(epochs=20, batch_size=64, lr=1e-3, checkpoint_path=os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_attention_real_gt.pth")):
    # Selezione automatica dell'acceleratore hardware: GPU CUDA (NVIDIA) se disponibile, altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===========================================================================")
    print(f"   ADDESTRAMENTO MODELLO ATTENZIONE SU GROUND TRUTH REALE NUSCENES")
    print(f"===========================================================================")
    print(f"Dispositivo Selezionato: {device}")

    # Percorso per il salvataggio dei tensori pre-elaborati con GT Reale in cache su disco
    cache_path = os.path.join("addestramento", "cached_dataset_per_zone.pth")
    if os.path.exists(cache_path):
        # Se la cache esiste, carica direttamente i tensori già estratti e pronti
        print(f"Caricamento dataset con GT Reale dalla cache: {cache_path}...")
        cache_data = torch.load(cache_path)
        patches, scalars, targets = cache_data["patches"], cache_data["scalars"], cache_data["targets"]
    else:
        # Altrimenti, estrae e ritaglia tutti i campioni dal dataset nuScenes da zero
        print("Generazione dataset con etichette GT Reale dai fotogrammi...")
        adapter = create_adapter("nuscenes", "./nuscenes")
        num_samples = adapter.get_num_samples()

        patches_list, scalars_list, targets_list = [], [], []

        # Scorre ciascun fotogramma del dataset
        for idx in range(num_samples):
            if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
                print(f"  Elaborazione: {idx+1}/{num_samples} frame...")

            frame_data = adapter.get_sample_data(idx)
            token = frame_data["sample_token"]
            
            # Estrae le 6 maschere 2D degli ostacoli reali (Auto, Camion, Pedoni, Bici, Moto, Barriere)
            target_masks = extract_ground_truth_masks(frame_data)

            # Estrae i layer semantici del terreno dalla Mappa HD
            semantic_map = frame_data['semantic_map']
            drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
            walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
            ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

            # Assembla gli 11 canali di input BEV (200x200):
            # 0: LiDAR, 1: Maschera Ombre, 2: Asfalto/Parcheggi, 3: Marciapiede, 4: Strisce, 5-10: Ostacoli Visibili
            input_channels = [
                frame_data.get('lidar_bev', np.zeros((200, 200))),
                frame_data.get('occlusion_mask', np.zeros((200, 200))),
                drivable_mask,
                walkway_mask,
                ped_crossing_mask
            ] + list(target_masks)

            input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

            # Carica le zone d'ombra estratte dal Raycaster per questo fotogramma
            occ_file = os.path.join("extracted_occlusions", f"{token}.json")
            if not os.path.exists(occ_file):
                continue
            with open(occ_file, "r") as f:
                occs = json.load(f)["occlusions"]

            # Itera su ciascuna zona d'ombra presente nel fotogramma
            for occ in occs:
                pts = occ.get("polygon_points_m", [])
                if len(pts) < 3:
                    continue
                pts_np = np.array(pts)
                poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

                # Calcola le percentuali di copertura semantica dell'ombra sul terreno
                occ_mask = rasterize_polygon(pts)
                tot_pixels = np.sum(occ_mask)
                road_fraction = np.sum(occ_mask * drivable_mask) / tot_pixels if tot_pixels > 0 else 0
                sidewalk_fraction = np.sum(occ_mask * walkway_mask) / tot_pixels if tot_pixels > 0 else 0
                ped_crossing_fraction = np.sum(occ_mask * ped_crossing_mask) / tot_pixels if tot_pixels > 0 else 0
                terrain_fraction = max(0.0, 1.0 - (road_fraction + sidewalk_fraction + ped_crossing_fraction))

                area = occ.get("area_sqm", 0.0)
                dist = occ.get("distance_m", 0.0)

                # Estrazione diretta della Ground Truth Reale 3D nuScenes a 6 classi (nessuna regola sintetica)
                gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)

                # Calcola il bounding box in pixel attorno all'ombra nella mappa 200x200
                px_x = np.clip(((poly_xy[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
                px_y = np.clip(((40.0 - poly_xy[:, 1]) / 0.4).astype(int), 0, 199)
                xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Ritaglia la patch locale a 11 canali e la ridimensiona a risoluzione fissa 64x64
                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                # Vettore dei 9 scalari fisici e semantici
                scalars_val = torch.tensor([area, dist, 2.0, 1.8, road_fraction, sidewalk_fraction, ped_crossing_fraction, 0.0, terrain_fraction], dtype=torch.float32)
                target_val = torch.tensor(gt_raw_6, dtype=torch.float32)

                patches_list.append(patch_res)
                scalars_list.append(scalars_val)
                targets_list.append(target_val)

        # Converte le liste in tensori PyTorch unificati
        patches = torch.stack(patches_list)
        scalars = torch.stack(scalars_list)
        targets = torch.stack(targets_list)
        # Salva la cache su disco per velocizzare i successivi avvii
        torch.save({"patches": patches, "scalars": scalars, "targets": targets}, cache_path)

    print(f"Campioni Totali: {len(patches)} | Batches per Epoca: {int(np.ceil(len(patches)/batch_size))}")

    # Creazione del Dataset PyTorch e del DataLoader con mescolamento casuale (shuffle=True)
    dataset = TensorDataset(patches, scalars, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Inizializzazione del Modello Neurale AttentionPerZoneModel su GPU
    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
    # Funzione di costo Asymmetric Loss (gamma_neg=4.0 per schiacciare i falsi allarmi, pos_weights per enfasi VRU)
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05, pos_weights=[2.0, 3.0, 3.0, 3.0, 3.0, 2.0])
    # Ottimizzatore AdamW con regolarizzazione Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler Cosine Annealing per ridurre dolcemente il Learning Rate fino all'ultima epoca
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nInizio Addestramento su GT Reale per {epochs} Epoche...\n")
    start_t = time.time()

    # Ciclo principale di Addestramento sulle 20 Epoche
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for p_b, s_b, t_b in loader:
            # Trasferisce il batch di dati sulla memoria della GPU
            p_b, s_b, t_b = p_b.to(device), s_b.to(device), t_b.to(device)
            
            # 1. Azzera i gradienti accumulati nel passo precedente
            optimizer.zero_grad()
            # 2. Forward Pass: calcola i logit predetti dal modello
            logits = model(p_b, s_b)
            # 3. Calcolo dell'errore asimmetrico confrontando i logit con la Ground Truth REALE
            loss = criterion(logits, t_b)
            # 4. Backward Pass: retropropagazione del gradiente lungo tutti i pesi della rete
            loss.backward()
            # 5. Ottimizzazione: aggiorna i pesi dei neuroni con AdamW
            optimizer.step()
            
            epoch_loss += loss.item() * len(p_b)

        # Aggiorna il Learning Rate a fine epoca secondo la curva cosinusoidale
        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoca {epoch+1:2d}/{epochs}] ASL Loss: {avg_loss:.5f} | LR: {cur_lr:.6f}")

    elapsed = time.time() - start_t
    print(f"\nAddestramento su GT Reale completato in {elapsed:.2f} secondi.")
    
    # Salva il dizionario completo del checkpoint (pesi addestrati, architettura, epoche e loss finale)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "AttentionPerZoneModel_RealGT",
        "epochs": epochs,
        "final_loss": avg_loss
    }, checkpoint_path)
    print(f"Pesi salvati con successo in: {checkpoint_path}")


# Blocco di esecuzione principale da terminale
if __name__ == "__main__":
    train_real_gt(epochs=20, batch_size=64)
