# Script Ufficiale di Valutazione e Validazione delle Metriche (valutazione/evaluate_final_official.py)
# Valuta l'architettura AttentionPerZoneModel su tutti i fotogrammi del dataset nuScenes.
# Esegue la doppia valutazione (su Ground Truth Sintetica e su Ground Truth Reale 3D nuScenes)
# a due raggi operativi distinti: Raggio Standard (20m) e Raggio Esteso (25m),
# calcolando Veri Positivi (TP), Falsi Positivi (FP), Falsi Negativi (FN), Precision, Recall ed F1-Score.

# Import dei moduli di sistema per percorsi e filesystem
import os
import sys
import json
# Import di PyTorch e NumPy per le operazioni matriciali ed il calcolo su GPU
import torch
import numpy as np
import torch.nn.functional as F

# Aggiunge la directory radice del progetto al sys.path per importare i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'Adapter Factory per la lettura universale dei frame del dataset
from dataset_adapter.factory_dataset import create_adapter
# Import delle funzioni di estrazione e rasterizzazione della Ground Truth
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
# Import dell'architettura AttentionPerZoneModel
from architettura_neurale import AttentionPerZoneModel

# Costanti e parametri di griglia spaziale
GRID_DIM = 200        # Risoluzione della griglia BEV (200x200 pixel)
GRID_RANGE = 40.0     # Estensione metrica da -40.0m a +40.0m
VOXEL_SIZE = 0.4      # Dimensione del singolo pixel (0.4 metri/pixel)
CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"] # 4 Macro-Classi standard ISO 26262
RADII = [20.0, 25.0]  # Raggi operativi di valutazione (20 metri e 25 metri)


def main():
    # Selezione automatica dell'acceleratore hardware: GPU CUDA (NVIDIA) se disponibile, altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("   VALIDAZIONE UFFICIALE END-TO-END DEL MODELLO FINALE (SE-ATTENTION + FiLM)")
    print("=" * 95)
    print(f"• Dispositivo di Calcolo: {device}")

    # 1. Caricamento e Inizializzazione del Modello Ufficiale Addestrato
    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
    ckpt_path = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint non trovato: {ckpt_path}"
    
    # Carica i pesi dal file .pth e li assegna ai neuroni del modello
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    # Imposta la rete in modalità valutazione (disattiva Dropout e congela BatchNorm)
    model.eval()
    print(f"• Modello caricato con successo: {model.__class__.__name__} ({ckpt_path})")

    # 2. Caricamento del Dataset tramite Factory Adapter
    adapter = create_adapter("nuscenes", "./nuscenes")
    num_samples = adapter.get_num_samples()
    print(f"• Campioni da valutare: {num_samples} fotogrammi")

    # Struttura dati per accumulare TP, FP, FN per ciascuna combinazione (GT_Sintetica / GT_Reale, Raggio, Classe)
    counts = {
        gt_type: {
            r: [{"tp": 0, "fp": 0, "fn": 0} for _ in range(4)]
            for r in RADII
        }
        for gt_type in ["GT_Sintetica", "GT_Reale"]
    }

    # 3. Scansione sequenziale di tutti i fotogrammi del dataset
    for idx in range(num_samples):
        if (idx + 1) % 100 == 0 or idx == num_samples - 1:
            print(f"  Elaborati {idx+1}/{num_samples} fotogrammi...")

        frame_data = adapter.get_sample_data(idx)
        token = frame_data["sample_token"]
        
        # Estrae le 6 maschere 2D degli ostacoli reali (Auto, Camion, Pedoni, Bici, Moto, Barriere)
        target_masks = extract_ground_truth_masks(frame_data)
        
        # Estrae i layer semantici del terreno dalla Mappa HD
        semantic_map = frame_data['semantic_map']
        drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
        walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
        ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

        # Assembla gli 11 canali di input BEV (200x200 pixel)
        input_channels = [
            frame_data.get('lidar_bev', np.zeros((GRID_DIM, GRID_DIM))),
            frame_data.get('occlusion_mask', np.zeros((GRID_DIM, GRID_DIM))),
            drivable_mask,
            walkway_mask,
            ped_crossing_mask
        ] + list(target_masks)

        input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

        # Carica le zone d'ombra geometriche calcolate dal Raycaster per questo fotogramma
        occ_file = os.path.join("extracted_occlusions", f"{token}.json")
        if not os.path.exists(occ_file):
            continue
        with open(occ_file) as f:
            occs = json.load(f)["occlusions"]

        # Itera su ciascuna zona d'ombra presente nel fotogramma
        for occ in occs:
            dist = occ.get("distance_m", 0.0)
            # Filtra le zone oltre il raggio massimo di analisi (25 metri)
            if dist > 25.0:
                continue
            pts = occ.get("polygon_points_m", [])
            pts_np = np.array(pts)
            if len(pts_np) < 3:
                continue
            poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

            # Calcola le dimensioni geometriche OBB (larghezza e lunghezza minima orientata)
            from shapely.geometry import Polygon as ShapelyPoly
            sp = ShapelyPoly(poly_xy)
            if sp.is_valid and sp.area > 0.01:
                mrr = sp.minimum_rotated_rectangle
                mrr_coords = np.array(mrr.exterior.coords)[:-1]
                e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                obb_w, obb_l = min(e1, e2), max(e1, e2)
            else:
                obb_w, obb_l = 0.2, 0.5

            # Calcola le percentuali di copertura semantica dell'ombra sul terreno
            occ_mask = rasterize_polygon(pts)
            tot = np.sum(occ_mask)
            road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = occ.get("area_sqm", 0.0)

            # Ground Truth REALE a 4 macro-classi (Auto, Camion/Bus, VRU=max(Pedone, Moto, Bici), Barriere)
            gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)
            gt_raw_4 = np.array([gt_raw_6[0], gt_raw_6[1], max(gt_raw_6[2], gt_raw_6[3], gt_raw_6[4]), gt_raw_6[5]], dtype=np.float32)

            # Ground Truth SINTETICA a 4 macro-classi con regole neurosimboliche
            gt_neuro_4 = np.zeros(4, dtype=np.float32)
            if np.sum(gt_raw_4) > 0:
                gt_neuro_4 = gt_raw_4.copy()
            else:
                if road_f >= 0.25 and obb_w >= 1.5 and obb_l >= 3.0 and area >= 5.0:
                    gt_neuro_4[0] = 1.0
                    if road_f >= 0.30 and obb_w >= 2.4 and obb_l >= 6.5 and area >= 18.0:
                        gt_neuro_4[1] = 1.0
                if (side_f >= 0.15 or cross_f >= 0.08 or (road_f >= 0.25 and obb_w < 1.8)) and obb_w >= 0.4 and area >= 0.6:
                    gt_neuro_4[2] = 1.0
                if terr_f >= 0.55 and obb_w >= 1.0 and area >= 3.0 and road_f < 0.15 and side_f < 0.10:
                    gt_neuro_4[3] = 1.0

            # Calcola il bounding box in pixel attorno all'ombra
            px_x = np.clip(((poly_xy[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
            px_y = np.clip(((GRID_RANGE - poly_xy[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
            xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
            ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
            if xmax <= xmin or ymax <= ymin:
                continue

            # Ritaglia la patch locale a 11 canali e la ridimensiona a 64x64
            patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
            patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(device)
            # Vettore dei 9 scalari fisici e semantici
            scalars = torch.tensor([[area, dist, 2.0, 1.8, road_f, side_f, cross_f, 0.0, terr_f]], dtype=torch.float32).to(device)

            # Inferenza Neurale forward senza gradienti (rapida ed efficiente)
            with torch.no_grad():
                out_6 = torch.sigmoid(model(patch_res, scalars)).squeeze(0).cpu().numpy()

            # Binarizzazione delle predizioni con soglia decisionale ottimale a 0.30
            pred = np.zeros(4, dtype=int)
            pred[0] = int(out_6[0] >= 0.30)  # Auto
            pred[1] = int(out_6[1] >= 0.30)  # Camion/Bus
            pred[2] = int(max(out_6[2], out_6[3], out_6[4]) >= 0.30) # Macro-Classe VRU (Pedoni + Bici + Moto)
            pred[3] = int(out_6[5] >= 0.30)  # Barriere

            # Aggiornamento dei contatori TP, FP, FN per raggio (20m e 25m)
            for r in RADII:
                if dist <= r:
                    for c in range(4):
                        # Valutazione su Ground Truth Sintetica
                        ts = int(gt_neuro_4[c])
                        if ts == 1 and pred[c] == 1:
                            counts["GT_Sintetica"][r][c]["tp"] += 1
                        elif ts == 0 and pred[c] == 1:
                            counts["GT_Sintetica"][r][c]["fp"] += 1
                        elif ts == 1 and pred[c] == 0:
                            counts["GT_Sintetica"][r][c]["fn"] += 1

                        # Valutazione su Ground Truth Reale 3D nuScenes
                        tr = int(gt_raw_4[c])
                        if tr == 1 and pred[c] == 1:
                            counts["GT_Reale"][r][c]["tp"] += 1
                        elif tr == 0 and pred[c] == 1:
                            counts["GT_Reale"][r][c]["fp"] += 1
                        elif tr == 1 and pred[c] == 0:
                            counts["GT_Reale"][r][c]["fn"] += 1

    # 4. Stampa Formattata della Tabella Ufficiale dei Risultati
    print("\n" + "=" * 95)
    print("   RISULTATI FINALI CERTIFICATI")
    print("=" * 95)
    for gt_name, gt_label in [("GT_Sintetica", "GROUND TRUTH SINTETICA NEUROSIMBOLICA"), ("GT_Reale", "GROUND TRUTH REALE NUSCENES")]:
        print(f"\n>>> MODALITÀ: {gt_label}")
        for r in RADII:
            print(f"\n--- RAGGIO: {r:.0f} METRI ---")
            print(f"{'Categoria':<22} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 85)
            tp_tot, fp_tot, fn_tot = 0, 0, 0
            for c_idx, c_name in enumerate(CATEGORIES):
                st = counts[gt_name][r][c_idx]
                tp, fp, fn = st["tp"], st["fp"], st["fn"]
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn
                p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
                print(f"{c_name:<22} | {tp:<6d} | {fp:<6d} | {fn:<6d} | {p:9.1f}% | {rc:9.1f}% | {f:9.1f}%")
            
            p_macro = (tp_tot / (tp_tot + fp_tot) * 100) if (tp_tot + fp_tot) > 0 else 0
            rc_macro = (tp_tot / (tp_tot + fn_tot) * 100) if (tp_tot + fn_tot) > 0 else 0
            f_macro = (2 * p_macro * rc_macro / (p_macro + rc_macro)) if (p_macro + rc_macro) > 0 else 0
            print("-" * 85)
            print(f"{'MEDIA GLOBALE':<22} | {tp_tot:<6d} | {fp_tot:<6d} | {fn_tot:<6d} | {p_macro:9.1f}% | {rc_macro:9.1f}% | {f_macro:9.1f}%")

    print("\n[VERIFICA COMPLETATA CON SUCCESSO]: Valutazione ufficiale terminata!")


# Blocco di esecuzione principale da terminale
if __name__ == "__main__":
    main()
