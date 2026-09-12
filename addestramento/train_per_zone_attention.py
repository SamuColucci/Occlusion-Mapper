# Script Ufficiale di Addestramento Unificato per AttentionPerZoneModel (addestramento/train_per_zone_attention.py)
# Addestra l'architettura neurale con Channel Attention (Squeeze-and-Excitation) e condizionamento FiLM
# supportando tutte le 4 modalità di supervisione dello studio di ablazione della tesi:
# 1. 'geometric'      : GT Sintetica Neurosimbolica con vincoli 3D (Spatially-Constrained / Physical-Aware)
# 2. 'semantic'       : GT Sintetica Neurosimbolica permissiva (Semantic-Affordance / Naïve Prior)
# 3. 'real'           : GT Reale nuScenes completa (18.682 zone, inclusi i vuoti)
# 4. 'positives_only' : GT Reale nuScenes addestrata ESCLUSIVAMENTE sulle zone con ostacoli reali
# 5. 'all'            : Esegue l'addestramento sequenziale completo di tutte le configurazioni.

import os
import sys
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Aggiunge la directory radice del progetto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
from ground_truth.ground_truth_extractor_synthetic import compute_synthetic_ground_truth
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel
from architettura_neurale.loss_functions import AsymmetricLoss


def train_attention_neuro(epochs=20, batch_size=64, lr=1e-3, gt_mode="geometric", checkpoint_path=None, split="train", force_extract=False):
    """
    Funzione di addestramento universale.
    Supporta:
      - gt_mode: 'geometric', 'semantic', 'hybrid', 'real', 'positives_only', 'all'.
      - split: 'train' (8 scene mini_train, default), 'val' (2 scene mini_val), 'all' (tutte).
    """
    if gt_mode == "all":
        print("\n" + "=" * 80)
        print(f"   AVVIO ADDESTRAMENTO SEQUENZIALE DI TUTTI I MODELLI (--mode all, split: {split.upper()})")
        print("=" * 80)
        modes_to_train = ["geometric", "semantic", "hybrid", "real", "positives_only"]
        for m in modes_to_train:
            train_attention_neuro(epochs=epochs, batch_size=batch_size, lr=lr, gt_mode=m, split=split, force_extract=force_extract)
        print("\n[COMPLETATO] Addestramento sequenziale di tutti i modelli terminato con successo!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mappatura automatica dei percorsi checkpoint
    if checkpoint_path is None:
        ckpt_map = {
            "geometric": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_geometric.pth"),
            "semantic": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_semantic.pth"),
            "hybrid": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
            "real": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth"),
            "positives_only": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_positive_only.pth"),
        }
        checkpoint_path = ckpt_map.get(gt_mode, os.path.join("pesi_modelli", f"per_zone_checkpoint_attention_{gt_mode}.pth"))

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"\n===========================================================================")
    print(f"   ADDESTRAMENTO ATTENTION-PER-ZONE: [{gt_mode.upper()}] (SPLIT: {split.upper()})")
    print(f"===========================================================================")
    print(f"• Dispositivo Selezionato: {device}")
    print(f"• Modalità Supervisione: {gt_mode.upper()}")
    print(f"• Split Dataset: {split.upper()} (Ufficiale nuScenes)")
    print(f"• Checkpoint di Output: {checkpoint_path}")

    # Gestione Cache differenziata per modalità e split
    split_suffix = f"_{split}" if split != "all" else ""
    if gt_mode in ["real", "positives_only"]:
        cache_path = os.path.join("addestramento", f"cached_dataset_per_zone{split_suffix}.pth")
    else:
        cache_path = os.path.join("addestramento", f"cached_dataset_neuro_{gt_mode}{split_suffix}.pth")

    if os.path.exists(cache_path) and not force_extract:
        print(f"• Caricamento dataset dalla cache: {cache_path}...")
        cache_data = torch.load(cache_path)
        patches, scalars, targets = cache_data["patches"], cache_data["scalars"], cache_data["targets"]

        # Se positives_only: filtra mantenendo SOLO le zone con almeno un ostacolo reale
        if gt_mode == "positives_only":
            pos_mask = (targets.sum(dim=1) > 0)
            patches = patches[pos_mask]
            scalars = scalars[pos_mask]
            targets = targets[pos_mask]
            print(f"  [Filtro Positives-Only]: Mantenuti {len(patches)} campioni con ostacoli reali (escluse zone vuote).")
    else:
        print(f"• Generazione dataset [{gt_mode}] (split: {split.upper()}) da zero...")
        adapter = create_adapter("nuscenes", "./nuscenes")
        num_samples = adapter.get_num_samples()

        from nuscenes.utils.splits import create_splits_scenes
        splits_dict = create_splits_scenes()
        train_scenes = set(splits_dict.get('mini_train', []))
        val_scenes = set(splits_dict.get('mini_val', []))

        patches_list, scalars_list, targets_list = [], [], []

        for idx in range(num_samples):
            sample = adapter.all_samples[idx]
            sc_name = adapter.nusc.get('scene', sample['scene_token'])['name']

            # Filtro per scena in base allo split richiesto
            if split == "train" and sc_name not in train_scenes:
                continue
            elif split == "val" and sc_name not in val_scenes:
                continue

            if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
                print(f"  Elaborazione: frame {idx+1}/{num_samples} (scena: {sc_name})...")

            frame_data = adapter.get_sample_data(idx)
            token = frame_data["sample_token"]
            
            target_masks = extract_ground_truth_masks(frame_data)
            semantic_map = frame_data['semantic_map']
            drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
            walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
            ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

            input_channels = [
                frame_data.get('lidar_bev', np.zeros((200, 200))),
                frame_data.get('occlusion_mask', np.zeros((200, 200))),
                drivable_mask,
                walkway_mask,
                ped_crossing_mask
            ] + list(target_masks)

            input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

            occ_file = os.path.join("extracted_occlusions", f"{token}.json")
            if not os.path.exists(occ_file):
                continue
            with open(occ_file) as f:
                occs = json.load(f)["occlusions"]

            import cv2
            dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * 0.4

            for occ in occs:
                pts = occ.get("polygon_points_m", [])
                pts_np = np.array(pts)
                if len(pts_np) < 3:
                    continue
                poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

                from shapely.geometry import Polygon as ShapelyPoly
                sp = ShapelyPoly(poly_xy)

                occ_mask = rasterize_polygon(pts)
                tot = np.sum(occ_mask)
                road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                area = float(occ.get("area_sqm", 0.0))
                dist = float(occ.get("distance_m", 0.0))

                # Calcolo prossimità al bordo strada (accosto / sosta entro 2.5m)
                min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
                roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

                # Step 1: Estrae la Ground Truth Reale nuScenes a 6 classi
                gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)

                # Se positives_only e la zona e vuota, la saltiamo
                if gt_mode == "positives_only" and np.sum(gt_raw_6) == 0:
                    continue

                # Step 2: Calcolo del target in base alla modalita
                if gt_mode in ["real", "positives_only"]:
                    gt_target = np.array(gt_raw_6, dtype=np.float32)
                else:
                    gt_target, _ = compute_synthetic_ground_truth(
                        sp=sp,
                        road_f=road_f,
                        side_f=side_f,
                        cross_f=cross_f,
                        roadside_f=roadside_f,
                        area=area,
                        dist=dist,
                        gt_raw_6=gt_raw_6,
                        mode=gt_mode
                    )

                px_x = np.clip(((poly_xy[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
                px_y = np.clip(((40.0 - poly_xy[:, 1]) / 0.4).astype(int), 0, 199)
                xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                # Calcolo larghezza e lunghezza effettive del varco dell'ombra tramite OBB
                if sp.is_valid and sp.area > 0.01:
                    mrr = sp.minimum_rotated_rectangle
                    mrr_coords = np.array(mrr.exterior.coords)[:-1]
                    e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                    e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                    obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))
                else:
                    obb_w, obb_l = 0.5, 0.5

                scalars = torch.tensor([area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f], dtype=torch.float32)
                target = torch.tensor(gt_target, dtype=torch.float32)

                patches_list.append(patch_res)
                scalars_list.append(scalars)
                targets_list.append(target)

        patches = torch.stack(patches_list)
        scalars = torch.stack(scalars_list)
        targets = torch.stack(targets_list)

        if gt_mode != "positives_only":
            torch.save({"patches": patches, "scalars": scalars, "targets": targets}, cache_path)
            print(f"• Cache salvata in: {cache_path}")

    # DataLoader
    dataset = TensorDataset(patches, scalars, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"• Campioni Totali: {len(dataset)} | Batches per Epoca: {len(dataloader)}")

    # Inizializzazione del Modello AttentionPerZoneModel
    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    # Funzione di costo Asymmetric Loss Bilanciata (gamma_pos=0.5 per compromesso ideale Recall/Precision)
    # Evita l'ipersensibilità e abbatte i falsi positivi mantenendo altissima la copertura
    if gt_mode in ["real", "positives_only"]:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.5, 2.5, 2.5, 2.0, 2.0, 2.5])
    elif gt_mode == "hybrid":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.2, 2.2, 2.0, 1.8, 1.8, 2.2])
    else:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nInizio Addestramento [{gt_mode.upper()}] per {epochs} Epoche...\n")
    start_t = time.time()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for p_b, s_b, t_b in dataloader:
            p_b, s_b, t_b = p_b.to(device), s_b.to(device), t_b.to(device)
            optimizer.zero_grad()
            logits = model(p_b, s_b)
            loss = criterion(logits, t_b, s_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(p_b)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoca {epoch+1:2d}/{epochs}] ASL Loss: {avg_loss:.5f} | LR: {cur_lr:.6f}")

    elapsed = time.time() - start_t
    print(f"\nAddestramento [{gt_mode.upper()}] completato in {elapsed:.2f} secondi.")

    ckpt_dict = {
        "model_state_dict": model.state_dict(),
        "architecture": f"AttentionPerZoneModel_{gt_mode}",
        "epochs": epochs,
        "final_loss": avg_loss,
        "gt_mode": gt_mode,
        "split": split
    }
    torch.save(ckpt_dict, checkpoint_path)
    print(f"[OK] Pesi salvati con successo in: {checkpoint_path}")

    # Se geometric, aggiorna anche il default del repository
    if gt_mode == "geometric":
        default_ckpt = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        torch.save(ckpt_dict, default_ckpt)
        print(f"[OK] Pesi standard aggiornati in: {default_ckpt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Addestramento AttentionPerZoneModel su GT Sintetica o Reale")
    parser.add_argument("--mode", "--gt_mode", dest="mode", type=str, default="hybrid",
                        choices=["geometric", "semantic", "hybrid", "real", "positives_only", "all"],
                        help="Modalità: 'geometric', 'semantic', 'hybrid', 'real', 'positives_only' o 'all' per addestrare tutti")
    parser.add_argument("--epochs", type=int, default=20, help="Numero di epoche (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch (default: 64)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                        help="Split nuScenes: 'train' (8 scene, default), 'val' (2 scene) o 'all' (10 scene)")
    parser.add_argument("--force", "--force_extract", dest="force", action="store_true", help="Forza la rigenerazione della cache dataset da zero")
    args = parser.parse_args()

    train_attention_neuro(epochs=args.epochs, batch_size=args.batch_size, gt_mode=args.mode, split=args.split, force_extract=args.force)
