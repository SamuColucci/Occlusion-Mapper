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
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPoly


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
        masks = cache_data.get("masks", torch.ones_like(targets))

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

        patches_list, scalars_list, masks_list, targets_list = [], [], [], []

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

            occ_file = os.path.join("extracted_occlusions", f"{token}.json")
            if not os.path.exists(occ_file):
                continue
            with open(occ_file) as f:
                occs = json.load(f)["occlusions"]

            # Maschera circolare metrica a 25m per tagliare rigorosamente TUTTI gli 11 canali di input
            r_grid = np.arange(200)
            gy, gx = np.meshgrid(r_grid, r_grid)
            mask_25m = (np.sqrt((gx - 99.5)**2 + (gy - 99.5)**2) * 0.4 <= 25.0).astype(np.float32)
            circle_25m = ShapelyPoint(0, 0).buffer(25.0)

            # 1. Canale 0: LiDAR Density BEV entro 25m
            lidar_bev = np.zeros((200, 200), dtype=np.float32)
            points = frame_data.get("points", np.zeros((0, 3)))
            if len(points) > 0:
                for px_m, py_m in points[:, :2]:
                    px = int((px_m + 40.0) / 0.4)
                    py = int((40.0 - py_m) / 0.4)
                    if 0 <= px < 200 and 0 <= py < 200:
                        lidar_bev[py, px] = 1.0
            lidar_bev = lidar_bev * mask_25m

            # 2. Canale 1: Ombre Raycasting cumulative entro 25m
            occlusion_mask = np.zeros((200, 200), dtype=np.float32)
            for occ in occs:
                pts = occ.get("polygon_points_m", [])
                if len(pts) >= 3:
                    sp_occ = ShapelyPoly(pts)
                    if not sp_occ.is_valid:
                        sp_occ = sp_occ.buffer(0)
                    sp_occ_25 = sp_occ.intersection(circle_25m)
                    if not sp_occ_25.is_empty and sp_occ_25.area >= 0.1:
                        sub_geoms = list(sp_occ_25.geoms) if sp_occ_25.geom_type == 'MultiPolygon' else [sp_occ_25]
                        for sg in sub_geoms:
                            coords = np.array(sg.exterior.coords)[:-1]
                            m = rasterize_polygon(coords)
                            occlusion_mask = np.maximum(occlusion_mask, m)
            occlusion_mask = occlusion_mask * mask_25m

            # 3. Canali 2, 3, 4: HD Map entro 25m
            semantic_map = frame_data['semantic_map']
            drivable_mask = (np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))) * mask_25m
            walkway_mask = (semantic_map.get('walkway', np.zeros_like(drivable_mask))) * mask_25m
            ped_crossing_mask = (semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))) * mask_25m

            # 4. Canali 5-10: Ostacoli visibili nuScenes entro 25m
            target_masks = extract_ground_truth_masks(frame_data) * mask_25m

            input_channels = [
                lidar_bev,
                occlusion_mask,
                drivable_mask,
                walkway_mask,
                ped_crossing_mask
            ] + list(target_masks)

            input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

            import cv2
            dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * 0.4
            boxes_by_token = {b.token: b for b in frame_data.get('boxes', [])}

            for occ in occs:
                pts = occ.get("polygon_points_m", [])
                pts_np = np.array(pts)
                if len(pts_np) < 3:
                    continue
                poly_xy = pts_np[:, :2]

                sp = ShapelyPoly(poly_xy)
                if not sp.is_valid or sp.area <= 0.01:
                    sp = sp.buffer(0)
                if not sp.is_valid or sp.area <= 0.01:
                    continue

                # Vincolo rigoroso: l'ombra viene ritagliata esattamente entro i 25 metri
                sp_25m = sp.intersection(circle_25m)
                if sp_25m.is_empty or sp_25m.area < 0.1:
                    continue

                occ_name = occ.get("object_name", None)
                occ_tok = occ.get("object_token", None)
                b_obj = boxes_by_token.get(occ_tok, None)
                occ_wlh = b_obj.wlh if (b_obj is not None and hasattr(b_obj, 'wlh')) else None

                sub_polys = list(sp_25m.geoms) if sp_25m.geom_type == 'MultiPolygon' else [sp_25m]
                for sp_sub in sub_polys:
                    if sp_sub.area < 0.1:
                        continue

                    vis_coords = np.array(sp_sub.exterior.coords)[:-1]
                    if len(vis_coords) < 3:
                        continue

                    # Feature semantiche calcolate STRETTAMENTE sulla maschera entro 25m
                    occ_mask = rasterize_polygon(vis_coords)
                    tot = np.sum(occ_mask)
                    road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                    side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                    cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                    terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                    area = float(sp_sub.area)
                    dist = float(np.hypot(sp_sub.centroid.x, sp_sub.centroid.y))

                    # Calcolo prossimità al bordo strada (accosto / sosta entro 2.5m)
                    min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
                    roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

                    # Step 1: Estrae la Ground Truth Reale nuScenes a 6 classi entro i 25m
                    gt_raw_6 = get_occlusion_ground_truth_target(target_masks, vis_coords)

                    # Se positives_only e la zona e vuota, la saltiamo
                    if gt_mode == "positives_only" and np.sum(gt_raw_6) == 0:
                        continue

                    # Step 2: Calcolo del target in base alla modalita (applicato al poligono 25m)
                    if gt_mode in ["real", "positives_only"]:
                        gt_target = np.array(gt_raw_6, dtype=np.float32)
                    else:
                        gt_target, _ = compute_synthetic_ground_truth(
                            sp=sp_sub,
                            sp_sample=sp_sub,
                            road_f=road_f,
                            side_f=side_f,
                            cross_f=cross_f,
                            roadside_f=roadside_f,
                            area=area,
                            dist=dist,
                            gt_raw_6=gt_raw_6,
                            mode=gt_mode,
                            occluder_name=occ_name,
                            occluder_wlh=occ_wlh,
                            use_occluder_filter=True
                        )

                    # Ritaglio patch calcolato strettamente sui pixel entro 25m
                    px_x = np.clip(((vis_coords[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
                    px_y = np.clip(((40.0 - vis_coords[:, 1]) / 0.4).astype(int), 0, 199)
                    xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
                    ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                    patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                    # Calcolo larghezza e lunghezza tramite OBB sul poligono 25m
                    mrr = sp_sub.minimum_rotated_rectangle
                    mrr_coords = np.array(mrr.exterior.coords)[:-1]
                    e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                    e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                    obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))

                    scalars = torch.tensor([area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f], dtype=torch.float32)
                    target = torch.tensor(gt_target, dtype=torch.float32)
                    occ_mask_t = AttentionPerZoneModel.build_compatibility_mask(occ_name, occ_wlh)

                    patches_list.append(patch_res)
                    scalars_list.append(scalars)
                    masks_list.append(occ_mask_t)
                    targets_list.append(target)

        patches = torch.stack(patches_list)
        scalars = torch.stack(scalars_list)
        masks = torch.stack(masks_list)
        targets = torch.stack(targets_list)

        if gt_mode != "positives_only":
            torch.save({"patches": patches, "scalars": scalars, "masks": masks, "targets": targets}, cache_path)
            print(f"• Cache salvata in: {cache_path}")

    # DataLoader
    dataset = TensorDataset(patches, scalars, masks, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"• Campioni Totali: {len(dataset)} | Batches per Epoca: {len(dataloader)}")

    # Inizializzazione del Modello AttentionPerZoneModel
    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    # Funzione di costo Asymmetric Loss Bilanciata (gamma_pos=0.5 per compromesso ideale Recall/Precision)
    # Evita l'ipersensibilità e abbatte i falsi positivi mantenendo altissima la copertura
    if gt_mode in ["real", "positives_only"]:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.5, 2.5, 2.5, 2.0, 2.0, 2.5])
    elif gt_mode == "hybrid":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.2, 2.2, 2.0, 1.8, 1.8, 3.5])
    else:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nInizio Addestramento [{gt_mode.upper()}] per {epochs} Epoche...\n")
    start_t = time.time()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for p_b, s_b, m_b, t_b in dataloader:
            p_b, s_b, m_b, t_b = p_b.to(device), s_b.to(device), m_b.to(device), t_b.to(device)
            optimizer.zero_grad()
            logits = model(p_b, s_b, occluder_mask=m_b)
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
