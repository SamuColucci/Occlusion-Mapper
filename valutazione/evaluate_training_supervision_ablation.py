# Studio di Ablazione Ufficiale: Impatto della Supervisione (GT Reale vs GT Neurosimbolica)
# Dimostra perche l'addestramento su GT Neurosimbolica e indispensabile per la guida difensiva.

import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel
from architettura_neurale.loss_functions import AsymmetricLoss

CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"]

def run_supervision_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("   STUDIO DI ABLAZIONE SULLA SUPERVISIONE: GT REALE VS GT NEUROSIMBOLICA")
    print("=" * 90)
    print(f"• Dispositivo di Calcolo: {device}")

    cache_real = os.path.join("addestramento", "cached_dataset_per_zone.pth")
    if os.path.exists(cache_real):
        print(f"• Caricamento dataset con supervisione GT Reale: {cache_real}...")
        data = torch.load(cache_real)
        patches, scalars, targets = data["patches"], data["scalars"], data["targets"]
    else:
        print("• Estrazione dataset con etichette GT Reale...")
        adapter = create_adapter("nuscenes", "./nuscenes")
        num_samples = adapter.get_num_samples()
        p_list, s_list, t_list = [], [], []

        for idx in range(num_samples):
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
            if not os.path.exists(occ_file): continue
            import json
            with open(occ_file) as f: occs = json.load(f)["occlusions"]

            for occ in occs:
                pts = occ.get("polygon_points_m", [])
                if len(pts) < 3: continue
                pts_np = np.array(pts)
                poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

                occ_mask = rasterize_polygon(pts)
                tot = np.sum(occ_mask)
                road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                area = occ.get("area_sqm", 0.0)
                dist = occ.get("distance_m", 0.0)

                gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)

                px_x = np.clip(((poly_xy[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
                px_y = np.clip(((40.0 - poly_xy[:, 1]) / 0.4).astype(int), 0, 199)
                xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin: continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                p_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                p_list.append(p_res)
                s_list.append(torch.tensor([area, dist, 2.0, 1.8, road_f, side_f, cross_f, 0.0, terr_f], dtype=torch.float32))
                t_list.append(torch.tensor(gt_raw_6, dtype=torch.float32))

        patches = torch.stack(p_list)
        scalars = torch.stack(s_list)
        targets = torch.stack(t_list)

    print(f"• Campioni Totali per-zone: {len(patches)}")

    # 1. Addestramento Modello con sola GT Reale
    model_real_gt = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05, pos_weights=[2.0, 3.0, 3.0, 3.0, 3.0, 2.0])
    optimizer = torch.optim.AdamW(model_real_gt.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    dataset = TensorDataset(patches, scalars, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("\nAddestramento Modello su sola GT Reale (20 epoche)...")
    for epoch in range(20):
        model_real_gt.train()
        epoch_loss = 0.0
        for p_b, s_b, t_b in loader:
            p_b, s_b, t_b = p_b.to(device), s_b.to(device), t_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model_real_gt(p_b, s_b), t_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(p_b)
        scheduler.step()
    print("✓ Addestramento su GT Reale completato.")

    # 2. Caricamento Modello Ufficiale addestrato su GT Neurosimbolica
    model_neuro_gt = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
    ckpt_neuro = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
    ckpt_data = torch.load(ckpt_neuro, map_location=device)
    model_neuro_gt.load_state_dict(ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data)
    model_neuro_gt.eval()
    model_real_gt.eval()

    print("\n" + "=" * 90)
    print("   CONFRONTO COMPARATIVO DI VALUTAZIONE (A 25 METRI)")
    print("=" * 90)

    adapter = create_adapter("nuscenes", "./nuscenes")
    num_samples = adapter.get_num_samples()

    # Contatori metriche
    results = {
        "Modello_Addestrato_Su_GT_Reale": {
            "Valutato_su_GT_Reale": {"tp": 0, "fp": 0, "fn": 0},
            "Valutato_su_GT_Sintetica": {"tp": 0, "fp": 0, "fn": 0}
        },
        "Modello_Addestrato_Su_GT_Neurosimbolica": {
            "Valutato_su_GT_Reale": {"tp": 0, "fp": 0, "fn": 0},
            "Valutato_su_GT_Sintetica": {"tp": 0, "fp": 0, "fn": 0}
        }
    }

    import json
    for idx in range(num_samples):
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
        if not os.path.exists(occ_file): continue
        with open(occ_file) as f: occs = json.load(f)["occlusions"]

        for occ in occs:
            dist = occ.get("distance_m", 0.0)
            if dist > 25.0: continue
            pts = occ.get("polygon_points_m", [])
            if len(pts) < 3: continue
            pts_np = np.array(pts)
            poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

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

            occ_mask = rasterize_polygon(pts)
            tot = np.sum(occ_mask)
            road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = occ.get("area_sqm", 0.0)

            gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)
            gt_raw_4 = np.array([gt_raw_6[0], gt_raw_6[1], max(gt_raw_6[2], gt_raw_6[3], gt_raw_6[4]), gt_raw_6[5]], dtype=np.float32)

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

            px_x = np.clip(((poly_xy[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
            px_y = np.clip(((40.0 - poly_xy[:, 1]) / 0.4).astype(int), 0, 199)
            xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
            ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
            if xmax <= xmin or ymax <= ymin: continue

            patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
            patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(device)
            scalars_t = torch.tensor([[area, dist, 2.0, 1.8, road_f, side_f, cross_f, 0.0, terr_f]], dtype=torch.float32).to(device)

            with torch.no_grad():
                out_real_m = torch.sigmoid(model_real_gt(patch_res, scalars_t)).squeeze(0).cpu().numpy()
                out_neuro_m = torch.sigmoid(model_neuro_gt(patch_res, scalars_t)).squeeze(0).cpu().numpy()

            pred_real_m = np.array([int(out_real_m[0]>=0.3), int(out_real_m[1]>=0.3), int(max(out_real_m[2],out_real_m[3],out_real_m[4])>=0.3), int(out_real_m[5]>=0.3)])
            pred_neuro_m = np.array([int(out_neuro_m[0]>=0.3), int(out_neuro_m[1]>=0.3), int(max(out_neuro_m[2],out_neuro_m[3],out_neuro_m[4])>=0.3), int(out_neuro_m[5]>=0.3)])

            for c in range(4):
                tr = int(gt_raw_4[c])
                ts = int(gt_neuro_4[c])

                # 1. Modello addestrato su GT Reale
                if tr == 1 and pred_real_m[c] == 1: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Reale"]["tp"] += 1
                elif tr == 0 and pred_real_m[c] == 1: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Reale"]["fp"] += 1
                elif tr == 1 and pred_real_m[c] == 0: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Reale"]["fn"] += 1

                if ts == 1 and pred_real_m[c] == 1: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Sintetica"]["tp"] += 1
                elif ts == 0 and pred_real_m[c] == 1: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Sintetica"]["fp"] += 1
                elif ts == 1 and pred_real_m[c] == 0: results["Modello_Addestrato_Su_GT_Reale"]["Valutato_su_GT_Sintetica"]["fn"] += 1

                # 2. Modello addestrato su GT Neurosimbolica
                if tr == 1 and pred_neuro_m[c] == 1: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Reale"]["tp"] += 1
                elif tr == 0 and pred_neuro_m[c] == 1: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Reale"]["fp"] += 1
                elif tr == 1 and pred_neuro_m[c] == 0: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Reale"]["fn"] += 1

                if ts == 1 and pred_neuro_m[c] == 1: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Sintetica"]["tp"] += 1
                elif ts == 0 and pred_neuro_m[c] == 1: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Sintetica"]["fp"] += 1
                elif ts == 1 and pred_neuro_m[c] == 0: results["Modello_Addestrato_Su_GT_Neurosimbolica"]["Valutato_su_GT_Sintetica"]["fn"] += 1

    print("\n" + "=" * 90)
    print("   TABELLA COMPARATIVA DI ABLAZIONE (RAGGIO 25 METRI)")
    print("=" * 90)
    print(f"{'Supervisione Addestramento':<35} | {'Valutato Su':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FN (Pericoli Persi)':<18}")
    print("-" * 115)

    for train_type, eval_dict in results.items():
        for eval_type, st in eval_dict.items():
            tp, fp, fn = st["tp"], st["fp"], st["fn"]
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
            label_train = "Sola GT Reale (Miope)" if "Reale" in train_type else "GT Neurosimbolica (Ufficiale)"
            label_eval = "GT Reale nuScenes" if "Reale" in eval_type else "GT Sintetica (Anticip.)"
            print(f"{label_train:<35} | {label_eval:<22} | {p:9.1f}% | {rc:9.1f}% | {f:9.1f}% | {fn:<18d}")

    print("=" * 115 + "\n")

if __name__ == "__main__":
    run_supervision_ablation()
