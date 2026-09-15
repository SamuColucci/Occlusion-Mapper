# Script di Calibrazione Automatica delle Soglie e Spatial BEV NMS
# (valutazione/tune_thresholds_and_nms.py)

import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
from shapely.geometry import Polygon as ShapelyPoly

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
from ground_truth.ground_truth_extractor_synthetic import compute_synthetic_ground_truth, to_macro_classes_4
from architettura_neurale import AttentionPerZoneModel

GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"]

# Raggio di collisione minima tipico per classe (in metri) per Spatial NMS
CLASS_NMS_DIST = {
    0: 3.5,  # Auto
    1: 5.0,  # Camion
    2: 1.2,  # VRU
    3: 1.5   # Barriera
}


def compute_polygon_iou(p1: ShapelyPoly, p2: ShapelyPoly) -> float:
    """Calcola l'Intersection-over-Union (IoU) tra due poligoni Shapely."""
    try:
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        if inter <= 0.0:
            return 0.0
        union = p1.area + p2.area - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Dispositivo: {device}")

    # Caricamento del modello Ibrido
    ckpt_path = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth")
    if not os.path.exists(ckpt_path):
        print(f"Errore: Checkpoint non trovato in {ckpt_path}")
        return

    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Selezione split di validazione nuScenes
    adapter = create_adapter("nuscenes", "./nuscenes")
    num_samples = adapter.get_num_samples()
    from nuscenes.utils.splits import create_splits_scenes
    splits_dict = create_splits_scenes()
    val_scenes = set(splits_dict.get('mini_val', []))

    valid_indices = []
    for idx in range(num_samples):
        sample = adapter.all_samples[idx]
        sc_name = adapter.nusc.get('scene', sample['scene_token'])['name']
        if sc_name in val_scenes:
            valid_indices.append(idx)

    print(f"• Fotogrammi di validazione: {len(valid_indices)}")
    print("• Estrazione predizioni continue e geometrie BEV...")

    # Struttura per memorizzare le predizioni organizzate per frame
    frames_data = []

    import cv2
    with torch.no_grad():
        for proc_i, idx in enumerate(valid_indices):
            frame_data = adapter.get_sample_data(idx)
            token = frame_data["sample_token"]
            target_masks = extract_ground_truth_masks(frame_data)
            semantic_map = frame_data['semantic_map']
            drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
            walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
            ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

            input_channels = [
                frame_data.get('lidar_bev', np.zeros((GRID_DIM, GRID_DIM))),
                frame_data.get('occlusion_mask', np.zeros((GRID_DIM, GRID_DIM))),
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

            dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * VOXEL_SIZE

            frame_occs = []

            for occ in occs:
                dist = occ.get("distance_m", 0.0)
                if dist > 25.0:
                    continue
                pts = occ.get("polygon_points_m", [])
                pts_np = np.array(pts)
                if len(pts_np) < 3:
                    continue
                poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])
                sp = ShapelyPoly(poly_xy)

                occ_mask = rasterize_polygon(pts)
                tot = np.sum(occ_mask)
                road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                area = float(occ.get("area_sqm", 0.0))

                min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
                roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

                # GT Targets
                gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)
                gt_real_4 = to_macro_classes_4(gt_raw_6)

                gt_hyb_6, _ = compute_synthetic_ground_truth(
                    sp=sp, road_f=road_f, side_f=side_f, cross_f=cross_f,
                    roadside_f=roadside_f, area=area, dist=dist, gt_raw_6=gt_raw_6, mode="hybrid"
                )
                gt_hyb_4 = to_macro_classes_4(gt_hyb_6)

                # Patch
                px_x = np.clip(((poly_xy[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                px_y = np.clip(((GRID_RANGE - poly_xy[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(device)

                if sp.is_valid and sp.area > 0.01:
                    mrr = sp.minimum_rotated_rectangle
                    mrr_coords = np.array(mrr.exterior.coords)[:-1]
                    e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                    e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                    obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))
                else:
                    obb_w, obb_l = 0.5, 0.5

                scalars = torch.tensor([[area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f]], dtype=torch.float32).to(device)
                out_6 = torch.sigmoid(model(patch_res, scalars)).squeeze(0).cpu().numpy()

                raw_probs_4 = np.array([
                    out_6[0],
                    out_6[1],
                    max(out_6[2], out_6[3], out_6[4]),
                    out_6[5]
                ], dtype=np.float32)

                centroid = np.array([sp.centroid.x, sp.centroid.y]) if sp.is_valid else np.mean(poly_xy, axis=0)

                frame_occs.append({
                    "poly": sp,
                    "centroid": centroid,
                    "area": area,
                    "dist": dist,
                    "probs": raw_probs_4,
                    "gt_real": gt_real_4,
                    "gt_hyb": gt_hyb_4
                })

            frames_data.append(frame_occs)

    total_occs = sum(len(f) for f in frames_data)
    print(f"• Elaborate {total_occs} zone d'ombra totali in {len(frames_data)} fotogrammi.")

    # =========================================================================
    # PARTE 1: RICERCA AUTOMATICA DELLE SOGLIE OTTIMALI
    # =========================================================================
    print("\n" + "=" * 80)
    print("   PARTE 1: OTTIMIZZAZIONE AUTOMATICA DELLE SOGLIE (Target: GT Ibrida & Reale)")
    print("=" * 80)

    # Scansione griglia per ciascuna classe su GT Ibrida
    best_thresholds_hyb = {}
    candidates = np.arange(0.12, 0.55, 0.02)

    all_probs = np.array([occ["probs"] for f in frames_data for occ in f])
    all_gt_hyb = np.array([occ["gt_hyb"] for f in frames_data for occ in f])
    all_gt_real = np.array([occ["gt_real"] for f in frames_data for occ in f])
    all_areas = np.array([occ["area"] for f in frames_data for occ in f])

    print(f"\nScansione soglie per classe (Ottimizzazione F1 con vincolo Recall >= 90%):")
    for c_i, c_name in enumerate(CATEGORIES):
        best_f1 = -1.0
        best_th = 0.25
        best_stats = None

        min_area = 8.0 if c_i == 1 else (3.5 if c_i == 0 else 0.0)

        for th in candidates:
            # Predizione con filtro geometrico
            p = ((all_probs[:, c_i] >= th) & (all_areas >= min_area)).astype(int)
            y = all_gt_hyb[:, c_i].astype(int)

            tp = np.sum((p == 1) & (y == 1))
            fp = np.sum((p == 1) & (y == 0))
            fn = np.sum((p == 0) & (y == 1))

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

            # Criterio: massimizzare F1 con Recall >= 90% (o massimo F1 assoluto se recall scende)
            if rec >= 0.90:
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
                    best_stats = (prec, rec, f1, tp, fp, fn)

        if best_stats is None:  # Se nessuna soglia ha recall >= 90%, prendi il miglior F1
            for th in candidates:
                p = ((all_probs[:, c_i] >= th) & (all_areas >= min_area)).astype(int)
                y = all_gt_hyb[:, c_i].astype(int)
                tp = np.sum((p == 1) & (y == 1))
                fp = np.sum((p == 1) & (y == 0))
                fn = np.sum((p == 0) & (y == 1))
                prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0
                rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
                    best_stats = (prec, rec, f1, tp, fp, fn)

        best_thresholds_hyb[c_i] = best_th
        p_b, r_b, f_b, tp_b, fp_b, fn_b = best_stats
        print(f"  • {c_name:<20}: Soglia Ottimale = {best_th:.2f} | Precision: {p_b*100:5.1f}% | Recall: {r_b*100:5.1f}% | F1: {f_b*100:5.1f}% (TP={tp_b}, FP={fp_b}, FN={fn_b})")

    # =========================================================================
    # PARTE 2: SPATIAL BEV NMS
    # =========================================================================
    print("\n" + "=" * 80)
    print("   PARTE 2: VALUTAZIONE SPATIAL BEV NON-MAXIMUM SUPPRESSION (NMS)")
    print("=" * 80)

    def evaluate_pipeline(use_nms=False, iou_thresh=0.20):
        """Valuta tutte le occ con le soglie ottimali, con o senza NMS."""
        metrics_hyb = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(4)}
        metrics_real = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(4)}

        for f_occs in frames_data:
            n_occs = len(f_occs)
            if n_occs == 0:
                continue

            # Step 1: Decisione iniziale con soglie ottimali e filtri geometrici
            preds_mask = np.zeros((n_occs, 4), dtype=int)
            for i, occ in enumerate(f_occs):
                for c in range(4):
                    th = best_thresholds_hyb[c]
                    min_a = 8.0 if c == 1 else (3.5 if c == 0 else 0.0)
                    if occ["probs"][c] >= th and occ["area"] >= min_a:
                        preds_mask[i, c] = 1

            # Step 2: Spatial BEV NMS per classe
            if use_nms:
                for c in range(4):
                    active_indices = [i for i in range(n_occs) if preds_mask[i, c] == 1]
                    if len(active_indices) <= 1:
                        continue

                    # Ordina per probabilità decrescente
                    active_indices.sort(key=lambda idx: f_occs[idx]["probs"][c], reverse=True)

                    suppressed = set()
                    for i_idx, curr_i in enumerate(active_indices):
                        if curr_i in suppressed:
                            continue
                        poly_i = f_occs[curr_i]["poly"]
                        cent_i = f_occs[curr_i]["centroid"]

                        for curr_j in active_indices[i_idx + 1:]:
                            if curr_j in suppressed:
                                continue
                            poly_j = f_occs[curr_j]["poly"]
                            cent_j = f_occs[curr_j]["centroid"]

                            # Test 1: Distanza tra baricentri
                            dist_cent = np.linalg.norm(cent_i - cent_j)
                            # Test 2: IoU poligonale
                            iou = compute_polygon_iou(poly_i, poly_j)

                            if iou > iou_thresh or dist_cent < CLASS_NMS_DIST[c]:
                                # Sopprimi il duplicato a minore confidenza
                                suppressed.add(curr_j)
                                preds_mask[curr_j, c] = 0

            # Step 3: Accumulo statistiche
            for i, occ in enumerate(f_occs):
                for c in range(4):
                    # Su GT Ibrida
                    pr = preds_mask[i, c]
                    gh = int(occ["gt_hyb"][c])
                    if pr == 1 and gh == 1: metrics_hyb[c]["tp"] += 1
                    elif pr == 1 and gh == 0: metrics_hyb[c]["fp"] += 1
                    elif pr == 0 and gh == 1: metrics_hyb[c]["fn"] += 1

                    # Su GT Reale
                    gr = int(occ["gt_real"][c])
                    if pr == 1 and gr == 1: metrics_real[c]["tp"] += 1
                    elif pr == 1 and gr == 0: metrics_real[c]["fp"] += 1
                    elif pr == 0 and gr == 1: metrics_real[c]["fn"] += 1

        return metrics_hyb, metrics_real

    def print_summary_table(title, m_dict):
        tp_tot = sum(m_dict[c]["tp"] for c in range(4))
        fp_tot = sum(m_dict[c]["fp"] for c in range(4))
        fn_tot = sum(m_dict[c]["fn"] for c in range(4))
        p_tot = (tp_tot / (tp_tot + fp_tot) * 100) if (tp_tot + fp_tot) > 0 else 0
        r_tot = (tp_tot / (tp_tot + fn_tot) * 100) if (tp_tot + fn_tot) > 0 else 0
        f_tot = (2 * p_tot * r_tot / (p_tot + r_tot)) if (p_tot + r_tot) > 0 else 0

        print(f"\n--- {title} ---")
        print(f"{'Categoria':<22} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
        print("-" * 80)
        for c in range(4):
            tp = m_dict[c]["tp"]
            fp = m_dict[c]["fp"]
            fn = m_dict[c]["fn"]
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            print(f"{CATEGORIES[c]:<22} | {tp:<6d} | {fp:<6d} | {fn:<6d} | {p:9.1f}% | {r:9.1f}% | {f1:9.1f}%")
        print("-" * 80)
        print(f"{'MEDIA GLOBALE':<22} | {tp_tot:<6d} | {fp_tot:<6d} | {fn_tot:<6d} | {p_tot:9.1f}% | {r_tot:9.1f}% | {f_tot:9.1f}%")
        return p_tot, r_tot, f_tot, fp_tot

    print("\n1. Risultati SENZA Spatial NMS:")
    m_hyb_base, m_real_base = evaluate_pipeline(use_nms=False)
    p_b_hyb, r_b_hyb, f_b_hyb, fp_b_hyb = print_summary_table("GT Sintetica Ibrida (Standard)", m_hyb_base)
    p_b_real, r_b_real, f_b_real, fp_b_real = print_summary_table("GT Reale nuScenes (Standard)", m_real_base)

    print("\n2. Risultati CON Spatial BEV NMS (IoU Thresh = 0.20, Distanza centroidi per classe):")
    m_hyb_nms, m_real_nms = evaluate_pipeline(use_nms=True, iou_thresh=0.20)
    p_n_hyb, r_n_hyb, f_n_hyb, fp_n_hyb = print_summary_table("GT Sintetica Ibrida (CON SPATIAL NMS)", m_hyb_nms)
    p_n_real, r_n_real, f_n_real, fp_n_real = print_summary_table("GT Reale nuScenes (CON SPATIAL NMS)", m_real_nms)

    print("\n" + "=" * 80)
    print("   CONFRONTO DIRETTO: IMPATTO DELLO SPATIAL BEV NMS")
    print("=" * 80)
    delta_fp_hyb = fp_n_hyb - fp_b_hyb
    delta_fp_real = fp_n_real - fp_b_real
    print(f"• GT Ibrida:  FP {fp_b_hyb} -> {fp_n_hyb} ({delta_fp_hyb:+d} FP) | Precision: {p_b_hyb:.1f}% -> {p_n_hyb:.1f}% | F1: {f_b_hyb:.1f}% -> {f_n_hyb:.1f}% | Recall: {r_b_hyb:.1f}% -> {r_n_hyb:.1f}%")
    print(f"• GT Reale:   FP {fp_b_real} -> {fp_n_real} ({delta_fp_real:+d} FP) | Precision: {p_b_real:.1f}% -> {p_n_real:.1f}% | F1: {f_b_real:.1f}% -> {f_n_real:.1f}% | Recall: {r_b_real:.1f}% -> {r_n_real:.1f}%")


if __name__ == "__main__":
    run_benchmark()
