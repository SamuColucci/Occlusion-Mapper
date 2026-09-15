# Script Ufficiale di Valutazione e Validazione delle Metriche (valutazione/evaluate_final_official.py)
# Valuta l'architettura AttentionPerZoneModel su tutti i fotogrammi del dataset nuScenes.
#
# Esegue il DOPPIO CONFRONTO per ciascun modello:
#   1. Valutazione su Ground Truth Reale nuScenes (Rilevamento Ostacoli Fisici 3D Reali)
#   2. Valutazione su Ground Truth Sintetica Neurosimbolica (Anticipazione del Rischio / Oggetti Plausibili)
#
# Supporta la valutazione di singoli modelli o di TUTTI i modelli dello studio di ablazione (--mode all):
#   - 'geometric'      : Modello addestrato su GT Sintetica Geometrica (Spatially-Constrained)
#   - 'semantic'       : Modello addestrato su GT Sintetica Semantica (Semantic-Affordance / Naïve Prior)
#   - 'real'           : Modello Baseline addestrato su GT Reale Completa (18k zone)
#   - 'positives_only' : Modello Baseline addestrato ESCLUSIVAMENTE sulle zone con ostacoli reali (4.9k zone)
#   - 'all'            : Esegue la valutazione comparativa ufficiale di tutti i modelli contemporaneamente.

import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import (
    extract_ground_truth_masks,
    rasterize_polygon,
    get_occlusion_ground_truth_target,
    extract_real_occlusion_ground_truth
)
from ground_truth.ground_truth_extractor_synthetic import (
    compute_synthetic_ground_truth,
    to_macro_classes_4,
    apply_occluder_compatibility_filter
)
from architettura_neurale import AttentionPerZoneModel
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPoly

GRID_DIM = 200        # 200x200 pixel BEV
GRID_RANGE = 40.0     # [-40m, +40m]
VOXEL_SIZE = 0.4      # 0.4 m/pixel
CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"]
RADII = [20.0, 25.0]


def find_checkpoint(model_key, custom_path=None):
    """Localizza il file di checkpoint per il modello specificato cercando nei percorsi standard."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    candidates = {
        "geometric": [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_geometric.pth"),
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        ],
        "semantic": [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_semantic.pth"),
            os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_attention_neuro_semantic.pth"),
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        ],
        "real": [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth"),
            os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_attention_real_gt.pth")
        ],
        "positives_only": [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_positive_only.pth"),
            os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_attention_positive_only.pth")
        ],
        "hybrid": [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
            os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_attention_neuro_hybrid.pth")
        ]
    }

    for p in candidates.get(model_key, []):
        if os.path.exists(p):
            return p
    return None


def print_table(title, counts_dict, gt_type):
    """Stampa la tabella formattata dei risultati per una specifica Ground Truth di riferimento."""
    print(f"\n--- {title} (Riferimento: {gt_type}) ---")
    for r in RADII:
        print(f"\n  [ Raggio di Analisi Operativa: {r:.0f} Metri ]")
        print("  " + "-" * 88)
        print(f"  {'Categoria Semantica':<24} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("  " + "-" * 88)

        tp_tot, fp_tot, fn_tot = 0, 0, 0
        for c_idx, c_name in enumerate(CATEGORIES):
            stats = counts_dict[r][c_idx]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            tp_tot += tp
            fp_tot += fp
            fn_tot += fn
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
            print(f"  {c_name:<24} | {tp:<6d} | {fp:<6d} | {fn:<6d} | {p:9.1f}% | {rc:9.1f}% | {f:9.1f}%")

        p_macro = (tp_tot / (tp_tot + fp_tot) * 100) if (tp_tot + fp_tot) > 0 else 0
        rc_macro = (tp_tot / (tp_tot + fn_tot) * 100) if (tp_tot + fn_tot) > 0 else 0
        f_macro = (2 * p_macro * rc_macro / (p_macro + rc_macro)) if (p_macro + rc_macro) > 0 else 0
        print("  " + "-" * 88)
        print(f"  {'MEDIA GLOBALE':<24} | {tp_tot:<6d} | {fp_tot:<6d} | {fn_tot:<6d} | {p_macro:9.1f}% | {rc_macro:9.1f}% | {f_macro:9.1f}%")


def evaluate_models(models_dict, syn_gt_strategy="geometric", split="val"):
    """
    Esegue la valutazione a passaggio singolo su GPU per tutti i modelli specificati,
    calcolando simultaneamente il TRIPLO CONFRONTO (su GT Reale, GT Sintetica Geom e GT Sintetica Sem).
    Supporta: 'val' (2 scene mai viste, default), 'train' (8 scene di training), 'all' (tutte).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n• Dispositivo di calcolo: {device}")
    print(f"• Modelli da valutare ({len(models_dict)}):")
    for k, m in models_dict.items():
        print(f"    - {k} ({m['ckpt_path']})")

    adapter = create_adapter("nuscenes", "./nuscenes")
    num_samples = adapter.get_num_samples()

    from nuscenes.utils.splits import create_splits_scenes
    splits_dict = create_splits_scenes()
    train_scenes = set(splits_dict.get('mini_train', []))
    val_scenes = set(splits_dict.get('mini_val', []))

    valid_frame_indices = []
    for idx in range(num_samples):
        sample = adapter.all_samples[idx]
        sc_name = adapter.nusc.get('scene', sample['scene_token'])['name']
        if split == "train" and sc_name not in train_scenes:
            continue
        elif split == "val" and sc_name not in val_scenes:
            continue
        valid_frame_indices.append(idx)

    print(f"• Split Dataset Selezionato: {split.upper()} ({len(valid_frame_indices)} su {num_samples} fotogrammi)")

    # Mappatura rapida token -> file probabilità bayesiane
    bayes_lookup = {}
    bayes_dir = os.path.join(os.path.dirname(__file__), "..", "extracted_occlusions_probabilities")
    if os.path.exists(bayes_dir):
        for fname in os.listdir(bayes_dir):
            if fname.endswith(".json"):
                tok = fname.replace(".json", "").split("_")[-1]
                bayes_lookup[tok] = os.path.join(bayes_dir, fname)

    # Struttura di conteggio per ciascun modello (Quadruplo Confronto simultaneo)
    counts = {}
    for m_key in models_dict:
        counts[m_key] = {
            gt_type: {r: [{"tp": 0, "fp": 0, "fn": 0} for _ in range(4)] for r in RADII}
            for gt_type in ["GT_Reale", "GT_Sintetica_Geom", "GT_Sintetica_Sem", "GT_Sintetica_Hyb"]
        }

    MODEL_KEY_MAP = {
        "hybrid": "NEURO_SIMB",
        "real": "REAL_GT",
        "positives_only": "POS_ONLY",
        "bayes": "BAYES",
        "geometric": "GEOMETRIC",
        "semantic": "SEMANTIC"
    }

    eval_cache = {
        "split": split,
        "frames": {}
    }

    print(f"\nInizio scansione split [{split.upper()}] (passaggio unico ultra-rapido per tutti i modelli)...")
    for proc_i, idx in enumerate(valid_frame_indices):
        if (proc_i + 1) % 50 == 0 or proc_i == len(valid_frame_indices) - 1:
            print(f"  Elaborati {proc_i + 1}/{len(valid_frame_indices)} fotogrammi...")

        frame_data = adapter.get_sample_data(idx)
        token = frame_data["sample_token"]

        frame_cache = {
            "sample_token": token,
            "models": {}
        }
        for m_k in models_dict:
            ui_m = MODEL_KEY_MAP.get(m_k, m_k.upper())
            frame_cache["models"][ui_m] = {
                gt_k: {
                    "stats": {c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0} for c in range(4)},
                    "zones": []
                }
                for gt_k in ["REAL", "NEURO_SIMB", "GEOMETRIC", "SEMANTIC"]
            }

        occ_file = os.path.join("extracted_occlusions", f"{token}.json")
        if not os.path.exists(occ_file):
            continue
        with open(occ_file) as f:
            occs = json.load(f)["occlusions"]

        # Maschera circolare metrica a 25m per tagliare rigorosamente TUTTI gli 11 canali di input
        r_grid = np.arange(GRID_DIM)
        gy, gx = np.meshgrid(r_grid, r_grid)
        mask_25m = (np.sqrt((gx - 99.5)**2 + (gy - 99.5)**2) * VOXEL_SIZE <= 25.0).astype(np.float32)
        circle_25m = ShapelyPoint(0, 0).buffer(25.0)

        # 1. Canale 0: LiDAR Density BEV entro 25m
        lidar_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        points = frame_data.get("points", np.zeros((0, 3)))
        if len(points) > 0:
            for px_m, py_m in points[:, :2]:
                px = int((px_m + GRID_RANGE) / VOXEL_SIZE)
                py = int((GRID_RANGE - py_m) / VOXEL_SIZE)
                if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                    lidar_bev[py, px] = 1.0
        lidar_bev = lidar_bev * mask_25m

        # 2. Canale 1: Ombre Raycasting cumulative entro 25m
        occlusion_mask = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
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

        bayes_occs = []
        if "bayes" in models_dict and token in bayes_lookup:
            try:
                with open(bayes_lookup[token], "r") as bf:
                    bayes_occs = json.load(bf).get("occlusions", [])
            except Exception:
                bayes_occs = []

        import cv2
        dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * VOXEL_SIZE
        all_frame_boxes = frame_data.get('boxes', [])
        boxes_by_token = {b.token: b for b in all_frame_boxes}
        occluder_tokens = {occ.get('object_token') for occ in occs if occ.get('object_token')}

        circle_25m = ShapelyPoint(0, 0).buffer(25.0)

        for occ_idx, occ in enumerate(occs):
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

                occ_mask = rasterize_polygon(vis_coords)
                tot = np.sum(occ_mask)
                road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                area = float(sp_sub.area)
                dist = float(np.hypot(sp_sub.centroid.x, sp_sub.centroid.y))

                # Calcolo prossimità bordo strada (accosto / parcheggio entro 2.5m)
                min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
                roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

                # 1. Target GT REALE a 4 macro-classi (escludendo categoricamente gli occludori)
                gt_raw_6, gt_real_4 = extract_real_occlusion_ground_truth(
                    all_frame_boxes, sp_sub, occluder_tokens=occluder_tokens
                )

                # 2. Target GT SINTETICA GEOMETRICA a 4 macro-classi (Fitting 3D)
                gt_geom_6, _ = compute_synthetic_ground_truth(
                    sp=sp_sub, sp_sample=sp_sub, road_f=road_f, side_f=side_f, cross_f=cross_f,
                    area=area, dist=dist, gt_raw_6=gt_raw_6, mode="geometric",
                    occluder_name=occ_name, occluder_wlh=occ_wlh, use_occluder_filter=True
                )
                gt_geom_4 = to_macro_classes_4(gt_geom_6)

                # 3. Target GT SINTETICA SEMANTICA a 4 macro-classi (Regole Naïve)
                gt_sem_6, _ = compute_synthetic_ground_truth(
                    sp=sp_sub, sp_sample=sp_sub, road_f=road_f, side_f=side_f, cross_f=cross_f,
                    area=area, dist=dist, gt_raw_6=gt_raw_6, mode="semantic",
                    occluder_name=occ_name, occluder_wlh=occ_wlh, use_occluder_filter=True
                )
                gt_sem_4 = to_macro_classes_4(gt_sem_6)

                # 4. Target GT SINTETICA IBRIDA a 4 macro-classi (Spatio-Semantic con sosta/accosto)
                gt_hyb_6, _ = compute_synthetic_ground_truth(
                    sp=sp_sub, sp_sample=sp_sub, road_f=road_f, side_f=side_f, cross_f=cross_f,
                    roadside_f=roadside_f, area=area, dist=dist, gt_raw_6=gt_raw_6, mode="hybrid",
                    occluder_name=occ_name, occluder_wlh=occ_wlh, use_occluder_filter=True
                )
                gt_hyb_4 = to_macro_classes_4(gt_hyb_6)

                # Ritaglio patch BEV 64x64 calcolato strettamente sui pixel entro 25m
                px_x = np.clip(((vis_coords[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                px_y = np.clip(((GRID_RANGE - vis_coords[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(device)

                mrr = sp_sub.minimum_rotated_rectangle
                mrr_coords = np.array(mrr.exterior.coords)[:-1]
                e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))

                scalars = torch.tensor([[area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f]], dtype=torch.float32).to(device)

            # Esegue inferenza simultanea su tutti i modelli
            with torch.no_grad():
                for m_key, m_info in models_dict.items():
                    ui_m = MODEL_KEY_MAP.get(m_key, m_key.upper())
                    if m_key == "bayes":
                        b_occ = bayes_occs[occ_idx] if occ_idx < len(bayes_occs) else {}
                        b_probs = b_occ.get("estimated_probabilities", {})
                        p_auto = float(b_probs.get("Auto", 0.0))
                        p_camion = float(max(b_probs.get("Camion", 0.0), b_probs.get("Bus", 0.0), b_probs.get("Rimorchio", 0.0)))
                        p_vru = float(max(b_probs.get("Pedone", 0.0), b_probs.get("Bicicletta", 0.0), b_probs.get("Moto", 0.0)))
                        p_barr = float(max(b_probs.get("Barriera", 0.0), b_probs.get("Cono", 0.0)))
                        p4 = [p_auto, p_camion, p_vru, p_barr]
                        pred_4 = np.array([
                            int(p_auto >= 0.28 and area >= 3.5),
                            int(p_camion >= 0.25 and area >= 8.0),
                            int(p_vru >= 0.25),
                            int(p_barr >= 0.26)
                        ], dtype=int)
                    else:
                        model = m_info["model"]
                        occluder_mask = AttentionPerZoneModel.build_compatibility_mask(occ_name, occ_wlh, device=device)
                        out_6 = torch.sigmoid(model(patch_res, scalars, occluder_mask=occluder_mask)).squeeze(0).cpu().numpy()

                        p4 = [
                            float(out_6[0]),
                            float(out_6[1]),
                            float(max(out_6[2], out_6[3], out_6[4])),
                            float(out_6[5])
                        ]
                        pred_4 = np.zeros(4, dtype=int)
                        pred_4[0] = int(out_6[0] >= 0.28 and area >= 3.5)
                        pred_4[1] = int(out_6[1] >= 0.25 and area >= 8.0)
                        pred_4[2] = int(max(out_6[2], out_6[3], out_6[4]) >= 0.25)
                        pred_4[3] = int(out_6[5] >= 0.26)

                    # Aggiorna conteggi globali per raggio
                    for r in RADII:
                        if dist <= r:
                            for c in range(4):
                                # A. Confronto su GT Reale nuScenes
                                tr = int(gt_real_4[c])
                                pr = int(pred_4[c])
                                if tr == 1 and pr == 1: counts[m_key]["GT_Reale"][r][c]["tp"] += 1
                                elif tr == 0 and pr == 1: counts[m_key]["GT_Reale"][r][c]["fp"] += 1
                                elif tr == 1 and pr == 0: counts[m_key]["GT_Reale"][r][c]["fn"] += 1

                                # B. Confronto su GT Sintetica Geometrica (3D Spatially-Constrained)
                                tg = int(gt_geom_4[c])
                                if tg == 1 and pr == 1: counts[m_key]["GT_Sintetica_Geom"][r][c]["tp"] += 1
                                elif tg == 0 and pr == 1: counts[m_key]["GT_Sintetica_Geom"][r][c]["fp"] += 1
                                elif tg == 1 and pr == 0: counts[m_key]["GT_Sintetica_Geom"][r][c]["fn"] += 1

                                # C. Confronto su GT Sintetica Semantica (Affordance Naïve)
                                ts = int(gt_sem_4[c])
                                if ts == 1 and pr == 1: counts[m_key]["GT_Sintetica_Sem"][r][c]["tp"] += 1
                                elif ts == 0 and pr == 1: counts[m_key]["GT_Sintetica_Sem"][r][c]["fp"] += 1
                                elif ts == 1 and pr == 0: counts[m_key]["GT_Sintetica_Sem"][r][c]["fn"] += 1

                                # D. Confronto su GT Sintetica Ibrida (Spazio-Semantica)
                                th = int(gt_hyb_4[c])
                                if th == 1 and pr == 1: counts[m_key]["GT_Sintetica_Hyb"][r][c]["tp"] += 1
                                elif th == 0 and pr == 1: counts[m_key]["GT_Sintetica_Hyb"][r][c]["fp"] += 1
                                elif th == 1 and pr == 0: counts[m_key]["GT_Sintetica_Hyb"][r][c]["fn"] += 1

                    # Aggiorna la cache del fotogramma per il visualizzatore
                    gt_map = {
                        "REAL": gt_real_4,
                        "NEURO_SIMB": gt_hyb_4,
                        "GEOMETRIC": gt_geom_4,
                        "SEMANTIC": gt_sem_4
                    }
                    pred_binary = [int(x) for x in pred_4]
                    max_class_idx = int(np.argmax(p4))
                    max_prob = float(p4[max_class_idx])

                    for gt_k, gt_curr in gt_map.items():
                        gt_binary = [int(x) for x in gt_curr]
                        st = frame_cache["models"][ui_m][gt_k]["stats"]
                        z_tps = [c for c in range(4) if pred_binary[c] == 1 and gt_binary[c] == 1]
                        z_fps = [c for c in range(4) if pred_binary[c] == 1 and gt_binary[c] == 0]
                        z_fns = [c for c in range(4) if pred_binary[c] == 0 and gt_binary[c] == 1]

                        for c in range(4):
                            if gt_binary[c] == 1: st[c]["gt_count"] += 1
                            if pred_binary[c] == 1: st[c]["pred_count"] += 1
                            if pred_binary[c] == 1 and gt_binary[c] == 1: st[c]["tp"] += 1
                            elif pred_binary[c] == 1 and gt_binary[c] == 0: st[c]["fp"] += 1
                            elif pred_binary[c] == 0 and gt_binary[c] == 1: st[c]["fn"] += 1

                        if len(z_tps) > 0 and len(z_fns) == 0:
                            v_label = f"TRUE POSITIVE ({CATEGORIES[z_tps[0]]})"; verdict = "TP"
                        elif len(z_tps) > 0 and len(z_fns) > 0:
                            v_label = f"PARZIALE: TP ({', '.join(CATEGORIES[c] for c in z_tps)}) + FN ({', '.join(CATEGORIES[c] for c in z_fns)})"; verdict = "TP/FN"
                        elif len(z_fns) > 0 and len(z_fps) == 0:
                            v_label = f"FALSE NEGATIVE ({CATEGORIES[z_fns[0]]})"; verdict = "FN"
                        elif len(z_fns) > 0 and len(z_fps) > 0:
                            v_label = f"FALSE NEGATIVE ({CATEGORIES[z_fns[0]]})"; verdict = "FN"
                        elif len(z_fps) > 0:
                            v_label = f"FALSE POSITIVE ({CATEGORIES[z_fps[0]]})"; verdict = "FP"
                        else:
                            v_label = "TRUE NEGATIVE (Zona Libera)"; verdict = "TN"

                        frame_cache["models"][ui_m][gt_k]["zones"].append({
                            "occ_idx": occ_idx,
                            "probs_4": [round(float(x), 4) for x in p4],
                            "pred_binary": pred_binary,
                            "gt_target_4": gt_binary,
                            "verdict": verdict,
                            "verdict_label": v_label,
                            "risk_score": round(max_prob, 4),
                            "pred_class": CATEGORIES[max_class_idx],
                            "pred_idx": max_class_idx
                        })

        # Calcola f1/rec/prec aggregate per questo frame
        for ui_m in frame_cache["models"]:
            for gt_k in frame_cache["models"][ui_m]:
                st = frame_cache["models"][ui_m][gt_k]["stats"]
                t_tp = sum(st[c]["tp"] for c in range(4))
                t_fp = sum(st[c]["fp"] for c in range(4))
                t_fn = sum(st[c]["fn"] for c in range(4))
                pr = (t_tp / (t_tp + t_fp) * 100.0) if (t_tp + t_fp) > 0 else 0.0
                rc = (t_tp / (t_tp + t_fn) * 100.0) if (t_tp + t_fn) > 0 else 0.0
                f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
                frame_cache["models"][ui_m][gt_k]["f1"] = round(f1, 2)
                frame_cache["models"][ui_m][gt_k]["recall"] = round(rc, 2)
                frame_cache["models"][ui_m][gt_k]["precision"] = round(pr, 2)

        eval_cache["frames"][str(idx)] = frame_cache

    # Stampa dettagliata delle tabelle di risultato per ciascun modello
    print("\n" + "=" * 95)
    print("   RISULTATI COMPLETI DELLA VALUTAZIONE UFFICIALE (TRIPLO CONFRONTO)")
    print("=" * 95)

    for m_key, m_info in models_dict.items():
        title = m_info["title"]
        print("\n" + "#" * 95)
        print(f"   MODELLO: {title}")
        print(f"   Checkpoint: {m_info['ckpt_path']}")
        print("#" * 95)

        # Tabella 1: Valutazione su GT Reale nuScenes
        print_table(f"1. Rilevamento Ostacoli Fisici Reali nuScenes", counts[m_key]["GT_Reale"], "GT Reale 3D")

        # Tabella 2: Valutazione su GT Sintetica Geometrica
        print_table(f"2. Anticipazione su GT Sintetica Geometrica (Fitting 3D)", counts[m_key]["GT_Sintetica_Geom"], "GT Sintetica Geometrica")

        # Tabella 3: Valutazione su GT Sintetica Semantica
        print_table(f"3. Anticipazione su GT Sintetica Semantica (Regole Naïve)", counts[m_key]["GT_Sintetica_Sem"], "GT Sintetica Semantica")

        # Tabella 4: Valutazione su GT Sintetica Ibrida
        print_table(f"4. Anticipazione su GT Sintetica Ibrida (Spazio-Semantica)", counts[m_key]["GT_Sintetica_Hyb"], "GT Sintetica Ibrida")

    # Tabella Comparativa Finale Riassuntiva a 25 metri
    print("\n" + "=" * 95)
    print(f"   TABELLA COMPARATIVA RIASSUNTIVA FINALE A 25 METRI [SPLIT: {split.upper()}] (PER LA TESI)")
    print("=" * 95)
    print(f"{'Modello Addestrato':<38} | {'Valutato su':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 95)

    for m_key, m_info in models_dict.items():
        name = m_info["short_name"]
        for gt_t, gt_desc in [("GT_Reale", "GT Reale nuScenes"),
                              ("GT_Sintetica_Geom", "GT Sintetica (Geometrica)"),
                              ("GT_Sintetica_Sem", "GT Sintetica (Semantica)"),
                              ("GT_Sintetica_Hyb", "GT Sintetica (Ibrida)")]:
            stats_list = counts[m_key][gt_t][25.0]
            tp = sum(s["tp"] for s in stats_list)
            fp = sum(s["fp"] for s in stats_list)
            fn = sum(s["fn"] for s in stats_list)
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
            print(f"{name:<38} | {gt_desc:<25} | {p:9.1f}% | {rc:9.1f}% | {f:9.1f}%")
        print("-" * 95)

    # Esportazione e aggiornamento automatico dei report Markdown per la tesi
    export_markdown_results(counts, models_dict, split, eval_cache)


def export_markdown_results(counts, models_dict, split, eval_cache=None):
    """
    Esporta automaticamente i risultati della valutazione in un report Markdown dedicato
    e aggiorna il file ufficiale documentazione/TABELLA_RISULTATI_COMPLETA.md.
    """
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "documentazione")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"RISULTATI_VALUTAZIONE_{split.upper()}.md")
    main_md_file = os.path.join(out_dir, "TABELLA_RISULTATI_COMPLETA.md")

    lines = []
    lines.append(f"\n## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: {split.upper()}] - {now_str}\n")
    lines.append(f"> **Split analizzato**: `{split.upper()}` (Ufficiale nuScenes)\n")
    lines.append("### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)\n")
    lines.append("| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")

    for m_key, m_info in models_dict.items():
        name = m_info["short_name"]
        for gt_t, gt_desc in [("GT_Reale", "GT Reale nuScenes"),
                              ("GT_Sintetica_Geom", "GT Sintetica (Geometrica 3D)"),
                              ("GT_Sintetica_Sem", "GT Sintetica (Semantica Naïve)"),
                              ("GT_Sintetica_Hyb", "GT Sintetica (Ibrida Spazio-Sem)")]:
            stats_list = counts[m_key][gt_t][25.0]
            tp = sum(s["tp"] for s in stats_list)
            fp = sum(s["fp"] for s in stats_list)
            fn = sum(s["fn"] for s in stats_list)
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
            lines.append(f"| **{name}** | {gt_desc} | {p:5.1f}% | {rc:5.1f}% | **{f:5.1f}%** |")

    lines.append("\n---\n")

    for m_key, m_info in models_dict.items():
        lines.append(f"\n### Modello: {m_info['title']}\n")
        for gt_t, gt_desc in [("GT_Reale", "1. GT Reale nuScenes (3D Reali)"),
                              ("GT_Sintetica_Geom", "2. GT Sintetica Geometrica (Fitting 3D)"),
                              ("GT_Sintetica_Sem", "3. GT Sintetica Semantica (Regole Naïve)"),
                              ("GT_Sintetica_Hyb", "4. GT Sintetica Ibrida (Spazio-Semantica)")]:
            lines.append(f"#### {gt_desc}\n")
            for r in RADII:
                lines.append(f"**Raggio {int(r)}m**\n")
                lines.append("| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
                tp_tot, fp_tot, fn_tot = 0, 0, 0
                for c, c_name in enumerate(CATEGORIES):
                    s = counts[m_key][gt_t][r][c]
                    tp, fp, fn = s["tp"], s["fp"], s["fn"]
                    tp_tot += tp; fp_tot += fp; fn_tot += fn
                    p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                    rc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                    f = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0
                    lines.append(f"| {c_name} | {tp} | {fp} | {fn} | {p:5.1f}% | {rc:5.1f}% | {f:5.1f}% |")
                p_m = (tp_tot / (tp_tot + fp_tot) * 100) if (tp_tot + fp_tot) > 0 else 0
                rc_m = (tp_tot / (tp_tot + fn_tot) * 100) if (tp_tot + fn_tot) > 0 else 0
                f_m = (2 * p_m * rc_m / (p_m + rc_m)) if (p_m + rc_m) > 0 else 0
                lines.append(f"| **MEDIA GLOBALE** | **{tp_tot}** | **{fp_tot}** | **{fn_tot}** | **{p_m:5.1f}%** | **{rc_m:5.1f}%** | **{f_m:5.1f}%** |\n")

    md_content = "\n".join(lines)

    # Salva il report dedicato per lo split corrente (sovrascrive in modo pulito senza duplicazioni)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Report Ufficiale di Valutazione nuScenes - Split {split.upper()}\n" + md_content)
    print(f"\n[REPORT SALVATO]: File Markdown aggiornato in: {out_file}")

    # Salva la cache JSON con tutti i frame precalcolati per il visualizzatore
    if eval_cache is not None:
        cache_split_path = os.path.join(os.path.dirname(__file__), f"cache_valutazione_{split.lower()}.json")
        cache_default_path = os.path.join(os.path.dirname(__file__), "cache_valutazione_ufficiale.json")
        with open(cache_split_path, "w", encoding="utf-8") as cf:
            json.dump(eval_cache, cf)
        if split.lower() == "all" or not os.path.exists(cache_default_path):
            with open(cache_default_path, "w", encoding="utf-8") as cf:
                json.dump(eval_cache, cf)
        print(f"[CACHE VALUTAZIONE SALVATA]:\n  - {cache_split_path}\n  - {cache_default_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione Ufficiale End-to-End con Confronto Multi-Target")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "geometric", "semantic", "hybrid", "real", "positives_only", "bayes"],
                        help="Quale modello valutare: 'geometric', 'semantic', 'hybrid', 'real', 'positives_only', 'bayes' o 'all' per tutti")
    parser.add_argument("--ckpt", type=str, default=None, help="Percorso checkpoint custom .pth (opzionale)")
    parser.add_argument("--gt_mode", type=str, default="hybrid", choices=["geometric", "semantic", "hybrid"],
                        help="Strategia della GT Sintetica di test: 'geometric', 'semantic' o 'hybrid'")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"],
                        help="Split nuScenes su cui valutare: 'val' (2 scene mai viste, default), 'train' o 'all'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = {
        "geometric": {
            "title": "Modello Attention su GT Sintetica Geometrica (Spatially-Constrained)",
            "short_name": "Attention + GT Sintetica (Geometric)",
            "default_file": "per_zone_checkpoint_attention_neuro_geometric.pth"
        },
        "semantic": {
            "title": "Modello Attention su GT Sintetica Semantica (Semantic-Affordance / Naïve)",
            "short_name": "Attention + GT Sintetica (Semantic)",
            "default_file": "per_zone_checkpoint_attention_neuro_semantic.pth"
        },
        "real": {
            "title": "Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)",
            "short_name": "Baseline Attention + GT Reale Completa",
            "default_file": "per_zone_checkpoint_attention_real_gt.pth"
        },
        "positives_only": {
            "title": "Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)",
            "short_name": "Baseline Attention + GT Reale (Positives Only)",
            "default_file": "per_zone_checkpoint_attention_positive_only.pth"
        },
        "hybrid": {
            "title": "Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)",
            "short_name": "Attention + GT Sintetica (Hybrid)",
            "default_file": "per_zone_checkpoint_attention_neuro_hybrid.pth"
        },
        "bayes": {
            "title": "Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)",
            "short_name": "Baseline Analitica (Bayes)",
            "default_file": None
        }
    }

    selected_keys = list(model_configs.keys()) if args.mode == "all" else [args.mode]
    models_to_run = {}

    for key in selected_keys:
        cfg = model_configs[key]
        if key == "bayes":
            models_to_run[key] = {
                "model": None,
                "title": cfg["title"],
                "short_name": cfg["short_name"],
                "ckpt_path": "extracted_occlusions_probabilities/*.json"
            }
            continue

        ckpt_path = args.ckpt if (len(selected_keys) == 1 and args.ckpt) else find_checkpoint(key)

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[ATTENZIONE]: Checkpoint per [{key}] non trovato su disco. Salto questo modello.")
            continue

        model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
        ckpt_data = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        model.load_state_dict(state_dict)
        model.eval()

        models_to_run[key] = {
            "model": model,
            "title": cfg["title"],
            "short_name": cfg["short_name"],
            "ckpt_path": ckpt_path
        }

    if not models_to_run:
        print("[ERRORE]: Nessun modello caricabile trovato. Controlla i checkpoint in 'pesi_modelli/'.")
        return

    evaluate_models(models_to_run, syn_gt_strategy=args.gt_mode, split=args.split)


if __name__ == "__main__":
    main()
