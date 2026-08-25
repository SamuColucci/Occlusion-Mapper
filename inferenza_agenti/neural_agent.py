# Agente Neurale Per-Zone di Inferenza per la Stima delle Occlusioni (inferenza_agenti/neural_agent.py)
# Carica i pesi addestrati (per_zone_checkpoint.pth / per_zone_checkpoint_focal_semantica.pth / per_zone_checkpoint_asl.pth)
# ed esegue l'inferenza del modello ibrido PerZoneModel.
# Per ciascuna zona d'ombra di ciascun fotogramma:
#   - Ritaglia il patch visivo 2D (11, 64, 64) dagli 11 canali d'ingresso BEV
#   - Estrae il vettore delle 9 feature scalari numeriche (4 geometriche + 5 semantiche)
#   - Esegue l'inferenza forward con la Sigmoid per ottenere le 6 probabilità di presenza
#   - Salva i risultati probabilistici nei file JSON della cartella extracted_occlusions_per_zone/

# Import dei moduli di sistema per la manipolazione dei percorsi e file
import os
import sys
import glob
import json
# Import di numpy per le operazioni matriciali sui canali ed i ritagli dei patch
import numpy as np
# Import di torch e funzionali per il calcolo neurale PyTorch e l'interpolazione bilineare
import torch
import torch.nn.functional as F

# Aggiunge la cartella radice del progetto al sys.path per consentire l'importazione dei moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import di cv2 (OpenCV) per la ricerca dei contorni delle superfici semantiche
import cv2
# Import delle primitive geometriche Shapely per l'intersezione delle sotto-zone
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
# Import del dataset neurale e della funzione di rasterizzazione dei poligoni
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetNeural, rasterize_polygon
# Import dell'architettura neurale PerZoneModel
from architettura_neurale.per_zone_model import PerZoneModel

# Funzione di utilità per fondere le maschere semantiche del terreno (numpy 2D o liste di poligoni) in un unico poligono Shapely (unary_union)
def _build_union_(layer_masks):
    if layer_masks is None or (isinstance(layer_masks, (list, tuple, np.ndarray)) and len(layer_masks) == 0):
        return None
    polys = []
    
    # Se la maschera è un array 2D numpy (200x200), converte i contorni pixel in coordinate metriche (metri)
    if isinstance(layer_masks, np.ndarray) and layer_masks.ndim == 2:
        contours, _ = cv2.findContours((layer_masks > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.squeeze(axis=1)
                # Trasformazione da pixel (0-199) a coordinate metriche (-40m a +40m)
                x_m = (pts[:, 0] * 0.4) - 40.0
                y_m = 40.0 - (pts[:, 1] * 0.4)
                p = ShapelyPolygon(np.column_stack((x_m, y_m)))
                if p.is_valid and p.area > 0.01:
                    polys.append(p)
    elif isinstance(layer_masks, (list, tuple)):
        for mask in layer_masks:
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        pts = cnt.squeeze(axis=1)
                        x_m = (pts[:, 0] * 0.4) - 40.0
                        y_m = 40.0 - (pts[:, 1] * 0.4)
                        p = ShapelyPolygon(np.column_stack((x_m, y_m)))
                        if p.is_valid and p.area > 0.01:
                            polys.append(p)
            elif isinstance(mask, (list, tuple)):
                for p_pts in mask:
                    if len(p_pts) >= 3:
                        p = ShapelyPolygon(p_pts)
                        if p.is_valid and p.area > 0.01:
                            polys.append(p)
    if not polys:
        return None
    # Unione booleana di tutti i poligoni validi appartenenti allo stesso strato semantico
    return unary_union(polys)

# Scompone la zona d'ombra principale in sotto-zone disgiunte in base al tipo di terreno semantico
def subdivide_occlusion_into_subzones(occ, semantic_map):
    poly_pts = occ.get("polygon_points_m", [])
    if len(poly_pts) < 3:
        return occ
    try:
        pts_arr = np.array(poly_pts)
        # polygon_points_m memorizza [y_ahead, x_right]. Convertiamo in [x_right, y_ahead] per Shapely
        pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
        poly = ShapelyPolygon(pts_xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return occ

    occ_area = float(poly.area)
    if occ_area <= 0:
        return occ

    # Ordine di priorità per l'assegnazione delle sotto-zone senza sovrapposizioni:
    # Strisce Pedonali -> Marciapiede -> Parcheggio -> Strada Carrabile
    layers_ordered = [
        ("ped_crossing", _build_union_(semantic_map.get("ped_crossing", []))),
        ("sidewalk", _build_union_(semantic_map.get("walkway", []))),
        ("other_flat", _build_union_(semantic_map.get("carpark_area", []))),
        ("driveable_surface", _build_union_(semantic_map.get("drivable_area", [])))
    ]

    sub_zones = []
    remaining_poly = poly

    for label, layer_geom in layers_ordered:
        if layer_geom is None or layer_geom.is_empty or remaining_poly.is_empty:
            continue
        try:
            # Calcola l'intersezione tra l'ombra residua ed il layer semantico del terreno
            inter = remaining_poly.intersection(layer_geom)
            if inter.is_empty or inter.area < 0.001:
                continue
            sub_area = float(inter.area)
            
            # Estrazione dei punti del poligono della sotto-zona (convertiti in [y_ahead, x_right] per la resa BEV)
            if inter.geom_type == 'Polygon':
                coords_xy = np.array(inter.exterior.coords)
            elif inter.geom_type == 'MultiPolygon':
                largest = max(inter.geoms, key=lambda p: p.area)
                coords_xy = np.array(largest.exterior.coords)
            else:
                coords_xy = pts_xy

            coords_yx = np.column_stack([coords_xy[:, 1], coords_xy[:, 0]])
            sz_pts = [[round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in coords_yx]

            sub_zones.append({
                "surface": label,
                "area_sqm": round(sub_area, 3),
                "area_fraction": round(min(sub_area / occ_area, 1.0), 4),
                "polygon_points_m": sz_pts
            })
            # Sottrae la porzione d'ombra appena assegnata per evitare sovrapposizioni nei layer successivi
            remaining_poly = remaining_poly.difference(inter)
        except Exception:
            continue

    # Terreno residuo non coperto da mappe semantiche specifiche
    if not remaining_poly.is_empty and remaining_poly.area >= 0.001:
        rem_area = float(remaining_poly.area)
        if remaining_poly.geom_type == 'Polygon':
            coords_xy = np.array(remaining_poly.exterior.coords)
        elif remaining_poly.geom_type == 'MultiPolygon':
            largest = max(remaining_poly.geoms, key=lambda p: p.area)
            coords_xy = np.array(largest.exterior.coords)
        else:
            coords_xy = pts_xy

        coords_yx = np.column_stack([coords_xy[:, 1], coords_xy[:, 0]])
        rem_pts = [[round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in coords_yx]

        sub_zones.append({
            "surface": "terrain",
            "area_sqm": round(rem_area, 3),
            "area_fraction": round(min(rem_area / occ_area, 1.0), 4),
            "polygon_points_m": rem_pts
        })

    occ_copy = dict(occ)
    occ_copy["sub_zones"] = sub_zones
    return occ_copy

class PerZoneOcclusionAgent:
    # Inizializzazione dell'Agente Neurale Per-Zone
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Inizializzazione PerZoneOcclusionAgent su: {self.device}")
        self.model = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        
        # Selezione automatica del miglior checkpoint disponibile se non fornito esplicitamente
        if checkpoint_path is None:
            raw_candidates = [
                "per_zone_checkpoint_focal_semantica.pth",
                "per_zone_checkpoint_asl.pth",
                "per_zone_checkpoint_surrounding.pth",
                "per_zone_checkpoint_focal.pth",
                "per_zone_checkpoint_semantica.pth",
                "per_zone_checkpoint.pth"
            ]
            candidates = []
            for c in raw_candidates:
                candidates.append(os.path.join("pesi_modelli", c))
                candidates.append(c)
            checkpoint_path = next((c for c in candidates if os.path.exists(c)), os.path.join("pesi_modelli", "per_zone_checkpoint_focal_semantica.pth"))

        self.checkpoint_used = checkpoint_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Pesi Per-Zone caricati con successo da: {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint '{checkpoint_path}' non trovato!")
            
        # Imposta la rete in modalità valutazione/inferenza (eval)
        self.model.eval()

    # Scorre tutti i fotogrammi del dataset e genera i file JSON probabilistici per-zone
    def process_dataset(self, dataset, out_dir="extracted_occlusions_per_zone"):
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nInizio generazione predizioni Agente Per-Zone (Patch + Scalari) per {len(dataset)} campioni...")

        for idx in range(len(dataset)):
            input_tensor, _ = dataset[idx]
            frame_data = dataset.adapter.get_sample_data(idx)
            sample_token = frame_data['sample_token']
            lidar_token = frame_data['lidar_token']
            scene_token = frame_data.get('scene_token', '')

            # Caricamento delle zone d'ombra dal file JSON base estratto dal RayCaster
            json_base_path = os.path.join("extracted_occlusions", f"{sample_token}.json")
            if not os.path.exists(json_base_path):
                continue
                
            with open(json_base_path, "r") as f:
                raw_data = json.load(f)
                raw_occlusions = raw_data.get("occlusions", [])

            per_zone_occlusions = []
            for raw_occ in raw_occlusions:
                pts = raw_occ.get("polygon_points_m", [])
                if len(pts) >= 3:
                    # Calcola le sotto-zone dinamiche tramite intersezione semantica col terreno
                    occ = subdivide_occlusion_into_subzones(raw_occ, frame_data["semantic_map"])
                if len(pts) >= 3:
                    # Bounding box in pixel attorno alla zona d'ombra nella griglia 200x200
                    pts_np = np.array(pts)
                    y_ahead = pts_np[:, 0]
                    x_right = pts_np[:, 1]
                    px_x = np.clip(((x_right + 40.0) / 0.4).astype(int), 0, 199)
                    px_y = np.clip(((40.0 - y_ahead) / 0.4).astype(int), 0, 199)
                    
                    xmin, xmax = max(0, np.min(px_x)), min(199, np.max(px_x))
                    ymin, ymax = max(0, np.min(px_y)), min(199, np.max(px_y))
                    
                    xmin, xmax = max(0, xmin - 2), min(199, xmax + 2)
                    ymin, ymax = max(0, ymin - 2), min(199, ymax + 2)
                    
                    if xmax <= xmin or ymax <= ymin:
                        continue
                        
                    # Ritaglio e resize del patch visivo dell'ombra a (11, 64, 64)
                    patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                    patch_resized = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device, dtype=torch.float32)
                    
                    # Estrazione dei 4 dati scalari numerici generali dell'ombra
                    area_sqm = float(occ.get("area_sqm", 0.0))
                    distance_m = float(occ.get("distance_m", 0.0))
                    occ_w = float(occ.get("occluder_width_m", 2.0)) if "occluder_width_m" in occ else 2.0
                    occ_h = float(occ.get("occluder_height_m", 1.8)) if "occluder_height_m" in occ else 1.8

                    # Recupero o scomposizione in sub-zone dell'ombra
                    sub_zones_raw = occ.get("sub_zones", [])
                    processed_sub_zones = []
                    
                    if sub_zones_raw:
                        global_probs_acc = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera"]}
                        total_weight = 0.0
                        
                        for sz in sub_zones_raw:
                            surf_name = sz.get("surface", "driveable_surface")
                            frac = float(sz.get("area_fraction", 1.0))
                            sz_area = float(sz.get("area_sqm", area_sqm * frac))
                            sz_w = float(sz.get("occlusion_width_m", occ_w))
                            
                            # One-Hot / Continuous Surface Flags
                            r_flag = 1.0 if surf_name == "driveable_surface" else 0.0
                            s_flag = 1.0 if surf_name == "sidewalk" else 0.0
                            c_flag = 1.0 if surf_name == "ped_crossing" else 0.0
                            p_flag = 1.0 if surf_name == "other_flat" else 0.0
                            t_flag = 1.0 if surf_name == "terrain" else 0.0
                            
                            # Feature scalari della specifica sotto-zona (9 scalari: 4 geometrici + 5 semantici)
                            sz_scalars = torch.tensor([[sz_area, distance_m, sz_w, occ_h, r_flag, s_flag, c_flag, p_flag, t_flag]], dtype=torch.float32).to(self.device)
                            
                            with torch.no_grad():
                                sz_logits = self.model(patch_resized, sz_scalars)
                                sz_probs_raw = torch.sigmoid(sz_logits).squeeze(0).cpu().numpy()
                                
                            sz_p_auto = float(sz_probs_raw[0])
                            sz_p_camion = float(sz_probs_raw[1])
                            sz_p_ped = float(sz_probs_raw[2])
                            sz_p_moto = float(sz_probs_raw[3])
                            sz_p_bici = float(sz_probs_raw[4])
                            sz_p_barriera = float(sz_probs_raw[5])
                            
                            sz_probs_dict = {
                                "Auto": round(sz_p_auto, 4),
                                "Pedone": round(sz_p_ped, 4),
                                "Camion": round(sz_p_camion, 4),
                                "Bicicletta": round(sz_p_bici, 4),
                                "Moto": round(sz_p_moto, 4),
                                "Bus": round(sz_p_camion * 0.6, 4),
                                "Rimorchio": round(sz_p_camion * 0.4, 4),
                                "Barriera": round(sz_p_barriera, 4),
                                "Cono": 0.0,
                                "Altro": 0.0
                            }
                            
                            new_sz = dict(sz)
                            new_sz["estimated_probabilities"] = sz_probs_dict
                            processed_sub_zones.append(new_sz)
                            
                            for k in global_probs_acc:
                                global_probs_acc[k] = max(global_probs_acc[k], sz_probs_dict[k])
                                
                        estimated_probabilities = {k: round(v, 4) for k, v in global_probs_acc.items()}
                    else:
                        # Inferenza singola se l'ombra non è suddivisa (9 scalari)
                        road_f = float(occ.get("road_fraction", 0.0))
                        side_f = float(occ.get("sidewalk_fraction", 0.0))
                        cross_f = float(occ.get("crosswalk_fraction", 0.0))
                        park_f = float(occ.get("carpark_fraction", 0.0))
                        terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f + park_f))
                        scalars = torch.tensor([[area_sqm, distance_m, occ_w, occ_h, road_f, side_f, cross_f, park_f, terr_f]], dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            logits = self.model(patch_resized, scalars)
                            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                            
                        estimated_probabilities = {
                            "Auto": round(float(probs[0]), 4),
                            "Camion": round(float(probs[1]), 4),
                            "Pedone": round(float(probs[2]), 4),
                            "Moto": round(float(probs[3]), 4),
                            "Bicicletta": round(float(probs[4]), 4),
                            "Bus": round(float(probs[1]) * 0.6, 4),
                            "Rimorchio": round(float(probs[1]) * 0.4, 4),
                            "Barriera": round(float(probs[5]), 4),
                            "Cono": 0.0,
                            "Altro": 0.0
                        }

                    new_occ = dict(occ)
                    new_occ["estimated_probabilities"] = estimated_probabilities
                    new_occ["sub_zones"] = processed_sub_zones
                    new_occ["per_zone_predicted"] = True
                    per_zone_occlusions.append(new_occ)

            # Salva i risultati probabilistici nel file JSON di output
            out_path = os.path.join(out_dir, f"occlusion_per_zone_{idx:04d}_{sample_token}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "sample_token": sample_token,
                    "lidar_token": lidar_token,
                    "scene_token": scene_token,
                    "model_checkpoint_used": getattr(self, "checkpoint_used", "N/A"),
                    "num_occlusions": len(per_zone_occlusions),
                    "occlusions": per_zone_occlusions
                }, f, indent=4)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(dataset):
                print(f"  Progresso Inferenza Per-Zone: {idx+1}/{len(dataset)} campioni elaborati...")

        print(f"Predizioni Agente Per-Zone completate e salvate in: {os.path.abspath(out_dir)}\n")


# Blocco principale di esecuzione se avviato da riga di comando
if __name__ == "__main__":
    dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot="./nuscenes")
    agent = PerZoneOcclusionAgent()  # Carica automaticamente il miglior checkpoint disponibile!
    agent.process_dataset(dataset)
