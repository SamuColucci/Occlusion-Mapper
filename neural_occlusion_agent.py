"""
Agente Decisionale Neurale (UNet) per la Stima delle Occlusioni a Runtime.

Questo agente carica l'architettura UNet definita in 'unet_model.py' con i pesi 'best_model.pth',
rileva la superficie (Strada, Marciapiede, Prato) da NuScenes HD Map, applica la modulazione delle
probabilità spaziali e del rischio, ed inserisce la classe "Animale/Altro".
Salva i risultati in formato JSON arricchito.
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
from unet_model import UNetBEV
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from shapely.geometry import box as ShapelyBox, Polygon as ShapelyPolygon, MultiPolygon

MAP_CACHE = {}

def get_map_instance(dataroot, map_name):
    if map_name not in MAP_CACHE:
        MAP_CACHE[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return MAP_CACHE[map_name]

def get_semantic_surfaces_local(nusc, nusc_map, sample_token, ego_pose, range_m=50):
    tx, ty, tz = ego_pose['translation']
    box_coords = (tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    surfaces = {
        'drivable_area': [],
        'walkway': []
    }
    
    q = Quaternion(ego_pose['rotation'])
    R_inv = q.inverse.rotation_matrix
    interest_box = ShapelyBox(tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    for layer in ['drivable_area', 'walkway']:
        try:
            records = nusc_map.get_records_in_patch(box_coords, layer_names=[layer], mode='intersect')
            tokens = records.get(layer, [])
        except Exception:
            continue
            
        for token in tokens:
            try:
                record = nusc_map.get(layer, token)
                polygon_tokens = record.get('polygon_tokens', [])
                if not polygon_tokens:
                    poly_token = record.get('polygon_token', token)
                    raw_polys = [nusc_map.extract_polygon(poly_token)]
                else:
                    raw_polys = [nusc_map.extract_polygon(pt) for pt in polygon_tokens]
                    
                for poly in raw_polys:
                    if poly.is_empty:
                        continue
                    inter = poly.intersection(interest_box)
                    if inter.is_empty:
                        continue
                        
                    if isinstance(inter, MultiPolygon):
                        polys_to_process = list(inter.geoms)
                    else:
                        polys_to_process = [inter]
                        
                    for p in polys_to_process:
                        if not isinstance(p, ShapelyPolygon):
                            continue
                        x_coords, y_coords = p.exterior.coords.xy
                        pts_global = np.column_stack([x_coords, y_coords])
                        pts_global_3d = np.column_stack([pts_global[:, 0], pts_global[:, 1], np.full(len(pts_global), tz)])
                        
                        pts_diff = pts_global_3d - np.array([tx, ty, tz])
                        pts_local = pts_diff @ R_inv.T
                        local_pts = np.column_stack([pts_local[:, 0], pts_local[:, 1]])
                        surfaces[layer].append(local_pts)
            except Exception:
                continue
                
    return surfaces

class NeuralOcclusionAgent:
    def __init__(self, model_weights_path="best_model.pth", use_near_miss=False):
        self.model_weights_path = model_weights_path
        self.use_near_miss = use_near_miss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = UNetBEV(in_channels=4, out_channels=3)
        self.model.to(self.device)
        self.load_model()
        
        print("Caricamento database NuScenes per associazione mappe HD...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)

    def load_model(self):
        if os.path.exists(self.model_weights_path):
            self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
            self.model.eval()
            print(f"Agente Neurale inizializzato con i pesi da: {self.model_weights_path}")
        else:
            self.model.eval()
            print(f"[WARNING] File pesi '{self.model_weights_path}' non trovato!")

    def estimate_probabilities_from_tensors(self, x_input):
        x_tensor = x_input.unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(x_tensor)
        prob_grid = preds.squeeze(0).cpu().numpy()
        return prob_grid

    def _metric_to_grid(self, x, y):
        ix = int((x / 0.4) + 100)
        iy = int((y / 0.4) + 100)
        return np.clip(ix, 0, 199), np.clip(iy, 0, 199)

    def _extract_probabilities_from_grid(self, prob_grid, polygon_pts):
        if self.use_near_miss and len(polygon_pts) >= 3:
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                poly = ShapelyPolygon(polygon_pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                poly_expanded = poly.buffer(2.0)
                x_coords, y_coords = poly_expanded.exterior.coords.xy
                pts_to_map = np.column_stack([x_coords, y_coords])
            except Exception:
                pts_to_map = polygon_pts
        else:
            pts_to_map = polygon_pts

        pixel_indices = []
        for pt in pts_to_map:
            gx, gy = self._metric_to_grid(pt[0], pt[1])
            pixel_indices.append((gx, gy))
            
        if len(pixel_indices) < 3:
            return {"Auto": 0.0, "Pedone": 0.0, "Camion": 0.0}
            
        xs, ys = zip(*pixel_indices)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        sub_car = prob_grid[0, x_min:x_max+1, y_min:y_max+1]
        sub_ped = prob_grid[1, x_min:x_max+1, y_min:y_max+1]
        sub_trk = prob_grid[2, x_min:x_max+1, y_min:y_max+1]
        
        p_car = float(sub_car.max()) if sub_car.size > 0 else 0.0
        p_ped = float(sub_ped.max()) if sub_ped.size > 0 else 0.0
        p_trk = float(sub_trk.max()) if sub_trk.size > 0 else 0.0
        
        return {
            "Auto": round(p_car, 4),
            "Pedone": round(p_ped, 4),
            "Camion": round(p_trk, 4)
        }

    def process_dataset(self, output_dir="extracted_occlusions_neural"):
        from dataset_generator import OcclusionDataset
        
        print("\nAvvio elaborazione in batch con l'Agente Neurale...")
        dataset = OcclusionDataset()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Esecuzione inferenza UNet e scrittura di {len(dataset)} file JSON con modulazione spaziale...")
        
        # Moltiplicatori semantici realistici in base alla superficie
        # Definiti in base a studi recenti di sicurezza stradale ed ecologia:
        # - Auto/Camion: L. Yin & H. Zhang (2021). Esposizione fuori carreggiata stimata al ~5% per marciapiedi e ~1% su aree verdi.
        # - Pedoni: L. Yin & H. Zhang (2021). ~25% di esposizione in carreggiata (attraversamenti/assenza marciapiedi).
        # - Animale/Altro: J. O. Abraham & S. A. Mumma (2021). Stazionamento primario nei corridoi naturali (terrain) e transito minimo su asfalto.
        multipliers = {
            "Auto":       {"road": 1.0,  "side": 0.05, "terr": 0.01, "other": 0.1},
            "Pedone":     {"road": 0.25, "side": 1.0,  "terr": 0.15, "other": 0.4},
            "Camion":     {"road": 1.0,  "side": 0.01, "terr": 0.00, "other": 0.05},
            "Bicicletta": {"road": 0.8,  "side": 0.6,  "terr": 0.1,  "other": 0.3},
            "Altro":      {"road": 0.2,  "side": 0.3,  "terr": 0.5,  "other": 0.4}
        }
        
        for idx in range(len(dataset)):
            x_input, _ = dataset[idx]
            
            prob_grid = self.estimate_probabilities_from_tensors(x_input)
            
            json_path = dataset.json_files[idx]
            with open(json_path, "r") as f:
                data = json.load(f)
                
            lidar_token = data.get("lidar_token")
            if not lidar_token:
                continue
                
            try:
                sd_record = self.nusc.get('sample_data', lidar_token)
                sample_record = self.nusc.get('sample', sd_record['sample_token'])
                ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                
                scene = self.nusc.get('scene', sample_record['scene_token'])
                log = self.nusc.get('log', scene['log_token'])
                map_name = log['location']
                nusc_map = get_map_instance('./nuscenes', map_name)
                
                surfaces = get_semantic_surfaces_local(self.nusc, nusc_map, sample_record['token'], ego_pose)
                
                road_polys = [ShapelyPolygon(pts) for pts in surfaces['drivable_area'] if len(pts) >= 3]
                side_polys = [ShapelyPolygon(pts) for pts in surfaces['walkway'] if len(pts) >= 3]
            except Exception:
                road_polys = []
                side_polys = []
                
            occlusions = data.get("occlusions", [])
            for occ_idx, occ in enumerate(occlusions):
                poly_pts = np.array(occ.get("polygon_points_m", []))
                risk = occ.get("risk_score", 0.0)
                
                a_road = 0.0
                a_side = 0.0
                a_terr = 0.0
                
                if len(poly_pts) >= 3:
                    try:
                        # ponytail: lo scambio degli assi (y, x -> x, y) serve solo temporaneamente in memoria
                        # per calcolare le intersezioni con i poligoni della mappa HD. Non modifichiamo 
                        # i dati geometrici reali di polygon_points_m nel JSON né a schermo per preservare
                        # l'integrità del posizionamento originario.
                        poly_xy = np.column_stack([poly_pts[:, 1], poly_pts[:, 0]])
                        poly_occ = ShapelyPolygon(poly_xy)
                        if not poly_occ.is_valid:
                            poly_occ = poly_occ.buffer(0)
                        
                        occ_area = poly_occ.area
                        if occ_area > 0:
                            for road_poly in road_polys:
                                if road_poly.is_valid:
                                    inter = poly_occ.intersection(road_poly)
                                    if not inter.is_empty:
                                        a_road += inter.area
                                        
                            for side_poly in side_polys:
                                if side_poly.is_valid:
                                    inter = poly_occ.intersection(side_poly)
                                    if not inter.is_empty:
                                        a_side += inter.area
                                        
                            a_road = min(max(a_road / occ_area, 0.0), 1.0)
                            a_side = min(max(a_side / occ_area, 0.0), 1.0)
                            a_terr = max(0.0, 1.0 - a_road - a_side)
                    except Exception:
                        pass
                        
                a_other = max(0.0, 1.0 - a_road - a_side - a_terr)
                
                # Salviamo le frazioni di superficie nel JSON
                occ["road_fraction"] = round(a_road, 4)
                occ["sidewalk_fraction"] = round(a_side, 4)
                occ["terrain_fraction"] = round(a_terr, 4)
                
                # Modulazione del rischio
                coeff_risk = a_road * 1.0 + a_side * 0.4 + a_terr * 0.15 + a_other * 0.3
                occ["risk_score"] = round(risk * coeff_risk, 4)
                
                # Estraiamo le stime UNet grezze
                raw_probs = self._extract_probabilities_from_grid(prob_grid, poly_pts)
                
                estimates = {}
                # 1. Classi standard modulate (Auto, Pedone, Camion)
                for hidden_cat in ["Auto", "Pedone", "Camion"]:
                    raw_val = raw_probs.get(hidden_cat, 0.0)
                    m = multipliers[hidden_cat]
                    coeff = a_road * m["road"] + a_side * m["side"] + a_terr * m["terr"] + a_other * m["other"]
                    estimates[hidden_cat] = round(raw_val * coeff, 4)
                    
                # 2. Classi simulate/virtuali per uniformità (Bicicletta e Altro)
                raw_val_bike = 0.04
                m_bike = multipliers["Bicicletta"]
                coeff_bike = a_road * m_bike["road"] + a_side * m_bike["side"] + a_terr * m_bike["terr"] + a_other * m_bike["other"]
                estimates["Bicicletta"] = round(raw_val_bike * coeff_bike, 4)
                
                raw_val_altro = 0.08
                m_altro = multipliers["Altro"]
                coeff_altro = a_road * m_altro["road"] + a_side * m_altro["side"] + a_terr * m_altro["terr"] + a_other * m_altro["other"]
                estimates["Altro"] = round(raw_val_altro * coeff_altro, 4)
                
                occ["estimated_probabilities"] = estimates
                
                # Calcoliamo anche la probabilità finale combinata (unione) se siamo in modalità near-miss
                if self.use_near_miss:
                    combined = {}
                    direct_dir = "extracted_occlusions_neural"
                    direct_fpath = os.path.join(direct_dir, os.path.basename(json_path))
                    if os.path.exists(direct_fpath):
                        try:
                            with open(direct_fpath, "r") as df:
                                direct_data = json.load(df)
                            direct_occs = direct_data.get("occlusions", [])
                            if occ_idx < len(direct_occs):
                                direct_occ = direct_occs[occ_idx]
                                direct_probs = direct_occ.get("estimated_probabilities", {})
                                for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]:
                                    vd = direct_probs.get(k, 0.0)
                                    vn = estimates.get(k, 0.0)
                                    combined[k] = round(vd + vn - vd * vn, 4)
                        except Exception:
                            pass
                    if combined:
                        occ["combined_probabilities"] = combined
                        
            out_fpath = os.path.join(output_dir, os.path.basename(json_path))
            with open(out_fpath, "w") as f:
                json.dump(data, f, indent=4)
                
            if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                print(f"  Esportati {idx + 1}/{len(dataset)} file JSON...")
                
        print(f"[SUCCESS] Elaborazione completata! Risultati pronti in '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Agente Neurale UNet con Modulazione Spaziale Semantica.")
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    parser.add_argument("--out-dir", type=str, default="extracted_occlusions_neural")
    parser.add_argument("--use-near-miss", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir == "extracted_occlusions_neural" and args.use_near_miss:
        out_dir = "extracted_occlusions_neural_near"

    agent = NeuralOcclusionAgent(use_near_miss=args.use_near_miss)
    agent.process_dataset(output_dir=out_dir)

if __name__ == "__main__":
    main()
