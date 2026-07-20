import os
import sys
import glob
import json
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.nuscenes import NuScenes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bayesian_occlusion_agent import BayesianOcclusionAgent

def box_to_bev_corners(box):
    corners = box.corners()
    return corners[:2, [0, 1, 5, 4]].T

def main():
    print("=== AVVIO VERIFICA FALSI POSITIVI CON VALORI ALTI (>= 90%) ===")
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    agent = BayesianOcclusionAgent()
    
    # Ripristina ricerca globale su tutto il dataset
    json_files = sorted(glob.glob("extracted_occlusions_probabilities/*.json"))
    
    total_high_predictions = 0
    false_positives_high = 0
    
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
            
        lidar_token = data.get("lidar_token")
        if not lidar_token:
            continue
            
        try:
            _, boxes_real, _ = nusc.get_sample_data(lidar_token)
        except Exception:
            continue
            
        for occ in data.get("occlusions", []):
            poly_pts = np.array(occ.get("polygon_points_m", []))
            if len(poly_pts) < 3:
                continue
                
            try:
                poly_xy = np.column_stack([poly_pts[:, 1], poly_pts[:, 0]])
                occ_poly = ShapelyPolygon(poly_xy)
                if not occ_poly.is_valid:
                    occ_poly = occ_poly.buffer(0)
            except Exception:
                continue
                
            source_token = occ.get("object_token", "")
            occ_dist = occ.get("distance_m", 0.0)
            estimated_probs = occ.get("estimated_probabilities", {})
            
            # Scorriamo le categorie con stima molto alta (>= 90%)
            for cat, prob in estimated_probs.items():
                # Consideriamo stime per categorie specifiche (non "Altro" generico)
                if prob >= 0.90 and cat != "Altro":
                    total_high_predictions += 1
                    
                    # Verifichiamo se c'è EFFETTIVAMENTE un oggetto di questa categoria nell'occlusione
                    is_object_really_there = False
                    for real_box in boxes_real:
                        if real_box.token == source_token:
                            continue
                        
                        try:
                            ann = nusc.get('sample_annotation', real_box.token)
                            cur_vis = ann.get('visibility_token', '1')
                        except Exception:
                            cur_vis = '1'
                            
                        # Solo se l'oggetto è attualmente occluso (visibilità 1 o 2)
                        if cur_vis in ['1', '2']:
                            bev = box_to_bev_corners(real_box)
                            real_fp = ShapelyPolygon(bev)
                            if not real_fp.is_valid:
                                real_fp = real_fp.buffer(0)
                                
                            occ_poly_expanded = occ_poly.buffer(0.2)
                            intersects = False
                            if real_fp.is_valid:
                                intersects = occ_poly_expanded.intersects(real_fp)
                            
                            real_cat = agent.classify_source(real_box.name)
                            if intersects and real_cat == cat:
                                is_object_really_there = True
                                break
                                    
                    if not is_object_really_there:
                        false_positives_high += 1
                        print(f"[FALSO POSITIVO] File: {os.path.basename(fpath)}")
                        print(f"  - Categoria stimata ad alta probabilità: {cat} ({prob*100}%)")
                        print(f"  - Ma nessun oggetto reale di tipo {cat} si trova nell'occlusione.")

    print("\n=== ESITO CONTROLLO FALSI POSITIVI ===")
    print(f"Numero totale di stime ad alta probabilità (>= 90%): {total_high_predictions}")
    print(f"Numero totale di Falsi Positivi con stime alte: {false_positives_high}")
    if total_high_predictions > 0:
        ratio = (false_positives_high / total_high_predictions) * 100
        print(f"Rapporto Falsi Positivi: {ratio:.2f}%")
    else:
        print("Nessuna stima alta rilevata.")
    print("=======================================")

if __name__ == "__main__":
    main()
