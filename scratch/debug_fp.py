import os
import sys
import json
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.nuscenes import NuScenes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bayesian_occlusion_agent import BayesianOcclusionAgent

def box_to_bev_corners(box):
    corners = box.corners()
    return corners[:2, [0, 1, 2, 3]].T

def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    agent = BayesianOcclusionAgent()
    
    fpath = "extracted_occlusions_probabilities/occlusion_sample_0379_99f903d3f74a4f76a05a4f1e10e49329.json"
    with open(fpath, "r") as f:
        data = json.load(f)
        
    lidar_token = data.get("lidar_token")
    _, boxes_real, _ = nusc.get_sample_data(lidar_token)
    
    print(f"=== ANALISI DETTAGLIATA FALSO POSITIVO 0379 ===")
    print(f"Lidar token: {lidar_token}")
    
    # Vediamo quali oggetti reali erano visti nel passato di questa scena!
    # Troviamo a quale scena appartiene questo sample
    sd_record = nusc.get('sample_data', lidar_token)
    sample_record = nusc.get('sample', sd_record['sample_token'])
    scene_token = sample_record['scene_token']
    
    # Troviamo tutti i sample precedenti di questa scena
    samples = nusc.field2token('sample', 'scene_token', scene_token)
    # Ordiniamo per timestamp
    samples = [nusc.get('sample', s) for s in samples]
    samples.sort(key=lambda s: s['timestamp'])
    
    # Costruiamo seen_instances del passato proprio come fa l'agente
    seen_instances = {}
    for s in samples:
        if s['timestamp'] >= sample_record['timestamp']:
            break
        # Prendiamo gli oggetti visibili
        s_lidar_token = s['data']['LIDAR_TOP']
        _, s_boxes, _ = nusc.get_sample_data(s_lidar_token)
        for b in s_boxes:
            try:
                ann = nusc.get('sample_annotation', b.token)
                vis = ann.get('visibility_token', '1')
                if vis in ['3', '4']:
                    seen_instances[ann['instance_token']] = agent.classify_source(b.name)
            except Exception:
                pass
                
    print(f"Oggetti visti nel passato (seen_instances): {seen_instances}")
    
    # Analizziamo le occlusioni che hanno stime >= 90%
    for occ_idx, occ in enumerate(data.get("occlusions", [])):
        estimated_probs = occ.get('estimated_probabilities', {})
        high_cats = [cat for cat, p in estimated_probs.items() if p >= 0.90]
        if not high_cats:
            continue
            
        poly_pts = np.array(occ.get("polygon_points_m", []))
        poly_xy = np.column_stack([poly_pts[:, 1], poly_pts[:, 0]])
        occ_poly = ShapelyPolygon(poly_xy)
        occ_poly_expanded = occ_poly.buffer(2.0)
        source_token = occ.get("object_token", "")
        
        print(f"\nOcclusione {occ_idx} (sorgente_token={source_token[:6]}):")
        print(f"  Stime alte rilevate: {estimated_probs}")
        
        # Vediamo quali oggetti reali hanno attivato il memory boost in questa occlusione
        for b in boxes_real:
            if b.token == source_token:
                continue
            bev = box_to_bev_corners(b)
            real_fp = ShapelyPolygon(bev)
            if not real_fp.is_valid:
                real_fp = real_fp.buffer(0)
                
            try:
                ann = nusc.get('sample_annotation', b.token)
                instance_token = ann['instance_token']
                cur_vis = ann.get('visibility_token', '1')
            except Exception:
                continue
                
            if instance_token in seen_instances:
                # Controlliamo se interseca
                inter_exp = occ_poly_expanded.intersects(real_fp) if real_fp.is_valid else False
                if inter_exp:
                    print(f"    -> Target in memoria rilevato vicino all'ombra:")
                    print(f"       Nome: {b.name}, Istanza: {instance_token[:6]}, Categoria: {seen_instances[instance_token]}")
                    print(f"       Visibilità Corrente: {cur_vis}")
                    print(f"       Interseca espanso: {inter_exp}")
                    print(f"       Interseca raw: {occ_poly.intersects(real_fp)}")
                    if real_fp.is_valid:
                        print(f"       Area intersezione raw: {occ_poly.intersection(real_fp).area:.3f}")

if __name__ == "__main__":
    main()
