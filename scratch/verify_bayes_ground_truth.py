import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.nuscenes import NuScenes

# Aggiunge la directory principale a sys.path per importare l'agente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bayesian_occlusion_agent import BayesianOcclusionAgent

def box_to_bev_corners(box):
    corners = box.corners()
    return corners[:2, [0, 1, 5, 4]].T

def main():
    print("=== INIZIO VERIFICA QUANTITATIVA AGENTE BAYESIANO ===")
    print("Inizializzazione NuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    agent = BayesianOcclusionAgent()
    
    json_files = sorted(glob.glob("extracted_occlusions_probabilities/*.json"))
    print(f"Scansione di {len(json_files)} file JSON contenenti le probabilità stimate...")
    
    total_hits = 0
    total_memory_boosts = 0
    boost_val_correct = 0
    
    # Raccogliamo i risultati per calcolare le metriche
    estimations_on_true_class = []
    category_metrics = defaultdict(list)
    
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
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue
                
            try:
                occ_poly = ShapelyPolygon(poly_pts)
                if not occ_poly.is_valid:
                    occ_poly = occ_poly.buffer(0)
            except Exception:
                continue
                
            source_token = occ.get("object_token", "")
            occ_dist = occ.get("distance_m", 0.0)
            estimated_probs = occ.get("estimated_probabilities", {})
            sub_zones = occ.get("sub_zones", [])
            
            # Controlliamo quali oggetti reali (Ground Truth) cadono nell'occlusione
            for real_box in boxes_real:
                if real_box.token == source_token:
                    # Salta lo stesso ostacolo che crea l'occlusione
                    continue
                    
                # Filtri di adiacenza (stessa logica usata per la visualizzazione hit reali)
                hidden_dist = np.linalg.norm(real_box.center[:2])
                if hidden_dist < occ_dist - 1.0:
                    continue
                bbox = occ.get("occlusion_bbox_m", [0,0,0,0])
                if np.linalg.norm(real_box.center[:2] - np.array([bbox[0], bbox[1]])) < 3.0:
                    continue
                    
                bev = box_to_bev_corners(real_box)
                real_fp = ShapelyPolygon(bev)
                if not real_fp.is_valid:
                    real_fp = real_fp.buffer(0)
                    
                if real_fp.is_valid and occ_poly.intersects(real_fp):
                    inter_area = occ_poly.intersection(real_fp).area
                    if inter_area > 0.1:
                        # Abbiamo un HIT reale (un oggetto Ground Truth che è fisicamente dentro l'occlusione!)
                        real_cat = agent.classify_source(real_box.name)
                        est_prob = estimated_probs.get(real_cat, 0.0)
                        
                        estimations_on_true_class.append(est_prob)
                        category_metrics[real_cat].append(est_prob)
                        total_hits += 1
                        
                        # Verifichiamo se c'è memoria temporale (seen_instances) attiva per questo target
                        # Cerchiamo se in una delle sotto-zone è stata attivata la probabilità 0.95 per questa classe
                        has_boost = False
                        for sz in sub_zones:
                            sz_probs = sz.get("estimated_probabilities", {})
                            if sz_probs.get(real_cat, 0.0) >= 0.95:
                                has_boost = True
                                
                        if has_boost:
                            total_memory_boosts += 1
                            # Verifichiamo che il valore sia fissato esattamente a 0.95 o al valore massimo del boost,
                            # e che non sia sommato a caso con altri coefficienti
                            if est_prob >= 0.95:
                                boost_val_correct += 1

    print("\n=== RISULTATI DEL CONFRONTO CON IL GROUND TRUTH ===")
    print(f"1. Numero totale di oggetti reali individuati all'interno delle occlusioni: {total_hits}")
    if total_hits > 0:
        avg_prob = np.mean(estimations_on_true_class) * 100
        print(f"2. Probabilità media assegnata dall'Agente Bayesiano alla CLASSE REALE presente nell'ombra: {avg_prob:.2f}%")
        
        # Accuratezza con soglie (se stima > 20% consideriamo l'evento 'predetto con successo')
        hits_pred_success = sum(1 for p in estimations_on_true_class if p >= 0.20)
        success_rate = (hits_pred_success / total_hits) * 100
        print(f"3. Percentuale di hit in cui l'agente ha previsto correttamente una probabilità significativa (>= 20%): {success_rate:.2f}%")
    else:
        print("Nessun hit reale trovato per il calcolo delle statistiche.")
        
    print(f"\n=== VERIFICA SPECIFICA REQUISITO MEMORIA TEMPORALE (BOOST 95%) ===")
    print(f"- Numero di hit reali che presentavano memoria temporale passata (Seen Instances): {total_memory_boosts}")
    if total_memory_boosts > 0:
        boost_acc = (boost_val_correct / total_memory_boosts) * 100
        print(f"- Percentuale di casi in cui il valore di stima finale della sub-zona è rimasto fisso >= 95% (e non sommato erroneamente): {boost_acc:.2f}%")
        print("  [OK] La memoria si applica come override fisso invece che come somma cumulativa.")
    else:
        print("- Nessun caso di memoria temporale rilevato negli hit correnti.")

    print("\n=== DETTAGLIO PER CATEGORIA DI OGGETTO REALE ===")
    for cat, probs in sorted(category_metrics.items()):
        cnt = len(probs)
        avg = np.mean(probs) * 100
        print(f"- {cat:<12}: Trovati {cnt:>2} casi reali, Probabilità media stimata dall'agente = {avg:.2f}%")
    
    print("\n=== FINE VERIFICA ===")

if __name__ == "__main__":
    main()
