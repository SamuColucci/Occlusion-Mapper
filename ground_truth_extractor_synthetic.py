# Modulo di Estrazione e Iniezione Sintetica della Ground Truth a 6 Classi (ground_truth_extractor_synthetic.py)
# Inietta ostacoli sintetici verosimili distribuiti su tutte le 6 classi semantiche
# (Auto, Camion/Bus, Pedone, Moto, Bicicletta, Barriera) in un sottoinsieme casuale di zone d'ombra ampie.

import json
import os
import numpy as np
import cv2

from ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, GRID_DIM, GRID_RANGE, VOXEL_SIZE

# Dimensioni verosimili degli ostacoli 2D (Lunghezza x Larghezza in metri)
OBSTACLE_SPECS_M = {
    0: ("Auto", 4.2, 1.8),         # Canale 0: Auto
    1: ("Camion/Bus", 7.5, 2.5),   # Canale 1: Camion / Bus
    2: ("Pedone", 0.6, 0.6),       # Canale 2: Pedone / Umano
    3: ("Moto", 2.0, 0.8),         # Canale 3: Moto
    4: ("Bicicletta", 1.6, 0.6),   # Canale 4: Bicicletta
    5: ("Barriera", 1.2, 0.5)      # Canale 5: Barriera / Cono
}

def generate_synthetic_injected_gt(frame_data, injection_rate=0.25, seed=42):
    """
    Prende la Ground Truth reale nuScenes a 6 canali (6, 200, 200) ed inietta ostacoli sintetici 
    verosimili estratti a caso su tutte le 6 classi in un sottoinsieme casuale (25%) di ombre ampie (>10m²).
    
    Regole di Verosimiglianza Terreno:
    - STRADA / PARCHEGGIO -> Estrazione casuale tra Auto (0), Camion/Bus (1), Moto (3)
    - MARCIAPIEDE / STRISCE -> Estrazione casuale tra Pedone (2), Bicicletta (4)
    - TERRENO / AI BORDI -> Estrazione casuale tra Barriera (5), Bicicletta (4), Pedone (2)
    """
    sample_token = frame_data.get("sample_token", "0")
    np.random.seed(seed + int(sample_token[:4], 16) % 1000)
    
    gt_masks = extract_ground_truth_masks(frame_data)
    synthetic_gt_masks = np.copy(gt_masks)
    
    json_path = os.path.join("extracted_occlusions", f"{sample_token}.json")
    if not os.path.exists(json_path):
        return synthetic_gt_masks, 0

    with open(json_path, "r") as f:
        occlusions = json.load(f).get("occlusions", [])

    injected_count = 0
    semantic_map = frame_data.get("semantic_map", {})
    walkway_mask = semantic_map.get('walkway', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
    ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
    drivable_mask = semantic_map.get('drivable_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))

    for occ in occlusions:
        poly_pts = occ.get("polygon_points_m", [])
        area = occ.get("area_sqm", 0.0)
        
        # Se l'ombra ha un'area sufficiente (> 10m²) ed il generatore casuale la seleziona (25% delle zone)
        if area >= 10.0 and np.random.rand() < injection_rate:
            pts_arr = np.array(poly_pts)
            pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
            
            cx = float(np.mean(pts_xy[:, 0]))
            cy = float(np.mean(pts_xy[:, 1]))
            
            # Converti il centro dell'ombra in pixel BEV
            px = int((cx + GRID_RANGE) / VOXEL_SIZE)
            py = int((GRID_RANGE - cy) / VOXEL_SIZE)
            
            if 10 <= px < GRID_DIM - 10 and 10 <= py < GRID_DIM - 10:
                # 1. Zona Marciapiede o Strisce Pedonali -> Ostacoli Ciclo-Pedonali (Pedoni o Bici)
                if walkway_mask[py, px] > 0.5 or ped_crossing_mask[py, px] > 0.5:
                    chosen_class = np.random.choice([2, 4], p=[0.7, 0.3])  # 70% Pedone, 30% Bici
                # 2. Zona Strada -> Veicoli a Motore (Auto, Camion/Bus, Moto)
                elif drivable_mask[py, px] > 0.5:
                    chosen_class = np.random.choice([0, 1, 3], p=[0.6, 0.25, 0.15])  # 60% Auto, 25% Camion, 15% Moto
                # 3. Zona Terreno Fuoristrada -> Barriere, Bici o Pedoni
                else:
                    chosen_class = np.random.choice([5, 4, 2], p=[0.5, 0.25, 0.25])  # 50% Barriera, 25% Bici, 25% Pedone

                c_name, length_m, width_m = OBSTACLE_SPECS_M[chosen_class]
                half_l = length_m / 2.0
                half_w = width_m / 2.0

                # Bounding Box dell'ostacolo sintetico verosimile attorno al centroide dell'ombra
                obstacle_box = np.array([
                    [cx - half_l, cy - half_w],
                    [cx + half_l, cy - half_w],
                    [cx + half_l, cy + half_w],
                    [cx - half_l, cy + half_w]
                ])
                
                obs_mask = rasterize_polygon(obstacle_box)
                synthetic_gt_masks[chosen_class] = np.maximum(synthetic_gt_masks[chosen_class], obs_mask)
                injected_count += 1

    return synthetic_gt_masks, injected_count

if __name__ == "__main__":
    from factory_dataset import create_adapter
    adapter = create_adapter("nuscenes", "./nuscenes")
    sample_data = adapter.get_sample_data(0)
    syn_gt, count = generate_synthetic_injected_gt(sample_data)
    print("Self-Test Ground Truth Extractor Sintetico a 6 Classi OK!")
    print(f"  Ostacoli verosimili iniettati nel frame 0: {count}")
    print(f"  Target positivi per classe nel frame 0: {np.sum(syn_gt > 0.5, axis=(1,2))}")
