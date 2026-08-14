# Modulo di Estrazione e Iniezione Sintetica della Ground Truth a 6 Classi (ground_truth_extractor_synthetic.py)
# Inietta ostacoli sintetici verosimili distribuiti su tutte le 6 classi semantiche
# (Auto, Camion/Bus, Pedone, Moto, Bicicletta, Barriera) STRICTLY INSIDE THE OCCLUSION SHADOW ZONES.

import json
import os
import numpy as np
import cv2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, GRID_DIM, GRID_RANGE, VOXEL_SIZE

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
    
    VINCOLO FONDAMENTALE DI SICUREZZA:
    Gli ostacoli sintetici vengono posizionati ed INTERSECATI RIGOROSAMENTE ED ESCLUSIVAMENTE 
    ALL'INTERNO DEI PIXEL DELLA ZONA D'OMBRA OCCLUSA (occ_mask), impedendo qualsiasi sconfinamento in zone note.
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
        
        # Se l'ombra ha un'area sufficiente (> 10m²) ed il generatore casuale la seleziona
        if area >= 10.0 and np.random.rand() < injection_rate:
            # 1. Rasterizza l'esatta maschera binaria 2D dell'ombra (200x200)
            occ_mask = rasterize_polygon(poly_pts)
            y_indices, x_indices = np.where(occ_mask > 0.5)
            
            if len(y_indices) < 10:
                continue
                
            # Calcola il baricentro interno della zona d'ombra in pixel BEV
            py_center = int(np.mean(y_indices))
            px_center = int(np.mean(x_indices))
            
            # 2. Selezione dinamica della classe in base alla semantica del terreno sottostante
            if walkway_mask[py_center, px_center] > 0.5 or ped_crossing_mask[py_center, px_center] > 0.5:
                chosen_class = np.random.choice([2, 4], p=[0.7, 0.3])  # 70% Pedone, 30% Bici
            elif drivable_mask[py_center, px_center] > 0.5:
                chosen_class = np.random.choice([0, 1, 3], p=[0.6, 0.25, 0.15])  # 60% Auto, 25% Camion, 15% Moto
            else:
                chosen_class = np.random.choice([5, 4, 2], p=[0.5, 0.25, 0.25])  # 50% Barriera, 25% Bici, 25% Pedone

            c_name, length_m, width_m = OBSTACLE_SPECS_M[chosen_class]
            
            # Converti raggio dell'ostacolo sintetico in pixel BEV (0.4m/pixel)
            radius_px_x = max(1, int(round((length_m / 2.0) / VOXEL_SIZE)))
            radius_px_y = max(1, int(round((width_m / 2.0) / VOXEL_SIZE)))
            
            # 3. Genera il rettangolo dell'ostacolo centrato in (px_center, py_center)
            obs_mask = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
            y_min = max(0, py_center - radius_px_y)
            y_max = min(GRID_DIM, py_center + radius_px_y + 1)
            x_min = max(0, px_center - radius_px_x)
            x_max = min(GRID_DIM, px_center + radius_px_x + 1)
            
            obs_mask[y_min:y_max, x_min:x_max] = 1.0
            
            # 4. VINCOLO RIGOROSO: Interseca l'ostacolo CON LA MASCHERA D'OMBRA (obs_mask * occ_mask)
            # Garantisce che NESSUN pixel dell'ostacolo esca fuori dalla zona occlusa!
            injected_obstacle_mask = obs_mask * occ_mask
            
            if np.any(injected_obstacle_mask > 0.5):
                synthetic_gt_masks[chosen_class] = np.maximum(synthetic_gt_masks[chosen_class], injected_obstacle_mask)
                injected_count += 1

    return synthetic_gt_masks, injected_count

if __name__ == "__main__":
    from dataset_adapter.factory_dataset import create_adapter
    adapter = create_adapter("nuscenes", "./nuscenes")
    sample_data = adapter.get_sample_data(0)
    syn_masks, count = generate_synthetic_injected_gt(sample_data)
    print(f"Test Iniezione Sintetica OK: Iniettati {count} ostacoli rigorosamente nelle zone d'ombra.")
