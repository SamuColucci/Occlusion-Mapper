# Modulo Centralizzato per l'Estrazione della Ground Truth BEV a 6 Canali (ground_truth_extractor.py)
# Converte le scatole 3D degli ostacoli nuScenes in maschere binarie BEV a 6 canali:
#   Canale 0: Auto
#   Canale 1: Camion / Bus / Rimorchio
#   Canale 2: Pedone / Umano
#   Canale 3: Moto (Veicolo a motore su strada)
#   Canale 4: Bicicletta (Ciclo-pedonale)
#   Canale 5: Barriera / Cono / Struttura

import numpy as np
import cv2

# Costanti di Griglia Spaziale Bird's Eye View (BEV)
GRID_DIM = 200     # Dimensioni della griglia immagine BEV: 200x200 pixel
GRID_RANGE = 40.0   # Campo visivo spaziale: da -40.0 metri a +40.0 metri attorno all'auto ego
VOXEL_SIZE = 0.4   # Risoluzione spaziale: ogni pixel rappresenta un quadrato di 0.4m x 0.4m


def rasterize_polygon(poly_pts, grid_dim=GRID_DIM, grid_range=GRID_RANGE, voxel_size=VOXEL_SIZE):
    """
    Trasforma un poligono espresso in coordinate metriche 2D (in metri [X, Y]) 
    in una matrice binaria Immagine 200x200 di float (valori 0.0 o 1.0) tramite OpenCV.
    """
    mask = np.zeros((grid_dim, grid_dim), dtype=np.float32)  # Inizializza matrice di zeri
    if poly_pts is None or len(poly_pts) < 3:
        return mask                                          # Controllo di sicurezza su poligoni validi (almeno 3 punti)
        
    pts_np = np.array(poly_pts)
    pts_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])   # Assicura l'ordine corretto [X_right, Y_ahead]
    
    # Proiezione dalle coordinate metriche in metri [X, Y] ai pixel dell'immagine BEV [px, py]
    px = np.clip(((pts_xy[:, 0] + grid_range) / voxel_size).astype(int), 0, grid_dim - 1)
    py = np.clip(((grid_range - pts_xy[:, 1]) / voxel_size).astype(int), 0, grid_dim - 1)
    
    pts_pixel = np.column_stack((px, py)).astype(np.int32)
    cv2.fillPoly(mask, [pts_pixel], 1.0)                     # Disegna e riempie il poligono interno con valore 1.0
    return mask


def extract_ground_truth_masks(frame_data):
    """
    Riceve il dizionario frame_data dal dataset adapter nuScenes ed estrae 
    il tensore target a 6 canali (6, 200, 200) contenente i poligoni d'ingombro 3D reali di ciascun ostacolo.
    """
    # Inizializza 6 matrici 200x200 vuote a zeri per le 6 classi semantiche
    target_cars = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)         # Canale 0: Auto
    target_trucks_buses = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32) # Canale 1: Camion/Bus/Trailer
    target_pedestrians = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)  # Canale 2: Pedoni/Umani
    target_motorcycles = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)  # Canale 3: Moto (Strada)
    target_bicycles = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)     # Canale 4: Bici (Ciclo-pedonale)
    target_structures = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)   # Canale 5: Barriere/Coni

    # Cicla su tutte le scatole 3D degli ostacoli nuScenes presenti nel fotogramma corrente
    for box in frame_data.get("boxes", []):
        # Estrazione del nome della categoria nuScenes dell'ostacolo
        if hasattr(box, 'name'):
            cat = box.name.lower()
        elif isinstance(box, dict):
            cat = box.get('name', '').lower()
        else:
            cat = ''

        # Estrazione dei 4 vertici dell'ingombro a terra 2D [X, Y] in metri
        if hasattr(box, 'bottom_corners'):
            corners = box.bottom_corners()[:2].T
        elif hasattr(box, 'corners'):
            corners = box.corners()[:2, :4].T
        elif isinstance(box, dict):
            corners = box.get('bottom_corners', [])
        else:
            corners = []

        # Se i vertici sono validi, rasterizza il poligono dell'ostacolo e lo unisce alla matrice di classe
        if len(corners) >= 3:
            mask = rasterize_polygon(corners)
            if "car" in cat:
                target_cars = np.maximum(target_cars, mask)
            elif "truck" in cat or "bus" in cat or "trailer" in cat:
                target_trucks_buses = np.maximum(target_trucks_buses, mask)
            elif "human" in cat or "pedestrian" in cat:
                target_pedestrians = np.maximum(target_pedestrians, mask)
            elif "motorcycle" in cat:
                target_motorcycles = np.maximum(target_motorcycles, mask)
            elif "bicycle" in cat:
                target_bicycles = np.maximum(target_bicycles, mask)
            else:
                target_structures = np.maximum(target_structures, mask)

    # Restituisce lo stack delle 6 matrici in un unico tensore binario 3D (6, 200, 200)
    return np.stack([
        target_cars,
        target_trucks_buses,
        target_pedestrians,
        target_motorcycles,
        target_bicycles,
        target_structures
    ], axis=0)


def get_occlusion_ground_truth_target(target_tensor, poly_pts):
    """
    Riceve il tensore target a 6 canali (6, 200, 200) ed i punti del poligono d'ombra [y_ahead, x_right].
    Restituisce un array binario 1D di dimensione 6: [Auto, Camion/Bus, Pedone, Moto, Bici, Barriera]
    dove il valore è 1.0 SOLO se un ostacolo reale nuScenes ricade all'interno di quella specifica ombra.
    """
    num_classes = target_tensor.shape[0] if hasattr(target_tensor, 'shape') else 6
    target_classes = np.zeros(num_classes, dtype=np.float32)
    if poly_pts is None or len(poly_pts) < 3:
        return target_classes
        
    pts_np = np.array(poly_pts)
    pts_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])
    poly_mask = rasterize_polygon(pts_xy) > 0.5  # Maschera binaria 2D della specifica zona d'ombra
    
    # Interroga ciascuno dei 6 canali GT SOLO ed ESCLUSIVAMENTE sui pixel dell'ombra:
    if np.any(poly_mask):
        tgt_np = target_tensor.numpy() if hasattr(target_tensor, 'numpy') else target_tensor
        for c in range(num_classes):
            if np.any(tgt_np[c][poly_mask] > 0.5):
                target_classes[c] = 1.0  # Assegna 1.0 SOLO se l'ostacolo è DENTRO l'ombra!
                
    return target_classes


if __name__ == "__main__":
    from factory_dataset import create_adapter
    adapter = create_adapter("nuscenes", "./nuscenes")
    sample_data = adapter.get_sample_data(0)
    gt_masks = extract_ground_truth_masks(sample_data)
    print("Self-Test Ground Truth Extractor (6 Canali) Completo!")
    print(f"  GT Masks Shape: {gt_masks.shape}")
    print(f"  Target positivi estratti per classe nel frame 0: {np.sum(gt_masks > 0.5, axis=(1,2))}")
