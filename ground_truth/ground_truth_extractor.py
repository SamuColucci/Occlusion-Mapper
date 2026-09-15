# Modulo Centralizzato per l'Estrazione della Ground Truth BEV a 6 Canali (ground_truth_extractor.py)
# Converte le scatole 3D degli ostacoli nuScenes in maschere binarie BEV a 6 canali:
#   Canale 0: Auto
#   Canale 1: Camion / Bus / Rimorchio
#   Canale 2: Pedone / Umano
#   Canale 3: Moto (Veicolo a motore su strada)
#   Canale 4: Bicicletta (Ciclo-pedonale)
#   Canale 5: Barriera / Cono / Struttura

# Import di numpy per la manipolazione di array multidimensionali e matrici di maschera
import numpy as np
# Import di cv2 (OpenCV) per le operazioni di rasterizzazione e disegno di poligoni su immagini
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
    # Inizializza la matrice binaria di zeri con la dimensione 200x200
    mask = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    # Controllo di sicurezza: se il poligono è nullo o ha meno di 3 vertici, restituisce la maschera vuota
    if poly_pts is None or len(poly_pts) < 3:
        return mask
        
    # Converte la lista di vertici in un array numpy
    pts_np = np.array(poly_pts)
    # Assicura l'ordine delle colonne per le coordinate [X_right, Y_ahead]
    pts_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])
    
    # Trasforma le coordinate metriche in metri [X, Y] nei corrispondenti indici pixel dell'immagine BEV [px, py]
    px = np.clip(((pts_xy[:, 0] + grid_range) / voxel_size).astype(int), 0, grid_dim - 1)
    py = np.clip(((grid_range - pts_xy[:, 1]) / voxel_size).astype(int), 0, grid_dim - 1)
    
    # Crea l'array dei punti in coordinate pixel a 32 bit per OpenCV
    pts_pixel = np.column_stack((px, py)).astype(np.int32)
    # Disegna e riempie l'area interna del poligono impostando il valore dei pixel a 1.0
    cv2.fillPoly(mask, [pts_pixel], 1.0)
    # Restituisce la matrice binaria 200x200 contenente il poligono rasterizzato
    return mask


def extract_ground_truth_masks(frame_data, exclude_tokens=None):
    """
    Riceve il dizionario frame_data dal dataset adapter nuScenes ed estrae 
    il tensore target a 6 canali (6, 200, 200) contenente i poligoni d'ingombro 3D reali di ciascun ostacolo.
    Se exclude_tokens è specificato (es. token degli occludori visibili), tali ostacoli vengono esclusi.
    """
    # Inizializza 6 matrici 200x200 vuote a zeri per ciascuna delle 6 classi semantiche
    target_cars = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)         # Canale 0: Auto
    target_trucks_buses = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32) # Canale 1: Camion/Bus/Trailer
    target_pedestrians = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)  # Canale 2: Pedoni/Umani
    target_motorcycles = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)  # Canale 3: Moto (Strada)
    target_bicycles = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)     # Canale 4: Bici (Ciclo-pedonale)
    target_structures = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)   # Canale 5: Barriere/Coni

    exclude_set = set(exclude_tokens) if exclude_tokens is not None else None

    # Cicla su tutte le scatole 3D degli ostacoli nuScenes presenti nel fotogramma corrente
    for box in frame_data.get("boxes", []):
        tok = getattr(box, 'token', '') if hasattr(box, 'token') else (box.get('token', '') if isinstance(box, dict) else '')
        if exclude_set is not None and tok in exclude_set:
            continue

        # Estrazione ed azzeramento del testo per il nome della categoria dell'ostacolo
        if hasattr(box, 'name'):
            cat = box.name.lower()
        elif isinstance(box, dict):
            cat = box.get('name', '').lower()
        else:
            cat = ''

        # Estrazione dei 4 vertici dell'ingombro a terra 2D [X, Y] espressi in metri
        if hasattr(box, 'corners'):
            c = box.corners()
            corners_xy = c[:2, [0, 1, 5, 4]].T # Footprint 2D BEV ordinato
            corners = corners_xy[:, [1, 0]]     # [Y, X] per rasterize_polygon
        elif hasattr(box, 'bottom_corners'):
            c = box.bottom_corners()[:2].T
            corners = c[:, [1, 0]]
        elif isinstance(box, dict):
            c = np.array(box.get('bottom_corners', []))
            corners = c[:, [1, 0]] if len(c) > 0 else []
        else:
            corners = []

        # Se i vertici sono validi (almeno 3 punti), rasterizza il poligono e lo unisce alla matrice di classe corretta
        if len(corners) >= 3:
            mask = rasterize_polygon(corners)
            # Assegnazione al canale corretto in base al nome della classe dell'ostacolo
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

    # Restituisce lo stack delle 6 matrici in un unico tensore binario 3D di forma (6, 200, 200)
    return np.stack([
        target_cars,
        target_trucks_buses,
        target_pedestrians,
        target_motorcycles,
        target_bicycles,
        target_structures
    ], axis=0)


def extract_real_occlusion_ground_truth(boxes, occ_polygon, occluder_tokens=None, min_overlap_area=0.05):
    """
    Estrae il target di Ground Truth Reale nuScenes per una specifica zona d'ombra (occ_polygon).
    Esclude rigorosamente tutti gli ostacoli occludenti (occluder_tokens) che generano l'ombra,
    evitando che l'ostacolo visibile in primo piano venga conteggiato come bersaglio nascosto.
    
    Ritorna:
      gt_6: array np float32 a 6 canali [Auto, Camion/Bus, Pedone, Moto, Bici, Barriere]
      gt_4: array np float32 a 4 macro-classi [Auto, Camion/Bus, VRU (Pedoni/Bici/Moto), Barriere]
    """
    from shapely.geometry import Polygon as ShapelyPoly
    from shapely.geometry.base import BaseGeometry

    gt_6 = np.zeros(6, dtype=np.float32)
    gt_4 = np.zeros(4, dtype=np.float32)

    if occ_polygon is None:
        return gt_6, gt_4

    if isinstance(occ_polygon, BaseGeometry):
        sp_occ = occ_polygon
    else:
        try:
            pts_np = np.array(occ_polygon)
            if len(pts_np) < 3:
                return gt_6, gt_4
            sp_occ = ShapelyPoly(pts_np[:, :2])
        except Exception:
            return gt_6, gt_4

    if not sp_occ.is_valid:
        sp_occ = sp_occ.buffer(0)
    if not sp_occ.is_valid or sp_occ.area < 0.01:
        return gt_6, gt_4

    exclude_set = set(occluder_tokens) if occluder_tokens is not None else set()

    for box in boxes:
        tok = getattr(box, 'token', '') if hasattr(box, 'token') else (box.get('token', '') if isinstance(box, dict) else '')
        if tok and tok in exclude_set:
            continue

        if hasattr(box, 'corners_3d'):
            c = box.corners_3d[:2, [0, 1, 5, 4]].T
        elif hasattr(box, 'corners'):
            c = box.corners()[:2, [0, 1, 5, 4]].T
        elif hasattr(box, 'bottom_corners'):
            c = box.bottom_corners()[:2].T
        elif isinstance(box, dict):
            c = np.array(box.get('bottom_corners', []))[:2].T if 'bottom_corners' in box else []
        else:
            continue

        if len(c) < 3:
            continue

        try:
            bp = ShapelyPoly(c)
            if not bp.is_valid:
                bp = bp.buffer(0)
            if not bp.is_valid or bp.area < 0.01:
                continue

            if sp_occ.intersects(bp):
                inter_area = sp_occ.intersection(bp).area
                if inter_area >= min_overlap_area:
                    name = getattr(box, 'name', '') if hasattr(box, 'name') else box.get('name', '')
                    cat = name.lower()
                    if "car" in cat:
                        gt_6[0] = 1.0
                        gt_4[0] = 1.0
                    elif "truck" in cat or "bus" in cat or "trailer" in cat:
                        gt_6[1] = 1.0
                        gt_4[1] = 1.0
                    elif "human" in cat or "pedestrian" in cat:
                        gt_6[2] = 1.0
                        gt_4[2] = 1.0
                    elif "motorcycle" in cat:
                        gt_6[3] = 1.0
                        gt_4[2] = 1.0
                    elif "bicycle" in cat:
                        gt_6[4] = 1.0
                        gt_4[2] = 1.0
                    else:
                        gt_6[5] = 1.0
                        gt_4[3] = 1.0
        except Exception:
            continue

    return gt_6, gt_4


def extract_real_occlusion_boxes(boxes, occlusions, max_range=25.0):
    """
    Estrae e separa gli oggetti nuScenes tra:
      - occluders: ostacoli visibili al LiDAR che generano coni d'ombra
      - occluded_boxes: ostacoli reali nuScenes che si trovano DENTRO le zone d'ombra (nascosti)
      - occluder_tokens: set di token degli oggetti occludenti
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    occ_polygons = []
    occluder_tokens = set()
    for occ in occlusions:
        pts = occ.get("polygon_points_m", [])
        if len(pts) >= 3:
            try:
                p = ShapelyPolygon(pts)
                if p.is_valid and p.area > 0.05:
                    occ_polygons.append((p, occ))
                    tok = occ.get("object_token", "")
                    if tok:
                        occluder_tokens.add(tok)
            except Exception:
                pass

    occluders = []
    occluded_boxes = []
    for box in boxes:
        dist = np.hypot(box.center[0], box.center[1])
        if dist > max_range:
            continue
        try:
            if hasattr(box, 'corners_3d'):
                c = box.corners_3d[:2, [0, 1, 5, 4]].T
            elif hasattr(box, 'corners'):
                c = box.corners()[:2, [0, 1, 5, 4]].T
            elif hasattr(box, 'bottom_corners'):
                c = box.bottom_corners()[:2].T
            else:
                continue
            box_poly = ShapelyPolygon(c)
            if not box_poly.is_valid:
                box_poly = box_poly.buffer(0)

            tok = getattr(box, 'token', '') if hasattr(box, 'token') else (box.get('token', '') if isinstance(box, dict) else '')
            if tok in occluder_tokens:
                occluders.append(box)
            elif any(op.intersects(box_poly) for op, _ in occ_polygons):
                occluded_boxes.append(box)
        except Exception:
            pass

    return occluders, occluded_boxes, occluder_tokens


def get_occlusion_ground_truth_target(target_tensor, poly_pts):
    """
    Riceve il tensore target a 6 canali (6, 200, 200) ed i punti del poligono d'ombra [y_ahead, x_right].
    Restituisce un array binario 1D di dimensione 6: [Auto, Camion/Bus, Pedone, Moto, Bici, Barriera]
    dove il valore è 1.0 SOLO se un ostacolo reale nuScenes ricade all'interno di quella specifica ombra.
    """
    # Determina il numero di classi dal tensore di target
    num_classes = target_tensor.shape[0] if hasattr(target_tensor, 'shape') else 6
    target_classes = np.zeros(num_classes, dtype=np.float32)
    # Verifica che i punti dell'ombra siano validi
    if poly_pts is None or len(poly_pts) < 3:
        return target_classes
        
    # Rasterizza la maschera binaria 2D della specifica zona d'ombra
    poly_mask = rasterize_polygon(poly_pts) > 0.5
    
    # Interroga ciascuno dei 6 canali GT SOLO ed ESCLUSIVAMENTE sui pixel ricadenti nell'ombra
    if np.any(poly_mask):
        tgt_np = target_tensor.numpy() if hasattr(target_tensor, 'numpy') else target_tensor
        for c in range(num_classes):
            # Se la maschera dell'ombra sovrappone almeno un pixel dell'ostacolo reale, imposta la presenza a 1.0
            if np.any(tgt_np[c][poly_mask] > 0.5):
                target_classes[c] = 1.0
                
    return target_classes


# Blocco principale di autoverifica (Self-Test) se eseguito direttamente
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dataset_adapter.factory_dataset import create_adapter
    # Inizializza l'adapter nuScenes per il test
    adapter = create_adapter("nuscenes", "./nuscenes")
    sample_data = adapter.get_sample_data(0)
    # Estrarre le maschere di Ground Truth dal primo fotogramma
    gt_masks = extract_ground_truth_masks(sample_data)
    print("Self-Test Ground Truth Extractor (6 Canali) Completo!")
    print(f"  GT Masks Shape: {gt_masks.shape}")
    print(f"  Target positivi estratti per classe nel frame 0: {np.sum(gt_masks > 0.5, axis=(1,2))}")
