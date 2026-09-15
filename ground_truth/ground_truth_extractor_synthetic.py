# Modulo di Estrazione e Calcolo della Ground Truth Sintetica Neurosimbolica (ground_truth_extractor_synthetic.py)
#
# Offre due strategie scientifiche distinte per la generazione della Ground Truth Sintetica:
# 1. GEOMETRIC (Spatially-Constrained / Physical-Aware):
#    Adatta gli oggetti al volume effettivo dell'ombra, richiedendo che il footprint 3D
#    rientri per almeno l'85% dentro l'ombra e verificando l'assenza di sovrapposizioni (collision-free).
# 2. SEMANTIC (Semantic-Affordance / Permissive Prior):
#    Assegna il valore 1 a TUTTE le classi teoricamente compatibili con la semantica del suolo
#    (Codice della Strada / Affordance pura), senza limitazioni fisiche di ingombro o incastro.

import os
import sys
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

# Elenco delle 6 classi standard
CLASS_NAMES = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bici", "Barriera"]

# Nomi canonici corrispondenti nel dataset nuScenes
NUSC_NAMES = [
    "vehicle.car",
    "vehicle.truck",
    "human.pedestrian.adult",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "movable_object.barrier"
]

# Dimensioni verosimili 3D degli ostacoli [larghezza, lunghezza, altezza] in metri
CLASS_WLH = {
    0: [1.9, 4.5, 1.5],  # Auto
    1: [2.5, 7.0, 3.2],  # Camion / Bus
    2: [0.6, 0.6, 1.7],  # Pedone
    3: [0.8, 2.1, 1.4],  # Moto
    4: [0.6, 1.7, 1.4],  # Bicicletta
    5: [0.5, 1.8, 1.0],  # Barriera
}


def get_prioritized_classes(road_f=0.0, side_f=0.0, cross_f=0.0, area=0.0):
    """
    Determina la sequenza ordinata di classi ammissibili in base alla semantica del suolo:
    - Su strada: Auto, poi Camion (se area >= 18m²), poi Moto.
    - Su marciapiede o strisce: Pedone, poi Bici.
    - Su terreno/bordo: Barriera, poi Bici, poi Pedone.
    """
    is_walkway = (side_f > 0.1) or (cross_f > 0.1)
    is_road = (road_f > 0.1)

    valid_classes = []
    if is_road:
        valid_classes.append(0)  # Auto
        if area >= 18.0:
            valid_classes.append(1)  # Camion
        valid_classes.append(3)  # Moto
        valid_classes.append(5)  # Barriera (cantieri stradali / coni)
    elif is_walkway:
        valid_classes.extend([2, 4, 5])  # Pedone, Bici, Barriera (cantieri marciapiede)
    else:
        valid_classes.extend([5, 4, 2])  # Barriera, Bici, Pedone
    return valid_classes


def apply_occluder_compatibility_filter(valid_classes, occluder_name=None, occluder_wlh=None, enabled=True):
    """
    Vincolo di Compatibilità Dimensionale tra Occludore e Bersaglio (Occluder Compatibility Constraint):
    1. Se l'occludore è un pedone, ciclista o moto (o sagoma molto stretta w < 0.95m):
       non può nascondere veicoli ingombranti (Auto, Camion) né barriere fisse.
       Ammessi solo altri VRU (Pedoni e Bici).
    2. Se l'occludore è un'auto normale (berlina/city car, altezza tipica 1.4-1.65m, minore della quota LiDAR di 1.84m):
       non può nascondere fisicamente camion pesanti o autobus alti > 3.0m (il fascio laser vi passerebbe sopra).
       Esclude quindi Camion/Bus (classe 1).
    3. Se l'occludore è un mezzo pesante (Truck, Bus, Trailer) o una struttura statica (edificio/muro static.manmade):
       può celare qualsiasi categoria.
    """
    if not enabled:
        return valid_classes

    if occluder_name is None and occluder_wlh is None:
        return valid_classes

    name_l = str(occluder_name).lower() if occluder_name is not None else ""
    occ_h = float(occluder_wlh[2]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None
    occ_w = float(occluder_wlh[0]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None

    is_human_or_bike = any(k in name_l for k in ["human", "pedestrian", "bicycle", "motorcycle"])
    is_narrow = (occ_w is not None and occ_w < 0.95)
    is_heavy_vehicle = any(k in name_l for k in ["truck", "bus", "trailer", "construction"])
    is_manmade = ("static.manmade" in name_l or "building" in name_l or "wall" in name_l)
    is_tall = (occ_h is not None and occ_h >= 1.50) # Calibrato: veicoli >= 1.50m possono celare mezzi commerciali leggeri

    # Caso 1: Occludore piccolo/stretto (Pedone, Ciclista, Moto)
    if is_human_or_bike or is_narrow:
        return [c for c in valid_classes if c in [2, 4]]

    # Caso 2: Occludore è un'auto molto bassa (non SUV, non furgone, non camion, non muro)
    if not (is_heavy_vehicle or is_manmade or is_tall):
        # Esclude Camion/Bus (classe 1)
        return [c for c in valid_classes if c != 1]

    # Caso 3: Occludore grande/alto (Muro, Camion, Bus, Trailer) -> ammette tutto
    return valid_classes


def get_occluder_filter_explanation(occluder_name=None, occluder_wlh=None):
    """
    Restituisce una descrizione concisa e scientifica del vincolo imposto dall'occludore.
    """
    if occluder_name is None and occluder_wlh is None:
        return "Nessun vincolo (occludore non identificato)"

    name_l = str(occluder_name).lower() if occluder_name is not None else ""
    occ_h = float(occluder_wlh[2]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None
    occ_w = float(occluder_wlh[0]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None

    is_human_or_bike = any(k in name_l for k in ["human", "pedestrian", "bicycle", "motorcycle"])
    is_narrow = (occ_w is not None and occ_w < 0.95)
    is_heavy_vehicle = any(k in name_l for k in ["truck", "bus", "trailer", "construction"])
    is_manmade = ("static.manmade" in name_l or "building" in name_l or "wall" in name_l)
    is_tall = (occ_h is not None and occ_h >= 2.5)

    if is_human_or_bike or is_narrow:
        return "Sagoma VRU stretta: ammessi solo Pedoni/Bici (esclusi veicoli e barriere)"
    if not (is_heavy_vehicle or is_manmade or is_tall):
        return "Sagoma Auto standard: Camion/Bus ESCLUSO (quota LiDAR 1.84m ne vedrebbe il tetto)"
    return "Sagoma Alta/Pesante: ammette qualsiasi categoria (può celare tutto)"



# =============================================================================
# 1. APPROCCIO GEOMETRICO (Spatially-Constrained / Physical-Aware)
# =============================================================================
def compute_synthetic_ground_truth_geometric(
    sp,
    road_f=0.0,
    side_f=0.0,
    cross_f=0.0,
    area=None,
    dist=0.0,
    gt_raw_6=None,
    existing_points=None,
    min_dist=3.0,
    sp_sample=None,
    max_attempts=20,
    fit_threshold=0.85,
    occluder_name=None,
    occluder_wlh=None,
    use_occluder_filter=True
):
    """
    Calcola la Ground Truth Sintetica Geometrica (Spatially-Constrained).
    Ogni classe ammissibile viene testata verificando che il footprint orientato entri
    all'85% dentro l'ombra e non collida con altri oggetti posizionati.
    """
    if area is None:
        area = float(sp.area) if (sp is not None and not sp.is_empty) else 0.0

    gt_neuro_6 = np.zeros(6, dtype=np.float32)
    placed_objects = []

    if gt_raw_6 is not None and np.sum(gt_raw_6) > 0:
        gt_neuro_6 = np.array(gt_raw_6, dtype=np.float32)
        return gt_neuro_6, placed_objects

    if sp is None or sp.is_empty or area < 0.1:
        return gt_neuro_6, placed_objects

    valid_classes = get_prioritized_classes(road_f=road_f, side_f=side_f, cross_f=cross_f, area=area)
    valid_classes = apply_occluder_compatibility_filter(
        valid_classes, occluder_name=occluder_name, occluder_wlh=occluder_wlh, enabled=use_occluder_filter
    )

    sample_poly = sp_sample if (sp_sample is not None and not sp_sample.is_empty) else sp
    minx, miny, maxx, maxy = sample_poly.bounds

    ext_pts = list(existing_points) if existing_points is not None else []

    for class_idx in valid_classes:
        wlh = CLASS_WLH.get(class_idx, [1.9, 4.5, 1.5])
        obj_name = NUSC_NAMES[class_idx]
        fits = False
        cur_min_dist = 0.8 if class_idx == 5 else min_dist

        # 1. Primo tentativo: baricentro / representative_point
        can_try_center = (not placed_objects) or (class_idx == 5 and all(sample_poly.representative_point().distance(op[2]) > cur_min_dist for op in placed_objects))
        if can_try_center:
            pt = sample_poly.representative_point()
            cx, cy = float(pt.x), float(pt.y)
            yaw = float(np.arctan2(cy, cx))
            dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_y, -s_y], [s_y, c_y]])
            footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

            dist_ok = all(pt.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > cur_min_dist for op in ext_pts)
            if dist_ok and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                gt_neuro_6[class_idx] = 1.0
                placed_objects.append((class_idx, obj_name, pt, yaw, wlh))
                fits = True

        # 2. Tentativi casuali se il baricentro fallisce o e gia occupato
        if not fits:
            attempts = 0
            cur_max_attempts = max_attempts * 2 if class_idx == 5 else max_attempts
            np.random.seed(int(area * 100) + int(dist * 100) + class_idx)
            while attempts < cur_max_attempts:
                pt_x = np.random.uniform(minx, maxx)
                pt_y = np.random.uniform(miny, maxy)
                pt_test = ShapelyPoint(pt_x, pt_y)
                if sample_poly.contains(pt_test):
                    cx, cy = pt_x, pt_y
                    yaw = float(np.arctan2(cy, cx))
                    dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
                    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                    c_y, s_y = np.cos(yaw), np.sin(yaw)
                    R = np.array([[c_y, -s_y], [s_y, c_y]])
                    footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

                    dist_ok_placed = all(pt_test.distance(op[2]) > cur_min_dist for op in placed_objects)
                    dist_ok_ext = all(pt_test.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > cur_min_dist for op in ext_pts)

                    if dist_ok_placed and dist_ok_ext and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                        gt_neuro_6[class_idx] = 1.0
                        placed_objects.append((class_idx, obj_name, pt_test, yaw, wlh))
                        fits = True
                        break
                attempts += 1

    return gt_neuro_6, placed_objects


# =============================================================================
# 2. APPROCCIO SEMANTICO (Semantic-Affordance / Permissive Prior)
# =============================================================================
def compute_synthetic_ground_truth_semantic(
    sp,
    road_f=0.0,
    side_f=0.0,
    cross_f=0.0,
    area=None,
    dist=0.0,
    gt_raw_6=None,
    sp_sample=None,
    occluder_name=None,
    occluder_wlh=None,
    use_occluder_filter=True
):
    """
    Calcola la Ground Truth Sintetica Semantica (Semantic-Affordance / Permissive Prior).
    Replica esattamente la logica della vecchia baseline:
    - Se ci sono ostacoli reali, li preserva.
    - Se ombra su asfalto con varco minimo: Auto = 1.0; se varco camion: Camion = 1.0.
    - Se marciapiede/strisce (o asfalto stretto): Pedoni = 1.0.
    - Se terreno/bordo incolto: Barriere = 1.0.
    Non esclude oggetti per collisione o sovrapposizione geometrica.
    """
    if area is None:
        area = float(sp.area) if (sp is not None and not sp.is_empty) else 0.0

    gt_neuro_6 = np.zeros(6, dtype=np.float32)
    placed_objects = []

    # 1. Se sono presenti ostacoli reali nuScenes annotati, li preserva
    if gt_raw_6 is not None and np.sum(gt_raw_6) > 0:
        gt_neuro_6 = np.array(gt_raw_6, dtype=np.float32)
        return gt_neuro_6, placed_objects

    if sp is None or sp.is_empty or area < 0.1:
        return gt_neuro_6, placed_objects

    # Calcolo OBB (Minimum Rotated Rectangle) per stimare larghezza e lunghezza dell'ombra
    if sp.is_valid and sp.area > 0.01:
        mrr = sp.minimum_rotated_rectangle
        mrr_coords = np.array(mrr.exterior.coords)[:-1]
        e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
        e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
        obb_w, obb_l = min(e1, e2), max(e1, e2)
    else:
        obb_w, obb_l = 0.2, 0.5

    terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))

    # Regola Auto e Camion su asfalto in varchi ampi
    if road_f >= 0.25 and obb_w >= 1.8 and obb_l >= 3.8 and area >= 8.0:
        gt_neuro_6[0] = 1.0
        if road_f >= 0.30 and obb_w >= 2.4 and obb_l >= 6.5 and area >= 18.0:
            gt_neuro_6[1] = 1.0

    # Regola Pedoni su marciapiedi o strisce pedonali (o su asfalto con varco stretto < 1.8m)
    if (side_f >= 0.15 or cross_f >= 0.08 or (road_f >= 0.25 and obb_w < 1.8)) and obb_w >= 0.4 and area >= 0.6:
        gt_neuro_6[2] = 1.0  # Pedoni

    # Regola Barriere su terreno incolto laterale
    if terr_f >= 0.55 and obb_w >= 1.0 and area >= 3.0 and road_f < 0.15 and side_f < 0.10:
        gt_neuro_6[5] = 1.0  # Barriere

    # Filtro di compatibilità dimensionale dell'occludore
    valid_sem = [c for c in range(6) if gt_neuro_6[c] > 0.5]
    valid_sem = apply_occluder_compatibility_filter(
        valid_sem, occluder_name=occluder_name, occluder_wlh=occluder_wlh, enabled=use_occluder_filter
    )
    for c in range(6):
        if c not in valid_sem:
            gt_neuro_6[c] = 0.0

    # Per il visualizzatore: genera i box per le classi accese
    sample_poly = sp_sample if (sp_sample is not None and not sp_sample.is_empty) else sp
    pt = sample_poly.representative_point()
    cx, cy = float(pt.x), float(pt.y)
    yaw = float(np.arctan2(cy, cx))
    for c_idx in range(6):
        if gt_neuro_6[c_idx] > 0.5:
            wlh = CLASS_WLH.get(c_idx, [1.9, 4.5, 1.5])
            placed_objects.append((c_idx, NUSC_NAMES[c_idx], pt, yaw, wlh))

    return gt_neuro_6, placed_objects


# =============================================================================
# 3. APPROCCIO IBRIDO (Spatio-Semantic / Geometric + Semantic Joint Constraint)
# =============================================================================
def compute_synthetic_ground_truth_hybrid(
    sp,
    road_f=0.0,
    side_f=0.0,
    cross_f=0.0,
    roadside_f=0.0,
    area=None,
    dist=0.0,
    gt_raw_6=None,
    existing_points=None,
    min_dist=3.0,
    sp_sample=None,
    max_attempts=20,
    fit_threshold=0.70,
    occluder_name=None,
    occluder_wlh=None,
    use_occluder_filter=True
):
    """
    Calcola la Ground Truth Sintetica IBRIDA Neuro-Simbolica (Geometrica + Semantica).
    Combina:
    1. Vincoli semantici di affordance stradale con margini permissivi anti-falsi-negativi:
       - Auto: ammessa su asfalto o su marciapiede/terreno adiacente (varco >= 1.55m, area >= 5.0m²).
       - Camion: ammesso su strada o cantiere bordo strada (varco >= 2.1m, area >= 13.5m²).
       - Pedoni e Bici: ammessi su marciapiede, crossing, o margine carreggiata.
       - Barriere e Coni: ammessi lungo strada, cantiere o marciapiede (area >= 0.5m²).
    2. Vincolo volumetrico 3D (Physical Containment permissivo):
       - Solo le classi semanticamente ammesse vengono testate per il fitting geometrico.
       - Il footprint orientato deve entrare per almeno il 70% dentro il poligono dell'ombra.
       - Collision-free rispetto agli altri oggetti già posizionati.
    3. Vincolo di compatibilità dimensionale tra occludore e bersaglio.
    """
    if area is None:
        area = float(sp.area) if (sp is not None and not sp.is_empty) else 0.0

    gt_neuro_6 = np.zeros(6, dtype=np.float32)
    placed_objects = []

    # Affordance veicoli: tiene conto della carreggiata e dell'accosto a bordo strada
    eff_road_car = max(road_f, 0.40 * roadside_f)
    eff_road_truck = max(road_f, 0.35 * roadside_f)

    # 1. Preservazione INCONDIZIONATA degli ostacoli reali nuScenes annotati.
    # Il gating fisico si applica solo alle label sintetiche (zone vuote), mai a oggetti realmente osservati.
    if gt_raw_6 is not None and np.sum(gt_raw_6) > 0:
        return np.array(gt_raw_6, dtype=np.float32), placed_objects

    if sp is None or sp.is_empty or area < 0.1:
        return gt_neuro_6, placed_objects

    # Calcolo OBB per stima larghezza/lunghezza del varco
    if sp.is_valid and sp.area > 0.01:
        mrr = sp.minimum_rotated_rectangle
        mrr_coords = np.array(mrr.exterior.coords)[:-1]
        e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
        e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
        obb_w, obb_l = min(e1, e2), max(e1, e2)
    else:
        obb_w, obb_l = 0.2, 0.5

    terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))

    # Vincoli di ammissibilità semantica congiunta con margini permissivi per sosta/accosto
    valid_classes = []
    # Auto: ammessa su asfalto o accostata al bordo strada entro 2.5m (varco >= 1.60m, area >= 5.5m²)
    if (road_f >= 0.08 or roadside_f >= 0.20) and obb_w >= 1.55 and obb_l >= 3.2 and area >= 5.0:
        valid_classes.append(0)  # Auto
    # Camion: su asfalto o stallo/cantiere a bordo strada entro 2.5m (sagoma coerente da mezzo pesante)
    if (road_f >= 0.12 or roadside_f >= 0.25) and obb_w >= 2.15 and obb_l >= 5.0 and area >= 14.0:
        valid_classes.append(1)  # Camion
    # Moto: su asfalto
    if road_f >= 0.12 and obb_w >= 0.7:
        valid_classes.append(3)  # Moto
    # Pedoni e Ciclisti: su marciapiedi, strisce pedonali, o zone fuori dalla carreggiata principale
    if (side_f >= 0.04 or cross_f >= 0.02 or (road_f < 0.25 and area >= 0.35) or (road_f >= 0.12 and obb_w < 1.6)) and area >= 0.35:
        valid_classes.append(2)  # Pedone
        valid_classes.append(4)  # Bici
    # Barriere da cantiere, jersey e coni: delimitano corsie e lavori su strada, marciapiedi o margini
    # Hanno ingombro compatto e lineare tipico delle transenne/coni
    if obb_w <= 2.4 and area >= 0.5:
        valid_classes.append(5)  # Barriera

    # Filtro di compatibilità dimensionale dell'occludore
    valid_classes = apply_occluder_compatibility_filter(
        valid_classes, occluder_name=occluder_name, occluder_wlh=occluder_wlh, enabled=use_occluder_filter
    )

    if not valid_classes:
        return gt_neuro_6, placed_objects

    sample_poly = sp_sample if (sp_sample is not None and not sp_sample.is_empty) else sp
    minx, miny, maxx, maxy = sample_poly.bounds
    ext_pts = list(existing_points) if existing_points is not None else []

    for class_idx in valid_classes:
        wlh = CLASS_WLH.get(class_idx, [1.9, 4.5, 1.5])
        obj_name = NUSC_NAMES[class_idx]
        fits = False
        cur_min_dist = 0.8 if class_idx == 5 else min_dist

        # Primo tentativo: baricentro / representative_point
        can_try_center = (not placed_objects) or (class_idx == 5 and all(sample_poly.representative_point().distance(op[2]) > cur_min_dist for op in placed_objects))
        if can_try_center:
            pt = sample_poly.representative_point()
            cx, cy = float(pt.x), float(pt.y)
            yaw = float(np.arctan2(cy, cx))
            dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_y, -s_y], [s_y, c_y]])
            footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

            dist_ok = all(pt.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > cur_min_dist for op in ext_pts)
            collides = any(footprint.intersects(op_footprint) for _, _, _, _, _, op_footprint in [(None, None, None, None, None, ShapelyPolygon(np.dot(np.array([[op[4][1]/2, op[4][0]/2], [op[4][1]/2, -op[4][0]/2], [-op[4][1]/2, -op[4][0]/2], [-op[4][1]/2, op[4][0]/2]]), np.array([[np.cos(op[3]), -np.sin(op[3])], [np.sin(op[3]), np.cos(op[3])]]).T) + [op[2].x, op[2].y])) for op in placed_objects])
            if dist_ok and (not collides) and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                gt_neuro_6[class_idx] = 1.0
                placed_objects.append((class_idx, obj_name, pt, yaw, wlh))
                fits = True

        # Tentativi casuali se il baricentro fallisce o è occupato
        if not fits:
            attempts = 0
            cur_max_attempts = max_attempts * 2 if class_idx == 5 else max_attempts
            np.random.seed(int(area * 100) + int(dist * 100) + class_idx)
            while attempts < cur_max_attempts:
                pt_x = np.random.uniform(minx, maxx)
                pt_y = np.random.uniform(miny, maxy)
                pt_test = ShapelyPoint(pt_x, pt_y)
                if sample_poly.contains(pt_test):
                    cx, cy = pt_x, pt_y
                    yaw = float(np.arctan2(cy, cx))
                    dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
                    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                    c_y, s_y = np.cos(yaw), np.sin(yaw)
                    R = np.array([[c_y, -s_y], [s_y, c_y]])
                    footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

                    dist_ok = all(pt_test.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > cur_min_dist for op in ext_pts)
                    collides = any(footprint.intersects(op_footprint) for _, _, _, _, _, op_footprint in [(None, None, None, None, None, ShapelyPolygon(np.dot(np.array([[op[4][1]/2, op[4][0]/2], [op[4][1]/2, -op[4][0]/2], [-op[4][1]/2, -op[4][0]/2], [-op[4][1]/2, op[4][0]/2]]), np.array([[np.cos(op[3]), -np.sin(op[3])], [np.sin(op[3]), np.cos(op[3])]]).T) + [op[2].x, op[2].y])) for op in placed_objects])
                    if dist_ok and (not collides) and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                        gt_neuro_6[class_idx] = 1.0
                        placed_objects.append((class_idx, obj_name, pt_test, yaw, wlh))
                        fits = True
                        break
                attempts += 1

    return gt_neuro_6, placed_objects


# =============================================================================
# FUNZIONE WRAPPER UNIFICATA (Default: mode='geometric')
# =============================================================================
def compute_synthetic_ground_truth(
    sp,
    road_f=0.0,
    side_f=0.0,
    cross_f=0.0,
    roadside_f=0.0,
    area=None,
    dist=0.0,
    gt_raw_6=None,
    existing_points=None,
    min_dist=3.0,
    sp_sample=None,
    max_attempts=20,
    fit_threshold=0.85,
    mode="geometric",
    occluder_name=None,
    occluder_wlh=None,
    use_occluder_filter=True
):
    """
    Punto di accesso unico e standardizzato per il calcolo della Ground Truth Sintetica.
    
    Parametro 'mode':
    - 'geometric' (default): Spatially-constrained con fitting volumetrico 3D.
    - 'semantic': Semantic-affordance permissivo (accende tutti i compatibili).
    - 'hybrid': Ibrido congiunto (Affordance semantica rigorosa con sosta/accosto + fitting 3D).
    """
    if mode == "semantic":
        return compute_synthetic_ground_truth_semantic(
            sp=sp,
            road_f=road_f,
            side_f=side_f,
            cross_f=cross_f,
            area=area,
            dist=dist,
            gt_raw_6=gt_raw_6,
            sp_sample=sp_sample,
            occluder_name=occluder_name,
            occluder_wlh=occluder_wlh,
            use_occluder_filter=use_occluder_filter
        )
    elif mode == "hybrid":
        return compute_synthetic_ground_truth_hybrid(
            sp=sp,
            road_f=road_f,
            side_f=side_f,
            cross_f=cross_f,
            roadside_f=roadside_f,
            area=area,
            dist=dist,
            gt_raw_6=gt_raw_6,
            existing_points=existing_points,
            min_dist=min_dist,
            sp_sample=sp_sample,
            max_attempts=max_attempts,
            fit_threshold=fit_threshold if fit_threshold != 0.85 else 0.80,
            occluder_name=occluder_name,
            occluder_wlh=occluder_wlh,
            use_occluder_filter=use_occluder_filter
        )
    else:
        return compute_synthetic_ground_truth_geometric(
            sp=sp,
            road_f=road_f,
            side_f=side_f,
            cross_f=cross_f,
            area=area,
            dist=dist,
            gt_raw_6=gt_raw_6,
            existing_points=existing_points,
            min_dist=min_dist,
            sp_sample=sp_sample,
            max_attempts=max_attempts,
            fit_threshold=fit_threshold,
            occluder_name=occluder_name,
            occluder_wlh=occluder_wlh,
            use_occluder_filter=use_occluder_filter
        )


def to_macro_classes_4(gt_6):
    """
    Converte un vettore a 6 classi [Auto, Camion, Pedone, Moto, Bici, Barriera]
    nelle 4 macro-classi standard ISO 26262 [Auto, Camion/Bus, VRU, Barriera].
    """
    return np.array([
        gt_6[0],
        gt_6[1],
        max(gt_6[2], gt_6[3], gt_6[4]),
        gt_6[5]
    ], dtype=np.float32)


if __name__ == "__main__":
    # Autoverifica comparativa delle due strategie
    test_poly = ShapelyPolygon([[0, 0], [10, 0], [10, 5], [0, 5]])
    
    gt_geom, placed_geom = compute_synthetic_ground_truth(test_poly, road_f=0.9, area=50.0, dist=10.0, mode="geometric")
    gt_sem, placed_sem = compute_synthetic_ground_truth(test_poly, road_f=0.9, area=50.0, dist=10.0, mode="semantic")

    print("==================================================================")
    print(" CONFRONTO STRATEGIE GROUND TRUTH SINTETICA NEUROSIMBOLICA")
    print("==================================================================")
    print(f"1. Strategia GEOMETRIC (Spatially-Constrained):")
    print(f"   Target 6 classi: {gt_geom} -> Macro 4 classi: {to_macro_classes_4(gt_geom)}")
    print(f"   Oggetti posizionati (collision-free): {len(placed_geom)}")
    for obj in placed_geom:
        print(f"     - {obj[1]} a ({obj[2].x:.2f}, {obj[2].y:.2f})")

    print(f"\n2. Strategia SEMANTIC (Semantic-Affordance / Permissive):")
    print(f"   Target 6 classi: {gt_sem} -> Macro 4 classi: {to_macro_classes_4(gt_sem)}")
    print(f"   Classi ammissibili accese a priori: {len(placed_sem)}")
    for obj in placed_sem:
        print(f"     - {obj[1]}")