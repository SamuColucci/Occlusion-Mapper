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
    elif is_walkway:
        valid_classes.extend([2, 4])  # Pedone, Bici
    else:
        valid_classes.extend([5, 4, 2])  # Barriera, Bici, Pedone
    return valid_classes


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
    fit_threshold=0.85
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

    sample_poly = sp_sample if (sp_sample is not None and not sp_sample.is_empty) else sp
    minx, miny, maxx, maxy = sample_poly.bounds

    ext_pts = list(existing_points) if existing_points is not None else []

    for class_idx in valid_classes:
        wlh = CLASS_WLH.get(class_idx, [1.9, 4.5, 1.5])
        obj_name = NUSC_NAMES[class_idx]
        fits = False

        # 1. Primo tentativo: baricentro / representative_point
        if not placed_objects:
            pt = sample_poly.representative_point()
            cx, cy = float(pt.x), float(pt.y)
            yaw = float(np.arctan2(cy, cx))
            dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_y, -s_y], [s_y, c_y]])
            footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

            dist_ok = all(pt.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > min_dist for op in ext_pts)
            if dist_ok and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                gt_neuro_6[class_idx] = 1.0
                placed_objects.append((class_idx, obj_name, pt, yaw, wlh))
                fits = True

        # 2. Tentativi casuali se il baricentro fallisce o e gia occupato
        if not fits:
            attempts = 0
            np.random.seed(int(area * 100) + int(dist * 100) + class_idx)
            while attempts < max_attempts:
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

                    dist_ok_placed = all(pt_test.distance(op[2]) > min_dist for op in placed_objects)
                    dist_ok_ext = all(pt_test.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > min_dist for op in ext_pts)

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
    sp_sample=None
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
    fit_threshold=0.70
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
    """
    if area is None:
        area = float(sp.area) if (sp is not None and not sp.is_empty) else 0.0

    gt_neuro_6 = np.zeros(6, dtype=np.float32)
    placed_objects = []

    # Affordance veicoli: tiene conto della carreggiata e dell'accosto a bordo strada
    eff_road_car = max(road_f, 0.40 * roadside_f)
    eff_road_truck = max(road_f, 0.35 * roadside_f)

    # 1. Preservazione degli ostacoli reali nuScenes annotati
    # Mantiene TUTTI gli ostacoli reali annotati a meno che non siano in cortili totalmente isolati
    if gt_raw_6 is not None and np.sum(gt_raw_6) > 0:
        gt_neuro_6 = np.array(gt_raw_6, dtype=np.float32)
        # Se la zona è completamente sperduta (niente asfalto e lontana più di 5m dalla strada), inibisce
        if eff_road_car < 0.04 and roadside_f < 0.05:
            gt_neuro_6[0] = 0.0
        if eff_road_truck < 0.04 and roadside_f < 0.05:
            gt_neuro_6[1] = 0.0
        if np.sum(gt_neuro_6) > 0:
            return gt_neuro_6, placed_objects

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

    if not valid_classes:
        return gt_neuro_6, placed_objects

    sample_poly = sp_sample if (sp_sample is not None and not sp_sample.is_empty) else sp
    minx, miny, maxx, maxy = sample_poly.bounds
    ext_pts = list(existing_points) if existing_points is not None else []

    for class_idx in valid_classes:
        wlh = CLASS_WLH.get(class_idx, [1.9, 4.5, 1.5])
        obj_name = NUSC_NAMES[class_idx]
        fits = False

        # Primo tentativo: baricentro / representative_point
        if not placed_objects:
            pt = sample_poly.representative_point()
            cx, cy = float(pt.x), float(pt.y)
            yaw = float(np.arctan2(cy, cx))
            dx, dy = wlh[1] / 2.0, wlh[0] / 2.0
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_y, -s_y], [s_y, c_y]])
            footprint = ShapelyPolygon(np.dot(corners, R.T) + [cx, cy])

            dist_ok = all(pt.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > min_dist for op in ext_pts)
            if dist_ok and sp.intersection(footprint).area >= footprint.area * fit_threshold:
                gt_neuro_6[class_idx] = 1.0
                placed_objects.append((class_idx, obj_name, pt, yaw, wlh))
                fits = True

        # Tentativi casuali se il baricentro fallisce o è occupato
        if not fits:
            attempts = 0
            np.random.seed(int(area * 100) + int(dist * 100) + class_idx)
            while attempts < max_attempts:
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

                    dist_ok_placed = all(pt_test.distance(op[2]) > min_dist for op in placed_objects)
                    dist_ok_ext = all(pt_test.distance(op if isinstance(op, ShapelyPoint) else ShapelyPoint(op[0], op[1])) > min_dist for op in ext_pts)

                    if dist_ok_placed and dist_ok_ext and sp.intersection(footprint).area >= footprint.area * fit_threshold:
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
    mode="geometric"
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
            sp_sample=sp_sample
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
            fit_threshold=fit_threshold if fit_threshold != 0.85 else 0.80
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
            fit_threshold=fit_threshold
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