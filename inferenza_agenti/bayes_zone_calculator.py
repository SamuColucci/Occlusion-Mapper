# Calcolatore Bayesiano Condizionato per Sotto-Zone Semantiche (bayes_zone_calculator.py)
# Modulo algebrico centrale per il calcolo delle probabilità a priori condizionate:
#   - Mappa le probabilità di base (BASE_PROB) per ciascuna superficie (carreggiata, marciapiede, strisce, ecc.)
#   - Calcola il varco minimo di passaggio in metri (W_sub) tramite l'Oriented Bounding Box (OBB)
#   - Modula la probabilità condizionata in base alle dimensioni del varco ed alla superficie in metri quadri
#   - Inietta il boost temporale al 95% se un ostacolo passato è ora nascosto nella sotto-zona d'ombra.

# Import di numpy per operazioni matriciali, algebriche e calcolo norme di vettori
import numpy as np
# Import di cv2 (OpenCV) per le operazioni di ricerca dei contorni sulle maschere 2D
import cv2
# Import delle primitive geometriche di Shapely per l'intersezione e la fusione di superfici
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# Probabilità a priori di base BASE_PROB raggruppate per tipo di superficie semantica del terreno
BASE_PROB = {
    # Auto: alta priorità su parcheggi/piazzali (0.65) e strada (0.30); quasi nulla su marciapiede e terreno
    "Auto":       {
        "driveable_surface": 0.30, 
        "sidewalk": 0.03, 
        "terrain": 0.01, 
        "other_flat": 0.65, 
        "ped_crossing": 0.10, 
        "other": 0.10
    },
    
    # Pedone: massima probabilità su strisce (0.60), marciapiedi (0.35) e prato (0.20); ridotta su asfalto
    "Pedone":     {
        "driveable_surface": 0.15, 
        "sidewalk": 0.35, 
        "terrain": 0.20, 
        "other_flat": 0.15, 
        "ped_crossing": 0.60, 
        "other": 0.25
    },
    
    # Camion: presente su strada (0.20) e parcheggi (0.15); nullo su marciapiede e terreno
    "Camion":     {
        "driveable_surface": 0.20, 
        "sidewalk": 0.01, 
        "terrain": 0.00, 
        "other_flat": 0.15, 
        "ped_crossing": 0.05, 
        "other": 0.05
    },
    
    # Bicicletta: distribuita tra marciapiedi (0.20), strisce (0.25) e strada (0.15)
    "Bicicletta": {
        "driveable_surface": 0.15, 
        "sidewalk": 0.20, 
        "terrain": 0.05, 
        "other_flat": 0.10, 
        "ped_crossing": 0.25, 
        "other": 0.10
    },
    
    # Moto: focalizzata su parcheggi (0.15) e carreggiata stradale (0.10)
    "Moto":       {
        "driveable_surface": 0.10, 
        "sidewalk": 0.02, 
        "terrain": 0.01, 
        "other_flat": 0.15, 
        "ped_crossing": 0.05, 
        "other": 0.05
    },
    
    # Bus: presente tipicamente solo su strada (0.08); nullo su marciapiedi e terreno
    "Bus":        {
        "driveable_surface": 0.08, 
        "sidewalk": 0.00, 
        "terrain": 0.00, 
        "other_flat": 0.02, 
        "ped_crossing": 0.01, 
        "other": 0.02
    },
    
    # Rimorchio: veicolo ingombrante, limitato a strada (0.05) e parcheggi (0.05)
    "Rimorchio":  {
        "driveable_surface": 0.05, 
        "sidewalk": 0.00, 
        "terrain": 0.00, 
        "other_flat": 0.05, 
        "ped_crossing": 0.00, 
        "other": 0.01
    },
    
    # Barriera: ostacolo statico tipico dei bordi stradali/marciapiedi (0.15) e terreno (0.20)
    "Barriera":   {
        "driveable_surface": 0.01, 
        "sidewalk": 0.15, 
        "terrain": 0.20, 
        "other_flat": 0.10, 
        "ped_crossing": 0.02, 
        "other": 0.15
    },
    
    # Cono: delimitazione stradale/cantiere, sparso su marciapiede (0.10) e strada (0.05)
    "Cono":       {
        "driveable_surface": 0.05, 
        "sidewalk": 0.10, 
        "terrain": 0.10, 
        "other_flat": 0.10, 
        "ped_crossing": 0.05, 
        "other": 0.10
    },
    
    # Altro: categoria residuale
    "Altro":      {
        "driveable_surface": 0.05, 
        "sidewalk": 0.05, 
        "terrain": 0.05, 
        "other_flat": 0.05, 
        "ped_crossing": 0.05, 
        "other": 0.05
    }
}

# Funzione ausiliaria per unificare una lista di poligoni vettoriali o una matrice 2D semantica (200x200) in un unico oggetto geometrico Shapely
# Converte le maschere 2D binarie in metri BEV (-40m .. +40m) e applica buffer(0) per riparare la topologia
def _build_union_(layer_data):
    if layer_data is None:
        return None
        
    # Se layer_data è una matrice 2D numpy (200x200) generata dall'Adapter nusc_map.get_map_mask
    if isinstance(layer_data, np.ndarray) and layer_data.ndim == 2:
        mask_uint8 = (layer_data > 0.5).astype(np.uint8)
        if not np.any(mask_uint8):
            return None
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_polys = []
        cell_size = 0.4  # 80.0m / 200 = 0.4m per pixel
        half_size = 40.0 # 40.0m
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.squeeze(axis=1)
                x_m = (pts[:, 0] * cell_size) - half_size
                y_m = half_size - (pts[:, 1] * cell_size)
                coords = np.column_stack((x_m, y_m))
                try:
                    sp = ShapelyPolygon(coords)
                    if not sp.is_valid:
                        sp = sp.buffer(0)
                    if not sp.is_empty and sp.area > 0.01:
                        valid_polys.append(sp)
                except Exception:
                    pass
        if not valid_polys:
            return None
        return unary_union(valid_polys)
        
    polys_list = layer_data if isinstance(layer_data, (list, np.ndarray)) else []
    if len(polys_list) == 0:
        return None
        
    valid_polys = []
    for p in polys_list:
        if hasattr(p, '__len__') and len(p) >= 3:
            try:
                poly = ShapelyPolygon(p)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    valid_polys.append(poly)
            except Exception:
                continue
    if not valid_polys:
        return None
    return unary_union(valid_polys)

# Calcola la larghezza minima (varco in metri W_sub) di un poligono di sotto-zona d'ombra
# Utilizza il rettangolo minimo orientato (Oriented Bounding Box - OBB) per misurare l'apertura reale
def width_filter(poly):
    if poly is None or poly.is_empty:
        return 0.0
    try:
        # Calcola il rettangolo minimo orientato che racchiude la sotto-zona
        rect = poly.minimum_rotated_rectangle
        if rect.is_empty:
            return 0.0
        coords = np.array(rect.exterior.coords)
        # Calcola la lunghezza dei lati del rettangolo orientato
        edges = [np.linalg.norm(coords[i] - coords[i+1]) for i in range(len(coords)-1)]
        if len(edges) >= 2:
            # La larghezza del varco W_sub è la dimensione del lato minore del rettangolo
            return float(min(edges[0], edges[1]))
        return 0.0
    except Exception:
        return 0.0

# Restituisce il moltiplicatore di inibizione/sblocco fisico in base all'ampiezza del varco W_sub
def get_width_coeff(w, category, is_road):
    # 1. Varco stretto (< 1.0m): impedisce l'accesso ai veicoli (auto, camion, bus)
    if w < 1.0:
        table = {
            "Auto": 0.00, "Camion": 0.00, "Bus": 0.00, "Rimorchio": 0.00,
            "Pedone": 1.00, "Bicicletta": 0.50, "Moto": 0.50, 
            "Barriera": 1.00, "Cono": 1.00, "Altro": 0.20
        }
    # 2. Varco (1.0m - 1.8m): la probabilità di trovare auto aumenta leggermente
    elif w < 1.8:
        table = {
            "Auto": 0.05, "Camion": 0.00, "Bus": 0.00, "Rimorchio": 0.00,
            "Pedone": 1.00, "Bicicletta": 1.00, "Moto": 1.00, 
            "Barriera": 1.00, "Cono": 1.00, "Altro": 0.30
        }
    # 3. Varco (1.8m - 2.5m): la probabilità di trovare auto aumenta al 50%
    elif w < 2.5:
        table = {
            "Auto": 0.50, "Camion": 0.20, "Bus": 0.20, "Rimorchio": 0.10,
            "Pedone": 1.00, "Bicicletta": 1.00, "Moto": 1.00, 
            "Barriera": 1.00, "Cono": 1.00, "Altro": 0.60
        }
    # 4. Varco (2.5m - 4.0m): la probabilità di trovare auto e camion aumenta
    elif w < 4.0:
        table = {
            "Auto": 1.00, "Camion": 0.50, "Bus": 0.80, "Rimorchio": 0.50,
            "Pedone": 0.80, "Bicicletta": 1.00, "Moto": 1.00, 
            "Barriera": 1.00, "Cono": 1.00, "Altro": 0.80
        }
    # 5. Varco (>= 4.0m): sblocco completo per tutte le categorie
    else:
        table = {
            "Auto": 1.00, "Camion": 1.00, "Bus": 1.00, "Rimorchio": 1.00,
            "Pedone": 0.60, "Bicicletta": 1.00, "Moto": 1.00, 
            "Barriera": 1.00, "Cono": 1.00, "Altro": 1.00
        }
        
    return table.get(category, 1.0)

# Calcola le probabilità condizionate Bayesiane per ciascuna sotto-zona d'ombra
def conditional_probablity_occlusion_zone(occ, semantic_map, currently_occluded_past_categories=None):
    poly_pts = occ.get("polygon_points_m", [])
    if len(poly_pts) < 3:
        occ["sub_zones"] = []
        occ["estimated_probabilities"] = {k: 0.0 for k in BASE_PROB}
        occ["risk_score"] = 0.0
        occ["boosted_categories"] = []
        occ["road_fraction"] = 0.0
        occ["sidewalk_fraction"] = 0.0
        occ["crosswalk_fraction"] = 0.0
        occ["carpark_fraction"] = 0.0
        occ["terrain_fraction"] = 0.0
        return occ
    
    try: 
        pts_arr = np.array(poly_pts)
        pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
        poly = ShapelyPolygon(pts_xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        occ["sub_zones"] = []
        occ["estimated_probabilities"] = {k: 0.0 for k in BASE_PROB}
        occ["risk_score"] = 0.0
        occ["boosted_categories"] = []
        occ["road_fraction"] = 0.0
        occ["sidewalk_fraction"] = 0.0
        occ["crosswalk_fraction"] = 0.0
        occ["carpark_fraction"] = 0.0
        occ["terrain_fraction"] = 0.0
        return occ

    occ_area = float(poly.area)
    if occ_area <= 0:
        occ["sub_zones"] = []
        occ["estimated_probabilities"] = {k: 0.0 for k in BASE_PROB}
        occ["risk_score"] = 0.0
        occ["boosted_categories"] = []
        occ["road_fraction"] = 0.0
        occ["sidewalk_fraction"] = 0.0
        occ["crosswalk_fraction"] = 0.0
        occ["carpark_fraction"] = 0.0
        occ["terrain_fraction"] = 0.0
        return occ

    # Priorità delle superfici per evitare sovrapposizioni (> 100%):
    # 1. Strisce Pedonali (ped_crossing)
    # 2. Marciapiede (sidewalk)
    # 3. Parcheggio (other_flat)
    # 4. Strada (driveable_surface)
    layers_ordered = [
        ("ped_crossing", _build_union_(semantic_map.get("ped_crossing", []))),
        ("sidewalk", _build_union_(semantic_map.get("walkway", []))),
        ("other_flat", _build_union_(semantic_map.get("carpark_area", []))),
        ("driveable_surface", _build_union_(semantic_map.get("drivable_area", [])))
    ]

    sub_zones = []
    remaining_poly = poly
    accumulated_sub_area = 0.0
    boosted_categories = set()

    fractions = {
        "ped_crossing": 0.0,
        "sidewalk": 0.0,
        "other_flat": 0.0,
        "driveable_surface": 0.0,
        "terrain": 0.0
    }

    # Ciclo di estrazione sequenziale delle sotto-zone seguendo l'ordine di priorità stabilito
    for label, layer_geom in layers_ordered:
        if layer_geom is None or layer_geom.is_empty or remaining_poly.is_empty:
            continue
            
        try:
            # Calcoliamo l'intersezione tra la porzione di ombra ancora non assegnata ed il layer semantico corrente
            inter = remaining_poly.intersection(layer_geom)
            if inter.is_empty:
                continue
            sub_area = float(inter.area)
            if sub_area < 0.001:
                continue
            
            # Accumuliamo l'area di sotto-zona trovata e ne salviamo la frazione rispetto all'area dell'ombra totale
            accumulated_sub_area += sub_area
            fractions[label] = sub_area / occ_area
            
            # Misuriamo l'ampiezza reale del varco in metri (W_sub) per questa specifica sotto-zona
            w_sub = width_filter(inter)
            is_road = (label == "driveable_surface" or label == "ped_crossing")

            # Calcolo delle probabilità condizionate Bayesiane per ciascuna delle 10 classi di ostacolo
            probs = {}
            for category in BASE_PROB:
                # Probabilità a priori di base per questo tipo di superficie (dal dizionario BASE_PROB)
                p_base = BASE_PROB[category].get(label, 0.10)
                # Coefficiente di inibizione/sblocco fisico in base alla larghezza del varco W_sub
                k_w = get_width_coeff(w_sub, category, is_road)
                
                # Scalamento dinamico in base all'Area Reale della Sotto-Zona (sub_area in m²):
                # Aumenta la capacità di nascondere veicoli SOLO se l'area effettivo di asfalto è > 10 m²
                k_area = 1.0
                if is_road and category in ["Auto", "Camion", "Bus"]:
                    if sub_area > 10.0:
                        k_area = min(1.0 + (sub_area - 10.0) / 20.0, 2.5)
                        
                # Probabilità congiunta limitata ad un massimo di 0.95 (95%)
                p = round(min(p_base * k_w * k_area, 0.95), 4)

                # Gestione della Memoria Temporale Persistente:
                # Se un oggetto di questa categoria era visibile nel passato ed ORA è occluso/nascosto nell'ombra,
                # la sua probabilità viene elevata al 95% (0.95)
                if currently_occluded_past_categories and category in currently_occluded_past_categories:
                    p = max(p, 0.95)
                    boosted_categories.add(category)

                probs[category] = p

            # Rimuoviamo la porzione d'ombra appena assegnata mediante differenza booleana (difference)
            # Questo garantisce che i layer successivi prendano solo lo spazio residuo (evitando sovrapposizioni > 100%)
            remaining_poly = remaining_poly.difference(inter)

            # Registriamo la sotto-zona arricchita con area, varco in metri e probabilità per classe
            sub_zones.append({
                "surface": label,
                "area_m2": round(sub_area, 3),
                "area_fraction": round(min(sub_area / occ_area, 1.0), 4),
                "occlusion_width_m": round(w_sub, 3),
                "estimated_probabilities": probs
            })
        except Exception:
            continue

    # Calcolo della superficie residua d'ombra (Terreno / Prato fuori dalla Mappa HD principale)
    rem_area = max(0.0, occ_area - accumulated_sub_area)
    if rem_area > 0.01:
        fractions["terrain"] = rem_area / occ_area
        w_sub = width_filter(remaining_poly if not remaining_poly.is_empty else poly)
        probs = {}
        for category in BASE_PROB:
            p_base = BASE_PROB[category].get("terrain", 0.10)
            k_w = get_width_coeff(w_sub, category, False)
            p = round(min(p_base * k_w, 1.0), 4)

            # Applicazione del Boost 95% anche sulla sotto-zona residua di terreno
            if currently_occluded_past_categories and category in currently_occluded_past_categories:
                p = max(p, 0.95)
                boosted_categories.add(category)

            probs[category] = p

        sub_zones.append({
            "surface": "terrain",
            "area_m2": round(rem_area, 3),
            "area_fraction": round(min(rem_area / occ_area, 1.0), 4),
            "occlusion_width_m": round(w_sub, 3),
            "estimated_probabilities": probs
        })

    occ["sub_zones"] = sub_zones
    occ["boosted_categories"] = list(boosted_categories)

    # Salvataggio delle frazioni di area disgiunte nel dizionario occ per la resa visuale dell'HUD
    occ["road_fraction"] = round(fractions["driveable_surface"], 4)
    occ["sidewalk_fraction"] = round(fractions["sidewalk"], 4)
    occ["crosswalk_fraction"] = round(fractions["ped_crossing"], 4)
    occ["carpark_fraction"] = round(fractions["other_flat"], 4)
    occ["terrain_fraction"] = round(fractions["terrain"], 4)

    # Aggregazione Max-Pooling di Sicurezza Automotrice:
    # La probabilità globale di un pericolo nell'ombra è pari al MAX riscontrato nelle sotto-zone
    if sub_zones:
        global_probs = {}
        for category in BASE_PROB:
            global_probs[category] = round(
                max(sz["estimated_probabilities"][category] for sz in sub_zones), 4
            )
        occ["estimated_probabilities"] = global_probs
    else:
        occ["estimated_probabilities"] = {k: 0.0 for k in BASE_PROB}

    # Calcolo del Risk Score complessivo pesato sul tipo di superficie
    a_road = occ["road_fraction"]
    a_side = occ["sidewalk_fraction"]
    a_cross = occ["crosswalk_fraction"]
    a_carp = occ["carpark_fraction"]
    a_terr = occ["terrain_fraction"]
    
    risk_coeff = (a_road * 1.0) + (a_cross * 0.8) + (a_carp * 0.5) + (a_side * 0.4) + (a_terr * 0.15)
    max_prob = max(occ["estimated_probabilities"].values()) if occ["estimated_probabilities"] else 0.0
    occ["risk_score"] = round(max_prob * risk_coeff, 4)

    return occ
