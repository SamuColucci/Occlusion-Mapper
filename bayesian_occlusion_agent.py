"""
Agente Decisionale Bayesiano per la Stima delle Occlusioni a Runtime.

Questo script implementa un agente probabilistico dinamico che arricchisce i poligoni delle
occlusioni stimate con le probabilità di contenere specifici oggetti (Auto, Pedoni, ecc.).
La logica si articola su tre meccanismi principali:
  1. Suddivisione geometrica dell'occlusione in sotto-zone semantiche tramite HD Map.
  2. Modulazione delle probabilità in base alla larghezza fisica del varco (coefficiente OBB).
  3. Memoria temporale causale (seen instances) basata sui frame passati della scena.
"""
import os
import csv
import json
import glob
import argparse
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# Probabilità di base semantica per superficie:
# Definisce P(Classe | Superficie Pura). Ad esempio, è molto probabile trovare un'auto
# in un parcheggio (other_flat: 0.65) o un pedone sulle strisce pedonali (ped_crossing: 0.60).
# Queste probabilità fungono da prior bayesiano e riflettono le frequenze del mondo reale.
_BASE_PROB = {
    # Auto: alta priorità su strada (0.30) e parcheggio (0.65); bassa su marciapiede (0.03) e prato (0.01)
    "Auto":       {"driveable_surface": 0.30, "sidewalk": 0.03, "terrain": 0.01, "other_flat": 0.65, "ped_crossing": 0.10, "other": 0.10},
    # Pedone: priorità massima su strisce (0.60), marciapiedi (0.35) e prato (0.20); ridotta su asfalto (0.15)
    "Pedone":     {"driveable_surface": 0.15, "sidewalk": 0.35, "terrain": 0.20, "other_flat": 0.15, "ped_crossing": 0.60, "other": 0.25},
    # Camion: presente su strada (0.20) e parcheggi (0.15); quasi nullo sui marciapiedi (0.01)
    "Camion":     {"driveable_surface": 0.20, "sidewalk": 0.01, "terrain": 0.00, "other_flat": 0.15, "ped_crossing": 0.05, "other": 0.05},
    # Bicicletta: moderatamente distribuita tra marciapiedi (0.20), strade (0.15) e strisce (0.25)
    "Bicicletta": {"driveable_surface": 0.15, "sidewalk": 0.20, "terrain": 0.05, "other_flat": 0.10, "ped_crossing": 0.25, "other": 0.10},
    # Moto: priorità simile alla bici ma leggermente più focalizzata sulla carreggiata stradale (0.10)
    "Moto":       {"driveable_surface": 0.10, "sidewalk": 0.02, "terrain": 0.01, "other_flat": 0.15, "ped_crossing": 0.05, "other": 0.05},
    # Bus: tipicamente presente solo su asfalto stradale (0.08); nullo su marciapiedi e terreno
    "Bus":        {"driveable_surface": 0.08, "sidewalk": 0.00, "terrain": 0.00, "other_flat": 0.02, "ped_crossing": 0.01, "other": 0.02},
    # Rimorchio: veicolo lungo e raro, ammesso su strada (0.05) e parcheggi (0.05)
    "Rimorchio":  {"driveable_surface": 0.05, "sidewalk": 0.00, "terrain": 0.00, "other_flat": 0.05, "ped_crossing": 0.00, "other": 0.01},
    # Barriera: ostacolo statico tipico di bordi stradali/marciapiedi (0.15) e terreno (0.20)
    "Barriera":   {"driveable_surface": 0.01, "sidewalk": 0.15, "terrain": 0.20, "other_flat": 0.10, "ped_crossing": 0.02, "other": 0.15},
    # Cono: elemento di cantiere e delimitazione, sparso su marciapiede (0.10), asfalto (0.05) e terreno (0.10)
    "Cono":       {"driveable_surface": 0.05, "sidewalk": 0.10, "terrain": 0.10, "other_flat": 0.10, "ped_crossing": 0.05, "other": 0.10},
    # Altro: categoria residuale per tutti gli oggetti non mappati
    "Altro":      {"driveable_surface": 0.05, "sidewalk": 0.05, "terrain": 0.10, "other_flat": 0.05, "ped_crossing": 0.02, "other": 0.10},
}



def get_width_coeff(occ_width_m, hidden_cat, a_road):
    """
    Coefficiente di inibizione fisica basato sulla larghezza orientata (OBB) del varco.
    
    Se la sotto-zona ricade sulla strada (driveable_surface), limitiamo la probabilità
    di presenza di oggetti voluminosi (Auto, Camion, Bus, Rimorchio) qualora la larghezza 
    dell'occlusione sia fisicamente inferiore all'ingombro del veicolo.
    I pedoni, le barriere e i coni non vengono penalizzati.
    """
    # Se la sotto-zona non ricade sulla carreggiata stradale (es. è un marciapiede o terreno),
    # non applichiamo vincoli fisici di larghezza poiché l'ingombro stradale delle corsie non è influente
    if a_road < 0.1:
        return 1.0
    
    w = occ_width_m
    
    # 1. Varco strettissimo (sotto il metro):
    # Impossibile contenere auto, camion, bus o rimorchi. Consentiti solo pedoni, bici, moto e ostacoli piccoli.
    if w < 1.0:
        table = {
            "Auto": 0.00, "Camion": 0.00, "Pedone": 1.00, "Bicicletta": 0.50, 
            "Moto": 0.50, "Bus": 0.00, "Rimorchio": 0.00, "Barriera": 1.00, "Cono": 1.00, "Altro": 0.20
        }
    # 2. Varco stretto (tra 1.0m e 1.8m):
    # Tipicamente troppo stretto per le auto standard (larghi circa 1.8m). Ammesse auto minuscole (es. Smart, coefficiente 0.05).
    # Esclusi camion, bus e rimorchi pesanti.
    elif w < 1.8:
        table = {
            "Auto": 0.05, "Camion": 0.00, "Pedone": 1.00, "Bicicletta": 1.00, 
            "Moto": 1.00, "Bus": 0.00, "Rimorchio": 0.00, "Barriera": 1.00, "Cono": 1.00, "Altro": 0.30
        }
    # 3. Varco medio (tra 1.8m e 2.5m):
    # Compatibile con auto standard (coefficiente 0.50). Inizia ad ammettere camion leggeri o bus piccoli (0.20).
    elif w < 2.5:
        table = {
            "Auto": 0.50, "Camion": 0.20, "Pedone": 1.00, "Bicicletta": 1.00, 
            "Moto": 1.00, "Bus": 0.20, "Rimorchio": 0.10, "Barriera": 1.00, "Cono": 1.00, "Altro": 0.60
        }
    # 4. Varco largo (tra 2.5m e 4.0m):
    # Perfettamente compatibile con le auto (1.00) e parzialmente con mezzi pesanti larghi (Camion 0.50, Bus 0.80).
    # Riduzione marginale dei pedoni al 0.80 poiché è meno comune vederli stazionare stabilmente al centro di una corsia larga.
    elif w < 4.0:
        table = {
            "Auto": 1.00, "Camion": 0.50, "Pedone": 0.80, "Bicicletta": 1.00, 
            "Moto": 1.00, "Bus": 0.80, "Rimorchio": 0.50, "Barriera": 1.00, "Cono": 1.00, "Altro": 0.80
        }
    # 5. Varco amplissimo (oltre i 4.0m):
    # Nessun vincolo fisico per i veicoli (1.00). La probabilità per i pedoni al centro è ridotta (0.60) a favore dei mezzi.
    else:
        table = {
            "Auto": 1.00, "Camion": 1.00, "Pedone": 0.60, "Bicicletta": 1.00, 
            "Moto": 1.00, "Bus": 1.00, "Rimorchio": 1.00, "Barriera": 1.00, "Cono": 1.00, "Altro": 1.00
        }
    
    return table.get(hidden_cat, 0.5)

class BayesianOcclusionAgent:
    """
    Agente probabilistico che analizza le occlusioni stimate (coni d'ombra LiDAR) 
    e assegna a ciascuna di esse un vettore di probabilità per 10 classi di oggetti stradali.
    """
    def __init__(self):
        print("Inizializzazione Agente Decisionale Bayesiano...")

    def classify_source(self, name):
        """
        Mappa le categorie testuali originali di NuScenes (oltre 23 classi)
        nelle nostre 10 macro-categorie decisionali semplificate.
        """
        name_lower = name.lower()
        if "car" in name_lower: return "Auto"
        if "truck" in name_lower: return "Camion"
        if "bus" in name_lower: return "Bus"
        if "pedestrian" in name_lower or "human" in name_lower: return "Pedone"
        if "bicycle" in name_lower: return "Bicicletta"
        if "motorcycle" in name_lower: return "Moto"
        if "trailer" in name_lower: return "Rimorchio"
        if "barrier" in name_lower: return "Barriera"
        if "trafficcone" in name_lower or "cone" in name_lower: return "Cono"
        return "Altro"

    def process_dataset(self, adapter, input_dir="extracted_occlusions", output_dir="extracted_occlusions_probabilities"):
        """
        Funzione principale di elaborazione batch:
          1. Ordina cronologicamente i campioni NuScenes scena per scena (causalità temporale) via Adapter.
          2. Carica i file delle occlusioni grezze.
          3. Ritaglia le sub-zone semantiche e calcola le probabilità condizionate.
          4. Applica la memoria temporale causale del passato ed esegue il salvataggio in output.
        """
        import math
        
        def box_to_bev_polygon(box):
            """Genera un poligono Shapely 2D BEV a partire da un oggetto 3D Box standard."""
            corners = box.corners()
            bev_corners = corners[:2, [0, 1, 5, 4]].T
            return ShapelyPolygon(bev_corners)

        json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
        csv_rows = []
        if not json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{input_dir}'!")
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Avvio elaborazione del dataset: arricchimento probabilistico di {len(json_files)} file...")
        
        # Mappa di lookup per trovare velocemente il file JSON di ciascun Lidar Token
        token_to_json = {}
        for fpath in json_files:
            with open(fpath, "r") as f:
                data = json.load(f)
            lt = data.get("lidar_token")
            if lt:
                token_to_json[lt] = fpath

        # Raggruppamento cronologico per Scena mediante l'Adapter
        scenes = adapter.get_scene_indices()

        count = 0
        # Ciclo principale scena per scena
        for scene_idx, (scene_token, sample_indices) in enumerate(scenes.items()):
            # Registro storico della memoria temporale: instance_token -> class_name.
            # Traccia quali istanze erano visibili nei frame passati della medesima scena.
            seen_instances = {} 
            
            for frame_idx, idx in enumerate(sample_indices):
                try:
                    frame_data = adapter.get_sample_data(idx)
                    lidar_token = frame_data["lidar_token"]
                except Exception as e:
                    print(f"Errore caricamento dati per sample index {idx}: {e}")
                    continue
                    
                fpath = token_to_json.get(lidar_token)
                if not fpath:
                    continue
                    
                with open(fpath, "r") as f:
                    data = json.load(f)
                    
                # Estrazione delle superfici semantiche locali passate dall'adapter
                try:
                    surfaces = frame_data["semantic_map"]
                    # Convertiamo i vettori 2D estratti della mappa HD in poligoni Shapely validi
                    road_polys = [ShapelyPolygon(pts) for pts in surfaces.get('drivable_area', []) if len(pts) >= 3]
                    side_polys = [ShapelyPolygon(pts) for pts in surfaces.get('walkway', []) if len(pts) >= 3]
                    carpark_polys = [ShapelyPolygon(pts) for pts in surfaces.get('carpark_area', []) if len(pts) >= 3]
                    crosswalk_polys = [ShapelyPolygon(pts) for pts in surfaces.get('ped_crossing', []) if len(pts) >= 3]
                except Exception as e:
                    road_polys, side_polys, carpark_polys, crosswalk_polys = [], [], [], []
                    
                # Estrazione footprint 2D BEV di tutti gli oggetti reali nel frame corrente (per tracciare la memoria)
                real_footprints = []
                for b in frame_data["boxes"]:
                    try:
                        # Genera il poligono 2D rettangolare completo della vettura/pedone
                        fp = box_to_bev_polygon(b)
                        if fp.is_valid and fp.area > 0.05:
                            # b.token in LocalBox è già il persistent instance_token!
                            real_footprints.append((b, fp, self.classify_source(b.name), b.token))
                    except Exception:
                        continue
                
                # Iterazione su tutte le occlusioni LiDAR stimate caricate dal file JSON
                occlusions = data.get("occlusions", [])
                for occ_idx, occ in enumerate(occlusions):
                    poly_pts = np.array(occ.get("polygon_points_m", []))
                    risk = occ.get("risk_score", 0.0)
                    
                    if len(poly_pts) < 3:
                        # Un poligono valido deve avere almeno 3 vertici
                        occ["sub_zones"] = []
                        occ["estimated_probabilities"] = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]}
                        continue

                    try:
                        # Creazione del poligono Shapely in coordinate locali BEV (invertendo X e Y)
                        poly_xy = np.column_stack([poly_pts[:, 1], poly_pts[:, 0]])
                        poly_occ = ShapelyPolygon(poly_xy)
                        if not poly_occ.is_valid:
                            # Risolve eventuali auto-intersezioni dei bordi
                            poly_occ = poly_occ.buffer(0)
                        occ_area = poly_occ.area
                    except Exception:
                        occ["sub_zones"] = []
                        occ["estimated_probabilities"] = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]}
                        continue

                    if occ_area <= 0:
                        occ["sub_zones"] = []
                        occ["estimated_probabilities"] = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]}
                        continue

                    # --- ESTRAZIONE DELLE SUB-ZONE GEOMETRICHE ---
                    # Uniamo i singoli poligoni di ciascun layer per velocizzare il calcolo dell'intersezione
                    from shapely.ops import unary_union
                    road_union = unary_union([p for p in road_polys if p.is_valid]) if road_polys else None
                    side_union = unary_union([p for p in side_polys if p.is_valid]) if side_polys else None
                    carpark_union = unary_union([p for p in carpark_polys if p.is_valid]) if carpark_polys else None
                    crosswalk_union = unary_union([p for p in crosswalk_polys if p.is_valid]) if crosswalk_polys else None

                    def extract_subzone(surf_poly):
                        """Taglia geometricamente l'area dell'occlusione mantenendo solo la porzione che ricade su una data superficie."""
                        if surf_poly is None:
                            return None
                        try:
                            # Esegue l'operazione booleana di intersezione
                            inter = poly_occ.intersection(surf_poly)
                            if inter.is_empty or inter.area < 0.01:
                                return None
                            return inter if inter.is_valid else inter.buffer(0)
                        except Exception:
                            return None

                    # Ritaglio delle 4 superfici HD Map all'interno del cono d'ombra
                    road_zone = extract_subzone(road_union)
                    side_zone = extract_subzone(side_union)
                    carpark_zone = extract_subzone(carpark_union)
                    crosswalk_zone = extract_subzone(crosswalk_union)

                    # Terreno/Prato (Terrain) = Calcolato per differenza escludendo le altre aree note dall'occlusione totale
                    try:
                        remainder = poly_occ
                        for z in [road_zone, side_zone, carpark_zone, crosswalk_zone]:
                            if z:
                                remainder = remainder.difference(z)
                        # Se la parte rimanente è valida e ha area significativa, è terreno
                        terr_zone = remainder if (not remainder.is_empty and remainder.area > 0.01) else None
                    except Exception:
                        terr_zone = None

                    def zone_to_coords(geom):
                        """Esporta i vertici di una sub-zona in formato lista [y, x] per il JSON."""
                        from shapely.geometry import MultiPolygon as MP
                        if isinstance(geom, MP):
                            # Se il ritaglio ha frammentato il poligono, manteniamo solo il frammento principale
                            geom = max(geom.geoms, key=lambda g: g.area)
                        try:
                            xy = np.array(geom.exterior.coords)
                            return [[round(float(p[1]), 3), round(float(p[0]), 3)] for p in xy]
                        except Exception:
                            return []

                    def zone_width(geom):
                        """Calcola la larghezza fisica basata sul rettangolo orientato minimo (OBB)."""
                        try:
                            min_rect = geom.minimum_rotated_rectangle
                            if min_rect.exterior:
                                x_r, y_r = min_rect.exterior.coords.xy
                                d1 = math.sqrt((x_r[0]-x_r[1])**2 + (y_r[0]-y_r[1])**2)
                                d2 = math.sqrt((x_r[1]-x_r[2])**2 + (y_r[1]-y_r[2])**2)
                                return min(d1, d2)
                        except Exception:
                            pass
                        return 0.0

                    def calc_probs(surface_label, w, memory_boost_cats):
                        """
                        Applica la formula di Bayes modulata per una specifica sub-zona semantica.
                        
                        Fasi:
                          1. Seleziona la base probabilistica semantica P_base per ciascun oggetto.
                          2. Applica il coefficiente fisico di ingombro basato sulla larghezza della corsia.
                          3. Se scatta la memoria temporale (seen instances nel passato), sovrascrive
                             la probabilità al 95% fisso (evitando accumuli lineari).
                        """
                        # Identificazione binaria (one-hot) del tipo di suolo della sub-zona
                        a_r = 1.0 if surface_label == "driveable_surface" else 0.0
                        a_s = 1.0 if surface_label == "sidewalk" else 0.0
                        a_cp = 1.0 if surface_label == "other_flat" else 0.0
                        a_cw = 1.0 if surface_label == "ped_crossing" else 0.0
                        a_t = 1.0 if surface_label == "terrain" else 0.0
                        # Categoria residuale 'other' per sicurezza matematica
                        a_o = max(0.0, 1.0 - a_r - a_s - a_cp - a_cw - a_t)
                        
                        probs = {}
                        for cat in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]:
                            # Recupera i prior semantici della classe corrente
                            b = _BASE_PROB[cat]
                            # Interpolazione lineare (media pesata) delle prior semantiche
                            p_base = (a_r*b["driveable_surface"] + a_s*b["sidewalk"] + a_cp*b["other_flat"] 
                                      + a_cw*b["ped_crossing"] + a_t*b["terrain"] + a_o*b["other"])
                            # Recupera il coefficiente di inibizione per l'ingombro fisico del mezzo
                            cw = get_width_coeff(w, cat, a_r)
                            # Calcola la probabilità di base modulata fisicamente
                            p = round(min(p_base * cw, 1.0), 4)
                            
                            # Se l'oggetto era visibile nel passato recente ed è scomparso in questo frame,
                            # applichiamo l'override del boost (bonus temporale condizionato) al 95% fisso
                            if cat in memory_boost_cats:
                                p = round(max(p, 0.95), 4)
                            probs[cat] = p
                        return probs

                    # --- VERIFICA DELLA MEMORIA TEMPORALE CAUSALE ---
                    # Controlliamo se un oggetto visto nei frame passati della stessa scena (cronologia nota)
                    # è ora scomparso (attualmente occluso, visibilità 1 o 2) ed interseca l'occlusione corrente
                    memory_boost_cats = set()
                    # Creiamo un cono d'ombra allargato di 20 cm per compensare tolleranze e rumori dei sensori
                    poly_occ_expanded = poly_occ.buffer(0.2)
                    for real_box, real_fp, real_cat, instance_token in real_footprints:
                        # Ignoriamo l'oggetto caster che sta proiettando questa specifica ombra
                        if real_box.token == occ.get("object_token", ""):
                            continue
                        # Verifichiamo se l'ID dell'istanza appartiene alla cronologia degli oggetti visti visibili
                        if instance_token in seen_instances:
                            # Il boost scatta solo se l'oggetto è attualmente occluso (visibilità 1 o 2)
                            cur_vis = real_box.visibility
                            if cur_vis in ['1', '2']:
                                # Se l'ingombro dell'oggetto interseca l'ombra espansa, attiviamo il boost
                                if poly_occ_expanded.intersects(real_fp):
                                    memory_boost_cats.add(real_cat)

                    # --- POPOLAMENTO DELLE SOTTO-ZONE NEL JSON ---
                    sub_zones = []
                    for label, geom in [("driveable_surface", road_zone), ("sidewalk", side_zone), 
                                        ("other_flat", carpark_zone), ("ped_crossing", crosswalk_zone), 
                                        ("terrain", terr_zone)]:
                        if geom is None:
                            continue
                        w = zone_width(geom)
                        coords = zone_to_coords(geom)
                        if not coords:
                            continue
                        sub_zones.append({
                            "surface": label,
                            "polygon_points_m": coords,
                            "area_m2": round(float(geom.area), 3),
                            "area_fraction": round(min(float(geom.area / occ_area), 1.0), 4),
                            "occlusion_width_m": round(w, 4),
                            "estimated_probabilities": calc_probs(label, w, memory_boost_cats)
                        })

                    occ["sub_zones"] = sub_zones

                    # --- FRAZIONI DI SUPPERFICIE E GEOMETRIA COMPLESSIVA ---
                    occ["occlusion_width_m"] = round(zone_width(poly_occ), 4)
                    occ["road_fraction"] = round(min((road_zone.area / occ_area) if road_zone else 0.0, 1.0), 4)
                    occ["sidewalk_fraction"] = round(min((side_zone.area / occ_area) if side_zone else 0.0, 1.0), 4)
                    occ["terrain_fraction"] = round(min((terr_zone.area / occ_area) if terr_zone else 0.0, 1.0), 4)
                    occ["carpark_fraction"] = round(min((carpark_zone.area / occ_area) if carpark_zone else 0.0, 1.0), 4)
                    occ["crosswalk_fraction"] = round(min((crosswalk_zone.area / occ_area) if crosswalk_zone else 0.0, 1.0), 4)

                    # --- MODULAZIONE DEL RISCHIO ---
                    # Calcola il coefficiente di rischio basato sulle superfici coperte (strada/strisce = alto rischio)
                    a_road = occ["road_fraction"]
                    a_side = occ["sidewalk_fraction"]
                    a_terr = occ["terrain_fraction"]
                    a_carpark = occ["carpark_fraction"]
                    a_crosswalk = occ["crosswalk_fraction"]
                    a_other = max(0.0, 1.0 - a_road - a_side - a_terr - a_carpark - a_crosswalk)
                    # Ponderazione del rischio (Strada: 1.0, Strisce: 0.8, Marciapiede: 0.4, Terreno: 0.15)
                    coeff_risk = a_road * 1.0 + a_side * 0.4 + a_carpark * 0.5 + a_crosswalk * 0.8 + a_terr * 0.15 + a_other * 0.3
                    occ["risk_score"] = round(risk * coeff_risk, 4)

                    # --- CALCOLO DELLE PROBABILITÀ GLOBALI DELLA ZONA D'OMBRA ---
                    # Effettuiamo una media pesata delle probabilità delle sub-zone in base alle rispettive aree
                    if sub_zones:
                        global_probs = {}
                        total_area = sum(sz["area_m2"] for sz in sub_zones)
                        for cat in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]:
                            global_probs[cat] = round(
                                sum(sz["estimated_probabilities"].get(cat, 0.0) * sz["area_m2"]
                                    for sz in sub_zones) / total_area, 4)
                        occ["estimated_probabilities"] = global_probs
                    else:
                        occ["estimated_probabilities"] = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]}
                    
                    # --- DETERMINAZIONE DELLA SINTESI TESTUALE PER LO STORICO ---
                    probs = occ["estimated_probabilities"]
                    best_cat = max(probs, key=probs.get)
                    best_prob = probs[best_cat]
                    
                    if best_prob >= 0.90:
                        sintesi = f"Forte indizio di presenza: {best_cat} ({best_prob*100:.1f}%) in base alla memoria temporale causale."
                    elif best_prob >= 0.20:
                        sintesi = f"Presenza potenziale: {best_cat} ({best_prob*100:.1f}%) stimata sulla superficie semantica di appoggio."
                    else:
                        sintesi = f"Area probabilmente libera o con ostacoli minori ({best_cat} al {best_prob*100:.1f}%)."
                        
                    # Registriamo i dati dell'occlusione corrente per scriverli successivamente nel file CSV storico
                    csv_rows.append({
                        "Scena": f"Scena {scene_idx+1} ({scene_token[:6]})",
                        "Frame_Num": frame_idx + 1,
                        "Lidar_Token": lidar_token,
                        "Occlusion_Index": occ_idx,
                        "Area_m2": round(occ_area, 3),
                        "Distanza_m": occ.get("distance_m", 0.0),
                        "Prob_Auto": probs["Auto"],
                        "Prob_Pedone": probs["Pedone"],
                        "Prob_Camion": probs["Camion"],
                        "Prob_Bicicletta": probs["Bicicletta"],
                        "Prob_Moto": probs["Moto"],
                        "Prob_Bus": probs["Bus"],
                        "Prob_Rimorchio": probs["Rimorchio"],
                        "Prob_Barriera": probs["Barriera"],
                        "Prob_Cono": probs["Cono"],
                        "Prob_Altro": probs["Altro"],
                        "Oggetto_Piu_Probabile": best_cat,
                        "Prob_Massima": best_prob,
                        "Sintesi": sintesi
                    })

                # Se non sono state stimate occlusioni nel frame corrente, inseriamo una riga descrittiva 'N/A' nello storico
                if not occlusions:
                    csv_rows.append({
                        "Scena": f"Scena {scene_idx+1} ({scene_token[:6]})",
                        "Frame_Num": frame_idx + 1,
                        "Lidar_Token": lidar_token,
                        "Occlusion_Index": "N/A",
                        "Area_m2": 0.0,
                        "Distanza_m": 0.0,
                        "Prob_Auto": 0.0,
                        "Prob_Pedone": 0.0,
                        "Prob_Camion": 0.0,
                        "Prob_Bicicletta": 0.0,
                        "Prob_Moto": 0.0,
                        "Prob_Bus": 0.0,
                        "Prob_Rimorchio": 0.0,
                        "Prob_Barriera": 0.0,
                        "Prob_Cono": 0.0,
                        "Prob_Altro": 0.0,
                        "Oggetto_Piu_Probabile": "N/A",
                        "Prob_Massima": 0.0,
                        "Sintesi": "Nessuna occlusione rilevata in questo frame."
                    })
                            
                # --- AGGIORNAMENTO MEMORIA TEMPORALE (CAUSALITÀ) ---
                # Aggiungiamo al registro seen_instances solo gli oggetti chiaramente visibili (visibilità 40-100%, ovvero classi 3 e 4)
                # L'aggiornamento avviene alla fine del frame corrente, rispettando il principio di non-anticipazione del futuro
                for real_box, _, real_cat, instance_token in real_footprints:
                    if instance_token is not None:
                        vis = real_box.visibility
                        if vis in ['3', '4']:
                            seen_instances[instance_token] = real_cat
                            
                # Scrittura del file JSON arricchito con i risultati probabilistici stimate
                out_fpath = os.path.join(output_dir, os.path.basename(fpath))
                with open(out_fpath, "w") as f:
                    json.dump(data, f, indent=4)
                    
                count += 1
                if count % 100 == 0:
                    print(f"  Elaborati {count} file JSON...")
                
        print(f"[SUCCESS] Elaborazione batch completata! {count} file salvati in '{output_dir}'.")

        # --- SCRITTURA DEL FILE CSV STORICO ---
        import csv
        csv_fpath = "report_storico_bayes.csv"
        fieldnames = [
            "Scena", "Frame_Num", "Lidar_Token", "Occlusion_Index", "Area_m2", "Distanza_m",
            "Prob_Auto", "Prob_Pedone", "Prob_Camion", "Prob_Bicicletta", "Prob_Moto",
            "Prob_Bus", "Prob_Rimorchio", "Prob_Barriera", "Prob_Cono", "Prob_Altro",
            "Oggetto_Piu_Probabile", "Prob_Massima", "Sintesi"
        ]
        try:
            with open(csv_fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[SUCCESS] Salvato report storico cumulativo in '{csv_fpath}'.")
        except Exception as e:
            print(f"[ERROR] Impossibile salvare il file CSV storico: {e}")

def main():
    """Entry point principale per l'esecuzione da linea di comando."""
    parser = argparse.ArgumentParser(description="Agente Bayesiano con Modulazione Spaziale Semantica.")
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    parser.add_argument("--in-dir", type=str, default="extracted_occlusions")
    parser.add_argument("--out-dir", type=str, default="extracted_occlusions_probabilities")
    args = parser.parse_args()

    agent = BayesianOcclusionAgent()
    from dataset_adapters import NuScenesAdapter
    adapter = NuScenesAdapter(dataroot="./nuscenes")
    agent.process_dataset(adapter, input_dir=args.in_dir, output_dir=args.out_dir)

if __name__ == "__main__":
    main()
