# Script per analizzare i file Json ottenuti durante l'estrazione delle zone occluse (inferenza_agenti/bayes_agent.py).
# Ora le analizziamo per calcolare le probabilità condizionate di trovare una determinata categoria di ostacoli 
# in base al tipo di superficie semantica del terreno ed alla memoria storica degli oggetti passati.

# Import dei moduli di sistema per la manipolazione dei percorsi e file
import os
import sys
# Import di numpy per calcoli algebrici ed operazioni vettoriali
import numpy as np
# Import di json per la lettura e deserializzazione dei file di predizione
import json
# Import delle primitive geometriche di Shapely per i controlli punto-in-poligono
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

# Aggiunge la cartella radice del progetto al sys.path per consentire l'importazione dei moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'adapter di fabbrica per la gestione agnostica del dataset
from dataset_adapter.factory_dataset import create_adapter
# Import del RayCaster per il tracciamento dei raggi
from raycaster.ray_caster import RayCaster
# Import della funzione di calcolo Bayesiano condizionato per la zona d'ombra
from inferenza_agenti.bayes_zone_calculator import conditional_probablity_occlusion_zone

# Metodo per il mapping tra i nomi delle categorie usati nel dataset e quelli usati nel resto del codice
def map_category_name(raw_name):
    # Converte il testo della categoria in lettere minuscole per rendere la ricerca agnostica dal maiuscolo/minuscolo
    name_lower = raw_name.lower()
    if "car" in name_lower or "vehicle" in name_lower:
        return "Auto"
    elif "human" in name_lower or "pedestrian" in name_lower:
        return "Pedone"
    elif "truck" in name_lower:
        return "Camion"
    elif "bicycle" in name_lower:
        return "Bicicletta"
    elif "motorcycle" in name_lower:
        return "Moto"
    elif "bus" in name_lower:
        return "Bus"
    elif "trailer" in name_lower:
        return "Rimorchio"
    elif "barrier" in name_lower:
        return "Barriera"
    elif "cone" in name_lower:
        return "Cono"
    else:
        return "Altro"

# Sfruttiamo l'adapter per rendere il codice indipendente dal dataset utilizzato
def main():
    # Scelta del dataset e del percorso
    dataset_name = "nuscenes"
    dataroot = "./nuscenes"
    print(f"Caricamento adattatore dataset: {dataset_name}...")
    # Istanzia l'adapter nuScenes
    adapter = create_adapter(dataset_name, dataroot)
    
    # Estraiamo gli indici dei frame raggruppati per scena
    scene_dict = adapter.get_scene_indices()
    # Creazione della directory di output per i JSON con le probabilità
    out_dir = "extracted_occlusions_probabilities"
    os.makedirs(out_dir, exist_ok=True)

    # Eseguiamo l'analisi per ogni scena
    print("Inizio estrazione zone occluse e calcolo delle probabilità...")
    boost_records = []

    # Scorre tutte le scene del dataset
    for scene_token, frame_indices in scene_dict.items():
        print(f"\n--- Elaborazione Scena: {scene_token} ({len(frame_indices)} frame) ---")
        
        # Reset temporaneo delle istanze viste per ogni scena (memoria temporale)
        seen_instances = {} 

        # Scansione cronologica dei frame della scena
        for idx in frame_indices:
            frame_data = adapter.get_sample_data(idx)
            token = frame_data["sample_token"]

            # Leggiamo il file JSON contenente le zone occluse estratte dal RayCaster
            json_path = os.path.join("extracted_occlusions", f"{token}.json")
            if not os.path.exists(json_path):
                print(f"  [WARN] File {json_path} non trovato, salto.")
                continue
            with open(json_path, "r") as f:
                raw_data = json.load(f)
                raw_occlusions = raw_data.get("occlusions", [])

            # Mappiamo per ogni specifica ombra quali categorie occluse viste nel passato ricadono DENTRO di essa
            occ_boost_map = {i: set() for i in range(len(raw_occlusions))}

            # Scorro tutti gli oggetti del frame corrente
            for box in frame_data["boxes"]:
                # Se l'oggetto è stato visto nel passato ed ORA è occluso (visibilità 1)
                if box.token in seen_instances and str(box.visibility) in ['1']:
                    cat_name = seen_instances[box.token]
                    box_pt = ShapelyPoint(box.center[0], box.center[1])
                    
                    # Verifichiamo QUALE SPECIFICA OMBRA contiene fisicamente la posizione dell'oggetto
                    for i, occ in enumerate(raw_occlusions):
                        pts = occ.get("polygon_points_m", [])
                        if len(pts) >= 3:
                            try:
                                occ_poly = ShapelyPolygon(pts)
                                if not occ_poly.is_valid:
                                    occ_poly = occ_poly.buffer(0)
                                # Se il centro dell'oggetto (o l'ombra) contiene il punto dell'oggetto
                                if occ_poly.contains(box_pt) or occ_poly.distance(box_pt) < 1.0:
                                    occ_boost_map[i].add(cat_name)
                                    break
                            except Exception:
                                pass

            # Calcoliamo le probabilità condizionate Bayesiane per ciascuna zona d'ombra
            enriched_occlusions = []
            frame_boosted_categories = set()

            for i, occ in enumerate(raw_occlusions):
                # Inviamo solo le categorie occluse che ricadono ESCLUSIVAMENTE DENTRO questa specifica ombra!
                specific_boost_cats = occ_boost_map.get(i, set())
                
                # Invocazione della funzione di calcolo Bayesiano condizionato per la zona d'ombra
                occ_prob = conditional_probablity_occlusion_zone(
                    occ, 
                    frame_data["semantic_map"], 
                    specific_boost_cats
                )
                enriched_occlusions.append(occ_prob)
                if occ_prob.get("boosted_categories"):
                    frame_boosted_categories.update(occ_prob["boosted_categories"])

            # Salviamo il file JSON contenente le zone d'ombra probabilistiche arricchite
            out_path = os.path.join(out_dir, f"occlusion_sample_{idx:04d}_{token}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "lidar_token": frame_data['lidar_token'],
                    "sample_token": token,
                    "scene_token": scene_token,
                    "occlusions": enriched_occlusions
                }, f, indent=4)

            # Aggiorniamo la memoria temporale (seen_instances) registrando gli oggetti ORA visibili (visibilità 2, 3, 4)
            # MA SOLO SE si trovano in prossimità (< 8.0 metri) di una zona d'ombra (potenziali futuri occlusi)
            for box in frame_data["boxes"]:
                if str(box.visibility) in ['2','3', '4']:
                    is_near_occlusion = False
                    for occ in raw_occlusions:
                        pts = occ.get("polygon_points_m", [])
                        if pts:
                            occ_center = np.mean(pts, axis=0)
                            if np.linalg.norm(box.center[:2] - occ_center) < 8.0:
                                is_near_occlusion = True
                                break
                    
                    if is_near_occlusion:
                        seen_instances[box.token] = map_category_name(box.name)
                    else:
                        seen_instances.pop(box.token, None)

            # Stampa di log a schermo: evidenzia i fotogrammi con Boost 95% per facilitare il collaudo visivo
            if frame_boosted_categories:
                boost_str = ", ".join(sorted(list(frame_boosted_categories)))
                boost_records.append({
                    "frame_idx": idx + 1,
                    "sample_token": token,
                    "categories": boost_str
                })
                print(f"  [BOOST 95%] Frame {idx+1}/{adapter.get_num_samples()} (Sample: {token[:12]}...) | Categorie Occluse nell'Ombra: {boost_str}")
            else:
                print(f"  Frame {idx+1}/{adapter.get_num_samples()} -> Salvato: {out_path}")
            
    # RECAP FINALE STAMPATO A SCHERMO
    print("\n" + "=" * 70)
    print("      RECAP FINALE: FRAME CON BOOST TEMPORALE 95% (MEMORIA ISTANZE)")
    print("=" * 70)
    print(f" Totale fotogrammi in cui un ostacolo passato è ora nascosto nell'ombra: {len(boost_records)}")
    print(" " + "-" * 66)
    
    if boost_records:
        for rec in boost_records:
            print(f"  • Frame {rec['frame_idx']:<4} | Sample: {rec['sample_token'][:16]}... | Categorie Occluse: {rec['categories']}")
    else:
        print("  Nessun caso di boost 95% riscontrato nel dataset.")

    print("=" * 70 + "\n")

# Blocco principale di esecuzione se avviato da riga di comando
if __name__ == "__main__":
    main()