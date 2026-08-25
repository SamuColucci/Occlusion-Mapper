# Script Principale di Estrazione Batch delle Zone Occluse LiDAR (estrazione_zone_occluse.py)
# Utilizza il RayCaster 3D per calcolare le ombre geometriche 2D BEV su tutti i fotogrammi nuScenes
# e salva le coordinate dei poligoni estratti nella cartella extracted_occlusions/ in formato JSON.

# Import dei moduli di sistema per la manipolazione dei percorsi
import os
import sys
# Import di json per la serializzazione e salvataggio dei poligoni d'ombra su disco
import json

# Inserisce la directory radice del progetto al primo posto in sys.path per consentire l'importazione dei pacchetti interni
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import della factory per la creazione dell'adattatore del dataset nuScenes
from dataset_adapter.factory_dataset import create_adapter
# Import del modulo principale RayCaster per la simulazione del fascio di raggi LiDAR
from raycaster.ray_caster import RayCaster

def main():
    # Scelta del dataset target da elaborare
    dataset_name = "nuscenes"
    # Definizione del percorso relativo della cartella dei dati grezzi nuScenes
    dataroot = "./nuscenes"

    print(f"Caricamento adattatore: {dataset_name}...")
    # Istanzia l'adattatore del dataset nuScenes tramite il Factory Pattern
    adapter = create_adapter(dataset_name, dataroot)
    # Recupera il numero totale di fotogrammi (sample) presenti nel dataset
    num_samples = adapter.get_num_samples()
    
    # Creazione della directory di output per il salvataggio dei JSON se non esiste già
    os.makedirs("extracted_occlusions", exist_ok=True)
    
    # Iterazione sequenziale su tutti i fotogrammi del dataset nuScenes
    for idx in range(num_samples):
        # Recupero del dizionario dei dati del fotogramma corrente dall'adattatore
        frame_data = adapter.get_sample_data(idx)
        # Estrazione del token univoco del fotogramma nuScenes
        token = frame_data["sample_token"]
        
        print(f"\n--- Elaborazione frame {idx+1}/{num_samples}: {token} ---")
        
        # Istanzia l'oggetto RayCaster passandogli i dati del fotogramma corrente
        caster = RayCaster(frame_data)
        # Generazione della maschera di occlusione binaria in coordinate polar-grid
        caster.get_occlusion_mask()
        # Estrazione dei poligoni d'ombra semplificati in metri rispetto al veicolo
        occlusions = caster.extract_polygons()
        
        # Salvataggio dei poligoni e del token LiDAR nel file JSON di output
        out_path = os.path.join("extracted_occlusions", f"{token}.json")
        with open(out_path, "w") as f:
            json.dump({"lidar_token": frame_data['lidar_token'], "occlusions": occlusions}, f, indent=4)
        
        print(f"Salvataggio completato ({len(occlusions)} poligoni): {out_path}")

# Blocco principale di esecuzione se il file viene lanciato direttamente da riga di comando
if __name__ == "__main__":
    main()