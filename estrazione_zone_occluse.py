import os
import json
from factory_dataset import create_adapter
from ray_caster import RayCaster

def main():
    # Scelta del dataset
    dataset_name = "nuscenes"
    # Definizione del path del dataset
    dataroot = "./nuscenes"

    print(f"Caricamento adattatore: {dataset_name}...")
    # Creazione dell'adattatore
    adapter = create_adapter(dataset_name, dataroot)
    num_samples = adapter.get_num_samples()
    
    # Creazione della directory di output
    os.makedirs("extracted_occlusions", exist_ok=True)
    # Iterazione su tutti i frame del dataset
    for idx in range(num_samples):
        # Recupero dei dati del frame
        frame_data = adapter.get_sample_data(idx)
        # Estrazione del token del frame
        token = frame_data["sample_token"]
        
        print(f"\n--- Elaborazione frame {idx+1}/{num_samples}: {token} ---")
        
        # Esecuzione del raycasting
        caster = RayCaster(frame_data)
        # Generazione della maschera di occlusione
        caster.get_occlusion_mask()
        # Estrazione dei poligoni di occlusione
        occlusions = caster.extract_polygons()
        
        # Salvataggio dei risultati
        out_path = os.path.join("extracted_occlusions", f"{token}.json")
        with open(out_path, "w") as f:
            json.dump({"lidar_token": frame_data['lidar_token'], "occlusions": occlusions}, f, indent=4)
        
        print(f"Salvataggio completato ({len(occlusions)} poligoni): {out_path}")

if __name__ == "__main__":
    main()