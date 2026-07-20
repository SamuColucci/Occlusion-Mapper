import sys
import os
import numpy as np

# Aggiungiamo la cartella corrente al path per l'importazione
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from factory_dataset import create_adapter
from occ3d_occlusion_explorer import SOTARayCaster
from ray_caster import RayCaster

def main():
    print("Inizializzazione del dataset NuScenes tramite l'Adapter...")
    # Usiamo il dataset mini che abbiamo usato in tutte le prove
    adapter = create_adapter("nuscenes", "./nuscenes")

    # Prendiamo il primo frame (sample) disponibile
    frame_data = adapter.get_sample_data(0)
    print(f"\nScaricato frame con {len(frame_data['points'])} punti LiDAR.")

    print("\n[1] Eseguo il VECCHIO algoritmo (SOTARayCaster)...")
    old_rc = SOTARayCaster(frame_data, verbose=False)
    old_grid = old_rc.generate_known_zone()
    
    print("[2] Eseguo il NUOVO algoritmo (RayCaster)...")
    new_rc = RayCaster(frame_data)
    new_grid = new_rc.generate_known_zone()

    print("--- RISULTATI DEL CONFRONTO ---")
    print(f"Dimensioni vecchia griglia: {old_grid.shape}")
    print(f"Dimensioni nuova griglia:   {new_grid.shape}")
    
    # Confronto bit a bit
    are_equal = np.array_equal(old_grid, new_grid)
    
    if are_equal:
        print("[SUCCESSO] Le due griglie sono IDENTICHE al 100%.")
    else:
        print("[ERRORE] CI SONO DELLE DIFFERENZE.")
        diff_count = np.sum(old_grid != new_grid)
        total_voxels = old_grid.size
        print(f"Differenze trovate: {diff_count} cubetti su {total_voxels} ({diff_count/total_voxels*100:.4f}%)")

        # 1. SALVATAGGIO IN JSON DELLE DIFFERENZE
        import json
        print("\nSalvataggio delle coordinate differenti in un file JSON...")
        diff_indices = np.where(old_grid != new_grid)
        diff_data = {
            "num_differenze": int(diff_count),
            "coordinate_differenti": [
                {"x": int(x), "y": int(y), "z": int(z)} 
                for x, y, z in zip(*diff_indices)
            ][:100]  # Salviamo solo le prime 100 per non far esplodere il file
        }
        with open("scratch/differenze_zone_note.json", "w") as f:
            json.dump(diff_data, f, indent=4)
        print("Salvato in 'scratch/differenze_zone_note.json'.")

        # 2. VISUALIZZATORE (Generazione di un'immagine delle differenze)
        import matplotlib.pyplot as plt
        print("Generazione dell'immagine di confronto (Bird's Eye View)...")
        
        # Schiacciamo il 3D in 2D prendendo il massimo lungo l'asse Z
        old_bev = np.max(old_grid, axis=2)
        new_bev = np.max(new_grid, axis=2)
        diff_bev = np.abs(old_bev - new_bev)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(old_bev, cmap='gray')
        axs[0].set_title('Vecchio (Occ3D)')
        
        axs[1].imshow(new_bev, cmap='gray')
        axs[1].set_title('Nuovo (RayCaster)')
        
        axs[2].imshow(diff_bev, cmap='hot')
        axs[2].set_title('Differenze (Rosso/Giallo = Errore)')

        # Salviamo l'immagine direttamente nella cartella degli artefatti per vederla qui in chat
        plot_path = r"C:\Users\samue\.gemini\antigravity-ide\brain\f6309c69-f5ea-4d36-98f2-8eb808743982\diff_plot.png"
        plt.savefig(plot_path)
        print(f"Immagine salvata in {plot_path}")

if __name__ == "__main__":
    main()
