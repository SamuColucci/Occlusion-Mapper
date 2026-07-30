"""
Script di utilità isolato e temporaneo per visualizzare i frame originali NuScenes.
Questo script mostra le immagini delle telecamere e la proiezione 3D del LiDAR
usando i metodi di rendering nativi dell'SDK.
"""
import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

def main():
    print("Inizializzazione NuScenes...")
    # Carica il dataset NuScenes mini situato nella cartella corrente
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=True)
    
    # Elaboriamo tutti i sample presenti nel dataset
    for sample_idx in range(len(nusc.sample)):
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']
        
        print(f"\n============================================================")
        print(f"Visualizzazione Frame #{sample_idx}")
        print(f"Sample Token: {sample_token}")
        print(f"============================================================")
        
        lidar_token = sample['data']['LIDAR_TOP']
        print(f"Sto renderizzando i dati del sensore LIDAR_TOP (Token: {lidar_token})...")
        print("Premi la 'X' in alto a destra sulla finestra o premi 'Q' per passare al frame successivo.")
        
        # Questo metodo nativo mostra la mappa LiDAR con i bounding box 3D annotati
        nusc.render_sample_data(
            lidar_token,
            with_anns=True,  # Mostra le annotazioni (bounding box)
            underlay_map=True, # Mostra la mappa sotto i punti
            use_flat_vehicle_coordinates=True # Allinea ai classici assi del veicolo (Ego)
        )
        plt.title(f"NuScenes - Dati LiDAR BEV - Sample #{sample_idx}", color='black', fontsize=12, fontweight='bold')
        
        # Mostra i grafici a schermo e blocca fino a che l'utente non chiude la finestra
        plt.show()

if __name__ == "__main__":
    main()
