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
    
    # Scegliamo un campione di interesse (es. il primo campione del dataset)
    sample_idx = 1
    if sample_idx >= len(nusc.sample):
        sample_idx = 0
        
    sample = nusc.sample[sample_idx]
    sample_token = sample['token']
    
    print(f"\n============================================================")
    print(f"Visualizzazione Frame #{sample_idx}")
    print(f"Sample Token: {sample_token}")
    print(f"============================================================")
    print("1. Chiusura del grafico precedente per mostrare il successivo.")
    print("2. Sto generando i rendering nativi...")
    
    # -------------------------------------------------------------------------
    # TEST 1: Rendering completo del campione (Multi-sensore a 360°)
    # -------------------------------------------------------------------------
    # Questo metodo nativo apre una figura con 6 telecamere e i radar/LiDAR proiettati.
    nusc.render_sample(sample_token)
    plt.title(f"NuScenes - Sample #{sample_idx} (Panoramica 360°)", color='black', fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # TEST 2: Nuvola LiDAR proiettata sulla Telecamera Frontale
    # -------------------------------------------------------------------------
    # Recuperiamo i token dei singoli canali
    cam_front_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Disegna la nuvola di punti LiDAR proiettata sopra la fotografia della CAM_FRONT
    plt.figure(figsize=(12, 7))
    nusc.render_pointcloud_in_image(
        sample_token,
        pointsensor_channel='LIDAR_TOP',
        camera_channel='CAM_FRONT',
        render_intensity=True,  # Mostra l'intensità di riflessione dei punti
        show_lidarseg=False
    )
    plt.title(f"Proiezione LiDAR (LIDAR_TOP) su Fotocamera Frontale (CAM_FRONT) - Sample #{sample_idx}", fontsize=10, fontweight='bold')
    
    # Mostra i grafici a schermo
    plt.show()

if __name__ == "__main__":
    main()
