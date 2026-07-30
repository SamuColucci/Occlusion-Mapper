"""
Script per visualizzare a schermo i 6 Canali di Input del Dataset Neurale BEV
più il canale Target Ground Truth per un fotogramma a scelta.
"""
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_generator_neural import OcclusionDatasetNeural

def main():
    print("\n--- VISUALIZZATORE DEGLI INPUT FISICI (6 CANALI BEV) ---")
    dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot="./nuscenes")
    
    # Selezioniamo il primo fotogramma (idx=0)
    sample_idx = 0
    input_tensor, target_tensor = dataset[sample_idx]
    
    # Convertiamo i tensor PyTorch in matrici NumPy
    input_maps = input_tensor.numpy()   # Shape: (6, 200, 200)
    target_map = target_tensor.numpy()[0] # Shape: (200, 200)

    channel_titles = [
        "Canale 0: Punti LiDAR (Laser)",
        "Canale 1: Zone d'Ombra (RayCaster)",
        "Canale 2: Asfalto Strada (Drivable Area)",
        "Canale 3: Marciapiedi (Walkway)",
        "Canale 4: Area Parcheggio (Carpark)",
        "Canale 5: Strisce Pedonali (Ped Crossing)"
    ]

    cmaps = ['cool', 'hot', 'gray', 'Greens', 'Purples', 'YlOrRd']

    # Creiamo la figura Matplotlib con 2 righe e 3 colonne per i 6 canali d'ingresso
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#0b0e14')
    fig.suptitle(f"ISPEZIONE DEI 6 CANALI D'INGRESSO BEV (200x200) | Sample #{sample_idx+1}", 
                 color='white', fontsize=14, fontweight='bold')

    axes = axes.flatten()

    for idx in range(6):
        ax = axes[idx]
        ax.set_facecolor('#05070a')
        
        # vmin e vmax fissati a 0.0 e 1.0 per evitare che le matrici vuote vengano colorate completamente di bianco
        im = ax.imshow(input_maps[idx], cmap=cmaps[idx], extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
        
        px_count = int(input_maps[idx].sum())
        ax.set_title(f"{channel_titles[idx]}\n[{px_count} px attivi]", color='white', fontsize=9, fontweight='bold', pad=8)
        ax.set_xlabel("Y Laterale (m)", color='#8a99ad', fontsize=8)
        ax.set_ylabel("X Longitudinale (m)", color='#8a99ad', fontsize=8)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, color='#1e222d', linestyle='--', alpha=0.4)

    plt.tight_layout()

    # Creiamo una seconda finestra per il Target Ground Truth
    fig_gt, ax_gt = plt.subplots(figsize=(7, 7), facecolor='#0b0e14')
    fig_gt.canvas.manager.set_window_title("Target Ground Truth (Ostacoli Occlusi Reali)")
    ax_gt.set_facecolor('#05070a')
    
    # Mostriamo lo sfondo dell'ombra e sovrapponiamo i pixel rossi dell'ostacolo reale
    ax_gt.imshow(input_maps[1], cmap='gray', alpha=0.3, extent=[-40, 40, -40, 40], origin='lower')
    ax_gt.imshow(target_map, cmap='Reds', alpha=0.8, extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
    
    ax_gt.set_title(f"TARGET GROUND TRUTH | Ostacoli Occlusi Reali: {int(target_map.sum())} px", 
                    color='white', fontsize=12, fontweight='bold')
    ax_gt.set_xlabel("Y Laterale (m)", color='white')
    ax_gt.set_ylabel("X Longitudinale (m)", color='white')
    ax_gt.tick_params(colors='white')
    ax_gt.grid(True, color='#1e222d', linestyle='--', alpha=0.4)

    print("Visualizzazione creata con successo! Apertura finestre grafiche...")
    plt.show()

if __name__ == "__main__":
    main()
