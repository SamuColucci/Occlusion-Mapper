"""
Script per visualizzare a schermo i 10 Canali Semantici di Input del Dataset Neurale BEV
con SFONDO NERO UNIFORME su tutti i subplots per la massima nitidezza visiva.
"""
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_generator_neural import OcclusionDatasetNeural

def make_dark_cmap(hex_color):
    """
    Crea una Colormap personalizzata dove:
    0.0 -> Nero profondo (#05070a)
    1.0 -> Colore brillante specificato
    """
    return ListedColormap(['#05070a', hex_color])

def main():
    print("\n--- VISUALIZZATORE DEGLI INPUT SEMANTICI (10 CANALI SFONDO NERO) ---")
    dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot="./nuscenes")
    
    # Selezioniamo il primo fotogramma (idx=0)
    sample_idx = 0
    input_tensor, target_tensor = dataset[sample_idx]
    
    input_maps = input_tensor.numpy()     # Shape: (10, 200, 200)
    target_map = target_tensor.numpy()[0] # Shape: (200, 200)

    channel_titles = [
        "Ch 0: LiDAR 2D",
        "Ch 1: Ombre RayCaster",
        "Ch 2: Asfalto (Drivable)",
        "Ch 3: Marciapiedi (Walkway)",
        "Ch 4: Strisce Pedonali",
        "Ch 5: Auto Visibili (Vis 2,3,4)",
        "Ch 6: Camion/Bus Visibili",
        "Ch 7: Moto/Bici Visibili",
        "Ch 8: Pedoni Visibili",
        "Ch 9: Barriere/Strutture Visibili"
    ]

    # Colori brillanti distinti per ciascun canale con sfondo sempre NERO
    dark_cmaps = [
        make_dark_cmap('#00f0ff'), # 0: LiDAR (Azzurro elettrico)
        make_dark_cmap('#ff0055'), # 1: Ombre RayCaster (Rosso/Rosa acceso)
        make_dark_cmap('#7a8b9e'), # 2: Asfalto (Grigio medio)
        make_dark_cmap('#00ff66'), # 3: Marciapiedi (Verde smeraldo)
        make_dark_cmap('#ffea00'), # 4: Strisce Pedonali (Giallo acceso)
        make_dark_cmap('#b55fe6'), # 5: Auto Visibili (Viola)
        make_dark_cmap('#3399ff'), # 6: Camion/Bus Visibili (Blu)
        make_dark_cmap('#ff9900'), # 7: Moto/Bici Visibili (Arancione)
        make_dark_cmap('#ff3333'), # 8: Pedoni Visibili (Rosso)
        make_dark_cmap('#00ccff')  # 9: Barriere/Strutture (Azzurro)
    ]

    # Figura principale con 2 righe e 5 colonne per i 10 canali d'ingresso
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5), facecolor='#0b0e14')
    fig.canvas.manager.set_window_title("Dataset Neurale BEV - Ispezione 10 Canali Semantici")
    fig.suptitle(f"ISPEZIONE DEI 10 CANALI SEMANTICI BEV (200x200) | Sample #{sample_idx+1}", 
                 color='white', fontsize=14, fontweight='bold')

    axes = axes.flatten()

    for idx in range(10):
        ax = axes[idx]
        ax.set_facecolor('#05070a')
        
        # Rendering con sfondo garantito nero 0.0
        im = ax.imshow(input_maps[idx], cmap=dark_cmaps[idx], extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
        
        px_count = int(input_maps[idx].sum())
        ax.set_title(f"{channel_titles[idx]}\n[{px_count} px attivi]", color='white', fontsize=9, fontweight='bold', pad=6)
        ax.set_xlabel("Y (m)", color='#8a99ad', fontsize=8)
        ax.set_ylabel("X (m)", color='#8a99ad', fontsize=8)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, color='#1e222d', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Seconda finestra per il Target Ground Truth
    fig_gt, ax_gt = plt.subplots(figsize=(7, 7), facecolor='#0b0e14')
    fig_gt.canvas.manager.set_window_title("Target Ground Truth (Ostacoli Occlusi Reali)")
    ax_gt.set_facecolor('#05070a')
    
    # Mostriamo lo sfondo dell'ombra in grigio scuro e sovrapponiamo i pixel rossi dell'ostacolo reale
    ax_gt.imshow(input_maps[1], cmap=make_dark_cmap('#333344'), alpha=0.5, extent=[-40, 40, -40, 40], origin='lower')
    ax_gt.imshow(target_map, cmap=make_dark_cmap('#ff0044'), alpha=0.9, extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
    
    ax_gt.set_title(f"TARGET GROUND TRUTH | Ostacoli Occlusi Reali (Visibilità '1'): {int(target_map.sum())} px", 
                    color='white', fontsize=11, fontweight='bold')
    ax_gt.set_xlabel("Y Laterale (m)", color='white')
    ax_gt.set_ylabel("X Longitudinale (m)", color='white')
    ax_gt.tick_params(colors='white')
    ax_gt.grid(True, color='#1e222d', linestyle='--', alpha=0.4)

    print("Visualizzazione creata con successo! Apertura finestre grafiche...")
    plt.show()

if __name__ == "__main__":
    main()
