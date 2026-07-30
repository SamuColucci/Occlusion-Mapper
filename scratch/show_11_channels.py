"""
Script per visualizzare a schermo gli 11 Canali Semantici di Input del Dataset Neurale BEV
(inclusa la classificazione semantica punto per punto del LiDAR: Terreno, Auto, Pedoni, Muri).
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
    print("\n--- VISUALIZZATORE DEGLI INPUT SEMANTICI (11 CANALI BEV CON LIDAR CLASSIFICATO) ---")
    dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot="./nuscenes")
    
    # Selezioniamo il primo fotogramma (idx=0)
    sample_idx = 0
    input_tensor, target_tensor = dataset[sample_idx]
    
    input_maps = input_tensor.numpy()     # Shape: (11, 200, 200)
    target_map = target_tensor.numpy()[0] # Shape: (200, 200)

    channel_titles = [
        "Ch 0: LiDAR Terreno/Asfalto",
        "Ch 1: LiDAR Auto Visibili",
        "Ch 2: LiDAR Pedoni Visibili",
        "Ch 3: LiDAR Edifici/Muri",
        "Ch 4: Ombre RayCaster",
        "Ch 5: Asfalto (Drivable)",
        "Ch 6: Marciapiedi (Walkway)",
        "Ch 7: Strisce Pedonali",
        "Ch 8: Sagome Auto Visibili",
        "Ch 9: Sagome Pedoni Visibili",
        "Ch 10: Sagome Barriere/Muri"
    ]

    dark_cmaps = [
        make_dark_cmap('#00f0ff'), # 0: LiDAR Terreno (Azzurro)
        make_dark_cmap('#b55fe6'), # 1: LiDAR Auto (Viola)
        make_dark_cmap('#ff3333'), # 2: LiDAR Pedoni (Rosso)
        make_dark_cmap('#00ccff'), # 3: LiDAR Edifici (Azzurro Chiaro)
        make_dark_cmap('#ff0055'), # 4: Ombre RayCaster (Rosso acceso)
        make_dark_cmap('#7a8b9e'), # 5: Asfalto (Grigio)
        make_dark_cmap('#00ff66'), # 6: Marciapiedi (Verde smeraldo)
        make_dark_cmap('#ffea00'), # 7: Strisce Pedonali (Giallo)
        make_dark_cmap('#b55fe6'), # 8: Sagome Auto (Viola)
        make_dark_cmap('#ff3333'), # 9: Sagome Pedoni (Rosso)
        make_dark_cmap('#00ccff')  # 10: Sagome Barriere (Azzurro)
    ]

    # Grid 3 righe x 4 colonne per contenere tutti gli 11 canali
    fig, axes = plt.subplots(3, 4, figsize=(19, 11), facecolor='#0b0e14')
    fig.canvas.manager.set_window_title("Dataset Neurale BEV - 11 Canali Semantici")
    fig.suptitle(f"ISPEZIONE DEGLI 11 CANALI SEMANTICI BEV (Con LiDAR Classificato) | Sample #{sample_idx+1}", 
                 color='white', fontsize=13, fontweight='bold')

    axes = axes.flatten()

    for idx in range(11):
        ax = axes[idx]
        ax.set_facecolor('#05070a')
        
        ax.imshow(input_maps[idx], cmap=dark_cmaps[idx], extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
        
        px_count = int(input_maps[idx].sum())
        ax.set_title(f"{channel_titles[idx]}\n[{px_count} px attivi]", color='white', fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel("Y (m)", color='#8a99ad', fontsize=7)
        ax.set_ylabel("X (m)", color='#8a99ad', fontsize=7)
        ax.tick_params(colors='white', labelsize=7)
        ax.grid(True, color='#1e222d', linestyle='--', alpha=0.3)

    # Disabilitiamo il 12° subplot (vuoto)
    axes[11].axis('off')

    plt.tight_layout()

    # Seconda finestra per la Target Ground Truth
    fig_gt, ax_gt = plt.subplots(figsize=(7, 7), facecolor='#0b0e14')
    fig_gt.canvas.manager.set_window_title("Target Ground Truth (Ostacoli Occlusi Reali)")
    ax_gt.set_facecolor('#05070a')
    
    ax_gt.imshow(input_maps[4], cmap=make_dark_cmap('#333344'), alpha=0.5, extent=[-40, 40, -40, 40], origin='lower')
    ax_gt.imshow(target_map, cmap=make_dark_cmap('#ff0044'), alpha=0.9, extent=[-40, 40, -40, 40], origin='lower', vmin=0.0, vmax=1.0)
    
    ax_gt.set_title(f"TARGET GROUND TRUTH | Ostacoli Occlusi Reali (Visibilità '1'): {int(target_map.sum())} px", 
                    color='white', fontsize=11, fontweight='bold')
    ax_gt.set_xlabel("Y Laterale (m)", color='white')
    ax_gt.set_ylabel("X Longitudinale (m)", color='white')
    ax_gt.tick_params(colors='white')
    ax_gt.grid(True, color='#1e222d', linestyle='--', alpha=0.4)

    print("Visualizzazione 11 Canali creata con successo! Apertura grafici...")
    plt.show()

if __name__ == "__main__":
    main()
