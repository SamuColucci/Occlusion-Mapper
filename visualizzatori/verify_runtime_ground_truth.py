# Visualizzatore Interattivo Ground Truth Reale vs Sintetica (verify_runtime_ground_truth.py)
# Mostra contemporaneamente ed in modo interattivo la Ground Truth Reale nuScenes 3D GT 
# e la Ground Truth Sintetica Inserita (Stress Test ad alta densità con 3.102 ostacoli).

import sys
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, get_occlusion_ground_truth_target
from ground_truth.ground_truth_extractor_synthetic import generate_synthetic_injected_gt, OBSTACLE_SPECS_M

CLASS_NAMES = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bicicletta", "Barriera"]
CLASS_COLORS = {
    "Auto": "#3B82F6",       # Blu
    "Camion/Bus": "#EC4899", # Rosa
    "Pedone": "#10B981",     # Verde Neon
    "Moto": "#F59E0B",       # Giallo/Arancio
    "Bicicletta": "#8B5CF6", # Viola
    "Barriera": "#64748B"    # Grigio
}

class GroundTruthVisualizer:
    def __init__(self, dataroot="./nuscenes"):
        print("\n" + "=" * 70)
        print("   INIZIALIZZAZIONE VISUALIZZATORE GROUND TRUTH (REALE VS SINTETICA)")
        print("=" * 70)
        print("Caricamento dataset nuScenes...")
        self.adapter = create_adapter("nuscenes", dataroot)
        self.nusc = self.adapter.nusc
        self.total_samples = self.adapter.get_num_samples()
        self.current_idx = 0
        
        # Configurazione Figure Matplotlib a 2 Pannelli Mappa + 1 HUD
        self.fig = plt.figure(figsize=(16, 9), facecolor='#0F172A')
        
        # Layout: GridSpec (HUD a Sinistra, GT Reale al Centro, GT Sintetica a Destra)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 2, 2], wspace=0.12)
        self.ax_hud = self.fig.add_subplot(gs[0])
        self.ax_real = self.fig.add_subplot(gs[1])
        self.ax_syn = self.fig.add_subplot(gs[2])
        
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)
        
        self.ax_hud.set_facecolor('#0F172A')
        self.ax_real.set_facecolor('#0F172A')
        self.ax_syn.set_facecolor('#0F172A')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI REGISTRATI (GROUND TRUTH)")
        print("=" * 50)
        print(" -> FRECCIA DX / FRECCIA SU  : Fotogramma Successivo")
        print(" -> FRECCIA SX / FRECCIA GIÙ : Fotogramma Precedente")
        print(" -> TASTO 'R'                : Rigenera Ostacoli Sintetici Casualita")
        print(" -> TASTO 'Q' / ESC          : Esci dal Visualizzatore")
        print("=" * 50 + "\n")
        
        self.plot_current()
        plt.show()

    def on_key(self, event):
        if event.key in ['right', 'up']:
            self.current_idx = (self.current_idx + 1) % self.total_samples
            self.plot_current()
        elif event.key in ['left', 'down']:
            self.current_idx = (self.current_idx - 1) % self.total_samples
            self.plot_current()
        elif event.key in ['r', 'R']:
            self.plot_current()
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def draw_gt_masks(self, ax, gt_masks, title_str, is_synthetic=False):
        ax.clear()
        ax.set_facecolor('#0B0C10')
        ax.axis('off')
        
        grid_dim = 200
        grid_range = 40.0
        voxel_size = 0.4
        
        # 1. RENDER HD MAP (Strada, Marciapiede, Strisce, Parcheggi)
        sem_map = self.sample_data.get('semantic_map', {})
        if sem_map:
            # Mask Drivable Area (Strada)
            if 'drivable_area' in sem_map and np.any(sem_map['drivable_area']):
                y_s, x_s = np.where(sem_map['drivable_area'])
                ax.scatter((x_s - 100) * 0.4, (y_s - 100) * 0.4, c='#1E293B', s=2, alpha=0.4, zorder=1)
            # Mask Walkway (Marciapiede)
            if 'walkway' in sem_map and np.any(sem_map['walkway']):
                y_w, x_w = np.where(sem_map['walkway'])
                ax.scatter((x_w - 100) * 0.4, (y_w - 100) * 0.4, c='#334155', s=2, alpha=0.5, zorder=2)
            # Mask Ped Crossing (Strisce)
            if 'ped_crossing' in sem_map and np.any(sem_map['ped_crossing']):
                y_p, x_p = np.where(sem_map['ped_crossing'])
                ax.scatter((x_p - 100) * 0.4, (y_p - 100) * 0.4, c='#0288D1', s=2, alpha=0.6, zorder=2)

        # 2. RENDER LIDAR POINT CLOUD
        pts = self.sample_data.get('points')
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c='#64748B', s=0.8, alpha=0.25, zorder=3)

        # 3. RENDER 3D BOUNDING BOXES REALI
        boxes = self.sample_data.get('boxes', [])
        for box in boxes:
            corners = box.corners_3d[:2, :]  # 2D corners (x, y)
            corners_closed = np.hstack([corners, corners[:, :1]])
            ax.plot(corners_closed[0, :], corners_closed[1, :], color='#00E5FF', linewidth=1.0, alpha=0.7, zorder=4)

        # 4. OVERLAY MASCHERE GROUND TRUTH (REALE O SINTETICA)
        for c in range(6):
            c_mask = gt_masks[c] > 0.5
            if np.any(c_mask):
                c_name = CLASS_NAMES[c]
                c_color = CLASS_COLORS[c_name]
                
                y_idx, x_idx = np.where(c_mask)
                x_m = (x_idx - grid_dim / 2.0) * voxel_size
                y_m = (y_idx - grid_dim / 2.0) * voxel_size
                
                ax.scatter(x_m, y_m, c=c_color, s=8, alpha=0.85, zorder=6, label=c_name)

        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_title(title_str, color='#F8FAFC', fontsize=11, fontweight='bold', pad=10)

    def plot_current(self):
        self.sample_data = self.adapter.get_sample_data(self.current_idx)
        scene_token = self.sample_data.get('scene_token', '')
        
        # Estrazione Ground Truth Reale nuScenes (6 Canali)
        real_gt_masks = extract_ground_truth_masks(self.sample_data)
        
        # Iniezione Ground Truth Sintetica (Stress Test ad Alta Densità)
        syn_gt_masks, syn_count = generate_synthetic_injected_gt(self.sample_data)
        
        # 1. Disegno Pannello Sinistro: GT REALE nuScenes
        self.draw_gt_masks(
            self.ax_real, real_gt_masks, 
            f"1. GROUND TRUTH REALE 3D (nuScenes GT)\nFrame {self.current_idx + 1}/{self.total_samples}"
        )
        
        # 2. Disegno Pannello Destro: GT SINTETICA (Stress Test)
        self.draw_gt_masks(
            self.ax_syn, syn_gt_masks, 
            f"2. GROUND TRUTH SINTETICA (Stress Test +{syn_count} Ostacoli)\nFrame {self.current_idx + 1}/{self.total_samples}"
        )
        
        # 3. Aggiornamento HUD Informativo di Sinistra
        self.ax_hud.clear()
        self.ax_hud.axis('off')
        
        self.ax_hud.text(0.05, 0.95, "ISPEZIONE GROUND TRUTH", color='#66FCF1', fontsize=12, fontweight='bold')
        self.ax_hud.text(0.05, 0.91, f"Fotogramma: {self.current_idx + 1} / {self.total_samples}", color='#94A3B8', fontsize=9)
        self.ax_hud.text(0.05, 0.88, f"Scene Token: {scene_token[:16]}...", color='#64748B', fontsize=8)
        
        self.ax_hud.text(0.05, 0.82, "―" * 32, color='#334155', fontsize=10)
        self.ax_hud.text(0.05, 0.78, "OSTACOLI REALI 3D (nuScenes):", color='#F8FAFC', fontsize=9, fontweight='bold')
        
        real_counts = np.sum(real_gt_masks > 0.5, axis=(1, 2))
        y_pos = 0.73
        for c in range(6):
            c_name = CLASS_NAMES[c]
            c_color = CLASS_COLORS[c_name]
            cnt = real_counts[c]
            status_str = f"{cnt} px positivi" if cnt > 0 else "Assente"
            self.ax_hud.text(0.08, y_pos, f"• {c_name:<12}: {status_str}", color=c_color, fontsize=8, fontweight='bold')
            y_pos -= 0.04
            
        self.ax_hud.text(0.05, y_pos - 0.01, "―" * 32, color='#334155', fontsize=10)
        y_pos -= 0.05
        
        self.ax_hud.text(0.05, y_pos, "STRESS TEST SINTETICO:", color='#F59E0B', fontsize=9, fontweight='bold')
        y_pos -= 0.04
        self.ax_hud.text(0.08, y_pos, f"Ostacoli Iniettati: +{syn_count} verosimili", color='#E2E8F0', fontsize=8)
        y_pos -= 0.04
        
        syn_counts = np.sum(syn_gt_masks > 0.5, axis=(1, 2))
        for c in range(6):
            c_name = CLASS_NAMES[c]
            c_color = CLASS_COLORS[c_name]
            cnt = syn_counts[c]
            self.ax_hud.text(0.08, y_pos, f"• {c_name:<12}: {cnt} px positivi", color=c_color, fontsize=8)
            y_pos -= 0.038

        self.ax_hud.text(0.05, 0.05, "FRECCE: Naviga Fotogrammi\nTASTO 'R': Rigenera Sintetici", color='#64748B', fontsize=8)
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    GroundTruthVisualizer(dataroot="./nuscenes")
