# Macro-Area 3D: Visualizzatore Comparativo Interattivo Affiancato (Bayes vs Per-Zone).
# Mostra contemporaneamente la Mappa Bayesiana e la Mappa Neurale Per-Zone in modo sincronizzato 1-a-1.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

class ComparisonRuntimeVisualizer:
    def __init__(self):
        self.dir_bayes = "extracted_occlusions_probabilities"
        self.dir_pz = "extracted_occlusions_per_zone"
        if not os.path.exists(self.dir_pz):
            self.dir_pz = "extracted_occlusions_neural"

        print("\nCaricamento dati per il Confronto Affiancato (Bayes vs Per-Zone)...")
        print("Inizializzazione NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        self.json_files = sorted(glob.glob(os.path.join(self.dir_bayes, "*.json")))
        if not self.json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{self.dir_bayes}'!")
            return
            
        self.current_file_idx = 0
        self.current_occ_idx = 0
        
        self.load_json_file()
        
        # Configurazione Matplotlib: 3 colonne (HUD a sinistra, Mappa Bayes al centro, Mappa Per-Zone a destra)
        self.fig, (self.ax_hud, self.ax_bayes, self.ax_pz) = plt.subplots(
            1, 3, figsize=(18, 9), facecolor='#0B0C10',
            gridspec_kw={'width_ratios': [1, 2, 2]}
        )
        self.fig.subplots_adjust(left=0.02, right=0.82, top=0.92, bottom=0.04, wspace=0.06)
        self.ax_hud.set_facecolor('#0B0C10')
        self.ax_bayes.set_facecolor('#0B0C10')
        self.ax_pz.set_facecolor('#0B0C10')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI COMPARATIVI")
        print("=" * 50)
        print(" -> MOUSE            : Evidenzia simultaneamente l'ombra in entrambe le mappe")
        print(" -> FRECCIA SU/GIÙ   : Cambia fotogramma")
        print(" -> FRECCIA DX/SX    : Scorri le zone d'ombra")
        print("=" * 50 + "\n")
        
        self.plot_current()
        plt.show()

    def load_json_file(self):
        json_path_bayes = self.json_files[self.current_file_idx]
        fname = os.path.basename(json_path_bayes)
        json_path_pz = os.path.join(self.dir_pz, fname)
        
        with open(json_path_bayes, 'r') as f:
            self.data_bayes = json.load(f)
            
        if os.path.exists(json_path_pz):
            with open(json_path_pz, 'r') as f:
                self.data_pz = json.load(f)
        else:
            self.data_pz = self.data_bayes

        self.occs_bayes = self.data_bayes.get('occlusions', [])
        self.occs_pz = self.data_pz.get('occlusions', [])
        self.current_occ_idx = 0
        self.lidar_token = self.data_bayes.get('lidar_token')

    def get_unicode_bar(self, val, length=12):
        filled = int(round(val * length))
        return '█' * filled + '░' * (length - filled)

    def lidar_to_ego(self, pts_lidar):
        if len(pts_lidar) == 0 or not self.lidar_token:
            return np.zeros((0, 2))
        try:
            sd = self.nusc.get('sample_data', self.lidar_token)
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            q_sensor = Quaternion(cs['rotation'])
            R_sensor = q_sensor.rotation_matrix[:2, :2]
            t_sensor = np.array(cs['translation'])[:2]
            return pts_lidar @ R_sensor.T + t_sensor
        except Exception:
            return pts_lidar

    def _draw_map(self, ax, occs, title_str):
        ax.clear()
        ax.axis('off')
        
        # 1. RENDER NATIVO OFFICIALE NUSCENES
        if self.lidar_token:
            try:
                self.nusc.render_sample_data(self.lidar_token, ax=ax, verbose=False)
            except Exception:
                pass
                
        ax.set_title(title_str, color='#66FCF1', fontsize=11, fontweight='bold', pad=12)

        if not occs:
            return

        # 2. Disegno delle occlusioni non selezionate
        for idx, occ in enumerate(occs):
            if idx == self.current_occ_idx:
                continue
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) >= 3:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                poly_ego = self.lidar_to_ego(poly_closed)
                ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#1E293B', alpha=0.15, zorder=4)

        # 3. Disegno dell'occlusione selezionata
        if self.current_occ_idx < len(occs):
            occ = occs[self.current_occ_idx]
            poly = np.array(occ.get('polygon_points_m', []))
            if len(poly) >= 3:
                poly_closed = np.vstack([poly, poly[0]])
                poly_ego = self.lidar_to_ego(poly_closed)
                ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#8B0000', alpha=0.50, zorder=5)
                ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#FF0055', linewidth=2.2, zorder=5)
                
                cx = float(np.mean(poly_ego[:, 0]))
                cy = float(np.mean(poly_ego[:, 1]))
                ax.text(cx, cy, f"#{self.current_occ_idx+1}", color='white', fontsize=9, fontweight='bold', ha='center', va='center', zorder=8)

    def plot_current(self):
        json_filename = os.path.basename(self.json_files[self.current_file_idx])
        
        # 1. Render Mappa Bayesiana
        self._draw_map(self.ax_bayes, self.occs_bayes, f"AGENTE BAYESIANO DINAMICO\nFrame: {json_filename}")
        
        # 2. Render Mappa Per-Zone
        ckpt_used = getattr(self, 'data_pz', {}).get("model_checkpoint_used", "N/A")
        self._draw_map(self.ax_pz, self.occs_pz, f"AGENTE NEURALE PER-ZONE\n(Checkpoint: {ckpt_used})")
        
        # 3. Costruzione Pannello HUD Comparativo Affiancato
        self.ax_hud.clear()
        self.ax_hud.axis('off')
        
        if not self.occs_bayes or self.current_occ_idx >= len(self.occs_bayes):
            self.fig.canvas.draw()
            return
            
        occ_b = self.occs_bayes[self.current_occ_idx]
        occ_pz = self.occs_pz[self.current_occ_idx] if self.current_occ_idx < len(self.occs_pz) else occ_b
        
        dist = occ_b.get('distance_m', 0.0)
        area = occ_b.get('area_sqm', 0.0)
        terrain_str = occ_b.get('terrain_type', 'Sconosciuto')
        
        bayes_probs = occ_b.get('estimated_probabilities', {})
        pz_probs = occ_pz.get('estimated_probabilities', {})
        
        hud_lines = []
        hud_lines.append(f"=== OCCLUSION #{self.current_occ_idx+1}/{len(self.occs_bayes)} ===")
        hud_lines.append(f"Distanza: {dist:.1f} m | Area: {area:.1f} m²")
        hud_lines.append(f"Terreno : {terrain_str}")
        hud_lines.append("=" * 30)
        hud_lines.append(" CONFRONTO PROBABILITÀ GLOBALI")
        hud_lines.append(" CATEGORIA   | BAYES  | PER-ZONE")
        hud_lines.append("-" * 30)
        
        for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Barriera"]:
            pb = bayes_probs.get(k, 0.0) * 100
            ppz = pz_probs.get(k, 0.0) * 100
            hud_lines.append(f" {k:<11}: {pb:>5.1f}%  vs  {ppz:>5.1f}%")
            
        b_best = max(bayes_probs, key=bayes_probs.get) if bayes_probs else "N/A"
        pz_best = max(pz_probs, key=pz_probs.get) if pz_probs else "N/A"
        
        hud_lines.append("=" * 30)
        hud_lines.append(f"BAYES   : {b_best} ({bayes_probs.get(b_best, 0.0)*100:.0f}%)")
        hud_lines.append(f"PER-ZONE: {pz_best} ({pz_probs.get(pz_best, 0.0)*100:.0f}%)")
        
        hud_box_str = "\n".join(hud_lines)
        
        self.ax_hud.text(
            0.05, 0.95, hud_box_str, color='white', fontsize=8.0, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.90, edgecolor='#00E5FF', boxstyle='round,pad=0.8'),
            va='top', ha='left', transform=self.ax_hud.transAxes
        )
        
        legend_elements = [
            Patch(facecolor='#8B0000', edgecolor='#FF0055', alpha=0.5, label='Occlusione Selezionata'),
            Patch(facecolor='#2C374E', edgecolor='#2C374E', alpha=0.4, label='Altre Occlusioni'),
            Patch(facecolor='#00E5FF', edgecolor='#00B0FF', alpha=0.6, label='Auto nuScenes'),
            Patch(facecolor='#FFD600', edgecolor='#FFAB00', alpha=0.6, label='Camion / Bus nuScenes'),
            Patch(facecolor='#FF1744', edgecolor='#D50000', alpha=0.6, label='Pedone nuScenes'),
        ]
        self.ax_pz.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), facecolor="#1E293B",
                          edgecolor="gray", fontsize=7.5, labelcolor="white")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            if self.occs_bayes:
                self.current_occ_idx = (self.current_occ_idx + 1) % len(self.occs_bayes)
                self.plot_current()
        elif event.key == 'left':
            if self.occs_bayes:
                self.current_occ_idx = (self.current_occ_idx - 1) % len(self.occs_bayes)
                self.plot_current()
        elif event.key == 'down':
            if len(self.json_files) > 1:
                self.current_file_idx = (self.current_file_idx + 1) % len(self.json_files)
                self.load_json_file()
                self.plot_current()
        elif event.key == 'up':
            if len(self.json_files) > 1:
                self.current_file_idx = (self.current_file_idx - 1) % len(self.json_files)
                self.load_json_file()
                self.plot_current()

    def on_mouse_move(self, event):
        if event.inaxes not in [self.ax_bayes, self.ax_pz]:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        point = ShapelyPoint(event.xdata, event.ydata)
        
        for idx, occ in enumerate(self.occs_bayes):
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) > 2:
                poly_ego = self.lidar_to_ego(poly_pts)
                shp_poly = ShapelyPolygon(poly_ego)
                if shp_poly.contains(point):
                    if self.current_occ_idx != idx:
                        self.current_occ_idx = idx
                        self.plot_current()
                    break

if __name__ == "__main__":
    ComparisonRuntimeVisualizer()
