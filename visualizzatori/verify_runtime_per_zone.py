# Macro-Area 3B: Esecuzione e Visualizzazione dell'Agente Per-Zone a Runtime.
# Visualizzatore Interattivo Semplificato per Agente Per-Zone (Mappa Nativa nuScenes + Ombre + Probabilità)
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

class PerZoneRuntimeVisualizer:
    def __init__(self, mode="per_zone"):
        self.modo_label = "NEURALE PER-ZONE (Patch 64x64 + Scalari)"
        self.in_dir = "extracted_occlusions_per_zone"
        if not os.path.exists(self.in_dir) or not glob.glob(os.path.join(self.in_dir, "*.json")):
            self.in_dir = "extracted_occlusions_neural"
            self.modo_label = "NEURALE UNET 2D (Fallback)"

        print(f"\nCaricamento dati in modalità: {self.modo_label}")
        print("Inizializzazione NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        self.json_files = sorted(glob.glob(os.path.join(self.in_dir, "*.json")))
        if not self.json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{self.in_dir}'!")
            print("Assicurati di aver generato i dati batch tramite l'opzione 7 della dashboard.")
            return
            
        self.current_file_idx = 0
        self.current_occ_idx = 0
        
        self.load_json_file()
        
        # Configurazione Matplotlib: 2 colonne (HUD a sinistra, Mappa Nativa nuScenes a destra)
        self.fig, (self.ax_hud, self.ax) = plt.subplots(
            1, 2, figsize=(15, 9), facecolor='#0B0C10',
            gridspec_kw={'width_ratios': [1, 2.5]}
        )
        self.fig.subplots_adjust(left=0.02, right=0.82, top=0.92, bottom=0.04, wspace=0.08)
        self.ax_hud.set_facecolor('#0B0C10')
        self.ax.set_facecolor('#0B0C10')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI REGISTRATI (PER-ZONE)")
        print("=" * 50)
        print(" -> MOUSE            : Passa sopra un'ombra per evidenziarla")
        print(" -> FRECCIA SU/GIÙ   : Cambia frame")
        print(" -> FRECCIA DX/SX    : Scorri manualmente le occlusioni")
        print("=" * 50 + "\n")
        
        self.plot_current()
        plt.show()

    def load_json_file(self):
        json_path = self.json_files[self.current_file_idx]
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.occlusions = self.data.get('occlusions', [])
        self.current_occ_idx = 0
        self.lidar_token = self.data.get('lidar_token')

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
            
            # Trasformazione esatta dal frame sensore LIDAR_TOP al frame Ego Vehicle
            pts_ego = pts_lidar @ R_sensor.T + t_sensor
            return pts_ego
        except Exception:
            return pts_lidar

    def plot_current(self):
        self.ax.clear()
        self.ax_hud.clear()
        self.ax_hud.axis('off')
        
        json_filename = os.path.basename(self.json_files[self.current_file_idx])
        
        # 1. RENDER NATIVO UFFICIALE NUSCENES (Strada HD Map, 3D Boxes, LiDAR PointCloud)
        if self.lidar_token:
            try:
                self.nusc.render_sample_data(self.lidar_token, ax=self.ax, underlay_map=True, verbose=False)
            except Exception as e:
                print(f"[WARN] Impossibile renderizzare il background nuScenes: {e}")
                
        if not self.occlusions:
            self.ax.set_title(f"FILE: {json_filename}\nNessuna occlusione registrata.", color='white')
            self.ax.axis('off')
            self.fig.canvas.draw()
            return
            
        # 2. Disegno di tutte le altre occlusioni NON selezionate (zorder=4)
        for idx, occ in enumerate(self.occlusions):
            if idx == self.current_occ_idx:
                continue
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) >= 3:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                poly_ego = self.lidar_to_ego(poly_closed)
                self.ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#0A2540', alpha=0.45, zorder=4)
                self.ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#00E5FF', linewidth=0.8, alpha=0.6, zorder=4)
                
                cx, cy = np.mean(poly_ego[:, 0]), np.mean(poly_ego[:, 1])
                self.ax.text(cx, cy, f"#{idx+1}", color='#64748B', fontsize=6, ha='center', va='center', zorder=5)

        SURF_COLORS = {
            "driveable_surface": ("#607D8B", "#37474F"),
            "sidewalk":          ("#8D6E63", "#5D4037"),
            "other_flat":        ("#7B1FA2", "#4A148C"),
            "ped_crossing":      ("#0288D1", "#01579B"),
            "terrain":           ("#388E3C", "#1B5E20"),
        }
        SURF_LABELS = {
            "driveable_surface": "Strada", 
            "sidewalk": "Marciapiede", 
            "other_flat": "Parcheggio", 
            "ped_crossing": "Strisce", 
            "terrain": "Terreno"
        }

        curr_occ = self.occlusions[self.current_occ_idx]
        curr_poly_pts = np.array(curr_occ.get('polygon_points_m', []))
        sub_zones = curr_occ.get('sub_zones', [])
        
        # 3. Disegno dell'occlusione selezionata
        if len(curr_poly_pts) >= 3:
            poly_closed = np.vstack([curr_poly_pts, curr_poly_pts[0]])
            poly_ego = self.lidar_to_ego(poly_closed)
            self.ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#8B0000', alpha=0.50, zorder=5)
            self.ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#FF0055', linewidth=2.2, zorder=5)
            
        # 4. Disegna le sotto-zone con i loro colori specifici sulla mappa BEV
        for sz in sub_zones:
            surf = sz.get('surface', 'terrain')
            fill_c, border_c = SURF_COLORS.get(surf, ("#555", "#333"))
            sz_pts = np.array(sz.get('polygon_points_m', []))
            if len(sz_pts) >= 3:
                sz_closed = np.vstack([sz_pts, sz_pts[0]])
                sz_ego = self.lidar_to_ego(sz_closed)
                self.ax.fill(sz_ego[:, 0], sz_ego[:, 1], color=fill_c, alpha=0.55, zorder=6)
                self.ax.plot(sz_ego[:, 0], sz_ego[:, 1], color=border_c, linewidth=1.5, zorder=6)
                cx = float(np.mean(sz_ego[:, 0]))
                cy = float(np.mean(sz_ego[:, 1]))
                frac_lbl = min(sz.get('area_fraction', 0.0), 1.0) * 100
                self.ax.text(cx, cy, f"{SURF_LABELS.get(surf, surf)}: {frac_lbl:.0f}%",
                             color='white', fontsize=7.0, fontweight='bold', ha='center', va='center',
                             bbox=dict(facecolor=fill_c, alpha=0.85, edgecolor=border_c, boxstyle='round,pad=0.3'),
                             zorder=7)
            
        if len(curr_poly_pts) >= 3:
            poly_ego = self.lidar_to_ego(curr_poly_pts)
            cx, cy = np.mean(poly_ego[:, 0]), np.mean(poly_ego[:, 1])
            self.ax.text(cx, cy, f"#{self.current_occ_idx+1}", color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', va='center', zorder=8)

        total_files = len(self.json_files)
        ckpt_used = self.data.get("model_checkpoint_used", "N/A")
        title_text = f"FRAME [{self.current_file_idx + 1}/{total_files}]: {json_filename}\nMODELLAZIONE PER-ZONE (Checkpoint Loss: {ckpt_used})"
        self.ax.set_title(title_text, color='#66FCF1', fontsize=11, fontweight='bold', pad=12)

        # --- Pannello HUD Sinistra ---
        dist = curr_occ.get('distance_m', 0.0)
        area = curr_occ.get('area_sqm', 0.0)
        terrain_str = curr_occ.get('terrain_type', 'Sconosciuto')
        sub_distrib = curr_occ.get('subzone_distribution', {})
        per_zone_probs = curr_occ.get('estimated_probabilities', {})
        
        hud_lines = []
        hud_lines.append(f"=== OCCLUSION #{self.current_occ_idx+1}/{len(self.occlusions)} ===")
        hud_lines.append(f"Distanza: {dist:.1f} m | Area: {area:.1f} m²")
        hud_lines.append(f"Terreno : {terrain_str}")
        hud_lines.append("-" * 30)
        
        for k_surf, v_pct in sub_distrib.items():
            hud_lines.append(f"{SURF_LABELS.get(k_surf, k_surf):<16}: {v_pct*100:>5.1f}%")
            
        hud_lines.append("=" * 30)
        hud_lines.append("PROB GLOBALI (Rete Per-Zone)")
        hud_lines.append("-" * 30)
        
        main_cats = ["Auto", "Pedone", "Camion", "Bicicletta"]
        for k in main_cats:
            v = per_zone_probs.get(k, 0.0)
            bar = self.get_unicode_bar(v)
            hud_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
            
        sec_cats = ["Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]
        for k in sec_cats:
            v = per_zone_probs.get(k, 0.0)
            if v > 0.005:
                bar = self.get_unicode_bar(v)
                hud_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
                
        best_cat = max(per_zone_probs, key=per_zone_probs.get) if per_zone_probs else "N/A"
        best_prob = per_zone_probs[best_cat] if per_zone_probs else 0.0
        
        hud_lines.append("-" * 30)
        hud_lines.append(f"SINTESI: {best_cat} ({best_prob*100:.0f}%) via Per-Zone CNN.")

        if sub_zones:
            hud_lines.append("=" * 30)
            hud_lines.append(" PROB PER SUB-ZONA")
            for sz in sub_zones:
                surf = sz.get('surface', '?')
                sz_probs = sz.get('estimated_probabilities', {})
                frac = min(sz.get('area_fraction', 0.0), 1.0)
                w = sz.get('occlusion_width_m', 0.0)
                hud_lines.append(f"--- {SURF_LABELS.get(surf, surf)} ({frac*100:.0f}% area, W={w:.1f}m) ---")
                
                for k in ["Auto", "Pedone", "Camion", "Bicicletta"]:
                    v = sz_probs.get(k, 0.0)
                    bar = self.get_unicode_bar(v)
                    hud_lines.append(f"  {k:<9}: {bar} {v*100:>5.1f}%")

        hud_box_str = "\n".join(hud_lines)
        
        self.ax_hud.text(
            0.05, 0.95, hud_box_str, color='white', fontsize=7.5, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.90, edgecolor='#00E5FF', boxstyle='round,pad=0.8'),
            va='top', ha='left', transform=self.ax_hud.transAxes
        )
        
        legend_elements = [
            Patch(facecolor='#8B0000', edgecolor='#FF0055', alpha=0.5, label='Occlusione Selezionata'),
            Patch(facecolor='#0A2540', edgecolor='#00E5FF', alpha=0.4, label='Altre Occlusioni'),
            Patch(facecolor='#607D8B', edgecolor='#37474F', alpha=0.6, label='Sub-zona Strada'),
            Patch(facecolor='#8D6E63', edgecolor='#5D4037', alpha=0.6, label='Sub-zona Marciapiede'),
            Patch(facecolor='#7B1FA2', edgecolor='#4A148C', alpha=0.6, label='Sub-zona Parcheggio'),
            Patch(facecolor='#0288D1', edgecolor='#01579B', alpha=0.6, label='Sub-zona Strisce'),
            Patch(facecolor='#388E3C', edgecolor='#1B5E20', alpha=0.6, label='Sub-zona Terreno'),
            Patch(facecolor='#00E5FF', edgecolor='#00B0FF', alpha=0.6, label='Auto nuScenes'),
            Patch(facecolor='#FFD600', edgecolor='#FFAB00', alpha=0.6, label='Camion / Bus nuScenes'),
            Patch(facecolor='#FF1744', edgecolor='#D50000', alpha=0.6, label='Pedone nuScenes'),
            Patch(facecolor='#00E676', edgecolor='#00C853', alpha=0.6, label='Moto / Bici nuScenes'),
            Patch(facecolor='#D500F9', edgecolor='#AA00FF', alpha=0.6, label='Barriera / Cono nuScenes'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.3, label='Ego Vehicle'),
        ]
        self.ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), facecolor="#1E293B",
                       edgecolor="gray", fontsize=7.5, labelcolor="white")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            if self.occlusions:
                self.current_occ_idx = (self.current_occ_idx + 1) % len(self.occlusions)
                self.plot_current()
        elif event.key == 'left':
            if self.occlusions:
                self.current_occ_idx = (self.current_occ_idx - 1) % len(self.occlusions)
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
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        point = ShapelyPoint(event.xdata, event.ydata)
        
        for idx, occ in enumerate(self.occlusions):
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
    PerZoneRuntimeVisualizer()
