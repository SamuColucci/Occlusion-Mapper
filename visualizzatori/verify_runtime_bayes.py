# Macro-Area 3A: Esecuzione e Visualizzazione dell'Agente Bayesiano a Runtime.
# Visualizzatore Interattivo Semplificato per Agente Bayesiano (Mappa Nativa nuScenes + Ombre + Probabilità)
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

class RuntimeBayesVisualizer:
    def __init__(self, in_dir=None):
        if in_dir is None:
            # ponytail: supporta l'avvio con parametro da riga di comando per riusare lo script in modalità diverse
            if len(sys.argv) > 1:
                self.in_dir = sys.argv[1]
            else:
                self.in_dir = "extracted_occlusions_probabilities"
        else:
            self.in_dir = in_dir
            
        if self.in_dir == "extracted_occlusions":
            self.modo_label = "GEOMETRIA RAW"
        elif self.in_dir == "extracted_occlusions_probabilities":
            self.modo_label = "UNIFICATO"
        else:
            self.modo_label = "PERSONALIZZATO"
            
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
        print("    CONTROLLI INTERATTIVI REGISTRATI (BAYES)")
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
            if len(poly_pts) > 0:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                poly_ego = self.lidar_to_ego(poly_closed)
                self.ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                self.ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#1E293B', alpha=0.15, zorder=4)
                
        # 3. Dati dell'occlusione selezionata
        occ = self.occlusions[self.current_occ_idx]
        poly = np.array(occ.get('polygon_points_m', []))
        name = occ.get('object_name', 'unknown')
        dist = occ.get('distance_m', 0.0)
        
        # ponytail: se visualizziamo geometria pura (senza probabilità), usa un colore arancione neon fisso
        if self.in_dir == "extracted_occlusions":
            color = "#FF3D00"
            max_b_val = 0.0
            bayes_probs = {}
        else:
            bayes_probs = occ.get('estimated_probabilities', {})
            max_b_val = max(bayes_probs.values()) if bayes_probs else 0.0
            if max_b_val < 0.20:
                color = "#00E676"  # Verde Neon
            elif max_b_val < 0.50:
                color = "#FFD600"  # Giallo Neon
            else:
                color = "#FF1744"  # Rosso Neon

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

        sub_zones = occ.get('sub_zones', [])
        
        # Disegna outline dell'occlusione selezionata in Ego Frame
        if len(poly) > 0:
            poly_closed = np.vstack([poly, poly[0]])
            poly_ego = self.lidar_to_ego(poly_closed)
            self.ax.plot(poly_ego[:, 0], poly_ego[:, 1], color=color, linewidth=2.5, zorder=5, alpha=0.6)
        
        # Disegna le sub-zone con colori distinti in Ego Frame
        for sz in sub_zones:
            surf = sz.get('surface', 'terrain')
            fill_c, border_c = SURF_COLORS.get(surf, ("#555", "#333"))
            sz_pts = np.array(sz.get('polygon_points_m', []))
            if len(sz_pts) >= 3:
                sz_closed = np.vstack([sz_pts, sz_pts[0]])
                sz_ego = self.lidar_to_ego(sz_closed)
                self.ax.fill(sz_ego[:, 0], sz_ego[:, 1], color=fill_c, alpha=0.45, zorder=5)
                self.ax.plot(sz_ego[:, 0], sz_ego[:, 1], color=border_c, linewidth=1.5, zorder=5)
                cx = float(np.mean(sz_ego[:, 0]))
                cy = float(np.mean(sz_ego[:, 1]))
                frac_lbl = min(sz.get('area_fraction', 0.0), 1.0) * 100
                self.ax.text(cx, cy,
                    f"{SURF_LABELS.get(surf, surf)}: {frac_lbl:.0f}%",
                    color='white', fontsize=7.0, ha='center', va='center',
                    bbox=dict(facecolor=fill_c, alpha=0.75, edgecolor=border_c, boxstyle='round,pad=0.3'),
                    zorder=9)

        # --- Pannello HUD Sinistra ---
        curr_occ = self.occlusions[self.current_occ_idx]
        dist = curr_occ.get('distance_m', 0.0)
        area = curr_occ.get('area_sqm', 0.0)
        
        bayes_lines = []
        bayes_lines.append(f"=== OCCLUSION #{self.current_occ_idx+1}/{len(self.occlusions)} ===")
        bayes_lines.append(f"Ostacolo: {name.split('.')[-1].upper()}")
        bayes_lines.append(f"Distanza: {dist:.1f} m | Area: {area:.1f} m²")
        
        if self.in_dir == "extracted_occlusions":
            # Mode: Raw Geometry HUD
            road_f = curr_occ.get('road_fraction', 0.0)
            side_f = curr_occ.get('sidewalk_fraction', 0.0)
            cross_f = curr_occ.get('crosswalk_fraction', 0.0)
            park_f = curr_occ.get('carpark_fraction', 0.0)
            terr_f = curr_occ.get('terrain_fraction', max(0.0, 1.0 - road_f - side_f - cross_f - park_f))
            
            bayes_lines.append("-" * 30)
            bayes_lines.append("COMPOSIZIONE TERRENO (RAYCASTER)")
            bayes_lines.append("-" * 30)
            
            SURF_LABELS_RAW = {
                "Strada": road_f,
                "Marciapiede": side_f,
                "Strisce Ped.": cross_f,
                "Parcheggio": park_f,
                "Terreno/Altro": terr_f
            }
            for lbl, val in SURF_LABELS_RAW.items():
                bar = self.get_unicode_bar(val)
                bayes_lines.append(f"{lbl:<14}: {bar} {val*100:>5.1f}%")
                
            bayes_lines.append("-" * 30)
            bayes_lines.append("STATO: Geometria Pura 2D")
            bayes_lines.append("Frazioni calcolate tramite")
            bayes_lines.append("intersezione con HD Map.")
            
            title_text = (
                f"GEOMETRIA ZONE OCCLUSE (RAYCASTER)\n"
                f"FILE: {json_filename} [{self.current_file_idx+1}/{len(self.json_files)}] | "
                f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m)"
            )
            
            legend_elements = [
                Patch(facecolor=color, edgecolor=color, alpha=0.6, label='Zona Occlusa Selezionata'),
                Patch(facecolor='#37474F', edgecolor='#455A64', alpha=0.6, label='Altre Zone Occluse'),
                Patch(facecolor='#00E5FF', edgecolor='#00B0FF', alpha=0.6, label='Auto nuScenes'),
                Patch(facecolor='#FFD600', edgecolor='#FFAB00', alpha=0.6, label='Camion / Bus nuScenes'),
                Patch(facecolor='#FF1744', edgecolor='#D50000', alpha=0.6, label='Pedone nuScenes'),
                Patch(facecolor='#00E676', edgecolor='#00C853', alpha=0.6, label='Moto / Bici nuScenes'),
                Patch(facecolor='#D500F9', edgecolor='#AA00FF', alpha=0.6, label='Barriera / Cono nuScenes'),
                Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.3, label='Ego Vehicle'),
            ]
        else:
            # Mode: Probabilistic Agents HUD
            terrain_str = curr_occ.get('terrain_type', 'Sconosciuto')
            sub_distrib = curr_occ.get('subzone_distribution', {})
            
            bayes_lines.append(f"Terreno : {terrain_str}")
            bayes_lines.append("-" * 30)
            
            for k_surf, v_pct in sub_distrib.items():
                bayes_lines.append(f"{SURF_LABELS.get(k_surf, k_surf):<16}: {v_pct*100:>5.1f}%")
                
            bayes_lines.append("=" * 30)
            bayes_lines.append("PROB GLOBALI (Agente Bayesiano)")
            bayes_lines.append("-" * 30)
            
            main_cats = ["Auto", "Pedone", "Camion", "Bicicletta"]
            for k in main_cats:
                v = bayes_probs.get(k, 0.0)
                bar = self.get_unicode_bar(v)
                bayes_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
                
            sec_cats = ["Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]
            for k in sec_cats:
                v = bayes_probs.get(k, 0.0)
                if v > 0.005:
                    bar = self.get_unicode_bar(v)
                    bayes_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
                    
            best_cat = max(bayes_probs, key=bayes_probs.get) if bayes_probs else "N/A"
            best_prob = bayes_probs[best_cat] if bayes_probs else 0.0
            
            if best_prob >= 0.90:
                sintesi_hud = f"SINTESI: {best_cat} ({best_prob*100:.0f}%) via Memoria Storica."
            elif best_prob >= 0.20:
                sintesi_hud = f"SINTESI: {best_cat} ({best_prob*100:.0f}%) via Semantica HD."
            else:
                sintesi_hud = f"SINTESI: Libera ({best_cat} {best_prob*100:.0f}%)."
                
            bayes_lines.append("-" * 30)
            bayes_lines.append(sintesi_hud)

            if sub_zones:
                bayes_lines.append("=" * 30)
                bayes_lines.append(" PROB PER SUB-ZONA")
                for sz in sub_zones:
                    surf = sz.get('surface', '?')
                    sz_probs = sz.get('estimated_probabilities', {})
                    frac = min(sz.get('area_fraction', 0.0), 1.0)
                    w = sz.get('occlusion_width_m', 0.0)
                    bayes_lines.append(f"--- {SURF_LABELS.get(surf, surf)} ({frac*100:.0f}% area, W={w:.1f}m) ---")
                    
                    for k in ["Auto", "Pedone", "Camion", "Bicicletta"]:
                        v = sz_probs.get(k, 0.0)
                        bar = self.get_unicode_bar(v)
                        bayes_lines.append(f"  {k:<9}: {bar} {v*100:>5.1f}%")

            title_text = (
                f"AGENTE BAYESIANO DINAMICO ({self.modo_label}) - RISULTATI STIMATI\n"
                f"FILE: {json_filename} [{self.current_file_idx+1}/{len(self.json_files)}] | "
                f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m)"
            )
            
            legend_elements = [
                Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.6, label='Rischio Basso (<20%)'),
                Patch(facecolor='#FFD600', edgecolor='#FFD600', alpha=0.6, label='Rischio Medio (20-50%)'),
                Patch(facecolor='#FF1744', edgecolor='#FF1744', alpha=0.6, label='Rischio Alto (>=50%)'),
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

        bayes_box_str = "\n".join(bayes_lines)
        self.ax.set_title(title_text, color='white', fontsize=10, fontweight='bold', pad=15)
        
        self.ax_hud.text(
            0.05, 0.95, bayes_box_str, color='white', fontsize=7.5, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.90, edgecolor=color, boxstyle='round,pad=0.8'),
            va='top', ha='left', transform=self.ax_hud.transAxes
        )
        
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
    RuntimeBayesVisualizer()
