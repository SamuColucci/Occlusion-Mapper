"""
Macro-Area 3B: Visualizzazione dell'Agente Neurale a Runtime (Solo Lettura Risultati).

Questo script legge in modo istantaneo le stime probabilistiche
semantiche pre-calcolate e salvate dall'Agente Neurale nella cartella 'extracted_occlusions_neural/'.
Non richiede PyTorch, non fa calcoli ed è velocissimo.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

MAP_CACHE = {}

def get_map_instance(dataroot, map_name):
    if map_name not in MAP_CACHE:
        print(f"Caricamento mappa NuScenes '{map_name}' in cache...")
        MAP_CACHE[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return MAP_CACHE[map_name]

def get_semantic_surfaces_local(nusc, nusc_map, sample_token, ego_pose, range_m=50):
    from shapely.geometry import box as ShapelyBox, Polygon as ShapelyPolygon, MultiPolygon
    tx, ty, tz = ego_pose['translation']
    box_coords = (tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    surfaces = {
        'drivable_area': [],
        'walkway': []
    }
    
    q = Quaternion(ego_pose['rotation'])
    R_inv = q.inverse.rotation_matrix
    interest_box = ShapelyBox(tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    for layer in ['drivable_area', 'walkway']:
        try:
            records = nusc_map.get_records_in_patch(box_coords, layer_names=[layer], mode='intersect')
            tokens = records.get(layer, [])
        except Exception:
            continue
            
        for token in tokens:
            try:
                record = nusc_map.get(layer, token)
                polygon_tokens = record.get('polygon_tokens', [])
                if not polygon_tokens:
                    poly_token = record.get('polygon_token', token)
                    raw_polys = [nusc_map.extract_polygon(poly_token)]
                else:
                    raw_polys = [nusc_map.extract_polygon(pt) for pt in polygon_tokens]
                    
                for poly in raw_polys:
                    if poly.is_empty:
                        continue
                    inter = poly.intersection(interest_box)
                    if inter.is_empty:
                        continue
                        
                    if isinstance(inter, MultiPolygon):
                        polys_to_process = list(inter.geoms)
                    else:
                        polys_to_process = [inter]
                        
                    for p in polys_to_process:
                        if not isinstance(p, ShapelyPolygon):
                            continue
                        x_coords, y_coords = p.exterior.coords.xy
                        pts_global = np.column_stack([x_coords, y_coords])
                        pts_global_3d = np.column_stack([pts_global[:, 0], pts_global[:, 1], np.full(len(pts_global), tz)])
                        
                        pts_diff = pts_global_3d - np.array([tx, ty, tz])
                        pts_local = pts_diff @ R_inv.T
                        local_pts = np.column_stack([pts_local[:, 0], pts_local[:, 1]])
                        surfaces[layer].append(local_pts)
            except Exception:
                continue
                
    return surfaces

class RuntimeNeuralOnlyVisualizer:
    def __init__(self, in_dir=None):
        if in_dir is None:
            # Selezione interattiva del tipo di prior
            print("\n" + "="*60)
            print("         SELEZIONE DI VISUALIZZAZIONE AGENTE NEURALE")
            print("="*60)
            print("  1. DIRETTO    (mostra le stime basate su intersezioni geometriche)")
            print("  2. NEAR-MISS  (mostra le stime basate sulla vicinanza entro 2 metri)")
            print("-"*60)
            scelta = input(" Inserisci la tua scelta [1-2, default=1]: ").strip()
            if scelta == '2':
                self.in_dir = "extracted_occlusions_neural_near"
                self.modo_label = "NEAR-MISS"
            else:
                self.in_dir = "extracted_occlusions_neural"
                self.modo_label = "DIRETTO"
        else:
            self.in_dir = in_dir
            self.modo_label = "PERSONALIZZATO"
            
        print(f"\nCaricamento dati in modalità: {self.modo_label}")
        print("Inizializzazione NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        # Scansione dei JSON pre-calcolati dall'Agente Neurale
        self.json_files = sorted(glob.glob(os.path.join(self.in_dir, "*.json")))
        if not self.json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{self.in_dir}'!")
            print("Assicurati di aver generato i risultati tramite l'opzione 7 della dashboard.")
            return
            
        self.current_file_idx = 0
        self.current_occ_idx = 0
        self.pc = np.zeros((0, 3))
        
        self.load_json_file()
        
        # Setup grafico
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='#0B0C10')
        self.ax.set_facecolor('#0B0C10')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI REGISTRATI (NEURALE)")
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
        
        self.ego_pose = None
        self.sample_record = None
        lidar_token = self.data.get('lidar_token')
        if lidar_token:
            try:
                pcl_path = self.nusc.get_sample_data_path(lidar_token)
                self.pc = np.fromfile(pcl_path, dtype=np.float32).reshape((-1, 5))[:, :3]
                
                sd_record = self.nusc.get('sample_data', lidar_token)
                self.ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                self.sample_record = self.nusc.get('sample', sd_record['sample_token'])
            except Exception:
                self.pc = np.zeros((0, 3))

    def get_unicode_bar(self, val):
        """Genera una barra di progresso orizzontale in formato Unicode."""
        filled_length = int(round(val * 10))
        return "█" * filled_length + "░" * (10 - filled_length)

    def plot_current(self):
        self.ax.clear()
        
        # Sfondo verde prato scuro (terrain) come piano base
        bg_color = "#102216"
        self.ax.set_facecolor(bg_color)
        self.fig.patch.set_facecolor("#0B0C10")
        
        # 0. Disegno della Superficie Stradale e delle aree limitrofe
        self.road_polys_shapely = []
        self.sidewalk_polys_shapely = []
        if self.ego_pose and self.sample_record:
            try:
                scene = self.nusc.get('scene', self.sample_record['scene_token'])
                log = self.nusc.get('log', scene['log_token'])
                map_name = log['location']
                nusc_map = get_map_instance('./nuscenes', map_name)
                
                surfaces = get_semantic_surfaces_local(
                    self.nusc, nusc_map, self.sample_record['token'], self.ego_pose, range_m=50
                )
                
                from shapely.geometry import Polygon as ShapelyPolygon
                
                # 1. Strada (Asfalto grigio scuro)
                for pts in surfaces['drivable_area']:
                    self.ax.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
                    self.ax.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
                    self.road_polys_shapely.append(ShapelyPolygon(pts))
                    
                # 2. Marciapiede (Grigio walkway)
                for pts in surfaces['walkway']:
                    self.ax.fill(pts[:, 0], pts[:, 1], color='#22252C', zorder=1, alpha=0.8)
                    self.ax.plot(pts[:, 0], pts[:, 1], color='#3A4250', linestyle=':', linewidth=0.6, zorder=1)
                    self.sidewalk_polys_shapely.append(ShapelyPolygon(pts))
            except Exception:
                pass
        
        # 1. Disegno dei Radar Rings (Cerchi Concentrici)
        for radius in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), radius, color='#2C374E', fill=False, linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
            self.ax.add_patch(circle)
            self.ax.text(0.5, radius, f"{radius}m", color='#475569', fontsize=7, ha='left', va='center', fontweight='bold', zorder=2)
            
        # 2. LiDAR in Sonar-Style (zorder=3 per stare sopra la strada)
        if len(self.pc) > 0:
            self.ax.scatter(self.pc[:, 1], self.pc[:, 0], s=0.12, c='#4A5E7D', alpha=0.25, zorder=3)
            
        # 3. Disegno dell'Ego-Vehicle stilizzato a centro (0, 0) (zorder=6 per stare sopra tutto)
        ego_rect = Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=1.5, edgecolor='#00E5FF', facecolor='#00E5FF', alpha=0.2, zorder=6)
        self.ax.add_patch(ego_rect)
        self.ax.plot(0, 0, color='#00E5FF', marker='^', markersize=4, zorder=6) # Freccia di heading
        
        json_filename = os.path.basename(self.json_files[self.current_file_idx])
        
        if not self.occlusions:
            self.ax.set_title(f"FILE: {json_filename}\nNessuna occlusione registrata.", color='white')
            self.ax.set_xlim(40, -40)
            self.ax.set_ylim(-40, 40)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.fig.canvas.draw()
            return
            
        # 4. Disegno di tutte le altre occlusioni NON selezionate in grigio-bluastro (zorder=4)
        for idx, occ in enumerate(self.occlusions):
            if idx == self.current_occ_idx:
                continue
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) > 0:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                self.ax.plot(poly_closed[:, 1], poly_closed[:, 0], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                self.ax.fill(poly_closed[:, 1], poly_closed[:, 0], color='#1E293B', alpha=0.15, zorder=4)
                
        # 5. Dati dell'occlusione selezionata (neon)
        occ = self.occlusions[self.current_occ_idx]
        poly = np.array(occ.get('polygon_points_m', []))
        bbox = occ.get('occlusion_bbox_m', [0,0,0,0])
        name = occ.get('object_name', 'unknown')
        dist = occ.get('distance_m', 0.0)
        
        # Lettura diretta dal JSON pre-calcolato ed allineato a monte
        a_road = occ.get("road_fraction", 0.0)
        a_side = occ.get("sidewalk_fraction", 0.0)
        a_terr = occ.get("terrain_fraction", 0.0)
        modulated_risk = occ.get("risk_score", 0.0)

        # Carichiamo le stime UNet modulate
        neural_probs = occ.get('estimated_probabilities', {})
        max_n_val = max(neural_probs.values()) if neural_probs else 0.0
        
        # Codifica colore in base alla probabilità semantica massima (allineato)
        if max_n_val < 0.20:
            color = "#00E676"  # Verde Neon
        elif max_n_val < 0.50:
            color = "#FFD600"  # Giallo Neon
        else:
            color = "#FF1744"  # Rosso Neon
            
        if len(poly) > 0:
            poly_closed = np.vstack([poly, poly[0]])
            self.ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5, zorder=5)
            self.ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color, alpha=0.25, zorder=5)
            
        # Generiamo le barre di avanzamento Unicode per la UNet
        neural_lines = []
        neural_lines.append(f"Strada  : {a_road*100:>5.1f}%")
        neural_lines.append(f"Marc.   : {a_side*100:>5.1f}%")
        neural_lines.append(f"Prato   : {a_terr*100:>5.1f}%")
        neural_lines.append("-" * 25)
        for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]:
            v = neural_probs.get(k, 0.0)
            bar = self.get_unicode_bar(v)
            neural_lines.append(f"{k:<10}: {bar} {v*100:>5.1f}%")
        neural_box_str = "\n".join(neural_lines)
            
        rect = Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], 
                         linewidth=1.0, edgecolor=color, facecolor='none', linestyle=':', zorder=6)
        self.ax.add_patch(rect)
        
        self.ax.set_xlim(40, -40)
        self.ax.set_ylim(-40, 40)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        title_text = (
            f"AGENTE NEURALE ({self.modo_label}) - STIME DI RISCHIO UNet\n"
            f"FILE: {json_filename} [{self.current_file_idx+1}/{len(self.json_files)}] | "
            f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m) | Pesi: best_model.pth"
        )
        self.ax.set_title(title_text, color='white', fontsize=10, fontweight='bold', pad=15)
        
        # Box HUD per le probabilità in basso a sinistra
        self.ax.text(
            35, -35, neural_box_str, color='white', fontsize=8, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.8'), zorder=10
        )
        
        # Legenda delle categorie geometriche aggiornata con Strada al posto di Altre Ombre
        legend_elements = [
            Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.4, label='Basso Rischio (<20%)'),
            Patch(facecolor='#FFD600', edgecolor='#FFD600', alpha=0.4, label='Rischio Medio (20-50%)'),
            Patch(facecolor='#FF1744', edgecolor='#FF1744', alpha=0.4, label='Alto Rischio (>=50%)'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.2, label='Ego Vehicle'),
            Patch(facecolor='#101216', edgecolor='#3A3632', alpha=0.9, label='Strada (Grigio Scuro)'),
            Patch(facecolor='#22252C', edgecolor='#3A4250', alpha=0.8, label='Marciapiede (Grigio Chiaro)'),
            Patch(facecolor='#102216', edgecolor='#1C3824', alpha=0.7, label='Prato (Verde Scuro)')
        ]
        self.ax.legend(handles=legend_elements, loc="lower right", facecolor="#1E293B", edgecolor="gray", fontsize=7, labelcolor="white")
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
        """Gestisce il passaggio del mouse sopra i poligoni d'ombra per evidenziare l'occlusione."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        point = ShapelyPoint(event.ydata, event.xdata)
        
        for idx, occ in enumerate(self.occlusions):
            poly_pts = occ.get('polygon_points_m', [])
            if len(poly_pts) > 2:
                shp_poly = ShapelyPolygon(poly_pts)
                if shp_poly.contains(point):
                    if self.current_occ_idx != idx:
                        self.current_occ_idx = idx
                        self.plot_current()
                    break

if __name__ == "__main__":
    RuntimeNeuralOnlyVisualizer()
