"""
Script di Confronto Comparativo Sincronizzato (Solo Lettura Risultati).

Questo visualizzatore affianca le predizioni pre-calcolate dell'Agente Bayesiano (sinistra)
e dell'Agente Neurale (destra) a partire dai JSON già salvati su disco.
Stampa inoltre in tempo reale sul terminale la tabella delle differenze (Delta) per ciascun frame.
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

class SincronizedComparisonVisualizer:
    def __init__(self, bayes_dir="extracted_occlusions_probabilities", bayes_near_dir="extracted_occlusions_probabilities_near",
                 neural_dir="extracted_occlusions_neural", neural_near_dir="extracted_occlusions_neural_near"):
        self.bayes_dir = bayes_dir
        self.bayes_near_dir = bayes_near_dir if os.path.exists(bayes_near_dir) else bayes_dir
        self.neural_dir = neural_dir
        self.neural_near_dir = neural_near_dir if os.path.exists(neural_near_dir) else neural_dir
        
        print("Inizializzazione NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        # Recuperiamo le liste dei file JSON
        self.bayes_files = sorted(glob.glob(os.path.join(self.bayes_dir, "*.json")))
        self.bayes_near_files = sorted(glob.glob(os.path.join(self.bayes_near_dir, "*.json")))
        self.neural_files = sorted(glob.glob(os.path.join(self.neural_dir, "*.json")))
        self.neural_near_files = sorted(glob.glob(os.path.join(self.neural_near_dir, "*.json")))
        
        if not self.bayes_files or not self.neural_files:
            print("[ERROR] Impossibile avviare il confronto! Assicurati di aver generato i dataset di base.")
            return
            
        # Sincronizziamo i file basandoci sui nomi dei file in comune a tutte e 4 le cartelle
        set_b = set(os.path.basename(f) for f in self.bayes_files)
        set_bn = set(os.path.basename(f) for f in self.bayes_near_files)
        set_n = set(os.path.basename(f) for f in self.neural_files)
        set_nn = set(os.path.basename(f) for f in self.neural_near_files)
        
        self.common_filenames = sorted(list(
            set_b.intersection(set_bn).intersection(set_n).intersection(set_nn)
        ))
        
        if not self.common_filenames:
            print("[ERROR] Nessun file JSON in comune trovato tra le 4 cartelle!")
            return
            
        self.current_file_idx = 0
        self.current_occ_idx = 0
        self.pc = np.zeros((0, 3))
        
        self.load_frame_data()
        
        # Setup grafico a due colonne
        self.fig, (self.ax_bayes, self.ax_neural) = plt.subplots(1, 2, figsize=(16, 9), facecolor='black')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        print("\n" + "=" * 50)
        print("          CONTROLLI INTERATTIVI REGISTRATI")
        print("=" * 50)
        print(" -> MOUSE            : Passa sopra un'ombra per evidenziarla")
        print(" -> FRECCIA SU/GIÙ   : Cambia frame all'unisono")
        print(" -> FRECCIA DX/SX    : Scorri manualmente le occlusioni")
        print("=" * 50 + "\n")
        
        self.plot_current()
        plt.show()

    def print_frame_differences(self):
        """Stampa a terminale una tabella comparativa dettagliata per ogni classe di ogni occlusione."""
        filename = self.common_filenames[self.current_file_idx]
        
        print("\n" + "=" * 115)
        print(f"[CONFRONTO COMPLETO COMPILATO DI DIFFERENZE FRAME {self.current_file_idx+1}/{len(self.common_filenames)}]: {filename}")
        print("=" * 115)
        print(f"{'Occlusione':<12} | {'Classe':<8} | {'Confronto DIRETTO (Bayes vs UNet)':<42} | {'Confronto NEAR-MISS (Bayes vs UNet)'}")
        print("-" * 115)
        
        for idx, occ_b in enumerate(self.occlusions_bayes):
            occ_b_near = self.occlusions_bayes_near[idx] if idx < len(self.occlusions_bayes_near) else {}
            occ_n = self.occlusions_neural[idx] if idx < len(self.occlusions_neural) else {}
            occ_n_near = self.occlusions_neural_near[idx] if idx < len(self.occlusions_neural_near) else {}
            
            bayes_dir = occ_b.get('estimated_probabilities', {})
            bayes_near = occ_b_near.get('estimated_probabilities', {})
            neural_dir = occ_n.get('estimated_probabilities', {})
            neural_near = occ_n_near.get('estimated_probabilities', {})
            
            for cls in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]:
                pb_d = bayes_dir.get(cls, 0.0)
                pb_n = bayes_near.get(cls, 0.0)
                pn_d = neural_dir.get(cls, 0.0)
                pn_n = neural_near.get(cls, 0.0)
                
                delta_d = pb_d - pn_d
                delta_n = pb_n - pn_n
                
                delta_d_str = f"{delta_d*100:+.1f}%"
                delta_n_str = f"{delta_n*100:+.1f}%"
                
                occ_label = f"{idx+1}/{len(self.occlusions_bayes)}"
                print(f"{occ_label:<12} | {cls[:8]:<8} | Bayes: {pb_d*100:>5.1f}% vs UNet: {pn_d*100:>5.1f}% (Diff: {delta_d_str:<6}) | Bayes: {pb_n*100:>5.1f}% vs UNet: {pn_n*100:>5.1f}% (Diff: {delta_n_str:<6})")
            print("-" * 115)
            
        print("=" * 115 + "\n")

    def load_frame_data(self):
        """Carica i dati dei file JSON sincronizzati per il frame corrente."""
        filename = self.common_filenames[self.current_file_idx]
        
        # Carica JSON Bayes Diretto
        with open(os.path.join(self.bayes_dir, filename), "r") as f:
            self.data_bayes = json.load(f)
            
        # Carica JSON Bayes Near-Miss
        with open(os.path.join(self.bayes_near_dir, filename), "r") as f:
            self.data_bayes_near = json.load(f)
            
        # Carica JSON Neurale Diretto
        with open(os.path.join(self.neural_dir, filename), "r") as f:
            self.data_neural = json.load(f)
            
        # Carica JSON Neurale Near-Miss
        with open(os.path.join(self.neural_near_dir, filename), "r") as f:
            self.data_neural_near = json.load(f)
            
        self.occlusions_bayes = self.data_bayes.get("occlusions", [])
        self.occlusions_bayes_near = self.data_bayes_near.get("occlusions", [])
        self.occlusions_neural = self.data_neural.get("occlusions", [])
        self.occlusions_neural_near = self.data_neural_near.get("occlusions", [])
        self.current_occ_idx = 0
        
        self.ego_pose = None
        self.sample_record = None
        
        # Stampa a console immediata delle differenze
        self.print_frame_differences()
        
        # Carica LiDAR (in comune)
        lidar_token = self.data_bayes.get("lidar_token")
        if lidar_token:
            try:
                pcl_path = self.nusc.get_sample_data_path(lidar_token)
                self.pc = np.fromfile(pcl_path, dtype=np.float32).reshape((-1, 5))[:, :3]
                
                sd_record = self.nusc.get('sample_data', lidar_token)
                self.ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                self.sample_record = self.nusc.get('sample', sd_record['sample_token'])
            except Exception:
                self.pc = np.zeros((0, 3))
        else:
            self.pc = np.zeros((0, 3))

    def get_unicode_bar(self, val):
        """Genera una barra di progresso orizzontale in formato Unicode."""
        filled_length = int(round(val * 10))
        return "█" * filled_length + "░" * (10 - filled_length)

    def plot_current(self):
        self.ax_bayes.clear()
        self.ax_neural.clear()
        
        # Sfondo verde prato scuro (terrain) come piano base
        bg_color = "#102216"
        self.ax_bayes.set_facecolor(bg_color)
        self.ax_neural.set_facecolor(bg_color)
        self.fig.patch.set_facecolor("#0B0C10")
        
        # 0. Disegno della Superficie Stradale (Drivable Area)
        self.road_polygons_shapely = []
        if self.ego_pose and self.sample_record:
            try:
                scene = self.nusc.get('scene', self.sample_record['scene_token'])
                log = self.nusc.get('log', scene['log_token'])
                map_name = log['location']
                nusc_map = get_map_instance('./nuscenes', map_name)
                
                self.road_polys_shapely = []
                self.sidewalk_polys_shapely = []
                
                surfaces = get_semantic_surfaces_local(
                    self.nusc, nusc_map, self.sample_record['token'], self.ego_pose, range_m=50
                )
                
                from shapely.geometry import Polygon as ShapelyPolygon
                
                # 1. Strada (Asfalto grigio scuro)
                for pts in surfaces['drivable_area']:
                    self.ax_bayes.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
                    self.ax_neural.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
                    self.ax_bayes.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
                    self.ax_neural.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
                    self.road_polys_shapely.append(ShapelyPolygon(pts))
                    
                # 2. Marciapiede (Grigio walkway)
                for pts in surfaces['walkway']:
                    self.ax_bayes.fill(pts[:, 0], pts[:, 1], color='#22252C', zorder=1, alpha=0.8)
                    self.ax_neural.fill(pts[:, 0], pts[:, 1], color='#22252C', zorder=1, alpha=0.8)
                    self.ax_bayes.plot(pts[:, 0], pts[:, 1], color='#3A4250', linestyle=':', linewidth=0.6, zorder=1)
                    self.ax_neural.plot(pts[:, 0], pts[:, 1], color='#3A4250', linestyle=':', linewidth=0.6, zorder=1)
                    self.sidewalk_polys_shapely.append(ShapelyPolygon(pts))
            except Exception:
                pass
        
        # 1. Disegno dei Radar Rings (Cerchi Concentrici) in entrambi i subplot
        for radius in [10, 20, 30, 40]:
            for ax in [self.ax_bayes, self.ax_neural]:
                circle = plt.Circle((0, 0), radius, color='#2C374E', fill=False, linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
                ax.add_patch(circle)
                # Etichetta sul raggio verticale
                ax.text(0.5, radius, f"{radius}m", color='#475569', fontsize=7, ha='left', va='center', fontweight='bold', zorder=2)
        
        # 2. Disegno del LiDAR di sfondo (Sonar-Style grigio-azzurro soffuso) (zorder=3 per stare sopra la strada)
        if len(self.pc) > 0:
            self.ax_bayes.scatter(self.pc[:, 1], self.pc[:, 0], s=0.12, c='#4A5E7D', alpha=0.25, zorder=3)
            self.ax_neural.scatter(self.pc[:, 1], self.pc[:, 0], s=0.12, c='#4A5E7D', alpha=0.25, zorder=3)
            
        # 3. Disegno dell'Ego-Vehicle stilizzato a centro (0, 0) (zorder=6 per stare sopra tutto)
        # Larghezza: 1.8m (asse X del grafico), Lunghezza: 4.0m (asse Y del grafico)
        for ax in [self.ax_bayes, self.ax_neural]:
            ego_rect = Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=1.5, edgecolor='#00E5FF', facecolor='#00E5FF', alpha=0.2, zorder=6)
            ax.add_patch(ego_rect)
            ax.plot(0, 0, color='#00E5FF', marker='^', markersize=4, zorder=6) # Freccia di heading
        
        filename = self.common_filenames[self.current_file_idx]
        
        if not self.occlusions_bayes:
            empty_t = f"FILE: {filename}\nNessuna occlusione registrata."
            self.ax_bayes.set_title(empty_t, color='white', fontsize=10)
            self.ax_neural.set_title(empty_t, color='white', fontsize=10)
            self.fig.canvas.draw()
            return
            
        # 4. Disegno di tutte le occlusioni NON selezionate a sinistra (Bayes) in modalità "Sfondo" (zorder=4)
        for idx, occ in enumerate(self.occlusions_bayes):
            if idx == self.current_occ_idx:
                continue
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) > 0:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                # Disegniamo con bordo sottile grigio scuro e riempimento minimo
                self.ax_bayes.plot(poly_closed[:, 1], poly_closed[:, 0], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                self.ax_bayes.fill(poly_closed[:, 1], poly_closed[:, 0], color='#1E293B', alpha=0.15, zorder=4)

        # 5. Disegno di tutte le occlusioni NON selezionate a destra (Neurale) in modalità "Sfondo" (zorder=4)
        for idx, occ in enumerate(self.occlusions_neural):
            if idx == self.current_occ_idx:
                continue
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) > 0:
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                self.ax_neural.plot(poly_closed[:, 1], poly_closed[:, 0], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                self.ax_neural.fill(poly_closed[:, 1], poly_closed[:, 0], color='#1E293B', alpha=0.15, zorder=4)

        # 6. Dati e disegno dell'occlusione selezionata (neon e HUD)
        occ_b = self.occlusions_bayes[self.current_occ_idx]
        occ_b_near = self.occlusions_bayes_near[self.current_occ_idx] if self.current_occ_idx < len(self.occlusions_bayes_near) else {}
        occ_n = self.occlusions_neural[self.current_occ_idx] if self.current_occ_idx < len(self.occlusions_neural) else {}
        occ_n_near = self.occlusions_neural_near[self.current_occ_idx] if self.current_occ_idx < len(self.occlusions_neural_near) else {}
        
        poly = np.array(occ_b.get('polygon_points_m', []))
        bbox = occ_b.get('occlusion_bbox_m', [0,0,0,0])
        name = occ_b.get('object_name', 'unknown')
        dist = occ_b.get('distance_m', 0.0)
        
        # Lettura diretta dal JSON pre-calcolato ed allineato a monte
        a_road = occ_b.get("road_fraction", 0.0)
        a_side = occ_b.get("sidewalk_fraction", 0.0)
        a_terr = occ_b.get("terrain_fraction", 0.0)
        modulated_risk = occ_b.get("risk_score", 0.0)

        # =====================================================================
        # SUBPLOT SINISTRA: BAYESIANO CLASSICO (EVIDENZIATO)
        # =====================================================================
        bayes_probs_dir = occ_b.get('estimated_probabilities', {})
        bayes_probs_near = occ_b_near.get('estimated_probabilities', {})
        
        max_b_val = max(list(bayes_probs_dir.values()) + list(bayes_probs_near.values())) if (bayes_probs_dir or bayes_probs_near) else 0.0
        
        if max_b_val < 0.20:
            b_color = "#00E676"  # Verde Neon (Basso rischio)
        elif max_b_val < 0.50:
            b_color = "#FFD600"  # Giallo Neon (Rischio medio)
        else:
            b_color = "#FF1744"  # Rosso Neon (Alto rischio)
            
        if len(poly) > 0:
            poly_closed = np.vstack([poly, poly[0]])
            self.ax_bayes.plot(poly_closed[:, 1], poly_closed[:, 0], color=b_color, linewidth=2.5, zorder=5)
            self.ax_bayes.fill(poly_closed[:, 1], poly_closed[:, 0], color=b_color, alpha=0.25, zorder=5)
            
        # Generiamo le linee testuali affiancate per Bayes con frazioni di superficie
        bayes_lines = []
        bayes_lines.append(f"Strada  : {a_road*100:>5.1f}%")
        bayes_lines.append(f"Marc.   : {a_side*100:>5.1f}%")
        bayes_lines.append(f"Prato   : {a_terr*100:>5.1f}%")
        bayes_lines.append("-" * 34)
        for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]:
            vd = bayes_probs_dir.get(k, 0.0)
            vn = bayes_probs_near.get(k, 0.0)
            vf = vd + vn - vd * vn
            bayes_lines.append(f"{k[:10]:<10} | Dir: {vd*100:>4.1f}% | Near: {vn*100:>4.1f}% | Fin: {vf*100:>4.1f}%")
        bayes_box_str = "\n".join(bayes_lines)
        
        rect_b = Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], 
                           linewidth=1.0, edgecolor=b_color, facecolor='none', linestyle=':', zorder=6)
        self.ax_bayes.add_patch(rect_b)
        
        self.ax_bayes.set_xlim(40, -40)
        self.ax_bayes.set_ylim(-40, 40)
        self.ax_bayes.set_aspect('equal')
        self.ax_bayes.axis('off')
        
        # Aggiunta della legenda HUD ad alto contrasto
        legend_elements = [
            Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.4, label='Basso Rischio (<20%)'),
            Patch(facecolor='#FFD600', edgecolor='#FFD600', alpha=0.4, label='Rischio Medio (20-50%)'),
            Patch(facecolor='#FF1744', edgecolor='#FF1744', alpha=0.4, label='Alto Rischio (>=50%)'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.2, label='Ego Vehicle'),
            Patch(facecolor='#101216', edgecolor='#3A3632', alpha=0.9, label='Strada (Grigio Scuro)'),
            Patch(facecolor='#22252C', edgecolor='#3A4250', alpha=0.8, label='Marciapiede (Grigio Chiaro)'),
            Patch(facecolor='#102216', edgecolor='#1C3824', alpha=0.7, label='Prato (Verde Scuro)')
        ]
        self.ax_bayes.legend(handles=legend_elements, loc='upper right', facecolor='#1E293B', edgecolor='gray', fontsize=7, labelcolor='white')
        
        # Titolo e Box informativo sovrapposto (Stile HUD)
        self.ax_bayes.set_title(
            f"1. AGENTE BAYESIANO (STATISTICO)\n"
            f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m) | Rischio Modulato: {modulated_risk:.2f}",
            color='white', fontsize=10, fontweight='bold', pad=12, loc='left'
        )
        # Box HUD per le probabilità in basso a sinistra
        self.ax_bayes.text(
            35, -35, bayes_box_str, color='white', fontsize=8, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.8, edgecolor=b_color, boxstyle='round,pad=0.8'), zorder=10
        )
        
        # =====================================================================
        # SUBPLOT DESTRA: NEURALE (UNet - EVIDENZIATO)
        # =====================================================================
        neural_probs_dir = occ_n.get('estimated_probabilities', {})
        neural_probs_near = occ_n_near.get('estimated_probabilities', {})
        
        max_n_val = max(list(neural_probs_dir.values()) + list(neural_probs_near.values())) if (neural_probs_dir or neural_probs_near) else 0.0
        
        if max_n_val < 0.20:
            n_color = "#00E676"  # Verde Neon (Basso rischio)
        elif max_n_val < 0.50:
            n_color = "#FFD600"  # Giallo Neon (Rischio medio)
        else:
            n_color = "#FF1744"  # Rosso Neon (Alto rischio)
            
        if len(poly) > 0:
            self.ax_neural.plot(poly_closed[:, 1], poly_closed[:, 0], color=n_color, linewidth=2.5, zorder=5)
            self.ax_neural.fill(poly_closed[:, 1], poly_closed[:, 0], color=n_color, alpha=0.25, zorder=5)
            
        # Generiamo le linee testuali affiancate per la UNet con frazioni di superficie
        neural_lines = []
        neural_lines.append(f"Strada  : {a_road*100:>5.1f}%")
        neural_lines.append(f"Marc.   : {a_side*100:>5.1f}%")
        neural_lines.append(f"Prato   : {a_terr*100:>5.1f}%")
        neural_lines.append("-" * 34)
        for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Altro"]:
            vd = neural_probs_dir.get(k, 0.0)
            vn = neural_probs_near.get(k, 0.0)
            vf = vd + vn - vd * vn
            neural_lines.append(f"{k[:10]:<10} | Dir: {vd*100:>4.1f}% | Near: {vn*100:>4.1f}% | Fin: {vf*100:>4.1f}%")
        neural_box_str = "\n".join(neural_lines)
        
        rect_n = Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], 
                           linewidth=1.0, edgecolor=n_color, facecolor='none', linestyle=':', zorder=6)
        self.ax_neural.add_patch(rect_n)
        
        self.ax_neural.set_xlim(40, -40)
        self.ax_neural.set_ylim(-40, 40)
        self.ax_neural.set_aspect('equal')
        self.ax_neural.axis('off')
        
        # Aggiunta della legenda HUD ad alto contrasto anche a destra
        self.ax_neural.legend(handles=legend_elements, loc='upper right', facecolor='#1E293B', edgecolor='gray', fontsize=7, labelcolor='white')
        
        self.ax_neural.set_title(
            f"2. AGENTE NEURALE (UNet)\n"
            f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m) | Pesi: best_model.pth", 
            color='white', fontsize=10, fontweight='bold', pad=12, loc='left'
        )
        # Box HUD per le probabilità in basso a sinistra
        self.ax_neural.text(
            35, -35, neural_box_str, color='white', fontsize=8, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.8, edgecolor=n_color, boxstyle='round,pad=0.8'), zorder=10
        )
        
        # Titolo superiore sincronizzato
        self.fig.suptitle(
            f"CONFRONTO AGENTI DI STIMA OCCLUSIONI BEV (DIRETTO vs NEAR-MISS)\n"
            f"Frame: {filename} [{self.current_file_idx+1}/{len(self.common_filenames)}] | Occlusione: {self.current_occ_idx+1}/{len(self.occlusions_bayes)}",
            color='white', fontsize=12, fontweight='bold', y=0.97
        )
        
        self.fig.canvas.draw()

    def on_key(self, event):
        # Scorrimento occlusioni
        if event.key == 'right':
            if self.occlusions_bayes:
                self.current_occ_idx = (self.current_occ_idx + 1) % len(self.occlusions_bayes)
                self.plot_current()
        elif event.key == 'left':
            if self.occlusions_bayes:
                self.current_occ_idx = (self.current_occ_idx - 1) % len(self.occlusions_bayes)
                self.plot_current()
                
        # Scorrimento frame
        elif event.key == 'down':
            if len(self.common_filenames) > 1:
                self.current_file_idx = (self.current_file_idx + 1) % len(self.common_filenames)
                self.load_frame_data()
                self.plot_current()
        elif event.key == 'up':
            if len(self.common_filenames) > 1:
                self.current_file_idx = (self.current_file_idx - 1) % len(self.common_filenames)
                self.load_frame_data()
                self.plot_current()

    def on_mouse_move(self, event):
        """Gestisce il passaggio del mouse sopra i poligoni d'ombra per evidenziare l'occlusione."""
        if event.inaxes not in [self.ax_bayes, self.ax_neural]:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        # Ricorda: l'asse X del grafico corrisponde a Y di NuScenes (laterale).
        # L'asse Y del grafico corrisponde a X di NuScenes (longitudinale).
        # Creiamo il punto in coordinate NuScenes metriche (X_nusc, Y_nusc) -> (ydata, xdata)
        point = ShapelyPoint(event.ydata, event.xdata)
        
        for idx, occ in enumerate(self.occlusions_bayes):
            poly_pts = occ.get('polygon_points_m', [])
            if len(poly_pts) > 2:
                # Creiamo il poligono Shapely per eseguire il test di collisione
                shp_poly = ShapelyPolygon(poly_pts)
                if shp_poly.contains(point):
                    if self.current_occ_idx != idx:
                        self.current_occ_idx = idx
                        self.plot_current()
                    break

if __name__ == "__main__":
    SincronizedComparisonVisualizer()
