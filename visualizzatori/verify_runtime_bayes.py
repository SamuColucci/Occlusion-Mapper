"""
Macro-Area 3A: Esecuzione e Visualizzazione dell'Agente Bayesiano a Runtime.

Questo script visualizza in modo interattivo le stime probabilistiche
calcolate dall'Agente Bayesiano classico (prior condizionate CSV + rischio temporale).
Non richiede l'installazione di PyTorch per essere eseguito.
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
        'walkway': [],
        'carpark_area': [],
        'ped_crossing': []
    }
    
    q = Quaternion(ego_pose['rotation'])
    R_inv = q.inverse.rotation_matrix
    interest_box = ShapelyBox(tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    for layer in ['drivable_area', 'walkway', 'carpark_area', 'ped_crossing']:
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

class RuntimeBayesVisualizer:
    def __init__(self, in_dir=None):
        if in_dir is None:
            self.in_dir = "extracted_occlusions_probabilities"
            self.modo_label = "UNIFICATO"
        else:
            self.in_dir = in_dir
            self.modo_label = "PERSONALIZZATO"
            
        print(f"\nCaricamento dati in modalità: {self.modo_label}")
        print("Inizializzazione NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        # Scansione dei file JSON probabilistici
        self.json_files = sorted(glob.glob(os.path.join(self.in_dir, "*.json")))
        if not self.json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{self.in_dir}'!")
            print("Assicurati di aver generato i dati batch tramite l'opzione 7 della dashboard.")
            return
            
        self.current_file_idx = 0
        self.current_occ_idx = 0
        self.pc = np.zeros((0, 3))
        
        self.load_json_file()
        
        # Configurazione Matplotlib: 2 colonne per separare rigorosamente HUD e Mappa
        self.fig, (self.ax_hud, self.ax) = plt.subplots(
            1, 2, figsize=(15, 9), facecolor='#0B0C10',
            gridspec_kw={'width_ratios': [1, 2.5]}
        )
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.08)
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
        self.ax_hud.clear()
        self.ax_hud.axis('off')
        
        # Sfondo verde prato scuro (terrain) come piano base
        bg_color = "#102216"
        self.ax.set_facecolor(bg_color)
        self.fig.patch.set_facecolor("#0B0C10")
        
        # 0. Disegno della Superficie Stradale e delle aree limitrofe
        self.road_polys_shapely = []
        self.sidewalk_polys_shapely = []
        self.carpark_polys_shapely = []
        self.crosswalk_polys_shapely = []
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
                    
                # 3. Parcheggio (Viola scuro carpark)
                for pts in surfaces['carpark_area']:
                    self.ax.fill(pts[:, 0], pts[:, 1], color='#251835', zorder=1, alpha=0.75)
                    self.ax.plot(pts[:, 0], pts[:, 1], color='#3F295B', linestyle=':', linewidth=0.6, zorder=1)
                    self.carpark_polys_shapely.append(ShapelyPolygon(pts))
                    
                # 4. Strisce/Attraversamenti (Blu-grigio ped_crossing)
                for pts in surfaces['ped_crossing']:
                    self.ax.fill(pts[:, 0], pts[:, 1], color='#182835', zorder=1, alpha=0.8)
                    self.ax.plot(pts[:, 0], pts[:, 1], color='#2B465D', linestyle='--', linewidth=0.8, zorder=1)
                    self.crosswalk_polys_shapely.append(ShapelyPolygon(pts))
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
                
        # 5. Dati dell'occlusione selezionata
        occ = self.occlusions[self.current_occ_idx]
        poly = np.array(occ.get('polygon_points_m', []))
        bbox = occ.get('occlusion_bbox_m', [0,0,0,0])
        name = occ.get('object_name', 'unknown')
        dist = occ.get('distance_m', 0.0)
        risk = occ.get('risk_score', 0.0)
        
        bayes_probs = occ.get('estimated_probabilities', {})
        max_b_val = max(bayes_probs.values()) if bayes_probs else 0.0
        
        # Codifica colore bordo globale in base alla probabilità massima
        if max_b_val < 0.20:
            color = "#00E676"  # Verde Neon
        elif max_b_val < 0.50:
            color = "#FFD600"  # Giallo Neon
        else:
            color = "#FF1744"  # Rosso Neon

        # Colori per sub-zona
        SURF_COLORS = {
            "driveable_surface": ("#607D8B", "#37474F"),   # Grigio asfalto
            "sidewalk":          ("#8D6E63", "#5D4037"),   # Marrone
            "other_flat":        ("#7B1FA2", "#4A148C"),   # Viola
            "ped_crossing":      ("#0288D1", "#01579B"),   # Blu
            "terrain":           ("#388E3C", "#1B5E20"),   # Verde prato
        }
        SURF_LABELS = {
            "driveable_surface": "Superficie Carrabile", 
            "sidewalk": "Marciapiede", 
            "other_flat": "Altro Piano", 
            "ped_crossing": "Strisce Pedonali", 
            "terrain": "Terreno"
        }

        sub_zones = occ.get('sub_zones', [])
        
        # Disegna outline del poligono intero
        if len(poly) > 0:
            poly_closed = np.vstack([poly, poly[0]])
            self.ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5, zorder=5, alpha=0.6)
        
        # Disegna le sub-zone con colori distinti
        for sz in sub_zones:
            surf = sz.get('surface', 'terrain')
            fill_c, border_c = SURF_COLORS.get(surf, ("#555", "#333"))
            sz_pts = np.array(sz.get('polygon_points_m', []))
            if len(sz_pts) >= 3:
                sz_closed = np.vstack([sz_pts, sz_pts[0]])
                self.ax.fill(sz_closed[:, 1], sz_closed[:, 0], color=fill_c, alpha=0.45, zorder=5)
                self.ax.plot(sz_closed[:, 1], sz_closed[:, 0], color=border_c, linewidth=1.5, zorder=5)
                # Etichetta al centroide della sub-zona
                cx = float(np.mean(sz_pts[:, 1]))
                cy = float(np.mean(sz_pts[:, 0]))
                frac_lbl = min(sz.get('area_fraction', 0.0), 1.0) * 100
                self.ax.text(cx, cy,
                    f"{SURF_LABELS.get(surf, surf)}: {frac_lbl:.0f}%",
                    color='white', fontsize=7.0, ha='center', va='center',
                    bbox=dict(facecolor=fill_c, alpha=0.75, edgecolor=border_c, boxstyle='round,pad=0.3'),
                    zorder=9)

        rect = Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],
                         linewidth=1.0, edgecolor=color, facecolor='none', linestyle=':', zorder=6)
        self.ax.add_patch(rect)

        # --- Pannello HUD ---
        bayes_lines = []
        a_road = occ.get('road_fraction', 0.0)
        a_side = occ.get('sidewalk_fraction', 0.0)
        a_carpark = occ.get('carpark_fraction', 0.0)
        a_crosswalk = occ.get('crosswalk_fraction', 0.0)
        a_terr = occ.get('terrain_fraction', 0.0)
        
        bayes_lines.append(f"Superficie Carrabile: {a_road*100:>5.1f}%")
        bayes_lines.append(f"Marciapiede         : {a_side*100:>5.1f}%")
        bayes_lines.append(f"Altro Piano         : {a_carpark*100:>5.1f}%")
        bayes_lines.append(f"Strisce Pedonali    : {a_crosswalk*100:>5.1f}%")
        bayes_lines.append(f"Terreno             : {a_terr*100:>5.1f}%")
        bayes_lines.append("=" * 30)
        bayes_lines.append(" PROB GLOBALI (media pesata area)")
        bayes_lines.append("-" * 30)
        
        # Mostriamo sempre le 4 categorie principali
        for k in ["Auto", "Pedone", "Camion", "Bicicletta"]:
            v = bayes_probs.get(k, 0.0)
            bar = self.get_unicode_bar(v)
            bayes_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
            
        # Mostriamo le categorie secondarie solo se significative (prob > 0.5%)
        sec_cats = ["Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]
        for k in sec_cats:
            v = bayes_probs.get(k, 0.0)
            if v > 0.005:
                bar = self.get_unicode_bar(v)
                bayes_lines.append(f"{k:<11}: {bar} {v*100:>5.1f}%")
                
        # Sintesi testuale dell'oggetto più probabile
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

        # Dettaglio per sub-zona
        if sub_zones:
            bayes_lines.append("=" * 30)
            bayes_lines.append(" PROB PER SUB-ZONA")
            for sz in sub_zones:
                surf = sz.get('surface', '?')
                sz_probs = sz.get('estimated_probabilities', {})
                frac = min(sz.get('area_fraction', 0.0), 1.0)
                w = sz.get('occlusion_width_m', 0.0)
                bayes_lines.append(f"--- {SURF_LABELS.get(surf, surf)} ({frac*100:.0f}% area, W={w:.1f}m) ---")
                
                # Mostra categorie principali + categorie secondarie se superiori a 1%
                for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera", "Cono", "Altro"]:
                    v = sz_probs.get(k, 0.0)
                    is_main = k in ["Auto", "Pedone", "Camion", "Bicicletta"]
                    if is_main or v > 0.01:
                        bar = self.get_unicode_bar(v)
                        bayes_lines.append(f"  {k:<9}: {bar} {v*100:>5.1f}%")

        bayes_box_str = "\n".join(bayes_lines)
        
        self.ax.set_xlim(40, -40)
        self.ax.set_ylim(-40, 40)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        title_text = (
            f"AGENTE BAYESIANO DINAMICO ({self.modo_label}) - RISULTATI STIMATI\n"
            f"FILE: {json_filename} [{self.current_file_idx+1}/{len(self.json_files)}] | "
            f"Sorgente: {name.split('.')[-1].upper()} ({dist:.1f}m)"
        )
        self.ax.set_title(title_text, color='white', fontsize=10, fontweight='bold', pad=15)
        
        # Box HUD disegnato all'interno del subplot dedicato a sinistra (ax_hud)
        # Garantisce al 100% che non possa mai sovrapporsi alla mappa a destra
        self.ax_hud.text(
            0.05, 0.95, bayes_box_str, color='white', fontsize=7.5, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.90, edgecolor=color, boxstyle='round,pad=0.8'),
            va='top', ha='left', transform=self.ax_hud.transAxes
        )
        
        # Legenda aggiornata
        legend_elements = [
            Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.4, label='Stima Bassa (<20%)'),
            Patch(facecolor='#FFD600', edgecolor='#FFD600', alpha=0.4, label='Stima Media (20-50%)'),
            Patch(facecolor='#FF1744', edgecolor='#FF1744', alpha=0.4, label='Stima Alta (>=50%)'),
            Patch(facecolor='#607D8B', edgecolor='#37474F', alpha=0.5, label='Sub-zona driveable_surface (Strada)'),
            Patch(facecolor='#8D6E63', edgecolor='#5D4037', alpha=0.5, label='Sub-zona sidewalk (Marciapiede)'),
            Patch(facecolor='#7B1FA2', edgecolor='#4A148C', alpha=0.5, label='Sub-zona other_flat (Parcheggio)'),
            Patch(facecolor='#0288D1', edgecolor='#01579B', alpha=0.5, label='Sub-zona ped_crossing (Strisce)'),
            Patch(facecolor='#388E3C', edgecolor='#1B5E20', alpha=0.5, label='Sub-zona terrain (Prato)'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.2, label='Ego Vehicle'),
        ]
        self.ax.legend(handles=legend_elements, loc="lower right", facecolor="#1E293B",
                       edgecolor="gray", fontsize=7, labelcolor="white")
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
    RuntimeBayesVisualizer()
