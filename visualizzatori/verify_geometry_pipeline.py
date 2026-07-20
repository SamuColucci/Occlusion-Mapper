"""
Macro-Area 1: Validazione Geometrica e Integrità del Dataset.

Questo script unificato esegue i controlli geometrici sulle zone occluse.
Supporta due modalità:
  - --mode check  : scansione automatica di integrità su tutto il dataset (ex verify_dataset.py)
  - --mode visual : visualizzatore BEV interattivo della geometria (ex validate_json_occlusion.py)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from matplotlib.path import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

MAP_CACHE = {}

def get_map_instance(dataroot, map_name):
    if map_name not in MAP_CACHE:
        print(f"Caricamento mappa NuScenes '{map_name}' in cache...")
        MAP_CACHE[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return MAP_CACHE[map_name]

def get_drivable_area_local(nusc, nusc_map, sample_token, ego_pose, range_m=50):
    from shapely.geometry import box as ShapelyBox, Polygon as ShapelyPolygon, MultiPolygon
    tx, ty, tz = ego_pose['translation']
    box_coords = (tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    try:
        records = nusc_map.get_records_in_patch(box_coords, layer_names=['drivable_area'], mode='intersect')
        tokens = records.get('drivable_area', [])
    except Exception:
        return []
        
    q = Quaternion(ego_pose['rotation'])
    R_inv = q.inverse.rotation_matrix
    
    interest_box = ShapelyBox(tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    local_polygons = []
    
    for token in tokens:
        try:
            record = nusc_map.get('drivable_area', token)
            polygon_tokens = record.get('polygon_tokens', [])
            for poly_token in polygon_tokens:
                poly = nusc_map.extract_polygon(poly_token)
                
                # Intersezione locale per ritagliare solo la strada vicina
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
                    
                    # Coordinate 3D globali
                    pts_global_3d = np.column_stack([pts_global[:, 0], pts_global[:, 1], np.full(len(pts_global), tz)])
                    
                    # Traslazione e rotazione inversa con quaternione
                    pts_diff = pts_global_3d - np.array([tx, ty, tz])
                    pts_local = pts_diff @ R_inv.T
                    
                    # x_local (laterale) -> colonna 0, y_local (longitudinale) -> colonna 1
                    local_pts = np.column_stack([pts_local[:, 0], pts_local[:, 1]])
                    local_polygons.append(local_pts)
        except Exception:
            continue
            
    return local_polygons

# =====================================================================
# 1. CODICE DI INTEGRITY CHECK AUTOMATICO
# =====================================================================
def run_integrity_check(nusc, json_files):
    print(f"Avvio verifica di integrità automatica su {len(json_files)} file JSON...")
    total_polygons = 0
    self_intersections = 0
    known_zone_violations = 0
    errors = 0

    for idx, fpath in enumerate(json_files):
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Errore di parsing JSON nel file {fpath}: {e}")
                errors += 1
                continue
                
        lidar_token = data.get("lidar_token")
        if not lidar_token:
            continue
            
        try:
            pcl_path = nusc.get_sample_data_path(lidar_token)
            pc = np.fromfile(pcl_path, dtype=np.float32).reshape((-1, 5))[:, :3]
        except Exception:
            pc = np.zeros((0, 3))
            
        occlusions = data.get("occlusions", [])
        for occ in occlusions:
            name = occ.get("object_name", "")
            token = occ.get("object_token", "")
            if "static" in name.lower() or "static" in token.lower():
                continue
                
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue
                
            total_polygons += 1
            poly_np = np.array(poly_pts)
            
            # Controllo auto-intersezioni Shapely
            try:
                sh_poly = ShapelyPolygon(poly_np)
                if not sh_poly.is_valid:
                    self_intersections += 1
            except Exception:
                self_intersections += 1

            # Controllo sovrapposizione zone note libere
            if len(pc) > 0:
                dist_m = occ["distance_m"]
                pt_dists = np.linalg.norm(pc[:, :2], axis=1)
                m_road = (pc[:, 2] > -1.6) & (pc[:, 2] < 0.2) & (pt_dists > (dist_m + 3.0))
                road_pts = pc[m_road, :2]
                
                if len(road_pts) > 0:
                    path = Path(poly_np)
                    inside = path.contains_points(road_pts)
                    num_inside = np.sum(inside)
                    if num_inside > 15:
                        known_zone_violations += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(json_files):
            print(f"  Verificati {idx + 1}/{len(json_files)} file...")

    print("\n" + "="*50)
    print("RESULTS INTEGRITY CHECK:")
    print(f"File totali verificati: {len(json_files)}")
    print(f"Poligoni estratti:      {total_polygons}")
    print(f"Auto-intersezioni:      {self_intersections}")
    print(f"Violazioni zone note:   {known_zone_violations}")
    print(f"Errori di parsing:      {errors}")
    print("="*50)

# =====================================================================
# 2. CODICE DI VISUALIZZAZIONE INTERATTIVA
# =====================================================================
class GeometryVisualizer:
    def __init__(self, nusc, json_files):
        self.nusc = nusc
        self.json_files = json_files
        self.current_file_idx = 0
        self.current_occ_idx = 0
        self.pc = np.zeros((0, 3))
        
        self.load_json_file()
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='#0B0C10')
        self.ax.set_facecolor('#0B0C10')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI REGISTRATI (GEOMETRIA)")
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
                
                # Recuperiamo ego pose
                sd_record = self.nusc.get('sample_data', lidar_token)
                self.ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                self.sample_record = self.nusc.get('sample', sd_record['sample_token'])
            except Exception:
                self.pc = np.zeros((0, 3))

    def plot_current(self):
        self.ax.clear()
        
        # 0. Disegno della Superficie Stradale (Drivable Area)
        if self.ego_pose and self.sample_record:
            try:
                scene = self.nusc.get('scene', self.sample_record['scene_token'])
                log = self.nusc.get('log', scene['log_token'])
                map_name = log['location']
                nusc_map = get_map_instance('./nuscenes', map_name)
                
                local_road_polys = get_drivable_area_local(
                    self.nusc, nusc_map, self.sample_record['token'], self.ego_pose, range_m=50
                )
                for pts in local_road_polys:
                    # Disegniamo l'asfalto in grigio fumo caldissimo/scuro neutro
                    self.ax.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
                    # Disegniamo i contorni / bordi stradali come linee tratteggiate discrete
                    self.ax.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
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
                
        # 5. Dati dell'occlusione selezionata (neon) (zorder=5)
        occ = self.occlusions[self.current_occ_idx]
        poly = np.array(occ.get('polygon_points_m', []))
        bbox = occ.get('occlusion_bbox_m', [0,0,0,0])
        name = occ.get('object_name', 'unknown')
        dist = occ.get('distance_m', 0.0)
        
        # Codifica dei colori per categoria
        color_map = {"car": "#00f0ff", "truck": "#00f0ff", "bus": "#00f0ff", "pedestrian": "#22c55e", "static": "#ec4899"}
        color = "#ffffff"
        for key, c_val in color_map.items():
            if key in name.lower():
                color = c_val
                break
                
        if len(poly) > 0:
            poly_closed = np.vstack([poly, poly[0]])
            self.ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5, zorder=5)
            self.ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color, alpha=0.25, zorder=5)
            
        rect = Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], 
                         linewidth=1.0, edgecolor=color, facecolor='none', linestyle=':', zorder=6)
        self.ax.add_patch(rect)
        
        self.ax.set_xlim(40, -40)
        self.ax.set_ylim(-40, 40)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        title_text = f"FILE: {json_filename} [{self.current_file_idx+1}/{len(self.json_files)}]\nSorgente: {name.split('.')[-1].upper()} | Distanza: {dist:.1f}m"
        self.ax.set_title(title_text, color='white', fontsize=10, fontweight='bold', pad=15)
        
        # Legenda delle categorie geometriche aggiornata con Strada al posto di Altre Ombre
        legend_elements = [
            Patch(facecolor='#00f0ff', edgecolor='#00f0ff', alpha=0.4, label='Veicoli (Ciano)'),
            Patch(facecolor='#22c55e', edgecolor='#22c55e', alpha=0.4, label='Pedoni (Verde)'),
            Patch(facecolor='#ec4899', edgecolor='#ec4899', alpha=0.4, label='Ostacoli Statici (Rosa)'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.2, label='Ego Vehicle'),
            Patch(facecolor='#101216', edgecolor='#3A3632', alpha=0.9, label='Strada (Grigio Scuro)')
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

# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Macro-Area 1: Validazione Geometrica delle Occlusioni.")
    parser.add_argument("--mode", type=str, default="visual", choices=["check", "visual"],
                        help="Modalità d'uso: 'check' per integrità automatica, 'visual' per visualizzatore BEV.")
    parser.add_argument("--dir", type=str, default="extracted_occlusions", help="Cartella dei JSON geometrici.")
    args = parser.parse_args()

    print("Inizializzazione NuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    
    json_files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not json_files:
        print(f"[ERROR] Nessun file JSON trovato nella cartella '{args.dir}'!")
        return

    if args.mode == "check":
        run_integrity_check(nusc, json_files)
    else:
        GeometryVisualizer(nusc, json_files)

if __name__ == "__main__":
    main()
