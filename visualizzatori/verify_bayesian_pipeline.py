"""
Macro-Area 2: Estrazione e Validazione dei Casi di Hit Bayesiano (Ground Truth).

Questo script unificato mostra i casi in cui un oggetto reale annotato (Ground Truth)
interseca geometricamente una zona d'ombra generata dalla pipeline, confermando
visivamente la bontà dei dati statistici calcolati.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
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

from matplotlib.patches import Rectangle, Patch

CATEGORY_MAP = {
    "human": "Pedone", "pedestrian": "Pedone",
    "vehicle.car": "Auto", "vehicle.truck": "Camion",
    "vehicle.bus": "Bus", "vehicle.motorcycle": "Moto",
    "vehicle.bicycle": "Bicicletta", "vehicle.construction": "Veicolo",
    "vehicle.trailer": "Veicolo", "vehicle.emergency": "Veicolo",
    "static": "Statico", "movable_object": "Oggetto",
}

def classify(name):
    for key, label in CATEGORY_MAP.items():
        if key in name.lower():
            return label
    return "Altro"

def box_to_bev_corners(box):
    corners = box.corners()
    return corners[:2, [0, 1, 2, 3]].T

def main():
    print("Inizializzazione NuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    
    json_files = sorted(glob.glob("extracted_occlusions/*.json"))
    print(f"Scansione di {len(json_files)} file alla ricerca di hit reali...")
    
    hits = []
    
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
        
        lidar_token = data.get("lidar_token")
        if not lidar_token:
            continue
        
        try:
            _, boxes_real, _ = nusc.get_sample_data(lidar_token)
        except Exception:
            continue
        
        for occ in data.get("occlusions", []):
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue
            
            source_token = occ.get("object_token", "")
            occ_dist = occ.get("distance_m", 0)
            
            try:
                occ_poly = ShapelyPolygon(poly_pts)
                if not occ_poly.is_valid:
                    occ_poly = occ_poly.buffer(0)
            except Exception:
                continue
            
            # Creiamo la versione espansa del cono d'ombra per i near-miss (tolleranza 2 metri)
            occ_poly_expanded = occ_poly.buffer(2.0)
            
            for real_box in boxes_real:
                if real_box.token == source_token:
                    continue
                # Filtri anti-falso positivo (direzionalità e vicinanza eccessiva)
                hidden_dist = np.linalg.norm(real_box.center[:2])
                if hidden_dist < occ_dist - 1.0:
                    continue
                bbox = occ.get("occlusion_bbox_m", [0,0,0,0])
                if np.linalg.norm(real_box.center[:2] - np.array([bbox[0], bbox[1]])) < 3.0:
                    continue
                
                try:
                    bev = box_to_bev_corners(real_box)
                    real_fp = ShapelyPolygon(bev)
                    if not real_fp.is_valid:
                        real_fp = real_fp.buffer(0)
                        
                    is_hit = False
                    inter_area = 0.0
                    if real_fp.is_valid and occ_poly.intersects(real_fp):
                        inter = occ_poly.intersection(real_fp).area
                        if inter > 0.1:
                            is_hit = True
                            inter_area = inter
                            
                    is_near = False
                    dist_m = 0.0
                    if not is_hit and real_fp.is_valid and occ_poly_expanded.intersects(real_fp):
                        dist_m = occ_poly.distance(real_fp)
                        if dist_m > 0.0:
                            is_near = True
                            
                    if is_hit or is_near:
                        hits.append({
                            "lidar_token": lidar_token,
                            "occ_poly": np.array(poly_pts),
                            "occ_name": occ.get("object_name", "?"),
                            "hidden_box_corners": bev,
                            "hidden_name": real_box.name,
                            "hidden_cat": classify(real_box.name),
                            "source_cat": classify(occ.get("object_name", "")),
                            "tipo": "diretto" if is_hit else "near-miss",
                            "inter_area": round(inter_area, 2),
                            "distance_m": round(dist_m, 2),
                            "occ_poly_shapely": occ_poly,
                            "real_fp_shapely": real_fp,
                            "all_occlusions": data.get("occlusions", []),
                        })
                except Exception:
                    continue
                    
    print(f"Trovati {len(hits)} casi di hit reali confermati nel dataset.\n")
    if not hits:
        print("Nessun hit trovato.")
        return
        
    # Visualizzatore interattivo
    current = [0]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0B0C10')
    
    def draw(idx):
        ax.clear()
        ax.set_facecolor('#0B0C10')
        h = hits[idx]
        
        # 0. Disegno della Superficie Stradale (Drivable Area)
        try:
            sd_record = nusc.get('sample_data', h["lidar_token"])
            ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
            sample_record = nusc.get('sample', sd_record['sample_token'])
            
            scene = nusc.get('scene', sample_record['scene_token'])
            log = nusc.get('log', scene['log_token'])
            map_name = log['location']
            nusc_map = get_map_instance('./nuscenes', map_name)
            
            local_road_polys = get_drivable_area_local(
                nusc, nusc_map, sample_record['token'], ego_pose, range_m=50
            )
            for pts in local_road_polys:
                # Disegniamo l'asfalto in grigio fumo caldissimo/scuro neutro
                ax.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
                # Disegniamo i contorni / bordi stradali come linee tratteggiate discrete
                ax.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
        except Exception:
            pass
        
        # 1. Disegno dei Radar Rings (Cerchi Concentrici)
        for radius in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), radius, color='#2C374E', fill=False, linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
            ax.add_patch(circle)
            ax.text(0.5, radius, f"{radius}m", color='#475569', fontsize=7, ha='left', va='center', fontweight='bold', zorder=2)
            
        # 2. LiDAR in Sonar-Style (zorder=3 per stare sopra la strada)
        try:
            pcl_path = nusc.get_sample_data_path(h["lidar_token"])
            pc = np.fromfile(pcl_path, dtype=np.float32).reshape((-1, 5))[:, :3]
            ax.scatter(pc[:, 1], pc[:, 0], s=0.12, c='#4A5E7D', alpha=0.25, zorder=3)
        except Exception:
            pass
            
        # 3. Disegno dell'Ego-Vehicle stilizzato a centro (0, 0) (zorder=6 per stare sopra tutto)
        ego_rect = Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=1.5, edgecolor='#00E5FF', facecolor='#00E5FF', alpha=0.2, zorder=6)
        ax.add_patch(ego_rect)
        ax.plot(0, 0, color='#00E5FF', marker='^', markersize=4, zorder=6) # Freccia di heading
        
        # 4. Disegno delle ombre degli ALTRI hit dello stesso frame in grigio (zorder=4)
        curr_poly = h["occ_poly"]
        for other_idx, other_h in enumerate(hits):
            if other_idx == idx:
                continue
            if other_h["lidar_token"] == h["lidar_token"]:
                other_poly = other_h["occ_poly"]
                if len(other_poly) > 0:
                    poly_closed = np.vstack([other_poly, other_poly[0]])
                    ax.plot(poly_closed[:, 1], poly_closed[:, 0], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                    ax.fill(poly_closed[:, 1], poly_closed[:, 0], color='#1E293B', alpha=0.15, zorder=4)
        
        # 5. Poligono ombra selezionato (rosso neon per hit diretto, giallo oro per near-miss) (zorder=5)
        color_occ = '#FFD600' if h["tipo"] == "near-miss" else '#FF1744'
        poly_closed = np.vstack([curr_poly, curr_poly[0]])
        ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color_occ, linewidth=2.5, zorder=5)
        ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color_occ, alpha=0.25, zorder=5)
        
        # 6. Footprint reale nascosto (verde neon per hit diretto, giallo oro per near-miss)
        bev = h["hidden_box_corners"]
        bev_closed = np.vstack([bev, bev[0]])
        color_fp = '#FFD600' if h["tipo"] == "near-miss" else '#00E676'
        ax.plot(bev_closed[:, 1], bev_closed[:, 0], color=color_fp, linewidth=2.5, zorder=6)
        ax.fill(bev_closed[:, 1], bev_closed[:, 0], color=color_fp, alpha=0.35, zorder=6)
        
        # 7. Disegno linea di connessione e distanza per Near-Miss
        if h["tipo"] == "near-miss":
            try:
                from shapely.ops import nearest_points
                p1, p2 = nearest_points(h["occ_poly_shapely"], h["real_fp_shapely"])
                # Asse X del grafico = coordinata y di Shapely (laterale)
                # Asse Y del grafico = coordinata x di Shapely (longitudinale)
                ax.plot([p1.y, p2.y], [p1.x, p2.x], color='#FFD600', linestyle='--', linewidth=1.5, zorder=6)
                
                # Calcolo del punto medio
                mx = (p1.y + p2.y) / 2.0
                my = (p1.x + p2.x) / 2.0
                ax.text(mx, my, f"{h['distance_m']:.1f}m", color='white', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='#1E293B', alpha=0.9, edgecolor='#FFD600', boxstyle='round,pad=0.2'), zorder=7)
            except Exception:
                pass
        
        ax.set_xlim(40, -40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if h["tipo"] == "near-miss":
            title_text = (
                f"VERIFICA NEAR-MISS REAL-TIME [{idx+1}/{len(hits)}]\n"
                f"Sorgente ombra: {h['occ_name'].split('.')[-1].upper()} | Nascosto: {h['hidden_name'].split('.')[-1].upper()} ({h['hidden_cat']})\n"
                f"Distanza Minima: {h['distance_m']} m | Frecce DX/SX o MOUSE per scorrere"
            )
        else:
            title_text = (
                f"VERIFICA HIT REAL-TIME [{idx+1}/{len(hits)}]\n"
                f"Sorgente ombra: {h['occ_name'].split('.')[-1].upper()} | Nascosto: {h['hidden_name'].split('.')[-1].upper()} ({h['hidden_cat']})\n"
                f"Intersezione: {h['inter_area']} m² | Frecce DX/SX o MOUSE per scorrere"
            )
        ax.set_title(title_text, color='white', fontsize=10, fontweight='bold', pad=15)
        
        # Legenda dei casi di hit e near-miss
        legend_elements = [
            Patch(facecolor='#FF1744', edgecolor='#FF1744', alpha=0.4, label='Hit Selezionato (Rosso)'),
            Patch(facecolor='#FFD600', edgecolor='#FFD600', alpha=0.4, label='Near-Miss Selezionato (Giallo)'),
            Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.4, label='Footprint GT Nascosto (Hit)'),
            Patch(facecolor='#00E5FF', edgecolor='#00E5FF', alpha=0.2, label='Ego Vehicle'),
            Patch(facecolor='#101216', edgecolor='#3A3632', alpha=0.9, label='Strada (Grigio Scuro)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', facecolor='#1E293B', edgecolor='gray', fontsize=7, labelcolor='white')
        fig.canvas.draw()
 
    def on_key(event):
        if event.key == 'right':
            current[0] = (current[0] + 1) % len(hits)
            draw(current[0])
        elif event.key == 'left':
            current[0] = (current[0] - 1) % len(hits)
            draw(current[0])

    def on_mouse_move(event):
        """Passando col mouse sopra un'ombra grigia, se c'è un hit associato ad essa in questo frame, lo seleziona."""
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        point = ShapelyPoint(event.ydata, event.xdata)
        
        curr_hit = hits[current[0]]
        for idx, h in enumerate(hits):
            # Cerca solo tra gli hit dello stesso frame (stesso lidar_token)
            if h["lidar_token"] == curr_hit["lidar_token"]:
                poly_pts = h["occ_poly"]
                if len(poly_pts) > 2:
                    shp_poly = ShapelyPolygon(poly_pts)
                    if shp_poly.contains(point):
                        if current[0] != idx:
                            current[0] = idx
                            draw(idx)
                        break

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    print("\n" + "=" * 50)
    print("    CONTROLLI INTERATTIVI REGISTRATI (HIT BAYES)")
    print("=" * 50)
    print(" -> MOUSE            : Passa sopra un hit grigio per attivarlo")
    print(" -> FRECCIA DX/SX    : Scorri manualmente gli hit reali")
    print("=" * 50 + "\n")
    
    draw(0)
    plt.show()

if __name__ == "__main__":
    main()
