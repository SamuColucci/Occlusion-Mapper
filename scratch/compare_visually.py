import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('file vecchi 21 luglio'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from factory_dataset import create_adapter
from occ3d_occlusion_explorer import SOTARayCaster
from ray_caster import RayCaster
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

MAP_CACHE = {}

def get_map_instance(dataroot, map_name):
    if map_name not in MAP_CACHE:
        MAP_CACHE[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return MAP_CACHE[map_name]

def get_semantic_surfaces_local(nusc, nusc_map, sample_token, ego_pose, range_m=50):
    from shapely.geometry import box as ShapelyBox, Polygon as ShapelyPolygon, MultiPolygon
    tx, ty, tz = ego_pose['translation']
    box_coords = (tx - range_m, ty - range_m, tx + range_m, ty + range_m)
    
    surfaces = {'drivable_area': [], 'walkway': []}
    q = Quaternion(ego_pose['rotation'])
    R_inv = q.inverse.rotation_matrix
    
    # Calibrazione del sensore (da Ego a Sensore)
    sample_record = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    cs_tx, cs_ty, cs_tz = cs_record['translation']
    cs_q = Quaternion(cs_record['rotation'])
    cs_R_inv = cs_q.inverse.rotation_matrix
    
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
                    if poly.is_empty: continue
                    inter = poly.intersection(interest_box)
                    if inter.is_empty: continue
                        
                    if isinstance(inter, MultiPolygon): polys_to_process = list(inter.geoms)
                    else: polys_to_process = [inter]
                        
                    for p in polys_to_process:
                        if not isinstance(p, ShapelyPolygon): continue
                        x_coords, y_coords = p.exterior.coords.xy
                        pts_global = np.column_stack([x_coords, y_coords])
                        pts_global_3d = np.column_stack([pts_global[:, 0], pts_global[:, 1], np.full(len(pts_global), tz)])
                        
                        # 1. Da Globale a Ego
                        pts_diff = pts_global_3d - np.array([tx, ty, tz])
                        pts_ego = pts_diff @ R_inv.T
                        
                        # 2. Da Ego a Sensore
                        pts_sensor_diff = pts_ego - np.array([cs_tx, cs_ty, cs_tz])
                        pts_sensor = pts_sensor_diff @ cs_R_inv.T
                        
                        local_pts = np.column_stack([pts_sensor[:, 0], pts_sensor[:, 1]])
                        surfaces[layer].append(local_pts)
            except Exception:
                continue
    return surfaces

def draw_map_and_hud(ax, nusc, ego_pose, sample_record, pc):
    # Sfondo verde prato scuro (terrain)
    bg_color = "#102216"
    ax.set_facecolor(bg_color)
    
    # 0. Disegno della Superficie Stradale (Drivable Area)
    try:
        scene = nusc.get('scene', sample_record['scene_token'])
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        nusc_map = get_map_instance('./nuscenes', map_name)
        
        surfaces = get_semantic_surfaces_local(
            nusc, nusc_map, sample_record['token'], ego_pose, range_m=50
        )
        
        # 1. Strada (Asfalto grigio scuro)
        for pts in surfaces['drivable_area']:
            ax.fill(pts[:, 0], pts[:, 1], color='#101216', zorder=1, alpha=0.9)
            ax.plot(pts[:, 0], pts[:, 1], color='#3A3632', linestyle=':', linewidth=0.8, zorder=1)
            
        # 2. Marciapiede (Grigio walkway)
        for pts in surfaces['walkway']:
            ax.fill(pts[:, 0], pts[:, 1], color='#22252C', zorder=1, alpha=0.8)
            ax.plot(pts[:, 0], pts[:, 1], color='#3A4250', linestyle=':', linewidth=0.6, zorder=1)
    except Exception:
        pass

    # 1. Disegno dei Radar Rings
    for radius in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), radius, color='#2C374E', fill=False, linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
        ax.add_patch(circle)
        ax.text(radius, 0.5, f"{radius}m", color='#475569', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=2)
    
    # 2. Disegno del LiDAR
    if len(pc) > 0:
        ax.scatter(pc[:, 0], pc[:, 1], s=0.12, c='#4A5E7D', alpha=0.25, zorder=3)
        
    # 3. Disegno dell'Ego-Vehicle
    ego_rect = Rectangle((-2.0, -0.9), 4.0, 1.8, linewidth=1.5, edgecolor='#00E5FF', facecolor='#00E5FF', alpha=0.2, zorder=6)
    ax.add_patch(ego_rect)
    ax.plot(0, 0, color='#00E5FF', marker='>', markersize=4, zorder=6)
    
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')

def main():
    adapter = create_adapter("nuscenes", "./nuscenes")
    frame_data = adapter.get_sample_data(0)
    token = frame_data["sample_token"]
    lidar_token = frame_data["lidar_token"]
    
    print(f"Elaborazione frame: {token}")
    
    print("Calcolo ombre SOTA (vecchio codice)...")
    sota_caster = SOTARayCaster(frame_data)
    sota_caster.generate_known_zone()
    sota_caster.generate_box_shadows()
    sota_caster.find_object_occlusion_wedges()
    sota_caster.extract_occlusion_zones()
    sota_2d = np.max(sota_caster.occluded_final, axis=2)
    
    print("Calcolo ombre Nuovo RayCaster...")
    new_caster = RayCaster(frame_data)
    new_2d = new_caster.get_occlusion_mask()
    
    # Inizializziamo NuScenes per prendere le mappe
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    sd_record = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
    sample_record = nusc.get('sample', token)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0B0C10')
    
    # Calcolo differenza per il terzo pannello
    diff = (sota_2d != new_2d).astype(float)
    
    # Creiamo un'immagine RGBA per sovrapporre le maschere con trasparenza
    # Rosso acceso per le ombre
    color_sota = np.zeros((*sota_2d.shape, 4))
    color_sota[sota_2d == 1] = [1.0, 0.09, 0.26, 0.4] # #FF1744 con alpha=0.4
    
    color_new = np.zeros((*new_2d.shape, 4))
    color_new[new_2d == 1] = [1.0, 0.84, 0.0, 0.4] # #FFD600 con alpha=0.4
    
    color_diff = np.zeros((*diff.shape, 4))
    color_diff[diff == 1] = [0.0, 0.9, 0.46, 0.5] # #00E676 con alpha=0.5
    
    # Per allineare imshow alla nuova orientazione:
    # La maschera (GRID_DIM, GRID_DIM) ha assi (X, Y).
    # Matplotlib imshow si aspetta (righe=Y, colonne=X).
    # Se facciamo mask.transpose(1, 0, 2), la X diventa colonna (sinistra-destra), la Y diventa riga (sopra-sotto).
    # Con origin='lower', le righe (Y) vanno dal basso verso l'alto.
    # Quindi l'extent deve essere [xmin, xmax, ymin, ymax] -> [-40, 40, -40, 40]
    extent = [-40, 40, -40, 40]
    
    for i, (ax, title, mask) in enumerate([
        (axes[0], '1. VECCHIO CODICE (SOTARayCaster)', color_sota),
        (axes[1], '2. NUOVO CODICE (RayCaster)', color_new),
        (axes[2], '3. DIFFERENZE (Nuovo - Vecchio)', color_diff)
    ]):
        draw_map_and_hud(ax, nusc, ego_pose, sample_record, frame_data["points"])
        # Sovrapponi la maschera! imshow usa extent [left, right, bottom, top]
        ax.imshow(mask.transpose(1, 0, 2), extent=extent, origin='lower', zorder=5)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=12, loc='left')
    
    fig.suptitle(
        f"CONFRONTO GEOMETRICO SINCRONIZZATO (SOTA vs NUOVO)\nFrame: {token}",
        color='white', fontsize=12, fontweight='bold', y=0.97
    )
    
    plt.tight_layout()
    plt.savefig('scratch/confronto_visivo_hud.png', dpi=150, facecolor='#0B0C10')
    print("Immagine salvata in scratch/confronto_visivo_hud.png")

if __name__ == "__main__":
    main()
