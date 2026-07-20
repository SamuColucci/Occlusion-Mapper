import os
import numpy as np
from shapely.geometry import box as ShapelyBox, Polygon as ShapelyPolygon, MultiPolygon
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

MAP_CACHE = {}

def get_map_instance(dataroot, map_name):
    """Gestisce il lazy loading delle mappe NuScenes per evitare ricaricamenti da disco."""
    if map_name not in MAP_CACHE:
        MAP_CACHE[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    return MAP_CACHE[map_name]

class LocalBox:
    """
    Rappresenta un oggetto 3D (box orientato) nel sistema di coordinate locale BEV (Ego frame).
    Espone l'interfaccia standard (center, corners, name, token, visibility) utilizzata 
    dal Ray-Caster e dall'Agente Bayesiano, rendendoli indipendenti dalle classi SDK di NuScenes.
    """
    def __init__(self, center, wlh, corners_3d, name, token, visibility="1"):
        self.center = np.array(center)
        self.wlh = np.array(wlh)
        self._corners_3d = np.array(corners_3d)  # shape: (3, 8)
        self.name = name
        self.token = token
        self.visibility = visibility

    def corners(self):
        """Ritorna gli 8 vertici 3D del box."""
        return self._corners_3d

class BaseDatasetAdapter:
    """
    Classe base astratta (interfaccia) per l'importazione di dataset multidimensionali.
    Ogni nuovo dataset (es. KITTI, Waymo, Carla) deve estendere questa classe.
    """
    def __init__(self, dataroot):
        self.dataroot = dataroot

    def get_num_samples(self) -> int:
        raise NotImplementedError

    def get_scene_indices(self) -> dict:
        """Ritorna un dizionario scene_token -> lista di indici ordinati cronologicamente."""
        raise NotImplementedError

    def get_sample_data(self, idx: int) -> dict:
        """
        Ritorna il frame standardizzato in coordinate locali BEV:
        {
            "lidar_token": str,
            "points": np.ndarray (N x 3),
            "boxes": list of LocalBox,
            "semantic_map": dict (layer_name -> list of np.ndarray of shape M x 2),
            "scene_token": str,
            "timestamp": int
        }
        """
        raise NotImplementedError

class NuScenesAdapter(BaseDatasetAdapter):
    """
    Adapter concreto per il dataset NuScenes (v1.0-mini).
    Carica i dati grezzi ed esegue la trasformazione geometrica delle coordinate nel frame Ego locale.
    """
    def __init__(self, dataroot="./nuscenes", version="v1.0-mini"):
        super().__init__(dataroot)
        self.version = version
        print(f"Inizializzazione NuScenesAdapter ({version}) su '{dataroot}'...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self._all_samples = self.nusc.sample

    def get_num_samples(self) -> int:
        return len(self._all_samples)

    def get_scene_indices(self) -> dict:
        from collections import defaultdict
        scenes = defaultdict(list)
        for idx, sample in enumerate(self._all_samples):
            scenes[sample['scene_token']].append((idx, sample['timestamp']))
        # Ordina ciascuna scena per timestamp
        sorted_scenes = {}
        for scene_token, idx_ts_list in scenes.items():
            idx_ts_list.sort(key=lambda x: x[1])
            sorted_scenes[scene_token] = [x[0] for x in idx_ts_list]
        return sorted_scenes

    def get_sample_data(self, idx: int) -> dict:
        sample = self._all_samples[idx]
        lidar_token = sample['data']['LIDAR_TOP']
        
        # 1. Caricamento dei punti LiDAR locali (NuScenes li memorizza già nel sensore LiDAR Ego frame)
        sd_record = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.dataroot, sd_record['filename'])
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        pts_local = pc[:, :3]

        # 2. Estrazione dei Bounding Box 3D in coordinate locali Ego BEV
        # nusc.get_sample_data restituisce le scatole orientate già rototraslate nel sensore Ego frame!
        _, boxes_ego, _ = self.nusc.get_sample_data(lidar_token)
        local_boxes = []
        for b in boxes_ego:
            # Recuperiamo l'instance token (tracking ID) e la visibilità dell'oggetto
            try:
                ann = self.nusc.get('sample_annotation', b.token)
                visibility = ann.get('visibility_token', '1')
                instance_token = ann.get('instance_token', b.token)
            except Exception:
                visibility = '1'
                instance_token = b.token
            
            lbox = LocalBox(
                center=b.center,
                wlh=b.wlh,
                corners_3d=b.corners(),
                name=b.name,
                token=instance_token,
                visibility=visibility
            )
            local_boxes.append(lbox)

        # 3. Estrazione e Rototraslazione delle Mappe HD locali
        surfaces = self._extract_semantic_surfaces_local(sample, sd_record)

        return {
            "lidar_token": lidar_token,
            "sample_token": sample['token'],
            "points": pts_local,
            "boxes": local_boxes,
            "semantic_map": surfaces,
            "scene_token": sample['scene_token'],
            "timestamp": sample['timestamp']
        }

    def _extract_semantic_surfaces_local(self, sample, sd_record, range_m=50) -> dict:
        """Estrae e proietta le mappe HD nel sistema di coordinate locale dell'auto (Ego frame)."""
        ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        sample_record = self.nusc.get('sample', sd_record['sample_token'])
        
        scene = self.nusc.get('scene', sample_record['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_name = log['location']
        nusc_map = get_map_instance(self.dataroot, map_name)
        
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
                            
                            # Trasformazione geometrica globale -> locale
                            pts_diff = pts_global_3d - np.array([tx, ty, tz])
                            pts_local = pts_diff @ R_inv.T
                            local_pts = np.column_stack([pts_local[:, 0], pts_local[:, 1]])
                            surfaces[layer].append(local_pts)
                except Exception:
                    continue
                    
        return surfaces
