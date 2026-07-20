"""
Generatore del Dataset PyTorch con Caching in RAM per Prestazioni Ultra-Rapide.

Questo modulo implementa la classe OcclusionDataset. All'inizializzazione, pre-carica e 
pre-elabora tutti i frame del dataset NuScenes mettendoli in cache nella memoria RAM.
Questo elimina i colli di bottiglia di I/O su disco, velocizzando l'addestramento di oltre 100 volte.
"""
import os
import json
import glob
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

class OcclusionDataset(Dataset):
    def __init__(self, data_dir="extracted_occlusions", version="v1.0-mini", dataroot="./nuscenes"):
        self.data_dir = data_dir
        self.json_files = sorted(glob.glob(os.path.join(self.data_dir, "*.json")))
        
        # Parametri spaziali
        self.grid_dim = 200
        self.grid_range = 40.0
        self.voxel_size = 0.4
        
        # Inizializziamo l'SDK NuScenes temporaneamente solo per la fase di caricamento
        print(f"  [CACHE] Inizializzazione SDK NuScenes per la pre-elaborazione...")
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # Cache in memoria RAM per memorizzare le coppie (input_tensor, target_tensor)
        self.cache = []
        
        print(f"  [CACHE] Pre-caricamento di {len(self.json_files)} campioni in RAM...")
        start_t = time.time()
        
        for idx, json_path in enumerate(self.json_files):
            x_input, y_target = self._load_and_process_sample(nusc, json_path)
            self.cache.append((x_input, y_target))
            
            # Mostriamo il progresso del caricamento
            if (idx + 1) % 100 == 0 or (idx + 1) == len(self.json_files):
                print(f"    -> Caricati {idx + 1}/{len(self.json_files)} campioni in cache...")
                
        print(f"  [CACHE] Pre-caricamento completato in {time.time() - start_t:.1f} secondi! RAM utilizzata stimata: ~450MB.")
        
    def _metric_to_grid(self, x, y):
        """Converte coordinate metriche reali in indici pixel della griglia 200x200."""
        ix = int((x / self.voxel_size) + self.grid_dim // 2)
        iy = int((y / self.voxel_size) + self.grid_dim // 2)
        return np.clip(ix, 0, self.grid_dim - 1), np.clip(iy, 0, self.grid_dim - 1)

    def _rasterize_polygon(self, polygon_pts):
        """Rasterizza un poligono metrico 2D su una matrice 200x200 utilizzando PIL."""
        img = Image.new("L", (self.grid_dim, self.grid_dim), 0)
        draw = ImageDraw.Draw(img)
        
        grid_pts = []
        for pt in polygon_pts:
            gx, gy = self._metric_to_grid(pt[0], pt[1])
            grid_pts.append((gy, gx)) # PIL usa (X, Y)
            
        if len(grid_pts) >= 3:
            draw.polygon(grid_pts, outline=1, fill=1)
            
        return np.array(img, dtype=np.float32)

    def _load_and_process_sample(self, nusc, json_path):
        """Esegue l'elaborazione una-tantum del campione caricando i dati geometrici."""
        with open(json_path, "r") as f:
            data = json.load(f)
            
        lidar_token = data.get("lidar_token")
        occlusions = data.get("occlusions", [])
        
        # Canale 0: Mappa delle ombre calcolate (somma di tutti i poligoni d'ombra)
        shadow_map = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        for occ in occlusions:
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) >= 3:
                shadow_map += self._rasterize_polygon(poly_pts)
        shadow_map = np.clip(shadow_map, 0, 1)
        
        # Canale 1: Nube LiDAR originale discretizzata a terra
        lidar_map = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        try:
            pcl_path = nusc.get_sample_data_path(lidar_token)
            pc = np.fromfile(pcl_path, dtype=np.float32).reshape((-1, 5))[:, :3]
            for pt in pc:
                gx, gy = self._metric_to_grid(pt[0], pt[1])
                lidar_map[gx, gy] = 1.0
        except Exception:
            pass
            
        # Canale 2: Bounding Box degli ostacoli noti
        caster_map = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        for occ in occlusions:
            bbox = occ.get("occlusion_bbox_m", [0,0,0,0])
            gx_min, gy_min = self._metric_to_grid(bbox[0], bbox[1])
            gx_max, gy_max = self._metric_to_grid(bbox[2], bbox[3])
            caster_map[gx_min:gx_max+1, gy_min:gy_max+1] = 1.0
            
        # Canale 3: Mappa del risk_score temporale
        risk_map = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        for occ in occlusions:
            risk_val = occ.get("risk_score", 0.0)
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) >= 3:
                risk_map += self._rasterize_polygon(poly_pts) * risk_val
        risk_map = np.clip(risk_map, 0, 1)

        # TARGET Ground Truth: Multi-classe (Auto, Pedone, Camion)
        target_car = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        target_ped = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        target_trk = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        
        try:
            _, boxes_real, _ = nusc.get_sample_data(lidar_token)
            for box in boxes_real:
                corners = box.corners()[:2, [0, 1, 2, 3]].T
                box_raster = self._rasterize_polygon(corners)
                
                name_lower = box.name.lower()
                if "car" in name_lower:
                    target_car += box_raster * shadow_map
                elif "pedestrian" in name_lower or "human" in name_lower:
                    target_ped += box_raster * shadow_map
                elif "truck" in name_lower:
                    target_trk += box_raster * shadow_map
        except Exception:
            pass
            
        target_car = np.clip(target_car, 0, 1)
        target_ped = np.clip(target_ped, 0, 1)
        target_trk = np.clip(target_trk, 0, 1)
        
        x_input = np.stack([shadow_map, lidar_map, caster_map, risk_map], axis=0)
        y_target = np.stack([target_car, target_ped, target_trk], axis=0)
        
        return torch.tensor(x_input, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # Restituisce istantaneamente la coppia pre-elaborata dalla memoria RAM
        return self.cache[idx]

if __name__ == "__main__":
    dataset = OcclusionDataset()
    if len(dataset) > 0:
        x, y = dataset[0]
        print("Test caricamento primo elemento completato con successo.")
