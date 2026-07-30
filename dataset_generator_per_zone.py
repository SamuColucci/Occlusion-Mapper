# Generatore del Dataset Neurale Per-Zone (OcclusionDatasetPerZone)
# Trasforma ogni cono d'ombra del dataset nuScenes in una coppia di addestramento:
# - Patch Visivo 2D (10, 64, 64) estratto dai 10 canali d'ingresso BEV
# - Vettore di 4 Feature Scalari Numeriche (area_sqm, distance_m, occluder_width_m, occluder_height_m)
# - Vettore di Target Ground Truth (5 classi binarie binarizzate dagli ostacoli 3D reali nuScenes)

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw
from factory_dataset import create_adapter
from ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target

# Parametri della griglia Bird's Eye View (BEV)
GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4

# Metodo per convertire un poligono da coordinate metriche a coordinate pixel
def rasterize_polygon(polygon_pts_m, grid_dim=GRID_DIM, grid_range=GRID_RANGE, voxel_size=VOXEL_SIZE):
    if len(polygon_pts_m) < 3:
        return np.zeros((grid_dim, grid_dim), dtype=np.float32)

    img_pts = []
    for x, y in polygon_pts_m:
        px = int((x + grid_range) / voxel_size)
        py = int((y + grid_range) / voxel_size)
        px = max(0, min(grid_dim - 1, px))
        py = max(0, min(grid_dim - 1, py))
        img_pts.append((px, py))

    img = Image.new('L', (grid_dim, grid_dim), 0)
    ImageDraw.Draw(img).polygon(img_pts, outline=1, fill=1)
    return np.array(img, dtype=np.float32)

class OcclusionDatasetNeural(Dataset):
    def __init__(self, dataset_name="nuscenes", dataroot="./nuscenes"):
        print(f"Inizializzazione OcclusionDatasetNeural ({dataset_name})...")
        self.adapter = create_adapter(dataset_name, dataroot)
        self.num_samples = self.adapter.get_num_samples()
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame_data = self.adapter.get_sample_data(idx)
        lidar_channel = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        points = frame_data.get("points", np.zeros((0, 3)))
        if len(points) > 0:
            for px_m, py_m in points[:, :2]:
                px = int((px_m + GRID_RANGE) / VOXEL_SIZE)
                py = int((py_m + GRID_RANGE) / VOXEL_SIZE)
                if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                    lidar_channel[py, px] = 1.0

        token = frame_data["sample_token"]
        json_path = os.path.join("extracted_occlusions", f"{token}.json")
        shadow_channel = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                raw_data = json.load(f)
                for occ in raw_data.get("occlusions", []):
                    pts = occ.get("polygon_points_m", [])
                    if len(pts) >= 3:
                        pts_arr = np.array(pts)
                        pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
                        mask = rasterize_polygon(pts_xy)
                        shadow_channel = np.maximum(shadow_channel, mask)

        semantic_map = frame_data.get("semantic_map", {})
        drivable_mask = semantic_map.get('drivable_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        walkway_mask = semantic_map.get('walkway', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        target_masks = extract_ground_truth_masks(frame_data)
        input_channels = [lidar_channel, shadow_channel, drivable_mask, walkway_mask, ped_crossing_mask]
        target_tensor = torch.tensor(target_masks, dtype=torch.float32)
        input_tensor = torch.tensor(np.stack(input_channels + list(target_masks), axis=0), dtype=torch.float32)
        return input_tensor, target_tensor

class OcclusionDatasetPerZone(Dataset):
    def __init__(self, dataset_name="nuscenes", dataroot="./nuscenes", patch_size=(64, 64)):
        # Carica il dataset base a 10 canali BEV
        self.base_dataset = OcclusionDatasetNeural(dataset_name=dataset_name, dataroot=dataroot)
        self.patch_size = patch_size
        self.samples = []
        
        print("\nEstrazione dei patch e delle feature per-zone su tutto il dataset...")
        self._build_samples()
        print(f"Dataset Per-Zone completato: Estratti {len(self.samples)} coni d'ombra totali.\n")

    def _build_samples(self):
        total_frames = len(self.base_dataset)
        for idx in range(total_frames):
            if (idx + 1) % 20 == 0 or (idx + 1) == total_frames:
                print(f"  Progresso Estrazione Patch: {idx+1}/{total_frames} fotogrammi elaborati...")
                
            # Estrazione dei tensori d'ingresso (10, 200, 200) e target (5, 200, 200) per il fotogramma corrente
            input_tensor, target_tensor = self.base_dataset[idx]
            frame_data = self.base_dataset.adapter.get_sample_data(idx)
            sample_token = frame_data['sample_token']
            
            # Caricamento del file JSON contenente le coordinate dei coni d'ombra per questo fotogramma
            json_path = os.path.join("extracted_occlusions", f"{sample_token}.json")
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, "r") as f:
                occlusions_data = json.load(f).get("occlusions", [])
                
            for occ in occlusions_data:
                pts = occ.get("polygon_points_m", [])
                if len(pts) < 3:
                    continue
                    
                # Calcolo del bounding box in pixel attorno alla zona d'ombra nella griglia 200x200 (0.4m/pixel)
                pts_np = np.array(pts)
                y_ahead = pts_np[:, 0]
                x_right = pts_np[:, 1]
                px_x = np.clip(((x_right + 40.0) / 0.4).astype(int), 0, 199)
                px_y = np.clip(((40.0 - y_ahead) / 0.4).astype(int), 0, 199)
                
                xmin, xmax = max(0, np.min(px_x)), min(199, np.max(px_x))
                ymin, ymax = max(0, np.min(px_y)), min(199, np.max(px_y))
                
                # Margine di sicurezza attorno all'ombra
                xmin, xmax = max(0, xmin - 2), min(199, xmax + 2)
                ymin, ymax = max(0, ymin - 2), min(199, ymax + 2)
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                    
                # Ritaglio del patch 2D dai 10 canali d'ingresso e ridimensionamento a (10, 64, 64)
                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_resized = F.interpolate(patch.unsqueeze(0), size=self.patch_size, mode='bilinear', align_corners=False).squeeze(0)
                
                # Estrazione delle 9 Feature Scalari Numeriche della zona d'ombra (4 geometriche + 5 semantiche)
                area_sqm = float(occ.get("area_sqm", 0.0))
                distance_m = float(occ.get("distance_m", 0.0))
                occ_w = float(occ.get("occluder_width_m", 2.0)) if "occluder_width_m" in occ else 2.0
                occ_h = float(occ.get("occluder_height_m", 1.8)) if "occluder_height_m" in occ else 1.8
                
                road_f = float(occ.get("road_fraction", 0.0))
                side_f = float(occ.get("sidewalk_fraction", 0.0))
                cross_f = float(occ.get("crosswalk_fraction", 0.0))
                park_f = float(occ.get("carpark_fraction", 0.0))
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f + park_f))
                
                scalars = torch.tensor([area_sqm, distance_m, occ_w, occ_h, road_f, side_f, cross_f, park_f, terr_f], dtype=torch.float32)
                
                # Calcolo della Target Ground Truth Reale per le 5 classi semantiche tramite il modulo dedicato ground_truth_extractor.py
                target_classes = get_occlusion_ground_truth_target(target_tensor, pts)
                target_vec = torch.tensor(target_classes, dtype=torch.float32)
                
                # Salvataggio del campione per-zone processato
                self.samples.append({
                    'patch': patch_resized,
                    'scalars': scalars,
                    'target': target_vec,
                    'sample_token': sample_token,
                    'polygon_points': pts,
                    'raw_occ': occ
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['patch'], s['scalars'], s['target']


if __name__ == "__main__":
    ds = OcclusionDatasetPerZone()
    if len(ds) > 0:
        p, sc, t = ds[0]
        print(f"Sample #0 -> Patch Shape: {p.shape} | Scalari: {sc} | Target Vector: {t}")
