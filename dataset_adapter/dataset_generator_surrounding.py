# Generatore di Dataset a Contesto Esterno Anello (dataset_generator_surrounding.py)
# Calcola gli scalari semantici del terreno ESCLUSIVAMENTE sull'anello circostante esterno (Ring Buffer 2.0m)
# mascherando come ignota la semantica interna al cono d'ombra.

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetNeural, rasterize_polygon, GRID_DIM, GRID_RANGE, VOXEL_SIZE
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, get_occlusion_ground_truth_target

def compute_surrounding_ring_scalars(poly_pts, frame_data, buffer_radius_m=2.0):
    """
    Calcola la percentuale di presenza di Strada, Marciapiede, Strisce, Parcheggio e Terreno 
    ESCLUSIVAMENTE sull'anello esterno circostante (buffer di 2.0 metri attorno all'ombra), 
    trattando come ignota la semantica interna.
    """
    scalars = np.zeros(9, dtype=np.float32)
    if poly_pts is None or len(poly_pts) < 3:
        return scalars

    pts_arr = np.array(poly_pts)
    pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
    try:
        poly_inner = ShapelyPolygon(pts_xy)
        if not poly_inner.is_valid:
            poly_inner = poly_inner.buffer(0)
    except Exception:
        return scalars

    occ_area = float(poly_inner.area)
    if occ_area <= 0:
        return scalars

    # 1. Geometria dell'ombra interna
    bounds = poly_inner.bounds
    width = float(bounds[2] - bounds[0])
    height = float(bounds[3] - bounds[1])
    center_x = float(np.mean(pts_xy[:, 0]))
    center_y = float(np.mean(pts_xy[:, 1]))
    dist = float(np.sqrt(center_x**2 + center_y**2))

    scalars[0] = min(occ_area / 50.0, 1.0)
    scalars[1] = min(dist / 40.0, 1.0)
    scalars[2] = min(width / 15.0, 1.0)
    scalars[3] = min(height / 15.0, 1.0)

    # 2. Generazione dell'Anello Esterno Circostante (Ring Buffer)
    try:
        poly_outer = poly_inner.buffer(buffer_radius_m)
        ring_poly = poly_outer.difference(poly_inner)
    except Exception:
        ring_poly = poly_inner

    ring_area = float(ring_poly.area)
    if ring_area <= 0.01:
        return scalars

    # 3. Intersezione della semantica dell'Anello Circostante Esterno
    semantic_map = frame_data.get("semantic_map", {})
    def get_layer_union(mask_name):
        mask_data = semantic_map.get(mask_name, None)
        if mask_data is None:
            return None
        if isinstance(mask_data, np.ndarray) and mask_data.ndim == 2:
            import cv2
            contours, _ = cv2.findContours((mask_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = cnt.squeeze(axis=1)
                    x_m = (pts[:, 0] * VOXEL_SIZE) - GRID_RANGE
                    y_m = GRID_RANGE - (pts[:, 1] * VOXEL_SIZE)
                    p = ShapelyPolygon(np.column_stack((x_m, y_m)))
                    if p.is_valid and p.area > 0.01:
                        polys.append(p)
            if polys:
                from shapely.ops import unary_union
                return unary_union(polys)
        return None

    layers = {
        'road': get_layer_union('drivable_area'),
        'sidewalk': get_layer_union('walkway'),
        'crosswalk': get_layer_union('ped_crossing'),
        'carpark': get_layer_union('carpark')
    }

    covered_ring_area = 0.0
    for idx, (lname, lpoly) in enumerate(layers.items()):
        if lpoly is not None:
            try:
                inter = ring_poly.intersection(lpoly)
                iarea = float(inter.area)
                scalars[4 + idx] = min(iarea / ring_area, 1.0)
                covered_ring_area += iarea
            except Exception:
                pass

    # Terreno anello circostante = area anello non coperta da altre superfici
    scalars[8] = min(max(0.0, ring_area - covered_ring_area) / ring_area, 1.0)
    return scalars

class OcclusionDatasetSurrounding(Dataset):
    """
    Dataset Per-Zone basato esclusivamente sul Contesto Semantico dell'Anello Circostante Esterno.
    """
    def __init__(self, dataset_name="nuscenes", dataroot="./nuscenes", patch_size=(64, 64)):
        self.base_dataset = OcclusionDatasetNeural(dataset_name=dataset_name, dataroot=dataroot)
        self.patch_size = patch_size
        self.samples = []
        
        print("\nEstrazione dei patch e delle feature basate sull'Anello Semantico Circostante...")
        self._build_samples()
        print(f"Dataset Contesto Esterno completato: Estratti {len(self.samples)} coni d'ombra totali.\n")

    def _build_samples(self):
        total_frames = len(self.base_dataset)
        for idx in range(total_frames):
            if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
                print(f"  Progresso Estrazione Anello Esterno: {idx+1}/{total_frames} fotogrammi elaborati...")
                
            input_tensor, target_tensor = self.base_dataset[idx]
            frame_data = self.base_dataset.adapter.get_sample_data(idx)
            sample_token = frame_data['sample_token']
            
            json_path = os.path.join("extracted_occlusions", f"{sample_token}.json")
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, "r") as f:
                occlusions = json.load(f).get("occlusions", [])
                
            for occ in occlusions:
                pts = occ.get("polygon_points_m", [])
                if len(pts) < 3:
                    continue
                    
                target_classes = get_occlusion_ground_truth_target(target_tensor, pts)
                surrounding_scalars = compute_surrounding_ring_scalars(pts, frame_data, buffer_radius_m=2.0)
                
                # Ritaglio patch visivo BEV 64x64
                pts_arr = np.array(pts)
                pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
                px = np.clip(((pts_xy[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                py = np.clip(((GRID_RANGE - pts_xy[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                
                cx, cy = int(np.mean(px)), int(np.mean(py))
                half_w, half_h = self.patch_size[0] // 2, self.patch_size[1] // 2
                
                x1, x2 = max(0, cx - half_w), min(GRID_DIM, cx + half_w)
                y1, y2 = max(0, cy - half_h), min(GRID_DIM, cy + half_h)
                
                patch = torch.zeros((11, self.patch_size[0], self.patch_size[1]), dtype=torch.float32)
                cropped = input_tensor[:, y1:y2, x1:x2]
                
                c_h, c_w = cropped.shape[1], cropped.shape[2]
                patch[:, :c_h, :c_w] = cropped
                
                self.samples.append({
                    'patch': patch,
                    'scalars': torch.tensor(surrounding_scalars, dtype=torch.float32),
                    'target': torch.tensor(target_classes, dtype=torch.float32),
                    'sample_token': sample_token,
                    'polygon_points': pts
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['patch'], s['scalars'], s['target']

if __name__ == "__main__":
    ds = OcclusionDatasetSurrounding()
    print(f"Self-Test Dataset Surrounding OK! Totale campioni: {len(ds)}")
