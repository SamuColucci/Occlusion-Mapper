# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - INPUT MULTIMODALI RETE NEURALE & RETE AUSILIARIA
# File: visualizzatori/vis_input_rete_neurale.py
#
# Dashboard Scientifica di Ispezione degli Input (Figure 6 Tesi):
# - MODALITÀ 11 CANALI BEV (Default):
#     * SINISTRA: Mappa BEV globale a 25m con auto ego, cerchi metrici e zone d'ombra;
#     * DESTRA: Griglia ad alta fedeltà con gli 11 canali tensoriali di input
#               [LiDAR, Ombre Raycaster, 3 Mappe HD, 6 Maschere Ostacoli Visibili]
#               + Grafico dei pesi Channel Attention (Squeeze-and-Excitation).
# - MODALITÀ RETE NEURALE AUSILIARIA (Attivabile con Toggle Button o tasto [A]):
#     * SINISTRA: Mappa BEV a 25m con zone d'ombra numerate e selezionabili (click/hover);
#     * DESTRA: Ispezione completa del flusso scalare topologico:
#               1. I 9 Descrittori Scalari Fisici (Area, Distanza, Varchi OBB, Suolo);
#               2. Architettura della Rete Ausiliaria MLP (9 -> 64 -> 128);
#               3. Modulazione Affine FiLM (Fattori Gamma e Beta estratti in tempo reale);
#               4. Vincolo Neuro-Simbolico & Ammissibilità Spazio-Semantica.
# ==============================================================================

import os
import sys
import time
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from shapely.geometry import Polygon as ShapelyPolygon, Point
from PIL import Image

# Impostazione percorso root del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset_adapter.factory_dataset import create_adapter
from raycaster.ray_caster import RayCaster
from ground_truth.ground_truth_extractor import (
    extract_ground_truth_masks,
    rasterize_polygon,
    get_occlusion_ground_truth_target
)
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")
GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4

CHANNEL_METADATA = [
    {"id": 0, "title": "C0: LiDAR Density BEV", "desc": "Punti laser 3D riflessi a terra", "cmap": "viridis"},
    {"id": 1, "title": "C1: Ombre Raycasting", "desc": "Maschera cumulativa zone cieche", "cmap": "YlOrRd"},
    {"id": 2, "title": "C2: HD-Map Carreggiata", "desc": "Drivable Area + Parcheggi", "cmap": "bone"},
    {"id": 3, "title": "C3: HD-Map Marciapiedi", "desc": "Walkway / camminamenti pedonali", "cmap": "Greens"},
    {"id": 4, "title": "C4: HD-Map Strisce", "desc": "Crosswalks / attraversamenti pedonali", "cmap": "PuBu"},
    {"id": 5, "title": "C5: Ostacoli: Auto", "desc": "Bounding box Auto già visibili", "cmap": "Blues"},
    {"id": 6, "title": "C6: Ostacoli: Camion/Bus", "desc": "Mezzi pesanti e trailer visibili", "cmap": "Purples"},
    {"id": 7, "title": "C7: Ostacoli: Pedoni", "desc": "Utenti vulnerabili visibili", "cmap": "Greens"},
    {"id": 8, "title": "C8: Ostacoli: Moto", "desc": "Motocicli su strada visibili", "cmap": "Oranges"},
    {"id": 9, "title": "C9: Ostacoli: Bici", "desc": "Ciclisti e corsie ciclabili", "cmap": "YlOrBr"},
    {"id": 10, "title": "C10: Ostacoli: Barriere", "desc": "Barriere, cantieri, coni e new jersey", "cmap": "Reds"}
]

SCALAR_METADATA = [
    {"idx": 0, "name": "Area Zona d'Ombra", "unit": "m²", "key": "area", "norm": 50.0, "color": "#0284C7"},
    {"idx": 1, "name": "Distanza dall'Ego", "unit": "m", "key": "dist", "norm": 25.0, "color": "#2563EB"},
    {"idx": 2, "name": "Larghezza Minima OBB (Varco)", "unit": "m", "key": "obb_w", "norm": 5.0, "color": "#7C3AED"},
    {"idx": 3, "name": "Lunghezza Massima OBB", "unit": "m", "key": "obb_l", "norm": 10.0, "color": "#9333EA"},
    {"idx": 4, "name": "Suolo Carreggiata (Asfalto)", "unit": "%", "key": "road_f", "norm": 1.0, "color": "#475569"},
    {"idx": 5, "name": "Suolo Marciapiede (Walkway)", "unit": "%", "key": "side_f", "norm": 1.0, "color": "#16A34A"},
    {"idx": 6, "name": "Suolo Attraversamento Pedonale", "unit": "%", "key": "cross_f", "norm": 1.0, "color": "#0D9488"},
    {"idx": 7, "name": "Prossimità Bordo Strada (Accosto)", "unit": "%", "key": "roadside_f", "norm": 1.0, "color": "#D97706"},
    {"idx": 8, "name": "Suolo Terreno Naturale / Verde", "unit": "%", "key": "terr_f", "norm": 1.0, "color": "#65A30D"}
]


class NeuralInputsVisualizer:
    def __init__(self, max_range=25.0, initial_aux=False):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: INPUT MULTIMODALI RETE NEURALE & RETE AUSILIARIA")
        print("=" * 80)

        self.max_range = float(max_range)
        self.show_auxiliary = bool(initial_aux)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Dispositivo di Calcolo: {self.device}")

        # Inizializzazione dataset nuScenes
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0
        self.selected_zone_idx = 0

        # Carica il modello Attention per valutare pesi SE e FiLM ausiliario
        self.model = self.load_model()

        # Setup stile grafico accademico
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Neural Network Multimodal Inputs & Auxiliary MLP")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None
        self.ui_buttons = []

        self.load_icons()
        self.load_frame(self.current_idx, broadcast=False)

        # Timer sincronizzazione bidirezionale
        self.sync_timer = self.fig.canvas.new_timer(interval=150)
        self.sync_timer.add_callback(self.check_sync_file)
        self.sync_timer.start()

    def load_model(self):
        """Carica il checkpoint ufficiale dell'architettura neurale AttentionPerZoneModel."""
        ckpt_candidates = [
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_geometric.pth"),
            os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth")
        ]
        ckpt_path = None
        for p in ckpt_candidates:
            if os.path.exists(p):
                ckpt_path = p
                break

        model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state)
                model.eval()
                print(f"• Modello neurale caricato con successo da: {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Errore caricamento checkpoint: {e}")
                model.eval()
        else:
            print("[WARN] Checkpoint neurale non trovato. Inizializzazione pesi random.")
            model.eval()
        return model

    def load_icons(self):
        """Carica le icone vettoriali stilizzate."""
        icons_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        self.icons = {
            'car_blue': Image.open(os.path.join(icons_dir, "car_blue.png")),
            'car_ego': Image.open(os.path.join(icons_dir, "car_ego.png")),
            'truck_purple': Image.open(os.path.join(icons_dir, "truck_purple.png")),
            'truck_cab_purple': Image.open(os.path.join(icons_dir, "truck_cab_purple.png")),
            'trailer_purple': Image.open(os.path.join(icons_dir, "trailer_purple.png")),
            'construction_orange': Image.open(os.path.join(icons_dir, "construction_orange.png")),
            'pedestrian_green': Image.open(os.path.join(icons_dir, "pedestrian_green.png")),
            'bicycle_orange': Image.open(os.path.join(icons_dir, "bicycle_orange.png")),
            'motorcycle_amber': Image.open(os.path.join(icons_dir, "motorcycle_amber.png")),
            'barrier_hazard': Image.open(os.path.join(icons_dir, "barrier_hazard.png")),
            'cart_trolley': Image.open(os.path.join(icons_dir, "cart_trolley.png")),
            'traffic_cone': Image.open(os.path.join(icons_dir, "traffic_cone.png")),
        }

    def get_ego_velocity(self, idx):
        try:
            s_curr = self.adapter.all_samples[idx]
            sd_curr = self.adapter.nusc.get('sample_data', s_curr['data']['LIDAR_TOP'])
            p_curr = self.adapter.nusc.get('ego_pose', sd_curr['ego_pose_token'])

            if idx + 1 < self.total_frames:
                s_next = self.adapter.all_samples[idx + 1]
                sd_next = self.adapter.nusc.get('sample_data', s_next['data']['LIDAR_TOP'])
                p_next = self.adapter.nusc.get('ego_pose', sd_next['ego_pose_token'])
                dt = abs(p_next['timestamp'] - p_curr['timestamp']) * 1e-6
                p_a, p_b = p_curr, p_next
            elif idx > 0:
                s_prev = self.adapter.all_samples[idx - 1]
                sd_prev = self.adapter.nusc.get('sample_data', s_prev['data']['LIDAR_TOP'])
                p_prev = self.adapter.nusc.get('ego_pose', sd_prev['ego_pose_token'])
                dt = abs(p_curr['timestamp'] - p_prev['timestamp']) * 1e-6
                p_a, p_b = p_prev, p_curr
            else:
                return 0.0

            if dt <= 0: return 0.0
            dx = p_b['translation'][0] - p_a['translation'][0]
            dy = p_b['translation'][1] - p_a['translation'][1]
            dz = p_b['translation'][2] - p_a['translation'][2]
            return float((np.sqrt(dx**2 + dy**2 + dz**2) / dt) * 3.6)
        except Exception:
            return 0.0

    def check_sync_file(self):
        try:
            if os.path.exists(SYNC_FILE):
                with open(SYNC_FILE, "r") as f:
                    content = f.read().strip()
                if content:
                    parts = content.split(",")
                    target_idx = int(parts[0])
                    source = parts[1] if len(parts) > 1 else ""
                    if source != "neural_inputs" and target_idx != self.current_idx and 0 <= target_idx < self.total_frames:
                        self.load_frame(target_idx, broadcast=False)
        except Exception:
            pass

    def _write_sync(self, idx):
        try:
            os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
            with open(SYNC_FILE, "w") as f:
                f.write(f"{idx},neural_inputs")
        except Exception:
            pass

    def load_frame(self, idx, broadcast=True):
        self.current_idx = max(0, min(self.total_frames - 1, idx))
        self.selected_zone_idx = 0
        if broadcast:
            self._write_sync(self.current_idx)

        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.sample_token = self.frame_data["sample_token"]
        self.ego_speed_kmh = self.get_ego_velocity(self.current_idx)

        cur_sample = self.adapter.all_samples[self.current_idx]
        sc_rec = self.adapter.nusc.get('scene', cur_sample['scene_token'])
        self.scene_name = sc_rec['name']
        log_rec = self.adapter.nusc.get('log', sc_rec['log_token'])
        self.location = log_rec['location']

        # 1. Estrazione degli 11 canali completi della scena BEV (11, 200, 200)
        self.extract_11_channels()

        # 2. Estrazione e calcolo delle zone d'ombra con i 9 scalari
        self.extract_occlusion_zones()

        # 3. Rileva pseudo-box statici (muri/edifici) con il clustering geometrico di ray_caster.py
        try:
            rc = RayCaster(self.frame_data, verbose=False)
            self.static_boxes = rc.detect_static_manmade_boxes(self.frame_data.get('boxes', []))
        except Exception:
            self.static_boxes = []

        # 4. Calcolo Pesi Channel Attention (Squeeze-and-Excitation) globali
        self.compute_channel_attention()

        self.render()

    def extract_11_channels(self):
        """Costruisce esattamente l'array tensoriale a 11 canali (11, 200, 200) del modello."""
        # 1. Canale 0: LiDAR Density BEV
        self.lidar_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        points = self.frame_data.get("points", np.zeros((0, 3)))
        if len(points) > 0:
            for px_m, py_m in points[:, :2]:
                px = int((px_m + GRID_RANGE) / VOXEL_SIZE)
                py = int((GRID_RANGE - py_m) / VOXEL_SIZE)
                if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                    self.lidar_bev[py, px] = 1.0

        # 2. Canale 1: Ombre Raycasting (Maschera cumulativa)
        self.occlusion_mask = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        token = self.frame_data.get("sample_token", "")
        json_path = os.path.join(ROOT_DIR, "extracted_occlusions", f"{token}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                raw_data = json.load(f)
                occs = raw_data.get("occlusions", []) if isinstance(raw_data, dict) else raw_data
                for occ in occs:
                    pts = occ.get("polygon_points_m", [])
                    if len(pts) >= 3:
                        pts_arr = np.array(pts)
                        pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
                        mask = rasterize_polygon(pts_xy)
                        self.occlusion_mask = np.maximum(self.occlusion_mask, mask)
        else:
            rc = RayCaster(self.frame_data, verbose=False)
            res = rc.calculate_occlusions(self.frame_data)
            for poly in res.get("occlusions", []):
                pts = poly.get("polygon_points", [])
                if len(pts) >= 3:
                    pts_arr = np.array(pts)
                    pts_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
                    mask = rasterize_polygon(pts_xy)
                    self.occlusion_mask = np.maximum(self.occlusion_mask, mask)

        # 3. Canali 2, 3, 4: HD Map (Drivable, Walkway, Ped Crossing)
        semantic_map = self.frame_data.get('semantic_map', {})
        self.drivable_mask = np.maximum(
            semantic_map.get('drivable_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)),
            semantic_map.get('carpark_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        )
        self.walkway_mask = semantic_map.get('walkway', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        self.ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))

        # 4. Canali 5-10: Target masks (Auto, Camion, Pedoni, Moto, Bici, Barriere)
        target_masks = extract_ground_truth_masks(self.frame_data)

        self.channels_11 = [
            self.lidar_bev,
            self.occlusion_mask,
            self.drivable_mask,
            self.walkway_mask,
            self.ped_crossing_mask
        ] + list(target_masks)

        self.input_tensor = torch.tensor(np.stack(self.channels_11, axis=0), dtype=torch.float32)

    def extract_occlusion_zones(self):
        """Estrae le zone d'ombra entro 25 metri e calcola i 9 descrittori scalari per ciascuna."""
        import cv2
        dt_road_map = cv2.distanceTransform((1 - self.drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * VOXEL_SIZE

        occ_file = os.path.join("extracted_occlusions", f"{self.sample_token}.json")
        self.occlusion_zones = []

        if os.path.exists(occ_file):
            with open(occ_file, "r") as f:
                data = json.load(f)
                raw_occs = data.get("occlusions", []) if isinstance(data, dict) else data
        else:
            raw_occs = []

        circle_25m = Point(0, 0).buffer(self.max_range - 0.4)

        for i, occ in enumerate(raw_occs):
            dist = float(occ.get("distance_m", 0.0))
            if dist > self.max_range:
                continue

            pts = occ.get("polygon_points_m", [])
            pts_np = np.array(pts)
            if len(pts_np) < 3:
                continue
            poly_xy = pts_np[:, :2]

            sp = ShapelyPolygon(poly_xy)
            if not sp.is_valid or sp.area <= 0.01:
                continue

            poly_vis = sp.intersection(circle_25m)
            if poly_vis.is_empty or poly_vis.area < 0.02:
                continue

            occ_mask = rasterize_polygon(pts)
            tot = np.sum(occ_mask)
            road_f = float(np.sum(occ_mask * self.drivable_mask) / tot if tot > 0 else 0.0)
            side_f = float(np.sum(occ_mask * self.walkway_mask) / tot if tot > 0 else 0.0)
            cross_f = float(np.sum(occ_mask * self.ped_crossing_mask) / tot if tot > 0 else 0.0)
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = float(occ.get("area_sqm", 0.0))

            min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
            roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

            if sp.is_valid and sp.area > 0.01:
                mrr = sp.minimum_rotated_rectangle
                mrr_coords = np.array(mrr.exterior.coords)[:-1]
                e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))
            else:
                obb_w, obb_l = 0.5, 0.5

            scalars = [area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f]

            # Verifiche vincolo neuro-simbolico fisico
            auto_allowed = (obb_w >= 1.7 and road_f >= 0.15)
            truck_allowed = (obb_w >= 2.3 and road_f >= 0.15)
            vru_allowed = True

            # Centroide per etichetta
            c_pt = sp.centroid
            cx, cy = float(c_pt.x), float(c_pt.y)

            self.occlusion_zones.append({
                "idx": len(self.occlusion_zones),
                "poly_xy": poly_xy,
                "shapely": sp,
                "center": (cx, cy),
                "scalars": scalars,
                "area": area,
                "dist": dist,
                "obb_w": obb_w,
                "obb_l": obb_l,
                "road_f": road_f,
                "side_f": side_f,
                "cross_f": cross_f,
                "roadside_f": roadside_f,
                "terr_f": terr_f,
                "auto_allowed": auto_allowed,
                "truck_allowed": truck_allowed,
                "vru_allowed": vru_allowed
            })

    def compute_channel_attention(self):
        """Calcola i pesi Channel Attention Squeeze-and-Excitation su scala globale."""
        try:
            with torch.no_grad():
                inp = self.input_tensor.unsqueeze(0).to(self.device)
                b, c, _, _ = inp.shape
                y = inp.view(b, c, -1).mean(dim=2)
                y_relu = torch.relu(self.model.input_se.fc1(y))
                self.se_weights = torch.sigmoid(self.model.input_se.fc2(y_relu)).squeeze().cpu().numpy()
        except Exception:
            self.se_weights = np.ones(11) * 0.5

    def draw_icon_object(self, ax, icon_name, center, length=3.0, deg=0.0, width=None,
                          flip_h=False, halo_color=None, halo_radius=None, zorder=8):
        """Disegna un'icona raster/vettoriale trasparente ruotata e scalata preservando le proporzioni naturali."""
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Circle
        from PIL import Image

        if halo_color and halo_radius:
            ax.add_patch(Circle(center, halo_radius, facecolor=halo_color, edgecolor='none', alpha=0.55, zorder=zorder - 1))

        if icon_name not in self.icons:
            return

        icon_img = self.icons[icon_name]
        if flip_h:
            icon_img = icon_img.transpose(Image.FLIP_LEFT_RIGHT)

        img_w, img_h = icon_img.size
        aspect = img_h / img_w

        if width is None:
            width = length / aspect

        tr = mtransforms.Affine2D().rotate_deg_around(center[0], center[1], deg) + ax.transData
        extent = [center[0] - width / 2.0, center[0] + width / 2.0,
                  center[1] - length / 2.0, center[1] + length / 2.0]
        ax.imshow(icon_img, extent=extent, transform=tr, zorder=zorder, origin='upper', clip_on=True)

    def draw_bev_map(self, ax):
        """Renderizza la mappa BEV accademica con strade, marciapiedi, ombre e ostacoli."""
        import scipy.ndimage as ndimage
        c_border = '#0F172A'
        ax.set_facecolor('#F8FAFC')
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-self.max_range, self.max_range)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.2)

        # Cerchi metrici di riferimento (10m, 20m, 25m)
        for r in [10.0, 20.0, 25.0]:
            ls = '-' if r == self.max_range else '--'
            lw = 1.3 if r == self.max_range else 0.8
            ec = c_border if r == self.max_range else '#94A3B8'
            c = Circle((0, 0), r, facecolor='none', edgecolor=ec, linestyle=ls, linewidth=lw, zorder=2)
            ax.add_patch(c)
            if r < self.max_range:
                ax.text(0, -r, f"{int(r)}m", color='#64748B', fontsize=7.5, fontweight='bold',
                        ha='center', va='center', zorder=3,
                        bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))

        # Disegno delle maschere semantiche HD di sfondo con map_rgba
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
        drivable_rot = np.rot90(self.drivable_mask, 3)
        walkway_rot  = np.rot90(self.walkway_mask, 3)
        crossing_rot = np.rot90(self.ped_crossing_mask, 3)
        h, w = drivable_rot.shape
        map_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        map_rgba[walkway_rot == 1] = [218, 224, 233, 245]
        map_rgba[drivable_rot == 1] = [125, 125, 125, 255]
        if np.any(crossing_rot):
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            stripe_diag = ((xx + yy) % 4) < 2
            border = (crossing_rot == 1) & (~ndimage.binary_erosion(crossing_rot == 1, structure=np.ones((3, 3))))
            map_rgba[crossing_rot == 1] = [90, 90, 90, 255]
            map_rgba[(crossing_rot == 1) & stripe_diag] = [255, 255, 255, 255]
            map_rgba[border == 1] = [255, 255, 255, 255]
        ax.imshow(map_rgba, extent=extent, origin='lower', zorder=2)

        # Punti LiDAR reali nuScenes entro raggio max_range
        pts = self.frame_data.get('points', None)
        if pts is not None and len(pts) > 0:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            pts_r = pts[dists <= self.max_range]
            if len(pts_r) > 4000:
                pts_sub = pts_r[::len(pts_r) // 4000]
            else:
                pts_sub = pts_r
            ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=1.1, c='#0F172A', alpha=0.55, zorder=3)

        # Disegno delle zone d'ombra
        self.drawn_occlusions = []
        for z_i, z in enumerate(self.occlusion_zones):
            poly_xy = z["poly_xy"]
            is_selected = (z_i == self.selected_zone_idx and self.show_auxiliary)

            fc = '#FEF08A' if is_selected else '#94A3B8'
            ec = '#B45309' if is_selected else '#475569'
            lw = 2.0 if is_selected else 1.0
            alpha = 0.65 if is_selected else 0.35

            patch = MplPolygon(poly_xy, closed=True, facecolor=fc, edgecolor=ec,
                               linewidth=lw, alpha=alpha, zorder=4)
            ax.add_patch(patch)

            # OBB per la zona selezionata
            if is_selected:
                sp = z["shapely"]
                if sp.is_valid and sp.area > 0.01:
                    mrr = sp.minimum_rotated_rectangle
                    mrr_coords = np.array(mrr.exterior.coords)
                    ax.plot(mrr_coords[:, 0], mrr_coords[:, 1], color='#D97706',
                            linewidth=1.8, linestyle='--', zorder=6)

            # Etichetta numerica zona (Z1, Z2, ...)
            cx, cy = z["center"]
            if -self.max_range < cx < self.max_range and -self.max_range < cy < self.max_range:
                lbl_color = '#FFFFFF' if is_selected else '#1E293B'
                bg_color = '#B45309' if is_selected else '#FFFFFF'
                ax.text(cx, cy, f"Z{z_i+1}", fontsize=7.2, fontweight='bold', color=lbl_color,
                        ha='center', va='center', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.18', facecolor=bg_color,
                                  edgecolor='#0F172A', linewidth=0.8, alpha=0.9))

            from matplotlib.path import Path as MplPath
            self.drawn_occlusions.append({
                'patch': patch,
                'path': MplPath(poly_xy),
                'meta': z,
                'default_fc': fc,
                'default_ec': ec,
                'default_lw': lw,
                'default_alpha': alpha
            })

        # Strutture statiche man-made (muri/edifici rilevati da clustering LiDAR)
        c_static_face = '#EF4444'
        c_static_edge = '#B91C1C'
        for sb in getattr(self, 'static_boxes', []):
            c_x = (sb.min_x + sb.max_x) / 2.0
            c_y = (sb.min_y + sb.max_y) / 2.0
            if np.hypot(c_x, c_y) <= self.max_range:
                rect = Rectangle((sb.min_x, sb.min_y), sb.max_x - sb.min_x, sb.max_y - sb.min_y,
                                 facecolor=c_static_face, edgecolor=c_static_edge,
                                 linewidth=1.8, linestyle='--', alpha=0.45, zorder=6)
                ax.add_patch(rect)

        # Disegno ostacoli reali con icone
        boxes = self.frame_data.get('boxes', [])
        coupled_truck_tokens = set()
        for b1 in boxes:
            if "truck" in b1.name.lower():
                c1 = b1.center[:2]
                for b2 in boxes:
                    if "trailer" in b2.name.lower():
                        c2 = b2.center[:2]
                        if np.linalg.norm(c1 - c2) <= (b1.wlh[1] + b2.wlh[1]) / 2.0 + 3.5:
                            coupled_truck_tokens.add(b1.token)
                            break

        for box in boxes:
            b_name = box.name.lower()
            if "barrier" in b_name:      cat = "Barriers"
            elif "pushable" in b_name or "pullable" in b_name: cat = "Carts"
            elif "trafficcone" in b_name or "cone" in b_name: cat = "Cones"
            elif "bicycle" in b_name:    cat = "Bicycles"
            elif "motorcycle" in b_name: cat = "Motorcycles"
            elif "pedestrian" in b_name or "human" in b_name: cat = "Pedestrians"
            elif "car" in b_name or "emergency" in b_name:    cat = "Cars"
            elif any(c in b_name for c in ["truck", "trailer", "bus", "construction"]): cat = "Trucks/Buses"
            else:
                continue

            corners_bev = box.corners_3d[:2, [0, 1, 5, 4]].T
            center_xy = np.mean(corners_bev, axis=0)
            dist = np.linalg.norm(center_xy)
            if dist > self.max_range:
                continue

            f_mid = (corners_bev[0] + corners_bev[1]) / 2.0
            r_mid = (corners_bev[2] + corners_bev[3]) / 2.0
            u_vec = f_mid - r_mid
            length = max(1.2, np.linalg.norm(u_vec))
            width = max(0.8, np.linalg.norm(corners_bev[1] - corners_bev[0]))
            deg = -np.degrees(np.arctan2(u_vec[0], u_vec[1]))

            if cat == "Pedestrians":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax, 'pedestrian_green', center=center_xy, length=2.8, deg=0.0, flip_h=flip,
                                      halo_color='#DCFCE7', halo_radius=1.5, zorder=8)
            elif cat == "Cars":
                c_l = min(4.8, max(3.8, length))
                self.draw_icon_object(ax, 'car_blue', center=center_xy, length=c_l, deg=deg, zorder=8)
            elif cat == "Trucks/Buses":
                if "construction" in b_name:
                    flip = (u_vec[0] < 0)
                    max_w = 4.5
                    exc_w = min(max_w, max(3.2, length * 0.65))
                    exc_l = exc_w * (345.0 / 537.0)
                    self.draw_icon_object(ax, 'construction_orange', center=center_xy, length=exc_l, width=exc_w, deg=0.0, flip_h=flip, zorder=8)
                elif "trailer" in b_name:
                    t_l = min(13.5, max(4.5, length))
                    t_w = min(2.6, max(1.8, width))
                    t_center = f_mid - (u_vec / length) * (t_l / 2.0)
                    self.draw_icon_object(ax, 'trailer_purple', center=t_center, length=t_l, width=t_w, deg=deg, zorder=8)
                elif "truck" in b_name:
                    if box.token in coupled_truck_tokens:
                        cab_len = min(4.0, length * 0.65)
                        cab_center = f_mid - (u_vec / length) * (cab_len / 2.0 + 0.35)
                        self.draw_icon_object(ax, 'truck_cab_purple', center=cab_center, length=cab_len, width=width * 1.05, deg=deg, zorder=8)
                    else:
                        tr_len = min(5.2, max(4.0, length)) if length <= 6.8 else min(7.8, max(5.5, length))
                        tr_w = min(2.1, max(1.7, width)) if length <= 6.8 else min(2.5, max(2.0, width))
                        self.draw_icon_object(ax, 'truck_purple', center=center_xy, length=tr_len, width=tr_w, deg=deg, zorder=8)
                else:
                    eff_len = min(7.8, max(5.5, length))
                    eff_w = min(2.5, max(2.0, width))
                    self.draw_icon_object(ax, 'truck_purple', center=center_xy, length=eff_len, width=eff_w, deg=deg, zorder=8)
            elif cat == "Bicycles":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax, 'bicycle_orange', center=center_xy, length=2.6, deg=0.0, flip_h=flip,
                                      halo_color='#FED7AA', halo_radius=1.5, zorder=8)
            elif cat == "Motorcycles":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax, 'motorcycle_amber', center=center_xy, length=2.6, deg=0.0, flip_h=flip,
                                      halo_color='#FDE68A', halo_radius=1.5, zorder=8)
            elif cat == "Barriers":
                self.draw_icon_object(ax, 'barrier_hazard', center=center_xy, length=2.2, deg=0.0, zorder=8)
            elif cat == "Carts":
                self.draw_icon_object(ax, 'cart_trolley', center=center_xy, length=2.2, deg=0.0,
                                      halo_color='#E0F2FE', halo_radius=1.3, zorder=8)
            elif cat == "Cones":
                self.draw_icon_object(ax, 'traffic_cone', center=center_xy, length=1.5, deg=0.0, zorder=8)

        # Ego Vehicle al centro
        self.draw_icon_object(ax, 'car_ego', center=[0.0, 0.0], length=5.0, deg=0.0, zorder=12)

        # Box telemetrico in basso a sinistra
        n_zones = len(self.occlusion_zones)
        ax.text(0.03, 0.035,
                f"Ego: {self.ego_speed_kmh:.1f} km/h  |  Scena: {self.scene_name}  |  "
                f"Zone Ombra: {n_zones}  |  Canali: 11  |  BEV: 200x200 (0.4m/px)",
                transform=ax.transAxes, fontsize=7.6, color='#475569', fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=0.9),
                zorder=20)

    def render(self):
        """Renderizza la dashboard completa in base alla modalità attiva."""
        self.fig.clf()
        self.ui_buttons = []

        c_border = '#0F172A'

        # Titolo in alto a sinistra
        self.fig.text(0.03, 0.962,
                      "Figure 6: Rappresentazione Multimodale degli Input (11 Canali BEV & Rete Ausiliaria)",
                      fontsize=10.5, fontweight='bold', color=c_border, ha='left', va='center')

        # Pulsante Toggle Rete Neurale Ausiliaria (Centro-Destra)
        aux_bg = '#4338CA' if self.show_auxiliary else '#F8FAFC'
        aux_fg = '#FFFFFF' if self.show_auxiliary else '#1E293B'
        aux_lbl = '★ RETE AUSILIARIA (MLP): ON' if self.show_auxiliary else '★ Mostra Rete Ausiliaria (MLP)'
        ax_aux = self.fig.add_axes([0.550, 0.942, 0.205, 0.034])
        btn_aux = Button(ax_aux, aux_lbl, color=aux_bg,
                         hovercolor='#3730A3' if self.show_auxiliary else '#E2E8F0')
        btn_aux.label.set_fontsize(7.8)
        btn_aux.label.set_fontweight('bold')
        btn_aux.label.set_color(aux_fg)
        btn_aux.on_clicked(self.toggle_auxiliary)
        self.ui_buttons.append((ax_aux, btn_aux))

        # Navigazione Frame (Destra)
        ax_prev = self.fig.add_axes([0.768, 0.942, 0.042, 0.034])
        btn_prev = Button(ax_prev, "◀ Prec", color='#F8FAFC', hovercolor='#E2E8F0')
        btn_prev.label.set_fontsize(8.0)
        btn_prev.label.set_fontweight('bold')
        btn_prev.label.set_color(c_border)
        btn_prev.on_clicked(lambda ev: self.on_key_step(-1))
        self.ui_buttons.append((ax_prev, btn_prev))

        self.fig.text(0.816, 0.959, "Frame:", fontsize=8.0, fontweight='bold', color=c_border, ha='left', va='center')
        ax_tb = self.fig.add_axes([0.852, 0.942, 0.052, 0.034])
        self.txt_frame = TextBox(ax_tb, '', initial=str(self.current_idx + 1),
                                 color='#F8FAFC', hovercolor='#E2E8F0')
        self.txt_frame.text_disp.set_color('#1D4ED8')
        self.txt_frame.text_disp.set_fontweight('bold')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ui_buttons.append((ax_tb, self.txt_frame))

        ax_next = self.fig.add_axes([0.914, 0.942, 0.052, 0.034])
        btn_next = Button(ax_next, "Succ ▶", color='#F8FAFC', hovercolor='#E2E8F0')
        btn_next.label.set_fontsize(8.0)
        btn_next.label.set_fontweight('bold')
        btn_next.label.set_color(c_border)
        btn_next.on_clicked(lambda ev: self.on_key_step(1))
        self.ui_buttons.append((ax_next, btn_next))

        if not self.show_auxiliary:
            self.render_mode_11_channels()
        else:
            self.render_mode_auxiliary_network()

        # Tooltip interattivo
        self.tooltip = self.ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#38BDF8", linewidth=1.5, alpha=0.95),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        # Footer con scorciatoie da tastiera
        self.fig.text(0.50, 0.015,
                      "[<- / ->] o [A / D]: Naviga Frame  |  [T / Spazio / M]: Toggle Rete Ausiliaria  |  "
                      "[Z / X]: Scorri Zone  |  [Click BEV]: Ispeziona Scalari & FiLM  |  [S]: Salva HD",
                      fontsize=8.0, color='#64748B', ha='center', va='center')

        self.fig.canvas.draw_idle()

    def render_mode_11_channels(self):
        """Renderizza la vista principale: Mappa BEV a sinistra + Griglia 11 canali a destra."""
        gs = GridSpec(1, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.06, top=0.91,
                      width_ratios=[1.1, 1.25], wspace=0.10)

        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.draw_bev_map(self.ax_map)

        # Sotto-griglia per gli 11 canali + 12esimo slot per Attention Weights
        gs_grid = GridSpec(3, 4, figure=self.fig, left=0.50, right=0.97, bottom=0.06, top=0.91,
                           wspace=0.18, hspace=0.28)

        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]

        for i in range(11):
            row = i // 4
            col = i % 4
            ax_c = self.fig.add_subplot(gs_grid[row, col])
            meta = CHANNEL_METADATA[i]
            ch_data = self.channels_11[i]

            ax_c.set_facecolor('#0F172A' if i == 0 else '#F8FAFC')
            ax_c.set_xticks([])
            ax_c.set_yticks([])
            for spine in ax_c.spines.values():
                spine.set_color('#0F172A')
                spine.set_linewidth(1.0)

            # Visualizza la mappa del canale (rot90 x3 per allineare l'orientamento alle mappe semantiche nuScenes)
            v_max = max(1.0, float(np.max(ch_data)))
            ax_c.imshow(np.rot90(ch_data, 3), extent=extent, cmap=meta["cmap"], vmin=0, vmax=v_max, origin='lower')

            # Indicatore peso attention per canale
            w_se = self.se_weights[i] if i < len(self.se_weights) else 0.5
            w_col = '#15803D' if w_se >= 0.50 else '#D97706'

            # Titolo pulito con badge attention
            ax_c.set_title(f"{meta['title']}\nSE Attn: {w_se:.2f}", fontsize=7.2, fontweight='bold',
                           color='#0F172A', pad=3)

            # Cerchio ego al centro
            ax_c.plot(0, 0, marker='o', color='#38BDF8', markersize=2.5, zorder=5)

        # 12esimo slot: Grafico a barre dei pesi Channel Attention (Squeeze-and-Excitation)
        ax_attn = self.fig.add_subplot(gs_grid[2, 3])
        ax_attn.set_facecolor('#FFFFFF')
        ax_attn.spines['top'].set_visible(False)
        ax_attn.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax_attn.spines[spine].set_color('#0F172A')
            ax_attn.spines[spine].set_linewidth(0.9)

        y_pos = np.arange(11)
        colors = ['#1D4ED8' if w >= 0.50 else '#94A3B8' for w in self.se_weights]
        ax_attn.barh(y_pos, self.se_weights[::-1], color=colors[::-1], height=0.75, edgecolor='#0F172A', linewidth=0.6)
        ax_attn.set_yticks(y_pos)
        ax_attn.set_yticklabels([f"C{10-i}" for i in range(11)], fontsize=6.8, fontweight='bold', color='#1E293B')
        ax_attn.set_xlim(0, 1.0)
        ax_attn.set_xticks([0, 0.5, 1.0])
        ax_attn.set_xticklabels(['0', '0.5', '1.0'], fontsize=6.5, color='#475569')
        ax_attn.set_title("Pesi Attention (SE)\nPer Canale", fontsize=7.2, fontweight='bold', color='#0F172A', pad=3)
        ax_attn.axvline(0.5, color='#DC2626', linestyle=':', linewidth=0.8, alpha=0.7)

    def render_mode_auxiliary_network(self):
        """Renderizza la vista Rete Neurale Ausiliaria: Mappa BEV a sinistra + Ispezione MLP/FiLM a destra."""
        gs = GridSpec(1, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.06, top=0.91,
                      width_ratios=[1.1, 1.25], wspace=0.10)

        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.draw_bev_map(self.ax_map)

        # Pannello destro suddiviso in 4 card verticali
        gs_right = GridSpec(4, 1, figure=self.fig, left=0.50, right=0.97, bottom=0.06, top=0.91,
                            height_ratios=[1.25, 0.85, 1.0, 0.9], hspace=0.30)

        ax_card1 = self.fig.add_subplot(gs_right[0, 0])
        ax_card2 = self.fig.add_subplot(gs_right[1, 0])
        ax_card3 = self.fig.add_subplot(gs_right[2, 0])
        ax_card4 = self.fig.add_subplot(gs_right[3, 0])

        c_border = '#0F172A'

        if not self.occlusion_zones:
            ax_card1.text(0.5, 0.5, "Nessuna zona d'ombra rilevata entro 25m in questo frame",
                          ha='center', va='center', fontsize=9.0, color='#64748B')
            for ax in [ax_card1, ax_card2, ax_card3, ax_card4]:
                ax.set_xticks([]); ax.set_yticks([])
            return

        active_zone = self.occlusion_zones[min(self.selected_zone_idx, len(self.occlusion_zones) - 1)]
        z_idx = active_zone["idx"]
        scalars_list = active_zone["scalars"]
        s_tensor = torch.tensor([scalars_list], dtype=torch.float32).to(self.device)

        # Esecuzione inferenza sulla rete ausiliaria (MLP + FiLM)
        with torch.no_grad():
            sc_feat = self.model.scalar_mlp(s_tensor)
            film_params = self.model.film_gen(sc_feat)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            sc_feat_np = sc_feat.squeeze().cpu().numpy()
            gamma_np = gamma.squeeze().cpu().numpy()
            beta_np = beta.squeeze().cpu().numpy()

        # =====================================================================
        # CARD 1: I 9 DESCRITTORI SCALARI TOPOLOGICI/GEOMETRICI
        # =====================================================================
        ax_card1.set_facecolor('#F8FAFC')
        for spine in ax_card1.spines.values():
            spine.set_color(c_border); spine.set_linewidth(1.0)
        ax_card1.set_xlim(0, 1); ax_card1.set_ylim(0, 1)
        ax_card1.set_xticks([]); ax_card1.set_yticks([])

        ax_card1.text(0.025, 0.91,
                      f"1. DESCRITTORI SCALARI TOPOLOGICI - ZONA Z{z_idx+1} (Tasti [Z]/[X] o Click BEV)",
                      fontsize=8.3, fontweight='bold', color='#1E293B', va='center')

        # Griglia 3x3 per visualizzare i 9 scalari in formato card
        for k, meta in enumerate(SCALAR_METADATA):
            r_i = k // 3
            c_i = k % 3
            x_box = 0.025 + c_i * 0.325
            y_box = 0.61 - r_i * 0.28
            w_box = 0.300
            h_box = 0.235

            val = scalars_list[meta["idx"]]
            if meta["unit"] == "%":
                val_str = f"{int(round(val * 100))}%"
            else:
                val_str = f"{val:.2f} {meta['unit']}"

            ax_card1.add_patch(FancyBboxPatch((x_box, y_box), w_box, h_box,
                                              boxstyle="round,pad=0.015,rounding_size=0.02",
                                              facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=0.9))

            ax_card1.text(x_box + 0.015, y_box + h_box - 0.065, meta["name"],
                          fontsize=6.8, color='#475569', fontweight='bold', va='center')

            ax_card1.text(x_box + 0.015, y_box + 0.075, val_str,
                          fontsize=9.2, color=meta["color"], fontweight='bold', va='center')

        # =====================================================================
        # CARD 2: ARCHITETTURA RETE AUSILIARIA MLP (9 -> 64 -> 128)
        # =====================================================================
        ax_card2.set_facecolor('#FFFFFF')
        for spine in ax_card2.spines.values():
            spine.set_color(c_border); spine.set_linewidth(1.0)
        ax_card2.set_xlim(0, 1); ax_card2.set_ylim(0, 1)
        ax_card2.set_xticks([]); ax_card2.set_yticks([])

        ax_card2.text(0.025, 0.82, "2. RETE AUSILIARIA MLP: Linear(9, 64) -> LayerNorm -> ReLU -> Linear(64, 128) -> LayerNorm -> ReLU",
                      fontsize=7.8, fontweight='bold', color='#4338CA', va='center')

        l2_norm = float(np.linalg.norm(sc_feat_np))
        mean_act = float(np.mean(sc_feat_np))
        max_act = float(np.max(sc_feat_np))

        stats_mlp = (f"Vettore Ausiliario: sc_feat ∈ ℝ¹²⁸   |   "
                     f"Norma L2: {l2_norm:.2f}   |   Attivazione Media: {mean_act:.2f}   |   Max: {max_act:.2f}")
        ax_card2.text(0.025, 0.54, stats_mlp, fontsize=7.6, color='#1E293B', va='center')

        # Visualizzazione spettrogramma 1D delle 128 feature estratte dall'MLP
        ax_feat = ax_card2.inset_axes([0.025, 0.12, 0.95, 0.28])
        ax_feat.imshow(sc_feat_np.reshape(1, 128), aspect='auto', cmap='magma', origin='upper')
        ax_feat.set_xticks([]); ax_feat.set_yticks([])
        for sp in ax_feat.spines.values():
            sp.set_color(c_border); sp.set_linewidth(0.8)

        # =====================================================================
        # CARD 3: MODULAZIONE MULTIMODALE FiLM (GAMMA & BETA)
        # =====================================================================
        ax_card3.set_facecolor('#F8FAFC')
        for spine in ax_card3.spines.values():
            spine.set_color(c_border); spine.set_linewidth(1.0)
        ax_card3.spines['top'].set_visible(False)
        ax_card3.spines['right'].set_visible(False)

        ax_card3.set_title("3. MODULAZIONE AFFINE FiLM: Linear(128, 256) → [γ_moltiplicativo, β_additivo]",
                           fontsize=8.0, fontweight='bold', color='#1E293B', pad=4, loc='left')

        # Istogramma comparativo dei coefficienti FiLM
        bins = np.linspace(-2.0, 2.0, 30)
        ax_card3.hist(gamma_np, bins=bins, alpha=0.65, color='#2563EB', label=f'γ (Fattore Scala, μ={gamma_np.mean():.2f})', edgecolor='#1D4ED8')
        ax_card3.hist(beta_np, bins=bins, alpha=0.65, color='#DC2626', label=f'β (Shift Additivo, μ={beta_np.mean():.2f})', edgecolor='#B91C1C')
        ax_card3.axvline(0.0, color='#64748B', linestyle='--', linewidth=0.8)
        ax_card3.legend(loc='upper right', fontsize=7.2, frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')
        ax_card3.set_xlim(-2.0, 2.0)
        ax_card3.tick_params(axis='both', which='major', labelsize=6.8)

        # =====================================================================
        # CARD 4: VINCOLO NEURO-SIMBOLICO & AMMISSIBILITÀ FISICA
        # =====================================================================
        ax_card4.set_facecolor('#FFFFFF')
        for spine in ax_card4.spines.values():
            spine.set_color(c_border); spine.set_linewidth(1.0)
        ax_card4.set_xlim(0, 1); ax_card4.set_ylim(0, 1)
        ax_card4.set_xticks([]); ax_card4.set_yticks([])

        ax_card4.text(0.025, 0.85, "4. VINCOLO NEURO-SIMBOLICO INTEGRATO (AFFORDANCE GATING)",
                      fontsize=8.0, fontweight='bold', color='#1E293B', va='center')

        # Stato delle 3 classi chiave
        classes_gating = [
            ("Auto / Van", active_zone["auto_allowed"], "Varco ≥ 1.7m & Asfalto ≥ 15%"),
            ("Camion / Bus", active_zone["truck_allowed"], "Varco ≥ 2.3m & Asfalto ≥ 15%"),
            ("VRU (Pedoni/Ciclisti)", active_zone["vru_allowed"], "Ammesso ovunque (Marciapiede / Strisce / Varchi stretti)")
        ]

        for i, (c_name, is_ok, rule_str) in enumerate(classes_gating):
            y_r = 0.58 - i * 0.24
            badge_c = '#15803D' if is_ok else '#DC2626'
            badge_bg = '#DCFCE7' if is_ok else '#FEE2E2'
            status_txt = "[OK] AMMESSO" if is_ok else "[NO] INIBITO"

            ax_card4.add_patch(FancyBboxPatch((0.025, y_r - 0.04), 0.22, 0.18,
                                              boxstyle="round,pad=0.015,rounding_size=0.02",
                                              facecolor=badge_bg, edgecolor=badge_c, linewidth=0.9))

            ax_card4.text(0.135, y_r + 0.05, f"{c_name}: {status_txt}",
                          fontsize=7.2, fontweight='bold', color=badge_c, ha='center', va='center')

            ax_card4.text(0.270, y_r + 0.05, f"Regola: {rule_str}",
                          fontsize=7.2, color='#475569', va='center')

    def toggle_auxiliary(self, event=None):
        """Attiva o disattiva la visualizzazione della rete ausiliaria."""
        self.show_auxiliary = not self.show_auxiliary
        self.render()

    def cycle_zone(self, delta):
        """Passa alla zona d'ombra precedente o successiva."""
        if not self.occlusion_zones:
            return
        n_zones = len(self.occlusion_zones)
        self.selected_zone_idx = (self.selected_zone_idx + delta) % n_zones
        if not self.show_auxiliary:
            self.show_auxiliary = True
        self.render()

    def on_key(self, event):
        """Gestione delle scorciatoie da tastiera."""
        if event.key in ['left', 'a', 'A']:
            self.on_key_step(-1)
        elif event.key in ['right', 'd', 'D']:
            self.on_key_step(1)
        elif event.key in ['z', 'Z', 'up']:
            self.cycle_zone(-1)
        elif event.key in ['x', 'X', 'down']:
            self.cycle_zone(1)
        elif event.key in ['space', 't', 'T', 'm', 'M']:
            self.toggle_auxiliary()
        elif event.key in ['s', 'S']:
            out_p = os.path.join(ROOT_DIR, "scratch", f"neural_inputs_frame_{self.current_idx+1}.png")
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            self.fig.savefig(out_p, dpi=300, bbox_inches='tight')
            print(f"[OK] Salvata visualizzazione HD in: {out_p}")

    def on_key_step(self, step):
        new_idx = self.current_idx + step
        if 0 <= new_idx < self.total_frames:
            self.load_frame(new_idx)

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            if 1 <= val <= self.total_frames:
                self.load_frame(val - 1)
        except ValueError:
            pass

    def on_mouse_click(self, event):
        """Seleziona una zona d'ombra con il click del mouse."""
        if event.inaxes != self.ax_map:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        pt = Point(x, y)

        for i, item in enumerate(self.drawn_occlusions):
            if item['path'].contains_point((x, y)):
                self.selected_zone_idx = i
                if not self.show_auxiliary:
                    self.show_auxiliary = True
                self.render()
                break

    def on_mouse_move(self, event):
        """Mostra tooltip informativo al passaggio del mouse."""
        if event.inaxes != self.ax_map:
            if self.tooltip and self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        found = False
        for item in self.drawn_occlusions:
            if item['path'].contains_point((x, y)):
                z = item['meta']
                z_i = z['idx']
                tt_text = (
                    f"ZONA D'OMBRA Z{z_i+1}\n"
                    f"• Area: {z['area']:.1f} m²  |  Distanza: {z['dist']:.1f} m\n"
                    f"• Varco OBB: {z['obb_w']:.2f}m x {z['obb_l']:.2f}m\n"
                    f"• Asfalto: {int(z['road_f']*100)}%  |  Marciapiede: {int(z['side_f']*100)}%\n"
                    f"• Auto Ammessa: {'Sì' if z['auto_allowed'] else 'No'}  |  Camion: {'Sì' if z['truck_allowed'] else 'No'}\n"
                    f"[Click per aprire Rete Ausiliaria]"
                )
                self.tooltip.xy = (x, y)
                self.tooltip.set_position((15, 15))
                self.tooltip.set_text(tt_text)
                self.tooltip.set_visible(True)
                found = True
                break

        if not found and self.tooltip and self.tooltip.get_visible():
            self.tooltip.set_visible(False)
        self.fig.canvas.draw_idle()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore Ufficiale Input Multimodali & Rete Ausiliaria Tesi")
    parser.add_argument("frame", nargs="?", type=int, default=None, help="Numero del frame iniziale (1-404, default: 1)")
    parser.add_argument("--aux", action="store_true", help="Mostra direttamente la Rete Neurale Ausiliaria (MLP)")
    parser.add_argument("--range", type=float, default=25.0, help="Raggio operativo BEV in metri (default: 25.0)")
    parser.add_argument("--save", type=str, default=None, help="Salva l'immagine ad alta risoluzione (300 DPI) ed esce")
    args = parser.parse_args()

    vis = NeuralInputsVisualizer(max_range=args.range, initial_aux=args.aux)
    if args.frame is not None and 1 <= args.frame <= vis.total_frames:
        vis.load_frame(args.frame - 1, broadcast=False)

    if args.save:
        out_p = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        vis.fig.savefig(out_p, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualizzazione salvata con successo in: {out_p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
