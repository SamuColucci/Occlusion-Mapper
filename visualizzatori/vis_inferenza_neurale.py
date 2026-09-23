# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - INFERENZA NEURALE & ANTICIPAZIONE RISCHIO (BEV NUSCENES)
# File: visualizzatori/vis_inferenza_neurale.py
#
# Figure 4 Tesi: Inferenza Neurale Live del Modello AttentionPerZoneModel nelle Zone Occluse
# Basato sulla stessa architettura, stile visivo, dimensioni e posizioni di:
# - visualizzatori/vis_raycasting_occlusioni.py (Figure 2)
# - visualizzatori/vis_ground_truth_occlusioni.py (Figure 3)
#
# DISTINZIONE VISIVA OGGETTI:
# - OSTACOLI VISIBILI AL LIDAR: icone con i loro colori standard (Auto Blu,
#   Camion Viola, Pedone Verde, Barriera Gialla/Nera) con Bounding Box 3D nuScenes;
# - OSTACOLI PREDETTI DAL MODELLO (in ombra): TUTTI I CORRISPETTIVI OGGETTI SONO
#   COLORATI DI ROSSO (Auto Rossa, Camion Rosso, Pedone Rosso, Barriera Rossa)
#   così da rendere la separazione percettiva immediata ed inequivocabile!
# ==============================================================================

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from scipy import ndimage
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
    rasterize_polygon
)
from ground_truth.ground_truth_extractor_synthetic import apply_occluder_compatibility_filter
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")
GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4

MODEL_CONFIGS = {
    "NEURO_SIMB": {
        "title": "Anticipazione Neuro-Simbolica Ibrida (Spazio 3D + HD-Map)",
        "short": "★ IBRIDO (SPAZIO+HDMAP)",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
        "badge_color": "#2563EB",
        "badge_bg": "#DBEAFE",
    },
    "GEOMETRIC": {
        "title": "Supervisione Geometrica (Fitting 3D)",
        "short": "★ GEOMETRICO (FITTING 3D)",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_geometric.pth"),
        "badge_color": "#0284C7",
        "badge_bg": "#E0F2FE",
    },
    "SEMANTIC": {
        "title": "Supervisione Semantica (Regole Naïve HD-Map)",
        "short": "★ SEMANTICO (REGOLE HDMAP)",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_semantic.pth"),
        "badge_color": "#E11D48",
        "badge_bg": "#FFF1F2",
    },
    "REAL_GT": {
        "title": "Supervisione Reale Completa (nuScenes 3D)",
        "short": "GT REALE (COMPLETA)",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth"),
        "badge_color": "#0D9488",
        "badge_bg": "#CCFBF1",
    },
    "POS_ONLY": {
        "title": "Supervisione Solo Zone Piene (Positives-Only)",
        "short": "SOLO POSITIVI (ZONE PIENE)",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_positive_only.pth"),
        "badge_color": "#D97706",
        "badge_bg": "#FEF3C7",
    }
}
MODEL_CONFIGS["HYBRID"] = MODEL_CONFIGS["NEURO_SIMB"]
MODEL_CONFIGS["GEOMETRICA"] = MODEL_CONFIGS["GEOMETRIC"]
MODEL_CONFIGS["SEMANTICA"] = MODEL_CONFIGS["SEMANTIC"]


def percorso_ckpt(model_key):
    """Restituisce il percorso assoluto del checkpoint del modello (relativo a ROOT_DIR)."""
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["NEURO_SIMB"])
    ckpt = cfg.get("ckpt")
    if not ckpt:
        return None
    return ckpt if os.path.isabs(ckpt) else os.path.join(ROOT_DIR, ckpt)


def modello_disponibile(model_key):
    """True se il file dei pesi del modello esiste su disco (nessun fallback su altri modelli)."""
    p = percorso_ckpt(model_key)
    return p is not None and os.path.exists(p)


class NeuralInferenceVisualizer:
    def __init__(self, max_range=25.0, initial_model="NEURO_SIMB"):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: INFERENZA NEURALE & ANTICIPAZIONE RISCHIO (BEV 25M)")
        print("=" * 80)

        self.max_range = float(max_range)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Dispositivo di Calcolo Attivo: {self.device}")

        # Inizializzazione Adapter nuScenes
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0
        init_key = initial_model.upper()
        if init_key in ["HYBRID", "IBRIDO", "NEURO_SIMB", "NEURO"]:
            self.current_model_key = "NEURO_SIMB"
        elif init_key in ["GEOMETRIC", "GEOMETRICA", "GEOM"]:
            self.current_model_key = "GEOMETRIC"
        elif init_key in ["SEMANTIC", "SEMANTICA", "SEM"]:
            self.current_model_key = "SEMANTIC"
        elif init_key in ["REAL", "REALE", "REAL_GT"]:
            self.current_model_key = "REAL_GT"
        elif init_key in ["POS", "POSITIVES", "POS_ONLY", "SOLO_POSITIVI"]:
            self.current_model_key = "POS_ONLY"
        else:
            self.current_model_key = init_key if init_key in MODEL_CONFIGS else "NEURO_SIMB"

        # Setup tipografia accademica
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Neural Risk Inference Visualizer")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None
        self.dropdown_open = False
        self.menu_buttons = []
        self.last_sync_idx = -1
        self.inference_latency_ms = 0.0

        # Cache modelli per commutazione istantanea senza rilettura da disco
        self.loaded_models = {}
        self.missing_reported = set()
        self._preload_model(self.current_model_key)

        self.load_icons()
        self.load_frame(self.current_idx, broadcast=False)

        # Timer sincronizzazione bidirezionale
        self.sync_timer = self.fig.canvas.new_timer(interval=150)
        self.sync_timer.add_callback(self.check_sync_file)
        self.sync_timer.start()

    def _preload_model(self, model_key):
        """Carica il checkpoint specificato con caching."""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        # Pesi mancanti: il modello e' segnalato come MANCANTE (nessun fallback su altri checkpoint)
        ckpt_path = percorso_ckpt(model_key)
        if not modello_disponibile(model_key):
            if model_key not in self.missing_reported:
                print(f"[MANCANTE] Pesi non trovati per {model_key}: {ckpt_path}")
                self.missing_reported.add(model_key)
            return None

        model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        self.loaded_models[model_key] = model
        print(f"• Modello [{model_key}] caricato con successo da: {ckpt_path}")
        return model

    def load_icons(self):
        """Carica le icone vettoriali standard e le rispettive versioni ROSSE per le predizioni."""
        icons_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        self.icons = {
            # Icone visibili LiDAR standard (uguali a vis_raycasting_occlusioni.py)
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
            # Versioni ROSSE per gli ostacoli predetti dalla rete neurale
            'car_red': Image.open(os.path.join(icons_dir, "car_red.png")),
            'pedestrian_red': Image.open(os.path.join(icons_dir, "pedestrian_red.png")),
            'truck_red': Image.open(os.path.join(icons_dir, "truck_red.png")),
            'trailer_red': Image.open(os.path.join(icons_dir, "trailer_red.png")),
            'barrier_red': Image.open(os.path.join(icons_dir, "barrier_red.png")),
            'bicycle_red': Image.open(os.path.join(icons_dir, "bicycle_red.png")),
        }

    def get_ego_velocity(self, idx):
        """Calcola la velocità istantanea dell'Ego Vehicle in km/h."""
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

            if dt <= 0:
                return 0.0
            dx = p_b['translation'][0] - p_a['translation'][0]
            dy = p_b['translation'][1] - p_a['translation'][1]
            dz = p_b['translation'][2] - p_a['translation'][2]
            return float((np.sqrt(dx**2 + dy**2 + dz**2) / dt) * 3.6)
        except Exception:
            return 0.0

    def check_sync_file(self):
        """Sincronizzazione automatica bidirezionale tra visualizzatori."""
        try:
            if os.path.exists(SYNC_FILE):
                with open(SYNC_FILE, "r") as f:
                    content = f.read().strip()
                if content:
                    parts = content.split(",")
                    target_idx = int(parts[0])
                    source = parts[1] if len(parts) > 1 else ""
                    if source != "neural" and target_idx != self.current_idx and 0 <= target_idx < self.total_frames:
                        self.load_frame(target_idx, broadcast=False)
        except Exception:
            pass

    def _write_sync(self, idx):
        try:
            os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
            with open(SYNC_FILE, "w") as f:
                f.write(f"{idx},neural")
        except Exception:
            pass

    def load_frame(self, idx, broadcast=True):
        """Carica il frame ed esegue l'inferenza del modello attivo su tutte le zone occluse."""
        self.current_idx = max(0, min(self.total_frames - 1, idx))
        if broadcast:
            self._write_sync(self.current_idx)

        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.sample_token = self.frame_data["sample_token"]
        self.ego_speed_kmh = self.get_ego_velocity(self.current_idx)

        # Informazioni di scena
        cur_sample = self.adapter.all_samples[self.current_idx]
        sc_rec = self.adapter.nusc.get('scene', cur_sample['scene_token'])
        self.scene_name = sc_rec['name']
        log_rec = self.adapter.nusc.get('log', sc_rec['log_token'])
        self.location = log_rec['location']

        # Rileva pseudo-box statici (muri/edifici) con il clustering geometrico di ray_caster.py
        try:
            rc = RayCaster(self.frame_data, verbose=False)
            self.static_boxes = rc.detect_static_manmade_boxes(self.frame_data.get('boxes', []))
        except Exception:
            self.static_boxes = []

        # Esecuzione inferenza neurale live sulle zone occluse
        self._run_neural_inference()
        self.render()

    def _run_neural_inference(self):
        """Esegue l'inferenza conforme al 100% con evaluate_final_official.py."""
        model = self._preload_model(self.current_model_key)
        self.inferred_occlusions = []
        if model is None:
            return

        occ_file = os.path.join("extracted_occlusions", f"{self.sample_token}.json")
        if not os.path.exists(occ_file):
            return

        with open(occ_file, "r") as f:
            occs = json.load(f)["occlusions"]

        # Maschera circolare metrica a 25m per tagliare rigorosamente TUTTI gli 11 canali di input
        r_grid = np.arange(GRID_DIM)
        gy, gx = np.meshgrid(r_grid, r_grid)
        mask_25m = (np.sqrt((gx - 99.5)**2 + (gy - 99.5)**2) * VOXEL_SIZE <= self.max_range).astype(np.float32)
        circle_25m = Point(0, 0).buffer(self.max_range)

        # 1. Canale 0: LiDAR Density BEV entro 25m
        lidar_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        points = self.frame_data.get("points", np.zeros((0, 3)))
        if len(points) > 0:
            for px_m, py_m in points[:, :2]:
                px = int((px_m + GRID_RANGE) / VOXEL_SIZE)
                py = int((GRID_RANGE - py_m) / VOXEL_SIZE)
                if 0 <= px < GRID_DIM and 0 <= py < GRID_DIM:
                    lidar_bev[py, px] = 1.0
        lidar_bev = lidar_bev * mask_25m

        # 2. Canale 1: Ombre Raycasting cumulative entro 25m
        occlusion_mask = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32)
        for occ in occs:
            pts = occ.get("polygon_points_m", [])
            if len(pts) >= 3:
                sp_occ = ShapelyPolygon(pts)
                if not sp_occ.is_valid:
                    sp_occ = sp_occ.buffer(0)
                sp_occ_25 = sp_occ.intersection(circle_25m)
                if not sp_occ_25.is_empty and sp_occ_25.area >= 0.1:
                    sub_geoms = list(sp_occ_25.geoms) if sp_occ_25.geom_type == 'MultiPolygon' else [sp_occ_25]
                    for sg in sub_geoms:
                        coords = np.array(sg.exterior.coords)[:-1]
                        m = rasterize_polygon(coords)
                        occlusion_mask = np.maximum(occlusion_mask, m)
        occlusion_mask = occlusion_mask * mask_25m

        # 3. Canali 2, 3, 4: HD Map entro 25m
        semantic_map = self.frame_data['semantic_map']
        drivable_mask = (np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))) * mask_25m
        walkway_mask = (semantic_map.get('walkway', np.zeros_like(drivable_mask))) * mask_25m
        ped_crossing_mask = (semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))) * mask_25m

        # 4. Canali 5-10: Ostacoli visibili nuScenes entro 25m
        target_masks = extract_ground_truth_masks(self.frame_data) * mask_25m

        input_channels = [
            lidar_bev,
            occlusion_mask,
            drivable_mask,
            walkway_mask,
            ped_crossing_mask
        ] + list(target_masks)

        input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

        start_time = time.perf_counter()
        cat_names_4 = ["Auto", "Camion/Bus", "VRU (Pedoni)", "Barriera"]

        # Mappa delle distanze euclidee dalla carreggiata (drivable area) in metri
        import cv2
        dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * VOXEL_SIZE

        # Mappa rapida per estrarre le dimensioni 3D dell'occludore se noto
        boxes_by_token = {b.token: b for b in self.frame_data.get('boxes', [])}
        for sb in getattr(self, 'static_boxes', []):
            boxes_by_token[sb.token] = sb

        for occ in occs:
            pts = occ.get("polygon_points_m", [])
            pts_np = np.array(pts)
            if len(pts_np) < 3:
                continue
            poly_xy = pts_np[:, :2]

            sp = ShapelyPolygon(poly_xy)
            if not sp.is_valid or sp.area <= 0.01:
                sp = sp.buffer(0)
            if not sp.is_valid or sp.area <= 0.01:
                continue

            # Filtro rigoroso: l'ombra viene ritagliata esattamente entro la circonferenza dei 25m
            poly_vis = sp.intersection(circle_25m)
            if poly_vis.is_empty or poly_vis.area < 0.1:
                continue

            occ_name = occ.get('object_name', None)
            occ_tok = occ.get('object_token', None)
            b_obj = boxes_by_token.get(occ_tok, None)
            if b_obj is not None:
                if hasattr(b_obj, 'wlh'):
                    occ_wlh = b_obj.wlh
                elif hasattr(b_obj, 'max_x'):
                    occ_wlh = [b_obj.max_x - b_obj.min_x, b_obj.max_y - b_obj.min_y, b_obj.max_z - b_obj.min_z]
                else:
                    occ_wlh = None
                if not occ_name and hasattr(b_obj, 'name'):
                    occ_name = b_obj.name
            else:
                occ_wlh = None

            sub_geoms = list(poly_vis.geoms) if poly_vis.geom_type == 'MultiPolygon' else [poly_vis]
            for poly_main in sub_geoms:
                if poly_main.area < 0.1:
                    continue

                vis_coords = np.array(poly_main.exterior.coords)[:-1]
                if len(vis_coords) < 3:
                    continue

                # Calcolo feature semantiche e geometriche STRETTAMENTE sulla porzione visibile entro 25m
                occ_mask = rasterize_polygon(vis_coords)
                tot = np.sum(occ_mask)
                road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
                side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
                cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
                terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                area = float(poly_main.area)
                dist = float(np.hypot(poly_main.centroid.x, poly_main.centroid.y))

                # Prossimità al bordo strada
                min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
                roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

                # Pipeline di ritaglio patch calcolata strettamente sui vertici entro 25m
                px_x = np.clip(((vis_coords[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                px_y = np.clip(((GRID_RANGE - vis_coords[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device)

                # Calcolo OBB strettamente sulla geometria 25m
                mrr = poly_main.minimum_rotated_rectangle
                mrr_coords = np.array(mrr.exterior.coords)[:-1]
                e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))

                # Vettore scalari: [area, dist, width, length, road_f, side_f, cross_f, roadside_f, terr_f]
                scalars = torch.tensor([[area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f]], dtype=torch.float32).to(self.device)

                occluder_mask = AttentionPerZoneModel.build_compatibility_mask(
                    occ_name, occ_wlh, road_f=road_f, roadside_f=roadside_f, device=self.device
                )
                with torch.no_grad():
                    out_6 = torch.sigmoid(model(patch_res, scalars, occluder_mask=occluder_mask)).squeeze(0).cpu().numpy()

                # Mappatura pura dell'output del modello a 4 macro-classi (nessuna modifica post-hoc)
                p4 = np.array([
                    out_6[0],
                    out_6[1],
                    max(out_6[2], out_6[3], out_6[4]),
                    out_6[5]
                ], dtype=np.float32)


                max_class_idx = int(np.argmax(p4))
                max_prob = float(p4[max_class_idx])
                pred_class = cat_names_4[max_class_idx]

                pt = poly_main.representative_point()
                cx_m, cy_m = float(pt.x), float(pt.y)

                self.inferred_occlusions.append({
                    "occ": occ,
                    "poly_xy": vis_coords,
                    "center": (cx_m, cy_m),
                    "area": area,
                    "dist": dist,
                    "road_f": road_f,
                    "side_f": side_f,
                    "cross_f": cross_f,
                    "roadside_f": roadside_f,
                    "min_d_road": min_d_road,
                    "scalars": scalars.squeeze(0).cpu().numpy(),
                    "probs_4": p4,
                    "pred_class": pred_class,
                    "max_prob": max_prob,
                    "risk_score": float(np.max(p4)),
                    "is_hazard": bool(np.max(p4) >= 0.35)
                })

        self.inference_latency_ms = (time.perf_counter() - start_time) * 1000.0

    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                         flip_h=False, halo_color=None, halo_radius=None):
        """Disegna un'icona raster ruotata e scalata preservando le proporzioni naturali (stile vis_raycasting)."""
        if halo_color and halo_radius:
            ax.add_patch(Circle(center, halo_radius, facecolor=halo_color, edgecolor='none', alpha=0.55, zorder=6))

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
        extent = [center[0] - width/2.0, center[0] + width/2.0,
                  center[1] - length/2.0, center[1] + length/2.0]
        ax.imshow(icon_img, extent=extent, transform=tr, zorder=8, origin='upper', clip_on=True)

    def render(self):
        """Renderizza la finestra grafica completa a 2 colonne in perfetto stile tesi."""
        self.fig.clf()
        self.fig.patch.set_facecolor('#FFFFFF')

        # Layout a griglia perfettamente identico a Figure 2 e Figure 3
        gs = GridSpec(3, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.925,
                      width_ratios=[1.1, 0.9], height_ratios=[1.25, 1.05, 1.20],
                      wspace=0.14, hspace=0.28)

        ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map = ax_map
        self.drawn_occlusions = []
        self.active_hovered = None

        ax_telemetry = self.fig.add_subplot(gs[0, 1])
        ax_objects = self.fig.add_subplot(gs[1, 1])
        ax_legend = self.fig.add_subplot(gs[2, 1])

        # Titolo visualizzatore pulito e conciso in alto a sinistra
        self.fig.text(0.03, 0.956,
                      f"Figure 4: Inferenza Neurale & Anticipazione Rischio (BEV Raggio {int(self.max_range)}m)",
                      fontsize=11.5, fontweight='bold', color='#0F172A', ha='left', va='center')

        # Costanti colori accademici (speculari a vis_raycasting_occlusioni.py)
        c_map_bg = '#F8FAFC'
        c_grid = '#E2E8F0'
        c_border = '#0F172A'
        c_car = '#1D4ED8'
        c_ped = '#16A34A'
        c_truck = '#9333EA'
        c_bike = '#EA580C'
        c_moto = '#D97706'
        c_barrier = '#F59E0B'
        c_cone_safe = '#64748B'
        c_cone_hazard = '#DC2626'
        c_static_face = '#EF4444' # Strutture statiche (Rosso)
        c_static_edge = '#B91C1C' # Bordo tratteggiato strutture statiche (Rosso scuro)

        # =====================================================================
        # 1. PANNELLO SINISTRO: MAPPA BEV AD ALTA FEDELTÀ (25M)
        # =====================================================================
        ax_map.set_facecolor(c_map_bg)
        ax_map.set_xlim(-self.max_range, self.max_range)
        ax_map.set_ylim(-self.max_range, self.max_range)
        ax_map.set_aspect('equal')
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        # Quadrettatura cartesiana metrica
        ticks = np.arange(-int(self.max_range), int(self.max_range) + 5, 5)
        for t in ticks:
            ax_map.axvline(t, color=c_grid, linewidth=0.6, zorder=1)
            ax_map.axhline(t, color=c_grid, linewidth=0.6, zorder=1)

        # Assi ortogonali a croce sull'ego-vehicle
        ax_map.axhline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)
        ax_map.axvline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)

        # Cerchi concentrici metrici (10m, 20m, 25m)
        for r in [10.0, 20.0, self.max_range]:
            c = Circle((0, 0), r, color=c_border, fill=False, linewidth=1.1, zorder=3)
            ax_map.add_patch(c)
            ax_map.text(0, -r + 1.1, f"{int(r)}m", color=c_border, fontsize=9.5,
                        ha='center', va='center', fontweight='bold', zorder=10,
                        bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))

        # HD-Map Stradale Vettoriale
        smap = self.frame_data.get('semantic_map', None)
        if smap is not None:
            drivable = np.rot90(smap.get('drivable_area', np.zeros((200, 200))), 3)
            walkway = np.rot90(smap.get('walkway', np.zeros((200, 200))), 3)
            crossing = np.rot90(smap.get('ped_crossing', np.zeros((200, 200))), 3)

            h, w = drivable.shape
            map_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            map_rgba[walkway == 1] = [218, 224, 233, 245]
            map_rgba[drivable == 1] = [125, 125, 125, 255]
            if np.any(crossing):
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                stripe_diag = ((xx + yy) % 4) < 2
                border = crossing & (~ndimage.binary_erosion(crossing, structure=np.ones((3, 3))))
                map_rgba[crossing == 1] = [90, 90, 90, 255]
                map_rgba[(crossing == 1) & stripe_diag] = [255, 255, 255, 255]
                map_rgba[border == 1] = [255, 255, 255, 255]

            ax_map.imshow(map_rgba, extent=[-40.0, 40.0, -40.0, 40.0], origin='lower', zorder=2)

        # LiDAR nuScenes reale entro 25m
        pts = self.frame_data.get('points', None)
        pts_dists = []
        if pts is not None and len(pts) > 0:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            mask_r = dists <= self.max_range
            pts_r = pts[mask_r]
            pts_dists = dists[mask_r]
            if len(pts_r) > 0:
                ax_map.scatter(pts_r[:, 0], pts_r[:, 1], s=1.1, c='#0F172A',
                               alpha=0.68, zorder=3, edgecolors='none')

        # Conteggi operativi per il grafico analitico
        counts_visibili = {"Auto": 0, "Camion/Bus": 0, "VRU (Pedoni)": 0, "Barriere": 0}
        counts_pred = {"Auto": 0, "Camion/Bus": 0, "VRU (Pedoni)": 0, "Barriere": 0}

        # ---------------------------------------------------------------------
        # F. ZONE OCCLUSE & OSTACOLI PREDETTI DAL MODELLO (TUTTI ROSSI!)
        # ---------------------------------------------------------------------
        for meta in self.inferred_occlusions:
            poly_xy = meta["poly_xy"]
            risk = meta["risk_score"]
            pred_cls = meta["pred_class"]
            is_haz = meta["is_hazard"]
            cx, cy = meta["center"]
            dist_c = float(np.hypot(cx, cy))

            if risk >= 0.40:
                # Pericolo elevato: allerta rossa evidente
                fc = '#FCA5A5'
                ec = '#DC2626'
                alpha_fill = min(0.55, 0.25 + risk * 0.30)
                lw = 1.3
            elif risk >= 0.20:
                # Rischio moderato: allerta ambra/arancione
                fc = '#FED7AA'
                ec = '#EA580C'
                alpha_fill = 0.32
                lw = 1.0
            else:
                # Basso rischio: sfumatura morbida
                fc = '#E2E8F0'
                ec = '#64748B'
                alpha_fill = 0.25
                lw = 0.8

            poly_patch = MplPolygon(poly_xy, closed=True, facecolor=fc, edgecolor=ec,
                                    linewidth=lw, alpha=alpha_fill, zorder=5)
            ax_map.add_patch(poly_patch)

            # Posiziona l'OGGETTO PREDETTO ROSSO all'interno di OGNI cono d'ombra visibile (entro 25m)
            if dist_c <= 24.8 and abs(cx) <= 24.8 and abs(cy) <= 24.8:
                if pred_cls == "Auto":
                    counts_pred["Auto"] += 1
                    self.draw_icon_object(ax_map, 'car_red', center=[cx, cy], length=4.0, deg=0.0)
                    h_off = 2.4
                elif pred_cls == "Camion/Bus":
                    counts_pred["Camion/Bus"] += 1
                    self.draw_icon_object(ax_map, 'truck_red', center=[cx, cy], length=5.5, deg=0.0)
                    h_off = 3.1
                elif "VRU" in pred_cls:
                    counts_pred["VRU (Pedoni)"] += 1
                    self.draw_icon_object(ax_map, 'pedestrian_red', center=[cx, cy], length=2.8, deg=0.0,
                                          halo_color='#FEE2E2', halo_radius=1.5)
                    h_off = 1.8
                else:
                    counts_pred["Barriere"] += 1
                    self.draw_icon_object(ax_map, 'barrier_red', center=[cx, cy], length=2.2, deg=0.0)
                    h_off = 1.5

                # Badge percentuale compatto sopra all'oggetto rosso
                b_bg = '#DC2626' if risk >= 0.40 else ('#EA580C' if risk >= 0.20 else '#475569')
                b_ec = '#991B1B' if risk >= 0.40 else ('#C2410C' if risk >= 0.20 else '#1E293B')
                ax_map.text(cx, cy + h_off, f"{int(risk*100)}%",
                            fontsize=7.2, fontweight='bold', color='#FFFFFF',
                            ha='center', va='bottom', zorder=12, clip_on=True,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor=b_bg,
                                      edgecolor=b_ec, linewidth=0.8, alpha=0.92))

            from matplotlib.path import Path as MplPath
            self.drawn_occlusions.append({
                'patch': poly_patch,
                'path': MplPath(poly_xy),
                'meta': meta,
                'default_fc': fc,
                'default_ec': ec,
                'default_lw': lw,
                'default_alpha': alpha_fill,
                'default_z': 5
            })

        # ---------------------------------------------------------------------
        # F2. STRUTTURE STATICHE MAN-MADE (Muri / Edifici rilevati da clustering LiDAR)
        # ---------------------------------------------------------------------
        for sb in getattr(self, 'static_boxes', []):
            c_x = (sb.min_x + sb.max_x) / 2.0
            c_y = (sb.min_y + sb.max_y) / 2.0
            if np.hypot(c_x, c_y) <= self.max_range:
                w = sb.max_x - sb.min_x
                h = sb.max_y - sb.min_y
                rect = Rectangle((sb.min_x, sb.min_y), w, h,
                                 facecolor=c_static_face, edgecolor=c_static_edge,
                                 linewidth=1.8, linestyle='--', alpha=0.45, zorder=6)
                ax_map.add_patch(rect)

        # ---------------------------------------------------------------------
        # G. OSTACOLI REALI NUSCENES (VISIBILI AL LIDAR) - STILE UGUALE A VIS_NUSCENES_RAW
        # ---------------------------------------------------------------------
        boxes = self.frame_data.get('boxes', [])

        # Rilevamento intelligente TIR articolati
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

            if "car" in b_name or "emergency" in b_name:
                counts_visibili["Auto"] += 1
                c_l = min(4.8, max(3.8, length))
                self.draw_icon_object(ax_map, 'car_blue', center=center_xy, length=c_l, deg=deg)
            elif "truck" in b_name or "trailer" in b_name or "bus" in b_name or "construction" in b_name:
                counts_visibili["Camion/Bus"] += 1
                if "construction" in b_name:
                    flip = (u_vec[0] < 0)
                    max_w = 4.5
                    exc_w = min(max_w, max(3.2, length * 0.65))
                    exc_l = exc_w * (345.0 / 537.0)
                    self.draw_icon_object(ax_map, 'construction_orange', center=center_xy,
                                          length=exc_l, width=exc_w, deg=0.0, flip_h=flip)
                elif "trailer" in b_name:
                    t_l = min(13.5, max(4.5, length))
                    t_w = min(2.6, max(1.8, width))
                    t_center = f_mid - (u_vec / length) * (t_l / 2.0)
                    self.draw_icon_object(ax_map, 'trailer_purple', center=t_center, length=t_l, width=t_w, deg=deg)
                elif "truck" in b_name:
                    if box.token in coupled_truck_tokens:
                        cab_len = min(4.0, length * 0.65)
                        cab_center = f_mid - (u_vec / length) * (cab_len / 2.0 + 0.35)
                        self.draw_icon_object(ax_map, 'truck_cab_purple', center=cab_center,
                                              length=cab_len, width=width * 1.05, deg=deg)
                    else:
                        tr_len = min(5.2, max(4.0, length)) if length <= 6.8 else min(7.8, max(5.5, length))
                        tr_w = min(2.1, max(1.7, width)) if length <= 6.8 else min(2.5, max(2.0, width))
                        self.draw_icon_object(ax_map, 'truck_purple', center=center_xy, length=tr_len, width=tr_w, deg=deg)
                else:
                    self.draw_icon_object(ax_map, 'truck_purple', center=center_xy, length=min(7.8, max(5.5, length)), deg=deg)
            elif "pedestrian" in b_name or "human" in b_name:
                counts_visibili["VRU (Pedoni)"] += 1
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'pedestrian_green', center=center_xy, length=2.8, deg=0.0,
                                      flip_h=flip, halo_color='#DCFCE7', halo_radius=1.5)
            elif "bicycle" in b_name:
                counts_visibili["VRU (Pedoni)"] += 1
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'bicycle_orange', center=center_xy, length=2.6, deg=0.0,
                                      flip_h=flip, halo_color='#FED7AA', halo_radius=1.5)
            elif "motorcycle" in b_name:
                counts_visibili["VRU (Pedoni)"] += 1
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'motorcycle_amber', center=center_xy, length=2.6, deg=0.0,
                                      flip_h=flip, halo_color='#FDE68A', halo_radius=1.5)
            elif "barrier" in b_name:
                counts_visibili["Barriere"] += 1
                self.draw_icon_object(ax_map, 'barrier_hazard', center=center_xy, length=2.2, deg=0.0)
            elif "pushable" in b_name or "pullable" in b_name:
                self.draw_icon_object(ax_map, 'cart_trolley', center=center_xy, length=2.2, deg=0.0)
            elif "trafficcone" in b_name or "cone" in b_name:
                self.draw_icon_object(ax_map, 'traffic_cone', center=center_xy, length=1.5, deg=0.0)

        # Ego Vehicle al centro
        self.draw_icon_object(ax_map, 'car_ego', center=[0.0, 0.0], length=5.0, deg=0.0)

        # Avviso centrato sulla BEV se i pesi del modello attivo non esistono
        if not modello_disponibile(self.current_model_key):
            ckpt_nome = os.path.basename(percorso_ckpt(self.current_model_key) or "")
            ax_map.text(0.5, 0.5, f"Pesi del modello mancanti:\n{ckpt_nome}",
                        transform=ax_map.transAxes, fontsize=11, fontweight='bold',
                        color='#475569', ha='center', va='center', zorder=40,
                        bbox=dict(boxstyle="round,pad=0.6", facecolor='#F1F5F9',
                                  edgecolor='#64748B', linewidth=1.2, alpha=0.95))

        # Tooltip interattivo per dettagli inferenza zona
        self.tooltip = ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#EF4444", linewidth=1.5, alpha=0.95),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        # =====================================================================
        # 2. PANNELLO DESTRA ALTO: TELEMETRIA EGO VEHICLE & METADATI INFERENZA
        # =====================================================================
        ax_telemetry.set_facecolor('#F8FAFC')
        for spine in ax_telemetry.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)

        ax_telemetry.text(0.035, 0.89, "EGO VEHICLE TELEMETRY & INFERENCE METADATA",
                          transform=ax_telemetry.transAxes, fontsize=9.0, fontweight='bold', color=c_border)

        # Box velocità Ego Vehicle più compatto a sinistra (x=0.025, w=0.20)
        ax_telemetry.add_patch(Rectangle((0.025, 0.08), 0.20, 0.74, transform=ax_telemetry.transAxes,
                                         facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2))
        ax_telemetry.text(0.125, 0.63, "VELOCITÀ EGO", transform=ax_telemetry.transAxes,
                          fontsize=7.8, fontweight='bold', color='#1D4ED8', ha='center', va='center')
        ax_telemetry.text(0.105, 0.36, f"{self.ego_speed_kmh:.1f}", transform=ax_telemetry.transAxes,
                          fontsize=18, fontweight='bold', color='#0F172A', ha='center', va='center')
        ax_telemetry.text(0.175, 0.31, "km/h", transform=ax_telemetry.transAxes,
                          fontsize=7.5, color='#64748B', va='bottom')

        # Badge modello attivo centrato ed elegante (non tocca il bordo destro)
        cfg_curr = MODEL_CONFIGS.get(self.current_model_key, MODEL_CONFIGS["NEURO_SIMB"])
        badge_bg = cfg_curr.get("badge_bg", "#DBEAFE")
        badge_col = cfg_curr.get("badge_color", "#2563EB")
        badge_short = cfg_curr.get("short", self.current_model_key)
        pesi_ok = modello_disponibile(self.current_model_key)
        if pesi_ok:
            badge_txt = f"MODELLO ATTIVO: {badge_short} (ATTENTION)"
        else:
            # Stile neutro grigio per segnalare i pesi mancanti
            badge_bg, badge_col = "#F1F5F9", "#64748B"
            badge_txt = f"MODELLO ATTIVO: {badge_short} — PESI MANCANTI"

        ax_telemetry.add_patch(FancyBboxPatch((0.28, 0.63), 0.66, 0.17, boxstyle="round,pad=0.015,rounding_size=0.03",
                                              transform=ax_telemetry.transAxes,
                                              facecolor=badge_bg, edgecolor=badge_col, linewidth=1.1, zorder=3))
        ax_telemetry.text(0.61, 0.715, badge_txt,
                          transform=ax_telemetry.transAxes,
                          fontsize=8.2, fontweight='bold', color=badge_col, ha='center', va='center')

        # Metadati descrittivi ben spaziati a sinistra senza overflow
        hazards_count = sum(1 for m in self.inferred_occlusions if m["is_hazard"])
        meta_lines = [
            f"Dataset: nuScenes (v1.0-trainval)  |  Scena: {self.scene_name}",
            f"Campione: {self.current_idx + 1} / {self.total_frames}  |  Latenza GPU: {self.inference_latency_ms:.1f} ms",
            f"Zone d'Ombra: {len(self.inferred_occlusions)}  |  Pericoli Predetti (Rossi): {hazards_count}"
        ]
        y_m = 0.49
        for line in meta_lines:
            ax_telemetry.text(0.27, y_m, line, transform=ax_telemetry.transAxes,
                              fontsize=8.0, color='#334155', va='center')
            y_m -= 0.14

        ax_telemetry.set_xlim(0, 1)
        ax_telemetry.set_ylim(0, 1)
        ax_telemetry.set_xticks([])
        ax_telemetry.set_yticks([])

        # =====================================================================
        # 3. PANNELLO DESTRA CENTRO: CONFRONTO OSTACOLI VISIBILI VS OCCLUSI ANTICIPATI
        # =====================================================================
        ax_objects.set_facecolor('#FFFFFF')
        for spine in ax_objects.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.0)
        ax_objects.spines['top'].set_visible(False)
        ax_objects.spines['right'].set_visible(False)

        cat_labels = ["Auto", "Camion/Bus", "VRU (Pedoni)", "Barriere"]
        y_vis = [counts_visibili["Auto"], counts_visibili["Camion/Bus"], counts_visibili["VRU (Pedoni)"], counts_visibili["Barriere"]]
        y_pred = [counts_pred["Auto"], counts_pred["Camion/Bus"], counts_pred["VRU (Pedoni)"], counts_pred["Barriere"]]

        x_ind = np.arange(len(cat_labels))
        width = 0.35

        rects_vis = ax_objects.bar(x_ind - width/2.0, y_vis, width, label='Ostacoli Visibili (LiDAR)',
                                   color='#0284C7', edgecolor=c_border, linewidth=1.1, zorder=3)
        rects_pred = ax_objects.bar(x_ind + width/2.0, y_pred, width, label='Ostacoli Predetti (AI - Rossi)',
                                    color='#DC2626', edgecolor=c_border, linewidth=1.1, zorder=3)

        max_val = max(max(y_vis, default=1), max(y_pred, default=1), 4)
        ax_objects.set_ylim(0, max_val * 1.40 + 3)
        ax_objects.set_xticks(x_ind)
        ax_objects.set_xticklabels(cat_labels, fontsize=8.5, fontweight='bold', color=c_border)
        ax_objects.set_ylabel(f"Numero Ostacoli (Raggio {int(self.max_range)}m)", fontsize=8.5, fontweight='bold', color=c_border)
        ax_objects.set_title(f"Consapevolezza Spaziale: Ostacoli Visibili (LiDAR) vs Predetti (AI - Rossi)",
                             fontsize=9.8, fontweight='bold', pad=10, color=c_border)
        ax_objects.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)

        leg = ax_objects.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=True,
            facecolor='#FFFFFF',
            edgecolor='#CBD5E1',
            framealpha=0.96,
            fontsize=8.3,
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=1.8
        )
        leg.get_frame().set_boxstyle("round,pad=0.25,rounding_size=0.2")
        leg.get_frame().set_linewidth(0.9)

        for bar in rects_vis:
            h = bar.get_height()
            if h > 0:
                ax_objects.text(bar.get_x() + bar.get_width()/2.0, h + 0.25, str(int(h)),
                                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0284C7')
        for bar in rects_pred:
            h = bar.get_height()
            if h > 0:
                ax_objects.text(bar.get_x() + bar.get_width()/2.0, h + 0.25, str(int(h)),
                                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#DC2626')

        # =====================================================================
        # 4. PANNELLO DESTRA BASSO: LEGENDA UFFICIALE (VISIBILI vs PREDETTI ROSSI)
        # =====================================================================
        ax_legend.set_facecolor('#FFFFFF')
        for spine in ax_legend.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_legend.set_title("Simbologia Ufficiale (Ostacoli Visibili vs Predetti Rossi)",
                            fontsize=9.8, fontweight='bold', pad=9, color=c_border)

        # 3 colonne strutturate e traslate a sinistra per evitare qualunque overflow:
        # Colonna 1: Ostacoli Visibili LiDAR (Colori originali)
        ax_legend.text(0.15, 0.90, "OSTACOLI VISIBILI (LIDAR)", transform=ax_legend.transAxes,
                       fontsize=7.6, fontweight='bold', color='#0284C7', ha='center', va='center')
        col1_items = [
            ('icon', 'car_ego', 'Ego Vehicle', 0.044),
            ('icon', 'car_blue', 'Auto Visibile', 0.044),
            ('icon', 'truck_purple', 'Camion Visibile', 0.035),
            ('icon', 'pedestrian_green', 'Pedone Visibile', 0.050),
        ]
        for i, it in enumerate(col1_items):
            cur_y = 0.72 - i * 0.18
            oi = OffsetImage(self.icons[it[1]], zoom=it[3])
            ab = AnnotationBbox(oi, (0.035, cur_y), xycoords='axes fraction', frameon=False, zorder=5)
            ax_legend.add_artist(ab)
            ax_legend.text(0.075, cur_y, it[2], transform=ax_legend.transAxes,
                           fontsize=7.4, color=c_border, va='center', fontweight='medium')

        # Colonna 2: Ostacoli Predetti dalla Rete (TUTTI ROSSI!)
        ax_legend.text(0.46, 0.90, "PREDETTI DAL MODELLO (ROSSI)", transform=ax_legend.transAxes,
                       fontsize=7.6, fontweight='bold', color='#DC2626', ha='center', va='center')
        col2_items = [
            ('icon', 'car_red', 'Auto Predetta', 0.044),
            ('icon', 'truck_red', 'Camion Predetto', 0.035),
            ('icon', 'pedestrian_red', 'Pedone Predetto', 0.050),
            ('icon', 'barrier_red', 'Barriera Predetta', 0.040),
        ]
        for i, it in enumerate(col2_items):
            cur_y = 0.72 - i * 0.18
            oi = OffsetImage(self.icons[it[1]], zoom=it[3])
            ab = AnnotationBbox(oi, (0.335, cur_y), xycoords='axes fraction', frameon=False, zorder=5)
            ax_legend.add_artist(ab)
            ax_legend.text(0.375, cur_y, it[2], transform=ax_legend.transAxes,
                           fontsize=7.4, color='#DC2626', va='center', fontweight='bold')

        # Colonna 3: Mappa Stradale & Coni d'Ombra & Strutture Statiche
        ax_legend.text(0.79, 0.90, "MAPPA & CONI D'OMBRA", transform=ax_legend.transAxes,
                       fontsize=7.6, fontweight='bold', color='#334155', ha='center', va='center')
        col3_items = [
            ('rect', '#FCA5A5', c_cone_hazard, 'Cono d\'Ombra Pericolo', 0.55),
            ('rect', '#64748B', '#475569', 'Cono d\'Ombra Libero', 0.25),
            ('dashed_rect', c_static_face, c_static_edge, 'Struttura Statica (Muro)', 0.45),
            ('rect', '#7D7D7D', c_border, 'Carreggiata (HD-Map)', 1.0),
            ('rect', '#DAE0E9', c_border, 'Marciapiede (Walkway)', 1.0),
        ]
        for i, it in enumerate(col3_items):
            cur_y = 0.74 - i * 0.145
            if it[0] == 'rect':
                r = Rectangle((0.635 - 0.016, cur_y - 0.020), 0.032, 0.038,
                              transform=ax_legend.transAxes, facecolor=it[1], edgecolor=it[2],
                              linewidth=0.8, alpha=it[4], zorder=5)
            elif it[0] == 'dashed_rect':
                r = Rectangle((0.635 - 0.016, cur_y - 0.020), 0.032, 0.038,
                              transform=ax_legend.transAxes, facecolor=it[1], edgecolor=it[2],
                              linewidth=1.3, linestyle='--', alpha=it[4], zorder=5)
            ax_legend.add_patch(r)
            ax_legend.text(0.670, cur_y, it[3], transform=ax_legend.transAxes,
                           fontsize=7.2, color=c_border, va='center', fontweight='medium')

        # =====================================================================
        # 5. MENU A TENDINA & CASELLA DI SALTO FRAME IN ALTO
        # =====================================================================
        ax_dd = self.fig.add_axes([0.565, 0.938, 0.280, 0.036])
        arrow = "▲" if getattr(self, 'dropdown_open', False) else "▼"
        curr_lbl = cfg_curr.get("short", self.current_model_key)
        self.btn_mode = Button(ax_dd, f"Modello: {curr_lbl}  {arrow}", color='#F8FAFC', hovercolor='#E2E8F0')
        self.btn_mode.label.set_fontsize(7.8)
        self.btn_mode.label.set_fontweight('bold')
        self.btn_mode.label.set_color(c_border)
        self.btn_mode.label.set_ha('center')
        self.btn_mode.on_clicked(self.toggle_dropdown)
        self.ax_dd = ax_dd

        ax_tb = self.fig.add_axes([0.915, 0.938, 0.055, 0.036])
        self.txt_frame = TextBox(ax_tb, 'Frame: ', initial=str(self.current_idx + 1),
                                 color='#F8FAFC', hovercolor='#E2E8F0')
        self.txt_frame.label.set_fontsize(8.5)
        self.txt_frame.label.set_fontweight('bold')
        self.txt_frame.label.set_color(c_border)
        self.txt_frame.text_disp.set_color('#1D4ED8')
        self.txt_frame.text_disp.set_fontweight('bold')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ax_tb = ax_tb

        # Barra inferiore comandi
        self.fig.text(0.50, 0.012,
                      "[<- / ->] o [A / D]: Frame  |  [M / Menu]: Cambia Variante Attention  |  [Hover Mouse]: Dettagli Rischio  |  [S]: Salva HD 300 DPI",
                      fontsize=8.0, color='#64748B', ha='center', style='italic')

        self.fig.canvas.draw_idle()

    # =========================================================================
    # GESTIONE MENU A TENDINA PER SELEZIONE DEI 4 MODELLI
    # =========================================================================
    def toggle_dropdown(self, event=None):
        self.dropdown_open = not getattr(self, 'dropdown_open', False)
        if self.dropdown_open:
            self._show_dropdown_menu()
        else:
            self._hide_dropdown_menu()

    def _show_dropdown_menu(self):
        self._hide_dropdown_menu()
        self.dropdown_open = True

        self.ax_menu = self.fig.add_axes([0.565, 0.748, 0.280, 0.188], facecolor='#FFFFFF', zorder=100)
        self.ax_menu.set_xticks([])
        self.ax_menu.set_yticks([])
        for spine in self.ax_menu.spines.values():
            spine.set_color('#0F172A')
            spine.set_linewidth(1.3)

        modes = [
            ("NEURO_SIMB", "1. ★ Modello Ibrido (Spazio 3D + HD-Map)"),
            ("GEOMETRIC", "2. ★ Modello Geometrico (Fitting 3D)"),
            ("SEMANTIC", "3. ★ Modello Semantico (Regole Naïve)"),
            ("REAL_GT", "4. Modello GT Reale (Supervisione Completa)"),
            ("POS_ONLY", "5. Modello Solo Positivi (Zone Piene)")
        ]

        y_offsets = [0.148, 0.111, 0.074, 0.037, 0.001]
        self.menu_buttons = []
        for (m_key, m_lbl), y_off in zip(modes, y_offsets):
            is_active = (self.current_model_key == m_key)
            prefix = " ●  " if is_active else " ○  "
            bg_col = '#EFF6FF' if is_active else '#FFFFFF'
            txt_col = '#1D4ED8' if is_active else '#334155'
            # Modelli senza pesi: restano selezionabili ma marcati in grigio
            if not modello_disponibile(m_key):
                m_lbl = m_lbl + "  [MANCANTE]"
                txt_col = '#94A3B8'

            ax_item = self.fig.add_axes([0.567, 0.748 + y_off, 0.276, 0.035], zorder=101)
            btn = Button(ax_item, prefix + m_lbl, color=bg_col, hovercolor='#E0F2FE')
            btn.label.set_fontsize(7.8)
            btn.label.set_fontweight('bold' if is_active else 'normal')
            btn.label.set_color(txt_col)
            btn.label.set_ha('left')
            btn.label.set_x(0.035)
            btn.on_clicked(lambda ev, k=m_key: self.select_model(k))
            self.menu_buttons.append((ax_item, btn, m_key))

        self.fig.canvas.draw_idle()

    def _hide_dropdown_menu(self):
        if hasattr(self, 'menu_buttons') and self.menu_buttons:
            for item in self.menu_buttons:
                try:
                    item[0].remove()
                except Exception:
                    pass
            self.menu_buttons = []

        if hasattr(self, 'ax_menu') and self.ax_menu:
            try:
                self.ax_menu.remove()
                self.ax_menu = None
            except Exception:
                pass
        self.dropdown_open = False
        self.fig.canvas.draw_idle()

    def select_model(self, model_key):
        self._hide_dropdown_menu()
        if model_key in MODEL_CONFIGS and model_key != self.current_model_key:
            self.current_model_key = model_key
            self._preload_model(self.current_model_key)
            self._run_neural_inference()
            self.render()

    def on_jump_frame(self, text):
        try:
            val = int(text)
            target = val - 1
            if 0 <= target < self.total_frames:
                self.load_frame(target, broadcast=True)
            else:
                self.txt_frame.set_val(str(self.current_idx + 1))
        except ValueError:
            self.txt_frame.set_val(str(self.current_idx + 1))

    def on_key(self, event):
        if self.dropdown_open:
            self._hide_dropdown_menu()

        if event.key in ['right', 'd']:
            if self.current_idx < self.total_frames - 1:
                self.load_frame(self.current_idx + 1, broadcast=True)
        elif event.key in ['left', 'a']:
            if self.current_idx > 0:
                self.load_frame(self.current_idx - 1, broadcast=True)
        elif event.key in ['m', 'M', 'b', 'B']:
            cycle = ["NEURO_SIMB", "GEOMETRIC", "SEMANTIC", "REAL_GT", "POS_ONLY"]
            cur_i = cycle.index(self.current_model_key) if self.current_model_key in cycle else 0
            next_m = cycle[(cur_i + 1) % len(cycle)]
            self.select_model(next_m)
        elif event.key in ['s', 'S']:
            out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, f"fig_4_neural_inference_{self.current_model_key.lower()}_sample{self.current_idx + 1}.png")
            self.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
            print(f"\n[SALVATA CON SUCCESSO]: Immagine Inferenza Neurale salvata in:\n  -> {fpath}")

    def on_mouse_click(self, event):
        if getattr(self, 'dropdown_open', False):
            for item in getattr(self, 'menu_buttons', []):
                ax_item, _, m_key = item
                if event.inaxes == ax_item:
                    self.select_model(m_key)
                    return
            if event.inaxes == getattr(self, 'ax_dd', None):
                return
            self._hide_dropdown_menu()
            return

    def on_mouse_move(self, event):
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            if self.active_hovered is not None:
                patch = self.active_hovered['patch']
                patch.set_facecolor(self.active_hovered['default_fc'])
                patch.set_edgecolor(self.active_hovered['default_ec'])
                patch.set_linewidth(self.active_hovered['default_lw'])
                patch.set_alpha(self.active_hovered['default_alpha'])
                self.active_hovered = None
                if hasattr(self, 'tooltip') and self.tooltip is not None:
                    self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        hit = None
        for item in reversed(self.drawn_occlusions):
            if item['path'].contains_point((x, y)):
                hit = item
                break

        if hit is not None:
            if self.active_hovered is not hit:
                if self.active_hovered is not None:
                    p = self.active_hovered['patch']
                    p.set_facecolor(self.active_hovered['default_fc'])
                    p.set_edgecolor(self.active_hovered['default_ec'])
                    p.set_linewidth(self.active_hovered['default_lw'])
                    p.set_alpha(self.active_hovered['default_alpha'])

                self.active_hovered = hit
                p = hit['patch']
                p.set_facecolor('#EF4444')
                p.set_edgecolor('#B91C1C')
                p.set_linewidth(2.2)
                p.set_alpha(0.85)

            meta = hit['meta']
            risk = meta['risk_score']
            pred_cls = meta['pred_class']
            probs = meta['probs_4']

            if risk >= 0.70:
                allerta_str = "CRITICA (Pericolo Imminente)"
            elif risk >= 0.35:
                allerta_str = "ATTENZIONE (Ostacolo Nascosto)"
            else:
                allerta_str = "SICURA (Spazio Libero)"

            # Calcolo composizione del suolo disgiunta (somma = 100%)
            rf_pct = int(round(meta.get('road_f', 0.0) * 100))
            sf_pct = int(round(meta.get('side_f', 0.0) * 100))
            cf_pct = int(round(meta.get('cross_f', 0.0) * 100))
            tf_pct = max(0, 100 - (rf_pct + sf_pct + cf_pct))

            suolo_parts = []
            if rf_pct > 0: suolo_parts.append(f"Strada {rf_pct}%")
            if sf_pct > 0: suolo_parts.append(f"Marciapiede {sf_pct}%")
            if cf_pct > 0: suolo_parts.append(f"Strisce {cf_pct}%")
            if tf_pct > 0: suolo_parts.append(f"Terreno/Verde {tf_pct}%")
            suolo_str = " | ".join(suolo_parts) if suolo_parts else "Terreno/Verde 100%"

            d_bordo = meta.get('min_d_road', 0.0)
            bordo_str = f"a contatto (d={d_bordo:.1f}m)" if d_bordo <= 0.1 else f"d={d_bordo:.1f}m dal cordolo"

            tooltip_text = (
                f"INFERENZA NEURALE ({self.current_model_key})\n"
                f"• Stato Allerta:   {allerta_str}\n"
                f"• Rischio Stimato: {int(risk * 100)}%\n"
                f"• Oggetto Predetto (Rosso): {pred_cls}\n"
                f"• Probabilità:     Auto: {int(probs[0]*100)}% | Camion: {int(probs[1]*100)}% | VRU: {int(probs[2]*100)}% | Barr: {int(probs[3]*100)}%\n"
                f"• Suolo Mappa (100%): {suolo_str}\n"
                f"• Vicinanza Carreggiata: {bordo_str} (Accosto: {int(meta.get('roadside_f', 0)*100)}%)\n"
                f"• Distanza da Ego: {meta['dist']:.1f} m | Area: {meta['area']:.1f} m²"
            )

            if hasattr(self, 'tooltip') and self.tooltip is not None:
                self.tooltip.xy = (x, y)
                ox = -240 if x > 2.0 else 15
                oy = -120 if y > 5.0 else 15
                self.tooltip.set_position((ox, oy))
                self.tooltip.set_text(tooltip_text)
                self.tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            if self.active_hovered is not None:
                patch = self.active_hovered['patch']
                patch.set_facecolor(self.active_hovered['default_fc'])
                patch.set_edgecolor(self.active_hovered['default_ec'])
                patch.set_linewidth(self.active_hovered['default_lw'])
                patch.set_alpha(self.active_hovered['default_alpha'])
                self.active_hovered = None
                if hasattr(self, 'tooltip') and self.tooltip is not None:
                    self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore Tesi Inferenza Neurale nelle Zone Occluse")
    parser.add_argument("frame", nargs="?", type=int, default=1, help="Numero del frame iniziale (1-404)")
    parser.add_argument("--model", "-m",
                        choices=["NEURO_SIMB", "GEOMETRIC", "SEMANTIC", "GEOMETRICA", "SEMANTICA", "REAL_GT", "POS_ONLY", "HYBRID"],
                        default="NEURO_SIMB",
                        help="Modello neurale iniziale da caricare ('NEURO_SIMB', 'GEOMETRIC', 'SEMANTIC', 'REAL_GT', 'POS_ONLY')")
    parser.add_argument("--save", action="store_true", help="Salva screenshot ad alta risoluzione 300 DPI ed esce")
    args = parser.parse_args()

    init_frame = max(1, min(404, args.frame))
    vis = NeuralInferenceVisualizer(max_range=25.0, initial_model=args.model.upper())
    vis.load_frame(init_frame - 1, broadcast=True)

    if args.save:
        out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, f"fig_4_neural_inference_{vis.current_model_key.lower()}.png")
        if hasattr(vis, 'ax_tb') and vis.ax_tb is not None:
            vis.ax_tb.set_visible(False)
        if hasattr(vis, 'ax_dd') and vis.ax_dd is not None:
            vis.ax_dd.set_visible(False)
        vis.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"\n[SALVATA CON SUCCESSO]: Immagine Inferenza Neurale salvata a 300 DPI in:\n  -> {fpath}")
        return

    plt.show()


if __name__ == "__main__":
    main()
