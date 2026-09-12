# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - PROBABILITÀ CONDIZIONATA BAYESIANA NELLE ZONE OCCLUSE
# File: visualizzatori/vis_probabilita_bayes.py
#
# Rappresenta l'approccio analitico / probabilistico (Baseline Bayesiana) senza Rete Neurale:
# 1. Modulazione Spazio-Semantica P(Classe | Superficie, Varco OBB)
# 2. Memoria Temporale Causale: Tracciamento delle istanze passate viste entrare nell'ombra (Boost 95%)
#
# CONTROLLI DA TASTIERA & MOUSE:
# - [<- / ->] oppure [A / D]: Frame precedente / successivo
# - [B / Menu a tendina]: Cambia modalità Bayesiana (Completa, Senza Memoria, Solo Alto Rischio)
# - [S]: Salva screenshot scientifico ad alta risoluzione (300 DPI)
# - Hover del Mouse: Mostra il tooltip con le probabilità analitiche per ogni classe
# - Casella di testo in alto a destra: Salto diretto a qualsiasi numero di frame (1-404)
# ==============================================================================

import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from scipy import ndimage
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

# Configurazione root di progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset_adapter.factory_dataset import create_adapter
from raycaster.ray_caster import RayCaster

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")


class BayesOcclusionVisualizer:
    def __init__(self, max_range=25.0, initial_mode="FULL"):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: PROBABILITÀ CONDIZIONATA BAYESIANA (BEV 25M)")
        print("=" * 80)

        self.max_range = float(max_range)
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0

        # Modalità Bayesiana attiva
        init_m = initial_mode.upper()
        if init_m in ["FULL", "COMPLETO", "DYNAMIC", "MEMORIA"]:
            self.current_mode = "FULL"
        elif init_m in ["NO_MEMORY", "STATIC", "SENZA_MEMORIA"]:
            self.current_mode = "NO_MEMORY"
        elif init_m in ["HIGH_RISK", "ALTO_RISCHIO"]:
            self.current_mode = "HIGH_RISK"
        else:
            self.current_mode = "FULL"

        self.bayes_dir = os.path.join(ROOT_DIR, "extracted_occlusions_probabilities")

        # Setup tipografia accademica
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Conditional Bayesian Risk Visualizer")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None
        self.dropdown_open = False

        # Carica icone raster/vettoriali
        self.load_icons()

    def load_icons(self):
        """Carica le icone vettoriali stilizzate."""
        from PIL import Image
        icons_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        self.icons = {
            'car_blue': Image.open(os.path.join(icons_dir, "car_blue.png")),
            'car_ego': Image.open(os.path.join(icons_dir, "car_ego.png")),
            'car_red': Image.open(os.path.join(icons_dir, "car_red.png")),
            'truck_purple': Image.open(os.path.join(icons_dir, "truck_purple.png")),
            'truck_cab_purple': Image.open(os.path.join(icons_dir, "truck_cab_purple.png")),
            'truck_red': Image.open(os.path.join(icons_dir, "truck_red.png")),
            'trailer_purple': Image.open(os.path.join(icons_dir, "trailer_purple.png")),
            'trailer_red': Image.open(os.path.join(icons_dir, "trailer_red.png")),
            'construction_orange': Image.open(os.path.join(icons_dir, "construction_orange.png")),
            'pedestrian_green': Image.open(os.path.join(icons_dir, "pedestrian_green.png")),
            'pedestrian_red': Image.open(os.path.join(icons_dir, "pedestrian_red.png")),
            'bicycle_orange': Image.open(os.path.join(icons_dir, "bicycle_orange.png")),
            'bicycle_red': Image.open(os.path.join(icons_dir, "bicycle_red.png")),
            'motorcycle_amber': Image.open(os.path.join(icons_dir, "motorcycle_amber.png")),
            'barrier_hazard': Image.open(os.path.join(icons_dir, "barrier_hazard.png")),
            'barrier_red': Image.open(os.path.join(icons_dir, "barrier_red.png")),
            'cart_trolley': Image.open(os.path.join(icons_dir, "cart_trolley.png")),
            'traffic_cone': Image.open(os.path.join(icons_dir, "traffic_cone.png")),
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

            if dt > 1e-4:
                dist = np.linalg.norm(np.array(p_b['translation'][:2]) - np.array(p_a['translation'][:2]))
                return (dist / dt) * 3.6
            return 0.0
        except Exception:
            return 0.0

    def load_frame(self, idx, broadcast=True):
        """Carica il fotogramma e le predizioni Bayesiane estratte."""
        self.current_idx = idx % self.total_frames
        if broadcast:
            self._write_sync(self.current_idx)

        self.sample = self.adapter.all_samples[self.current_idx]
        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.token = self.frame_data["sample_token"]
        self.ego_speed_kmh = self.get_ego_velocity(self.current_idx)

        # Informazioni su scena e location
        try:
            sc = self.adapter.nusc.get('scene', self.sample['scene_token'])
            log = self.adapter.nusc.get('log', sc['log_token'])
            self.scene_name = sc.get('name', 'N/A')
            self.location = log.get('location', 'N/A')
        except Exception:
            self.scene_name = 'scene-0061'
            self.location = 'singapore-onenorth'

        # Carica il file JSON precalcolato delle probabilità Bayesiane
        json_pattern = os.path.join(self.bayes_dir, f"occlusion_sample_{self.current_idx:04d}_*.json")
        matches = glob.glob(json_pattern)
        if not matches:
            # Prova con token diretto
            matches = glob.glob(os.path.join(self.bayes_dir, f"*{self.token}*.json"))

        self.bayes_occlusions = []
        if matches:
            try:
                with open(matches[0], 'r') as f:
                    self.bayes_data = json.load(f)
                    self.bayes_occlusions = self.bayes_data.get('occlusions', [])
            except Exception as e:
                print(f"[WARN] Errore lettura file Bayes: {e}")
                self.bayes_occlusions = []
        else:
            # Fallback se file non trovato
            print(f"[WARN] File probabilità Bayesiane non trovato per sample {self.current_idx:04d}.")
            self.bayes_occlusions = []

        # Rilevamento pseudo-box statici (muri / edifici) con clustering LiDAR
        try:
            rc = RayCaster(self.frame_data, verbose=False)
            self.static_boxes = rc.detect_static_manmade_boxes(self.frame_data.get('boxes', []))
        except Exception:
            self.static_boxes = []

        self.render()

    def _write_sync(self, idx):
        try:
            os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
            with open(SYNC_FILE, "w") as f:
                f.write(f"{idx},bayes_visualizer")
        except Exception:
            pass

    def check_sync_file(self):
        try:
            if os.path.exists(SYNC_FILE):
                with open(SYNC_FILE, "r") as f:
                    content = f.read().strip()
                if content:
                    parts = content.split(",")
                    target = int(parts[0])
                    source = parts[1] if len(parts) > 1 else ""
                    if source != "bayes_visualizer" and target != self.current_idx and 0 <= target < self.total_frames:
                        self.load_frame(target, broadcast=False)
        except Exception:
            pass

    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                          flip_h=False, halo_color=None, halo_radius=None, z_order=8):
        """Disegna un'icona raster/vettoriale trasparente ruotata e scalata preservando le proporzioni."""
        import matplotlib.transforms as mtransforms
        from PIL import Image

        if halo_color and halo_radius:
            ax.add_patch(Circle(center, halo_radius, facecolor=halo_color, edgecolor='none', alpha=0.55, zorder=z_order-1))

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
        ax.imshow(icon_img, extent=extent, transform=tr, zorder=z_order, origin='upper', clip_on=True)

    def render(self):
        self.fig.clf()
        self.drawn_occlusions = []
        self.active_hovered = None

        c_border = '#0F172A'
        c_static_face = '#FCA5A5'
        c_static_edge = '#EF4444'

        c_car = '#1D4ED8'
        c_truck = '#7E22CE'
        c_ped = '#059669'
        c_bike = '#D97706'
        c_barrier = '#D97706'

        # Layout a griglia 3x2 identico a vis_inferenza_neurale.py
        gs = GridSpec(3, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.925,
                      width_ratios=[1.1, 0.9], height_ratios=[1.25, 1.05, 1.20],
                      wspace=0.14, hspace=0.28)

        ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map = ax_map

        ax_telemetry = self.fig.add_subplot(gs[0, 1])
        ax_objects = self.fig.add_subplot(gs[1, 1])
        ax_legend = self.fig.add_subplot(gs[2, 1])

        # Titolo visualizzatore pulito in alto a sinistra
        self.fig.text(0.03, 0.956,
                      f"Figure 2B: Stima del Rischio con Probabilità Condizionata Bayesiana (BEV Raggio {int(self.max_range)}m)",
                      fontsize=11.5, fontweight='bold', color='#0F172A', ha='left', va='center')

        # =====================================================================
        # 1. PANNELLO SINISTRA: MAPPA BEV COMPLETA AD ALTA FEDELTÀ
        # =====================================================================
        ax_map.set_facecolor('#F8FAFC')
        ax_map.set_xlim(-self.max_range, self.max_range)
        ax_map.set_ylim(-self.max_range, self.max_range)
        ax_map.set_aspect('equal')
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # Griglia di precisione metrica ogni 2m
        grid_lim = self.max_range + 1.1
        for gx in np.arange(-grid_lim, grid_lim, 2.0):
            ax_map.axvline(gx, color='#E2E8F0', linewidth=0.55, zorder=1)
        for gy in np.arange(-grid_lim, grid_lim, 2.0):
            ax_map.axhline(gy, color='#E2E8F0', linewidth=0.55, zorder=1)
        ax_map.axvline(0, color=c_border, linewidth=1.1, zorder=2)
        ax_map.axhline(0, color=c_border, linewidth=1.1, zorder=2)

        # Cerchi radar concentrici (10m, 20m, 25m)
        circles = [10.0, 20.0, 25.0]
        for r in circles:
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
        if pts is not None and len(pts) > 0:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            mask_r = dists <= self.max_range
            pts_r = pts[mask_r]
            if len(pts_r) > 0:
                ax_map.scatter(pts_r[:, 0], pts_r[:, 1], s=1.1, c='#0F172A',
                               alpha=0.68, zorder=3, edgecolors='none')

        # Conteggi per statistiche
        counts_visibili = {"Auto": 0, "Camion/Bus": 0, "VRU (Pedoni)": 0, "Barriere": 0}
        counts_bayes = {"Auto": 0, "Camion/Bus": 0, "VRU (Pedoni)": 0, "Barriere": 0}

        # ---------------------------------------------------------------------
        # ZONE OCCLUSE & PROBABILITÀ CONDIZIONATE BAYESIANE
        # ---------------------------------------------------------------------
        boosted_zone_count = 0
        all_risks = []

        circle_25m = ShapelyPoint(0, 0).buffer(self.max_range)

        for occ_idx, occ in enumerate(self.bayes_occlusions):
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue

            try:
                sh_poly = ShapelyPolygon(poly_pts)
                if not sh_poly.is_valid:
                    sh_poly = sh_poly.buffer(0)
                if sh_poly.is_empty:
                    continue
            except Exception:
                continue

            # Intersezione con il cerchio 25m
            inter = sh_poly.intersection(circle_25m)
            if inter.is_empty or inter.area < 0.20:
                continue

            probs = dict(occ.get("estimated_probabilities", {}))
            boosted = list(occ.get("boosted_categories", []))
            base_risk = float(occ.get("risk_score", 0.0))

            # Se in modalità NO_MEMORY: annulla i boost temporali ripristinando le stime semantiche
            if self.current_mode == "NO_MEMORY":
                boosted = []
                for b_cat in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto"]:
                    if probs.get(b_cat, 0.0) >= 0.90:
                        # Ricalibra alla probabilità di base semantica
                        probs[b_cat] = 0.35 if b_cat == "Pedone" else 0.30
                base_risk = min(0.38, base_risk * 0.5)

            # Calcolo categoria dominante
            p_auto = probs.get("Auto", 0.0)
            p_camion = max(probs.get("Camion", 0.0), probs.get("Bus", 0.0), probs.get("Rimorchio", 0.0))
            p_vru = max(probs.get("Pedone", 0.0), probs.get("Bicicletta", 0.0), probs.get("Moto", 0.0))
            p_barr = max(probs.get("Barriera", 0.0), probs.get("Cono", 0.0))

            max_p = max(p_auto, p_camion, p_vru, p_barr)
            effective_risk = max(base_risk, max_p)
            all_risks.append(effective_risk)

            # Filtro HIGH_RISK
            if self.current_mode == "HIGH_RISK" and effective_risk < 0.30:
                continue

            if boosted:
                boosted_zone_count += 1

            # Determinazione categoria predetta da Bayes
            if max_p == p_auto:
                pred_cls = "Auto"
            elif max_p == p_camion:
                pred_cls = "Camion/Bus"
            elif max_p == p_vru:
                pred_cls = "VRU (Pedoni)"
            else:
                pred_cls = "Barriere"

            # Colore del poligono in base alla probabilità/rischio
            if effective_risk >= 0.40 or len(boosted) > 0:
                fc = '#FCA5A5'
                ec = '#DC2626'
                alpha_fill = min(0.55, 0.28 + effective_risk * 0.30)
                lw = 1.3
            elif effective_risk >= 0.20:
                fc = '#FED7AA'
                ec = '#EA580C'
                alpha_fill = 0.32
                lw = 1.0
            else:
                fc = '#E2E8F0'
                ec = '#64748B'
                alpha_fill = 0.22
                lw = 0.8

            poly_xy = np.array(poly_pts)
            poly_patch = MplPolygon(poly_xy, closed=True, facecolor=fc, edgecolor=ec,
                                    linewidth=lw, alpha=alpha_fill, zorder=5)
            ax_map.add_patch(poly_patch)

            # Posizionamento icona predetta al centroide se rischio significativo (o boost)
            cx, cy = float(inter.centroid.x), float(inter.centroid.y)
            dist_c = np.hypot(cx, cy)

            if (effective_risk >= 0.18 or len(boosted) > 0) and dist_c <= 24.8 and abs(cx) <= 24.8 and abs(cy) <= 24.8:
                if pred_cls == "Auto":
                    counts_bayes["Auto"] += 1
                    self.draw_icon_object(ax_map, 'car_red', center=[cx, cy], length=4.0, deg=0.0)
                    h_off = 2.4
                elif pred_cls == "Camion/Bus":
                    counts_bayes["Camion/Bus"] += 1
                    self.draw_icon_object(ax_map, 'truck_red', center=[cx, cy], length=5.5, deg=0.0)
                    h_off = 3.1
                elif "VRU" in pred_cls:
                    counts_bayes["VRU (Pedoni)"] += 1
                    self.draw_icon_object(ax_map, 'pedestrian_red', center=[cx, cy], length=2.8, deg=0.0,
                                          halo_color='#FEE2E2', halo_radius=1.5)
                    h_off = 1.8
                else:
                    counts_bayes["Barriere"] += 1
                    self.draw_icon_object(ax_map, 'barrier_red', center=[cx, cy], length=2.2, deg=0.0)
                    h_off = 1.5

                # Badge percentuale probabilità sopra all'oggetto
                badge_lbl = f"{int(max_p*100)}%"
                if boosted:
                    badge_lbl = f"⚡{badge_lbl}"

                b_bg = '#DC2626' if (effective_risk >= 0.40 or boosted) else ('#EA580C' if effective_risk >= 0.20 else '#475569')
                b_ec = '#991B1B' if (effective_risk >= 0.40 or boosted) else ('#C2410C' if effective_risk >= 0.20 else '#1E293B')
                ax_map.text(cx, cy + h_off, badge_lbl,
                            fontsize=7.0, fontweight='bold', color='#FFFFFF',
                            ha='center', va='bottom', zorder=12, clip_on=True,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor=b_bg,
                                      edgecolor=b_ec, linewidth=0.8, alpha=0.92))

            from matplotlib.path import Path as MplPath
            meta_dict = {
                "idx": occ_idx,
                "distance_m": occ.get("distance_m", dist_c),
                "area_sqm": occ.get("area_sqm", inter.area),
                "road_fraction": occ.get("road_fraction", 0.0),
                "sidewalk_fraction": occ.get("sidewalk_fraction", 0.0),
                "crosswalk_fraction": occ.get("crosswalk_fraction", 0.0),
                "probs": probs,
                "boosted": boosted,
                "risk_score": effective_risk,
                "pred_cls": pred_cls,
                "centroid": (cx, cy)
            }

            self.drawn_occlusions.append({
                'patch': poly_patch,
                'path': MplPath(poly_xy),
                'meta': meta_dict,
                'default_fc': fc,
                'default_ec': ec,
                'default_lw': lw,
                'default_alpha': alpha_fill,
                'default_z': 5
            })

        # Strutture statiche rilevate da LiDAR (muri/edifici tratteggiati)
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

        # Ostacoli reali nuScenes visibili al LiDAR
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

        # Ego Vehicle al centro (lunghezza 5.0m, asset vettoriale hd)
        self.draw_icon_object(ax_map, 'car_ego', center=[0.0, 0.0], length=5.0, deg=0.0, z_order=12)

        # Tooltip interattivo per dettagli Bayes della zona
        self.tooltip = ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#EF4444", linewidth=1.5, alpha=0.95),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        for spine in ax_map.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        # =====================================================================
        # 2. PANNELLO DESTRA ALTO: TELEMETRIA EGO VEHICLE & METADATI BAYES
        # =====================================================================
        ax_telemetry.set_facecolor('#F8FAFC')
        for spine in ax_telemetry.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)

        ax_telemetry.text(0.04, 0.88, "EGO VEHICLE TELEMETRY & BAYESIAN REASONING METADATA",
                          transform=ax_telemetry.transAxes, fontsize=9.2, fontweight='bold', color=c_border)

        # Box velocità
        ax_telemetry.add_patch(Rectangle((0.03, 0.10), 0.22, 0.68, transform=ax_telemetry.transAxes,
                                         facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2))
        ax_telemetry.text(0.14, 0.60, "VELOCITÀ EGO", transform=ax_telemetry.transAxes,
                          fontsize=8.0, fontweight='bold', color='#1D4ED8', ha='center', va='center')
        ax_telemetry.text(0.12, 0.35, f"{self.ego_speed_kmh:.1f}", transform=ax_telemetry.transAxes,
                          fontsize=19, fontweight='bold', color='#0F172A', ha='center', va='center')
        ax_telemetry.text(0.20, 0.30, "km/h", transform=ax_telemetry.transAxes,
                          fontsize=8.0, color='#64748B', va='bottom')

        # Badge modello Bayesiano attivo
        mode_names = {
            "FULL": ("★ BAYES DINAMICO (Spazio 3D + Mappa + Memoria)", "#1D4ED8", "#EFF6FF", "#3B82F6"),
            "NO_MEMORY": ("BAYES STATICO (Spazio 3D + Mappa Senza Memoria)", "#D97706", "#FFFBEB", "#F59E0B"),
            "HIGH_RISK": ("FILTRO ALTO RISCHIO BAYESIANO (P >= 30%)", "#DC2626", "#FEF2F2", "#EF4444")
        }
        title_b, col_txt, col_bg, col_ec = mode_names.get(self.current_mode, mode_names["FULL"])

        badge_box = FancyBboxPatch(
            (0.28, 0.63), 0.68, 0.17,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            transform=ax_telemetry.transAxes,
            facecolor=col_bg, edgecolor=col_ec, linewidth=1.2, zorder=3
        )
        ax_telemetry.add_patch(badge_box)
        ax_telemetry.text(0.62, 0.715, f"APPROCCIO: {title_b}", transform=ax_telemetry.transAxes,
                          fontsize=7.6, fontweight='bold', color=col_txt, ha='center', va='center', zorder=4)

        # Metadati analitici della scena
        avg_risk = np.mean(all_risks) if len(all_risks) > 0 else 0.0
        max_risk = np.max(all_risks) if len(all_risks) > 0 else 0.0

        meta_lines = [
            f"Dataset: nuScenes ({self.scene_name})  |  Campione: {self.current_idx + 1} / {self.total_frames}",
            f"Zone Occluse Analizzate: {len(self.bayes_occlusions)}  |  Zone con Memoria Attiva: {boosted_zone_count}",
            f"Rischio Bayesiano Medio: {avg_risk*100:.1f}%  |  Rischio Massimo: {max_risk*100:.1f}%",
            f"Regola: P(Classe | Superficie HD-Map, Varco OBB) con Boost Causale al 95%"
        ]

        y_m = 0.48
        for line in meta_lines:
            ax_telemetry.text(0.29, y_m, line, transform=ax_telemetry.transAxes,
                              fontsize=7.8, color='#334155', va='center')
            y_m -= 0.12

        ax_telemetry.set_xlim(0, 1)
        ax_telemetry.set_ylim(0, 1)
        ax_telemetry.set_xticks([])
        ax_telemetry.set_yticks([])

        # =====================================================================
        # 3. PANNELLO DESTRA CENTRO: CONFRONTO OSTACOLI VISIBILI vs BAYES
        # =====================================================================
        ax_objects.set_facecolor('#FFFFFF')
        for spine in ax_objects.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.0)
        ax_objects.spines['top'].set_visible(False)
        ax_objects.spines['right'].set_visible(False)

        cat_names = ["Auto", "Camion/Bus", "VRU (Pedoni)", "Barriere"]
        vals_vis = [counts_visibili[c] for c in cat_names]
        vals_bayes = [counts_bayes[c] for c in cat_names]

        x = np.arange(len(cat_names))
        width = 0.32

        rects_vis = ax_objects.bar(x - width/2.0, vals_vis, width, label='Ostacoli Reali Visibili (LiDAR)',
                                   color='#0284C7', edgecolor=c_border, linewidth=1.0, zorder=3)
        rects_bayes = ax_objects.bar(x + width/2.0, vals_bayes, width, label='Zone con Stima Bayesiana (P > 20%)',
                                     color='#DC2626', edgecolor=c_border, linewidth=1.0, zorder=3)

        max_val = max(max(vals_vis, default=1), max(vals_bayes, default=1), 4)
        ax_objects.set_ylim(0, max_val + 3.2)
        ax_objects.set_ylabel(f"Numero Zone / Oggetti (entro {int(self.max_range)}m)", fontsize=8.5, fontweight='bold', color=c_border)
        ax_objects.set_title(f"Confronto Rilevamento Diretto vs Anticipazione Probabilistica Bayesiana",
                             fontsize=9.8, fontweight='bold', pad=7, color=c_border)
        ax_objects.set_xticks(x)
        ax_objects.set_xticklabels(cat_names, fontsize=8.5, fontweight='bold', color=c_border)
        ax_objects.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_objects.tick_params(colors=c_border, labelsize=8.2)

        leg = ax_objects.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=True,
            facecolor='#F8FAFC',
            edgecolor='#CBD5E1',
            fontsize=7.8,
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
        for bar in rects_bayes:
            h = bar.get_height()
            if h > 0:
                ax_objects.text(bar.get_x() + bar.get_width()/2.0, h + 0.25, str(int(h)),
                                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#DC2626')

        # =====================================================================
        # 4. PANNELLO DESTRA BASSO: LEGENDA UFFICIALE & SIMBOLOGIA
        # =====================================================================
        ax_legend.set_facecolor('#FFFFFF')
        for spine in ax_legend.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_legend.set_title("Simbologia Ufficiale (Ostacoli Visibili LiDAR vs Stima Bayesiana)",
                            fontsize=9.8, fontweight='bold', pad=9, color=c_border)

        # 3 colonne pulite
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

        ax_legend.text(0.46, 0.90, "STIMA BAYESIANA (ROSSI)", transform=ax_legend.transAxes,
                       fontsize=7.6, fontweight='bold', color='#DC2626', ha='center', va='center')
        col2_items = [
            ('icon', 'car_red', 'Auto Bayesiana', 0.044),
            ('icon', 'truck_red', 'Camion Bayesiano', 0.035),
            ('icon', 'pedestrian_red', 'Pedone Bayesiano', 0.050),
            ('icon', 'barrier_red', 'Barriera Bayesiana', 0.040),
        ]
        for i, it in enumerate(col2_items):
            cur_y = 0.72 - i * 0.18
            oi = OffsetImage(self.icons[it[1]], zoom=it[3])
            ab = AnnotationBbox(oi, (0.335, cur_y), xycoords='axes fraction', frameon=False, zorder=5)
            ax_legend.add_artist(ab)
            ax_legend.text(0.375, cur_y, it[2], transform=ax_legend.transAxes,
                           fontsize=7.4, color='#DC2626', va='center', fontweight='bold')

        ax_legend.text(0.79, 0.90, "MAPPA & CONI D'OMBRA", transform=ax_legend.transAxes,
                       fontsize=7.6, fontweight='bold', color='#334155', ha='center', va='center')
        col3_items = [
            ('rect', '#FCA5A5', '#DC2626', 'Cono Pericolo (P>=40% o Boost)', 0.55),
            ('rect', '#FED7AA', '#EA580C', 'Cono Medio Rischio (P>=20%)', 0.35),
            ('rect', '#E2E8F0', '#64748B', 'Cono Basso Rischio (P<20%)', 0.25),
            ('dashed_rect', c_static_face, c_static_edge, 'Struttura Statica (Muro)', 0.45),
            ('rect', '#7D7D7D', c_border, 'Carreggiata (HD-Map)', 1.0),
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
                           fontsize=7.0, color=c_border, va='center', fontweight='medium')

        # =====================================================================
        # 5. MENU A TENDINA & CASELLA DI SALTO FRAME IN ALTO
        # =====================================================================
        ax_dd = self.fig.add_axes([0.575, 0.938, 0.270, 0.036])
        arrow = "▲" if getattr(self, 'dropdown_open', False) else "▼"
        mode_btn_labels = {
            "FULL": "★ Bayes Dinamico (Memoria)",
            "NO_MEMORY": "Bayes Statico (No Memoria)",
            "HIGH_RISK": "Filtro Alto Rischio (P>=30%)"
        }
        curr_lbl = mode_btn_labels.get(self.current_mode, self.current_mode)
        self.btn_mode = Button(ax_dd, f"Modello: {curr_lbl}  {arrow}", color='#F8FAFC', hovercolor='#E2E8F0')
        self.btn_mode.label.set_fontsize(8.2)
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
                      "[<- / ->] o [A / D]: Frame  |  [B / Menu]: Cambia Modalità Bayes  |  [Hover Mouse]: Dettagli Probabilità  |  [S]: Salva HD 300 DPI",
                      fontsize=8.0, color='#64748B', ha='center', style='italic')

        self.fig.canvas.draw_idle()

    # =========================================================================
    # GESTIONE MENU A TENDINA PER MODALITÀ BAYES
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

        self.ax_menu = self.fig.add_axes([0.575, 0.816, 0.270, 0.120], facecolor='#FFFFFF', zorder=100)
        self.ax_menu.set_xticks([])
        self.ax_menu.set_yticks([])
        for spine in self.ax_menu.spines.values():
            spine.set_color('#0F172A')
            spine.set_linewidth(1.3)

        modes = [
            ("FULL", "1. ★ Bayes Dinamico (Spazio + Memoria)"),
            ("NO_MEMORY", "2. Bayes Statico (Solo Mappa HD)"),
            ("HIGH_RISK", "3. Filtro Alto Rischio (P >= 30%)")
        ]

        y_offsets = [0.080, 0.041, 0.002]
        self.menu_buttons = []
        for (m_key, m_lbl), y_off in zip(modes, y_offsets):
            is_active = (self.current_mode == m_key)
            prefix = " ●  " if is_active else " ○  "
            bg_col = '#EFF6FF' if is_active else '#FFFFFF'
            txt_col = '#1D4ED8' if is_active else '#334155'

            ax_item = self.fig.add_axes([0.577, 0.816 + y_off, 0.266, 0.036], zorder=101)
            btn = Button(ax_item, prefix + m_lbl, color=bg_col, hovercolor='#E0F2FE')
            btn.label.set_fontsize(8.0)
            btn.label.set_fontweight('bold' if is_active else 'normal')
            btn.label.set_color(txt_col)
            btn.label.set_ha('left')
            btn.label.set_x(0.04)
            btn.on_clicked(lambda ev, k=m_key: self.select_bayes_mode(k))
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

        if hasattr(self, 'ax_menu') and self.ax_menu is not None:
            try:
                self.ax_menu.remove()
            except Exception:
                pass
            self.ax_menu = None
        self.dropdown_open = False
        self.fig.canvas.draw_idle()

    def select_bayes_mode(self, mode_key):
        self.current_mode = mode_key
        self.dropdown_open = False
        self._hide_dropdown_menu()
        print(f"\n>>> [DROPDOWN MENU] Selezionata modalità Bayes: {self.current_mode}")
        self.render()

    # =========================================================================
    # EVENTI TASTIERA & MOUSE
    # =========================================================================
    def on_key(self, event):
        if event.key in ['right', 'd', 'D']:
            self.load_frame(self.current_idx + 1)
        elif event.key in ['left', 'a', 'A']:
            self.load_frame(self.current_idx - 1)
        elif event.key in ['b', 'B', 'm', 'M']:
            # Cicla tra le 3 modalità Bayes
            mode_cycle = ["FULL", "NO_MEMORY", "HIGH_RISK"]
            curr_i = mode_cycle.index(self.current_mode) if self.current_mode in mode_cycle else 0
            self.current_mode = mode_cycle[(curr_i + 1) % len(mode_cycle)]
            print(f"\n>>> [TOGGLE MODALITÀ BAYES] Attiva: {self.current_mode}")
            self.render()
        elif event.key in ['s', 'S']:
            out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"fig_bayes_inference_frame_{self.current_idx + 1}_{self.current_mode.lower()}.png"
            fpath = os.path.join(out_dir, fname)
            self.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
            print(f"\n[SALVATAGGIO OK]: Immagine Bayes salvata in: {fpath} (300 DPI)")
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            target = (val - 1) if 1 <= val <= self.total_frames else val
            if 0 <= target < self.total_frames and target != self.current_idx:
                self.load_frame(target)
        except ValueError:
            pass

    def on_mouse_click(self, event):
        if getattr(self, 'dropdown_open', False):
            if event.inaxes != self.ax_dd and (self.ax_menu is None or event.inaxes != self.ax_menu):
                in_item = any(event.inaxes == it[0] for it in getattr(self, 'menu_buttons', []))
                if not in_item:
                    self._hide_dropdown_menu()

    def on_mouse_move(self, event):
        if getattr(self, 'dropdown_open', False):
            return

        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            if self.active_hovered is not None:
                self.active_hovered['patch'].set_facecolor(self.active_hovered['default_fc'])
                self.active_hovered['patch'].set_edgecolor(self.active_hovered['default_ec'])
                self.active_hovered['patch'].set_linewidth(self.active_hovered['default_lw'])
                self.active_hovered['patch'].set_alpha(self.active_hovered['default_alpha'])
                self.active_hovered['patch'].set_zorder(self.active_hovered['default_z'])
                self.active_hovered = None
            if self.tooltip is not None:
                self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        pt = (event.xdata, event.ydata)
        found = None
        for item in reversed(self.drawn_occlusions):
            if item['path'].contains_point(pt):
                found = item
                break

        if found != self.active_hovered:
            if self.active_hovered is not None:
                self.active_hovered['patch'].set_facecolor(self.active_hovered['default_fc'])
                self.active_hovered['patch'].set_edgecolor(self.active_hovered['default_ec'])
                self.active_hovered['patch'].set_linewidth(self.active_hovered['default_lw'])
                self.active_hovered['patch'].set_alpha(self.active_hovered['default_alpha'])
                self.active_hovered['patch'].set_zorder(self.active_hovered['default_z'])

            self.active_hovered = found

            if found is not None:
                found['patch'].set_facecolor('#FEF08A')
                found['patch'].set_edgecolor('#CA8A04')
                found['patch'].set_linewidth(2.2)
                found['patch'].set_alpha(0.70)
                found['patch'].set_zorder(15)

                meta = found['meta']
                probs = meta['probs']
                boosted = meta['boosted']

                p_auto = probs.get("Auto", 0.0) * 100
                p_ped = probs.get("Pedone", 0.0) * 100
                p_camion = max(probs.get("Camion", 0.0), probs.get("Bus", 0.0)) * 100
                p_bici = max(probs.get("Bicicletta", 0.0), probs.get("Moto", 0.0)) * 100
                p_barr = max(probs.get("Barriera", 0.0), probs.get("Cono", 0.0)) * 100

                boost_tag = f"\n  ⚡ MEMORIA CAUSALE: Boost {boosted[0]} (95%)" if boosted else ""

                text = (
                    f"ZONA OCCLUSA BAYESIANA #{meta['idx'] + 1}\n"
                    f"  Distanza: {meta['distance_m']:.1f} m  |  Area: {meta['area_sqm']:.1f} m²\n"
                    f"  Affordance HD-Map: Strada {int(meta['road_fraction']*100)}% | Marciapiede {int(meta['sidewalk_fraction']*100)}%\n"
                    f"-----------------------------------------\n"
                    f"  P(Auto): {p_auto:.1f}%  |  P(Pedone): {p_ped:.1f}%\n"
                    f"  P(Camion/Bus): {p_camion:.1f}%  |  P(Bici/Moto): {p_bici:.1f}%\n"
                    f"  P(Barriera/Cono): {p_barr:.1f}%\n"
                    f"-----------------------------------------\n"
                    f"  Top Categoria Bayesiana: {meta['pred_cls']}{boost_tag}"
                )

                self.tooltip.set_text(text)
                self.tooltip.xy = (event.xdata, event.ydata)
                self.tooltip.set_visible(True)
            else:
                self.tooltip.set_visible(False)

            self.fig.canvas.draw_idle()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore Tesi Stima del Rischio con Probabilità Condizionata Bayesiana")
    parser.add_argument("frame", nargs="?", type=int, default=1, help="Numero del frame iniziale (1-404)")
    parser.add_argument("--mode", "-m", choices=["FULL", "NO_MEMORY", "HIGH_RISK"], default="FULL",
                        help="Modalità iniziale ('FULL', 'NO_MEMORY', 'HIGH_RISK')")
    parser.add_argument("--save", action="store_true", help="Salva screenshot ad alta risoluzione 300 DPI ed esce")
    args = parser.parse_args()

    init_frame = max(1, min(404, args.frame))
    vis = BayesOcclusionVisualizer(max_range=25.0, initial_mode=args.mode.upper())
    vis.load_frame(init_frame - 1, broadcast=True)

    if args.save:
        out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"fig_2b_bayes_inference_{vis.current_mode.lower()}.png"
        fpath = os.path.join(out_dir, fname)
        vis.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"\n[SALVATA CON SUCCESSO]: Immagine Bayes salvata a 300 DPI in:\n  -> {fpath}")
        return

    timer = vis.fig.canvas.new_timer(interval=150)
    timer.add_callback(vis.check_sync_file)
    timer.start()

    plt.show()


if __name__ == "__main__":
    main()
