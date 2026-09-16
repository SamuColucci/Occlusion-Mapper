# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - GROUND TRUTH NELLE ZONE OCCLUSE (BEV NUSCENES)
# File: visualizzatori/vis_ground_truth_occlusioni.py
#
# Mostra ESCLUSIVAMENTE gli elementi della Ground Truth presenti nelle Zone Occluse:
# - MODALITÀ GT REALE: Visualizza solo gli ostacoli reali annotati nuScenes 3D
#                      che ricadono all'interno dei coni d'ombra / occlusioni.
# - MODALITÀ GT SINTETICA: Visualizza gli ostacoli sintetici verosimili generati
#                          all'interno delle zone d'ombra in base ai vincoli fisici e stradali.
#
# CONTROLLI DA TASTIERA:
# - [B]: Alterna istantaneamente la Ground Truth tra "REALE" e "SINTETICA"
# - [<- / ->] oppure [A / D]: Frame precedente / successivo
# - [S]: Salva screenshot scientifico HD a 300 DPI
# - Casella di testo in alto a destra: Salto diretto a qualsiasi numero di frame
# ==============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from scipy import ndimage
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

# Impostazione percorso root del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset_adapter.factory_dataset import create_adapter
from raycaster.ray_caster import RayCaster
from ground_truth.ground_truth_extractor_synthetic import compute_synthetic_ground_truth, get_occluder_filter_explanation
from ground_truth.ground_truth_extractor import rasterize_polygon, extract_real_occlusion_boxes

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")

class SyntheticBox:
    """Rappresenta un ostacolo 3D sintetico verosimile generato all'interno di una zona d'ombra."""
    def __init__(self, name, center, wlh, yaw=0.0, token="syn"):
        self.name = name
        self.center = np.array(center, dtype=float)
        self.wlh = np.array(wlh, dtype=float)  # [width, length, height]
        self.yaw = float(yaw)
        self.token = str(token)

        w, l, h = self.wlh[0], self.wlh[1], self.wlh[2]
        x_corners = l / 2.0 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2.0 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2.0 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])
        corners = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners[0, :] += self.center[0]
        corners[1, :] += self.center[1]
        corners[2, :] += self.center[2]
        self.corners_3d = corners

    def corners(self):
        return self.corners_3d


class GroundTruthOcclusionVisualizer:
    def __init__(self, max_range=25.0, initial_mode="NEURO_SIMB"):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: GROUND TRUTH NELLE ZONE OCCLUSE (BEV 25M)")
        print("=" * 80)

        self.max_range = float(max_range)
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0
        init_m = initial_mode.upper()
        if init_m in ["HYBRID", "IBRIDO", "SINTETICA_HYBRID", "NEURO_SIMB", "NEURO", "SINTETICA"]:
            self.gt_mode = "NEURO_SIMB"
        elif init_m in ["GEOM", "GEOMETRIC", "GEOMETRICA", "SINTETICA_GEOM"]:
            self.gt_mode = "GEOMETRICA"
        elif init_m in ["SEM", "SEMANTIC", "SEMANTICA", "SINTETICA_SEM"]:
            self.gt_mode = "SEMANTICA"
        elif init_m in ["REALE", "REAL"]:
            self.gt_mode = "REALE"
        elif init_m in ["REALE_POSITIVES", "POSITIVES", "POS_ONLY"]:
            self.gt_mode = "REALE_POSITIVES"
        else:
            self.gt_mode = "NEURO_SIMB"

        self.use_occluder_filter = True
        self.show_occluders = True

        # Setup tipografia accademica
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Ground Truth & Neurosymbolic Risk Visualizer")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None

        # Carica icone vettoriali stilizzate ad alta risoluzione
        self.load_icons()

    def load_icons(self):
        """Carica le icone vettoriali e genera versioni con outline colorato per occluder/occluded."""
        from PIL import Image
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
        """Carica i dati del fotogramma ed aggiorna la vista."""
        self.current_idx = idx % self.total_frames
        if broadcast:
            self._write_sync(self.current_idx)

        self.sample = self.adapter.all_samples[self.current_idx]
        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.token = self.frame_data["sample_token"]
        self.ego_speed_kmh = self.get_ego_velocity(self.current_idx)

        # Informazioni su scena e log
        try:
            sc = self.adapter.nusc.get('scene', self.sample['scene_token'])
            log = self.adapter.nusc.get('log', sc['log_token'])
            self.scene_name = sc.get('name', 'N/A')
            self.location = log.get('location', 'N/A')
        except Exception:
            self.scene_name = 'scene-0061'
            self.location = 'singapore-onenorth'

        # Carica occlusioni reali da raycasting
        occ_file = os.path.join("extracted_occlusions", f"{self.token}.json")
        self.occlusions = []
        if os.path.exists(occ_file):
            try:
                with open(occ_file) as f:
                    self.occlusions = json.load(f).get("occlusions", [])
            except Exception:
                pass

        # Rileva pseudo-box statici (muri/edifici) con il clustering di ray_caster.py
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
                f.write(f"{idx},gt_occlusion")
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
                    if source != "gt_occlusion" and target != self.current_idx and 0 <= target < self.total_frames:
                        self.load_frame(target, broadcast=False)
        except Exception:
            pass

    # =========================================================================
    # ESTRAZIONE OSTACOLI GROUND TRUTH NELLE ZONE OCCLUSE
    # =========================================================================
    def get_ground_truth_occluded_boxes(self):
        """
        In modalità REALE: restituisce (occluders, occluded_boxes)
          - occluders: box nuScenes 3D che GENERANO un'occlusione (visibili, ma bloccano la vista)
          - occluded_boxes: box nuScenes 3D che si trovano DENTRO le zone d'ombra (nascosti)
        In modalità SINTETICA: restituisce ([], syn_boxes)
          - syn_boxes: ostacoli sintetici verosimili generati dentro le ombre
        """
        all_boxes = self.frame_data.get('boxes', [])
        real_occluders, real_occluded, _ = extract_real_occlusion_boxes(
            all_boxes, self.occlusions, max_range=self.max_range
        )

        if self.gt_mode in ["REALE", "REALE_POSITIVES"]:
            return real_occluders, real_occluded

        else:
            # Modalità SINTETICA NEURO-SIMBOLICA (Spazio 3D + HD-Map entro 25m)
            syn_boxes = []

            occ_polygons = []
            for occ in self.occlusions:
                pts = occ.get("polygon_points_m", [])
                if len(pts) >= 3:
                    sp = ShapelyPolygon(pts)
                    if not sp.is_valid:
                        sp = sp.buffer(0)
                    if sp.is_valid and sp.area > 0.05:
                        occ_polygons.append((sp, occ))

            circle_25m = ShapelyPoint(0, 0).buffer(self.max_range)
            all_generated_points = []
            if "SEM" in self.gt_mode:
                gt_mode_param = "semantic"
            elif "GEOM" in self.gt_mode:
                gt_mode_param = "geometric"
            else:
                gt_mode_param = "hybrid"

            # Precalcolo per la distanza dalla carreggiata (accosto bordo strada)
            import cv2
            smap = self.frame_data.get('semantic_map', {})
            drivable_m = smap.get('drivable_area', np.zeros((200, 200)))
            walkway_m = smap.get('walkway', np.zeros((200, 200)))
            crossing_m = smap.get('ped_crossing', np.zeros((200, 200)))
            dt_road = cv2.distanceTransform((1 - drivable_m).astype(np.uint8), cv2.DIST_L2, 3) * 0.4

            # Mappa rapida per estrarre le dimensioni 3D dell'occludore se noto
            boxes_by_token = {b.token: b for b in self.frame_data.get('boxes', [])}
            for sb in getattr(self, 'static_boxes', []):
                boxes_by_token[sb.token] = sb

            for idx, (poly, occ) in enumerate(occ_polygons):
                poly_vis = poly.intersection(circle_25m)
                if poly_vis.is_empty or poly_vis.area < 0.2:
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

                # ponytail: se l'intersezione genera un MultiPolygon, consideriamo le componenti significative
                sub_polys = list(poly_vis.geoms) if poly_vis.geom_type == 'MultiPolygon' else [poly_vis]

                for s_idx, sp_sub in enumerate(sub_polys):
                    if sp_sub.area < 0.2:
                        continue

                    coords_sub = np.array(sp_sub.exterior.coords)[:-1]
                    occ_mask = rasterize_polygon(coords_sub)
                    tot_px = np.sum(occ_mask)

                    if tot_px > 0:
                        road_f = float(np.sum(occ_mask * drivable_m) / tot_px)
                        side_f = float(np.sum(occ_mask * walkway_m) / tot_px)
                        cross_f = float(np.sum(occ_mask * crossing_m) / tot_px)
                        min_d_r = float(np.min(dt_road[occ_mask > 0]))
                        roadside_f = max(0.0, 1.0 - (min_d_r / 2.5))
                    else:
                        road_f = float(occ.get('road_fraction', 0.0))
                        side_f = float(occ.get('sidewalk_fraction', 0.0))
                        cross_f = float(occ.get('crosswalk_fraction', 0.0))
                        roadside_f = float(occ.get('roadside_fraction', 0.0))

                    area_sub = float(sp_sub.area)
                    dist_sub = float(np.hypot(sp_sub.centroid.x, sp_sub.centroid.y))

                    # Calcolo centralizzato della Ground Truth sintetica neurosimbolica strettamente entro 25m
                    _, placed_objects = compute_synthetic_ground_truth(
                        sp=sp_sub,
                        sp_sample=sp_sub,
                        road_f=road_f,
                        side_f=side_f,
                        cross_f=cross_f,
                        roadside_f=roadside_f,
                        area=area_sub,
                        dist=dist_sub,
                        existing_points=all_generated_points,
                        min_dist=3.0,
                        mode=gt_mode_param,
                        occluder_name=occ_name,
                        occluder_wlh=occ_wlh,
                        use_occluder_filter=getattr(self, 'use_occluder_filter', True)
                    )

                    for (class_idx, obj_name, pt, yaw, wlh) in placed_objects:
                        cx, cy = float(pt.x), float(pt.y)
                        if np.hypot(cx, cy) <= self.max_range:
                            all_generated_points.append(pt)
                            sb = SyntheticBox(obj_name, [cx, cy, 0.5], wlh, yaw=yaw, token=f"syn_{idx}_{s_idx}_{int(abs(cx*10))}")
                            syn_boxes.append(sb)

            return real_occluders, syn_boxes

    # =========================================================================
    # RENDERING CON ICONE VETTORIALI STILIZZATE E MINIMALI
    # =========================================================================
    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                         flip_h=False, halo_color=None, halo_radius=None, z_order=8):
        """Disegna un'icona raster/vettoriale trasparente ruotata e scalata preservando le proporzioni naturali."""
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Circle
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

        # Palette accademica ad alto contrasto
        c_bg = '#FFFFFF'
        c_grid = '#F1F5F9'
        c_border = '#0F172A'
        c_cone = '#8A99AD'          # Grigio ardesia coni d'ombra
        c_cone_edge = '#64748B'
        c_static_face = '#FCA5A5'   # Rosso chiaro muri/edifici
        c_static_edge = '#EF4444'

        c_car = '#1D4ED8'
        c_truck = '#7E22CE'
        c_ped = '#059669'
        c_bike = '#D97706'
        c_moto = '#EA580C'
        c_barrier = '#D97706'

        # Colori esclusivi per il ruolo nella GT Reale (non usati altrove)
        c_occluder = '#F97316'   # Arancione acceso  - chi genera l'ombra
        c_occluded  = '#8B5CF6'  # Viola medio       - chi è nascosto nell'ombra

        # Layout a griglia: pannello destro leggermente più largo per la legenda
        gs = GridSpec(3, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.925,
                      width_ratios=[1.1, 0.9], height_ratios=[1.25, 1.05, 1.20],
                      wspace=0.14, hspace=0.28)

        ax_map = self.fig.add_subplot(gs[:, 0])
        ax_telemetry = self.fig.add_subplot(gs[0, 1])
        ax_objects = self.fig.add_subplot(gs[1, 1])
        ax_legend = self.fig.add_subplot(gs[2, 1])

        # Titolo visualizzatore con indicazione Ground Truth attiva
        self.fig.text(0.03, 0.956,
                      f"Figure 3: Ground Truth Zone Occluse (BEV {int(self.max_range)}m)",
                      fontsize=10.5, fontweight='bold', color='#0F172A', ha='left', va='center')

        # =====================================================================
        # 1. PANNELLO SINISTRA: MAPPA BEV COMPLETA AD ALTA FEDELTÀ
        # =====================================================================
        ax_map.set_facecolor(c_bg)
        ax_map.set_xlim(-self.max_range, self.max_range)
        ax_map.set_ylim(-self.max_range, self.max_range)
        ax_map.set_aspect('equal')

        # Sfondo cartesiano metrico
        ticks = np.arange(-int(self.max_range), int(self.max_range) + 5, 5)
        ax_map.set_xticks(ticks)
        ax_map.set_yticks(ticks)
        ax_map.set_xticklabels([])
        ax_map.set_yticklabels([])
        ax_map.tick_params(colors='#CBD5E1', length=0)
        ax_map.grid(True, color=c_grid, linewidth=0.8, linestyle='-', zorder=1)

        # Assi ortogonali cartesiani
        ax_map.axhline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)
        ax_map.axvline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)

        # Cerchi concentrici metrici (10m, 20m, 25m)
        for r in [10.0, 20.0, self.max_range]:
            c = Circle((0, 0), r, color=c_border, fill=False, linewidth=1.1, zorder=3)
            ax_map.add_patch(c)
            ax_map.text(0, -r + 1.1, f"{int(r)}m", color=c_border, fontsize=9.5,
                        ha='center', va='center', fontweight='bold', zorder=10,
                        bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))

        # Mappa Stradale Vettoriale HD-Map (Drivable Area, Walkway & Strisce Pedonali)
        smap = self.frame_data.get('semantic_map', None)
        if smap is not None:
            drivable = np.rot90(smap.get('drivable_area', np.zeros((200, 200))), 3)
            walkway = np.rot90(smap.get('walkway', np.zeros((200, 200))), 3)
            crossing = np.rot90(smap.get('ped_crossing', np.zeros((200, 200))), 3)

            h, w = drivable.shape
            map_rgba = np.zeros((h, w, 4), dtype=np.uint8)

            # 1. Marciapiedi ed aree pedonali in grigio chiaro (#DAE0E9)
            map_rgba[walkway == 1] = [218, 224, 233, 245]
            # 2. Carreggiata stradale asfaltata in grigio scuro solido (valore 125)
            map_rgba[drivable == 1] = [125, 125, 125, 255]
            # 3. Attraversamenti pedonali: Strisce pedonali zebrate ad alta visibilità
            if np.any(crossing):
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                stripe_diag = ((xx + yy) % 4) < 2
                border = crossing & (~ndimage.binary_erosion(crossing, structure=np.ones((3, 3))))
                map_rgba[crossing == 1] = [90, 90, 90, 255]
                map_rgba[(crossing == 1) & stripe_diag] = [255, 255, 255, 255]
                map_rgba[border == 1] = [255, 255, 255, 255]

            ax_map.imshow(map_rgba, extent=[-40.0, 40.0, -40.0, 40.0], origin='lower', zorder=2)

        # Punti LiDAR REALI nuScenes filtrati entro max_range
        pts = self.frame_data.get('points', None)
        pts_dists = []
        if pts is not None and len(pts) > 0:
            all_dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            mask = (all_dists <= self.max_range)
            pts_bev = pts[mask]
            pts_dists = all_dists[mask]

            if len(pts_bev) > 4000:
                pts_sub = pts_bev[::len(pts_bev)//4000]
            else:
                pts_sub = pts_bev

            ax_map.scatter(pts_sub[:, 0], pts_sub[:, 1], s=1.1, c='#0F172A', alpha=0.60, zorder=4)

        # =====================================================================
        # OSTACOLI GROUND TRUTH NELLE ZONE OCCLUSE (SOLO DENTRO LE OMBRE!)
        # =====================================================================
        occluders, occluded_boxes = self.get_ground_truth_occluded_boxes()

        # Poligoni reali dei Coni d'Ombra (Raycasting Occlusions strettamente entro 25m)
        circle_25m = ShapelyPoint(0, 0).buffer(self.max_range)
        for occ in self.occlusions:
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue
            poly_arr = np.array(poly_pts)
            p = ShapelyPolygon(poly_arr)
            if not p.is_valid:
                p = p.buffer(0)
            p_vis = p.intersection(circle_25m)
            if p_vis.is_empty or p_vis.area < 0.05:
                continue
            
            has_occluded = False
            boxes_in_zone = []
            for box in occluded_boxes:
                corners_bev = box.corners_3d[:2, [0, 1, 5, 4]].T
                if p_vis.intersects(ShapelyPolygon(corners_bev)):
                    has_occluded = True
                    boxes_in_zone.append(box)

            # Se in modalità REALE_POSITIVES, mostra SOLO le zone d'ombra che ospitano ostacoli reali
            if self.gt_mode == "REALE_POSITIVES" and not has_occluded:
                continue

            face_c = '#FED7AA' if has_occluded else c_cone  # Arancione tenue per le ombre con oggetti
            edge_c = '#EA580C' if has_occluded else c_cone_edge
            lw = 1.6 if has_occluded else 0.9

            occ_tok = occ.get("object_token", "")
            occ_wlh = None
            if occ_tok:
                for b in self.frame_data.get('boxes', []):
                    if b.token == occ_tok:
                        occ_wlh = b.wlh
                        break
                if occ_wlh is None:
                    for sb in getattr(self, 'static_boxes', []):
                        if sb.token == occ_tok:
                            occ_wlh = [sb.max_x - sb.min_x, sb.max_y - sb.min_y, sb.max_z - sb.min_z]
                            break

            sub_geoms = list(p_vis.geoms) if p_vis.geom_type == 'MultiPolygon' else [p_vis]
            for geom in sub_geoms:
                if geom.area < 0.05:
                    continue
                c_pts = np.array(geom.exterior.coords)
                patch = MplPolygon(c_pts, closed=True,
                                   facecolor=face_c, edgecolor=edge_c,
                                   linewidth=lw, alpha=0.50, zorder=5)
                ax_map.add_patch(patch)

                self.drawn_occlusions.append({
                    'patch': patch,
                    'data': occ,
                    'polygon': c_pts,
                    'default_face': face_c,
                    'default_edge': edge_c,
                    'default_alpha': 0.50,
                    'default_z': 5,
                    'has_occluded': has_occluded,
                    'occluded_boxes': boxes_in_zone,
                    'occ_wlh': occ_wlh
                })

        # Strutture statiche man-made (muri/edifici)
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

        all_gt_boxes = occluded_boxes  # per il conteggio istogramma

        counts = {"Cars": 0, "Pedestrians": 0, "Trucks/Buses": 0, "Bicycles": 0, "Motorcycles": 0,
                  "Barriers": 0, "Carts": 0, "Cones": 0}
        counts_occ = {k: 0 for k in counts}  # conteggio occluders separato

        # Rilevamento intelligente dei camion articolati (su tutti i box)
        all_boxes_flat = occluders + occluded_boxes
        coupled_truck_tokens = set()
        for b1 in all_boxes_flat:
            if "truck" in b1.name.lower():
                c1 = b1.center[:2]
                for b2 in all_boxes_flat:
                    if "trailer" in b2.name.lower():
                        c2 = b2.center[:2]
                        if np.linalg.norm(c1 - c2) <= (b1.wlh[1] + b2.wlh[1]) / 2.0 + 3.5:
                            coupled_truck_tokens.add(b1.token)
                            break

        def _draw_box(box, z_order=8):
            """Disegna l'oggetto usando le stesse dimensioni e logica del visualizzatore raycasting."""
            try:
                corners_bev = box.corners_3d[:2, [0, 1, 5, 4]].T
            except Exception:
                return False
            center_xy = np.mean(corners_bev, axis=0)
            if np.linalg.norm(center_xy) > self.max_range:
                return False

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
                return False

            f_mid = (corners_bev[0] + corners_bev[1]) / 2.0
            r_mid = (corners_bev[2] + corners_bev[3]) / 2.0
            u_vec = f_mid - r_mid
            length = max(1.2, np.linalg.norm(u_vec))
            width = max(0.8, np.linalg.norm(corners_bev[1] - corners_bev[0]))
            deg = -np.degrees(np.arctan2(u_vec[0], u_vec[1]))

            if cat == "Pedestrians":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'pedestrian_green', center=center_xy, length=2.8, deg=0.0, flip_h=flip, z_order=z_order)
            elif cat == "Cars":
                c_l = min(4.8, max(3.8, length))
                self.draw_icon_object(ax_map, 'car_blue', center=center_xy, length=c_l, deg=deg, z_order=z_order)
            elif cat == "Trucks/Buses":
                if "construction" in b_name:
                    flip = (u_vec[0] < 0)
                    exc_w = min(4.5, max(3.2, length * 0.65))
                    exc_l = exc_w * (345.0 / 537.0)
                    self.draw_icon_object(ax_map, 'construction_orange', center=center_xy, length=exc_l, width=exc_w, deg=0.0, flip_h=flip, z_order=z_order)
                elif "trailer" in b_name:
                    t_l = min(13.5, max(4.5, length))
                    t_w = min(2.6, max(1.8, width))
                    t_center = f_mid - (u_vec / length) * (t_l / 2.0)
                    self.draw_icon_object(ax_map, 'trailer_purple', center=t_center, length=t_l, width=t_w, deg=deg, z_order=z_order)
                elif "truck" in b_name:
                    if box.token in coupled_truck_tokens:
                        cab_len = min(4.0, length * 0.65)
                        cab_center = f_mid - (u_vec / length) * (cab_len / 2.0 + 0.35)
                        self.draw_icon_object(ax_map, 'truck_cab_purple', center=cab_center, length=cab_len, width=width * 1.05, deg=deg, z_order=z_order)
                    else:
                        if length <= 6.8:
                            tr_len = min(5.2, max(4.0, length))
                            tr_w = min(2.1, max(1.7, width))
                        else:
                            tr_len = min(7.8, max(5.5, length))
                            tr_w = min(2.5, max(2.0, width))
                        self.draw_icon_object(ax_map, 'truck_purple', center=center_xy, length=tr_len, width=tr_w, deg=deg, z_order=z_order)
                else:
                    eff_len = min(7.8, max(5.5, length))
                    eff_w = min(2.5, max(2.0, width))
                    self.draw_icon_object(ax_map, 'truck_purple', center=center_xy, length=eff_len, width=eff_w, deg=deg, z_order=z_order)
            elif cat == "Bicycles":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'bicycle_orange', center=center_xy, length=2.6, deg=0.0, flip_h=flip, z_order=z_order)
            elif cat == "Motorcycles":
                flip = (u_vec[0] < 0)
                self.draw_icon_object(ax_map, 'motorcycle_amber', center=center_xy, length=2.6, deg=0.0, flip_h=flip, z_order=z_order)
            elif cat == "Barriers":
                self.draw_icon_object(ax_map, 'barrier_hazard', center=center_xy, length=2.2, deg=0.0, z_order=z_order)
            elif cat == "Carts":
                self.draw_icon_object(ax_map, 'cart_trolley', center=center_xy, length=2.2, deg=0.0, z_order=z_order)
            elif cat == "Cones":
                self.draw_icon_object(ax_map, 'traffic_cone', center=center_xy, length=1.5, deg=0.0, z_order=z_order)
            return cat

        # --- Disegna OCCLUDERS (ostacoli che generano le zone d'ombra) ---
        for box in occluders:
            b_name = box.name.lower()
            if "barrier" in b_name:      cat = "Barriers"
            elif "pushable" in b_name or "pullable" in b_name: cat = "Carts"
            elif "trafficcone" in b_name or "cone" in b_name: cat = "Cones"
            elif "bicycle" in b_name:    cat = "Bicycles"
            elif "motorcycle" in b_name: cat = "Motorcycles"
            elif "pedestrian" in b_name or "human" in b_name: cat = "Pedestrians"
            elif "car" in b_name or "emergency" in b_name:    cat = "Cars"
            elif any(c in b_name for c in ["truck", "trailer", "bus", "construction"]): cat = "Trucks/Buses"
            else: cat = None
            if cat:
                counts_occ[cat] += 1
            if getattr(self, 'show_occluders', True):
                _draw_box(box, z_order=8)

        # --- Disegna OCCLUDED (uguali, l'evidenziazione è nell'ombra arancione) ---
        for box in occluded_boxes:
            result = _draw_box(box, z_order=9)
            if result:
                counts[result] += 1


        # Ego Vehicle al centro (0, 0)
        self.draw_icon_object(ax_map, 'car_ego', center=[0.0, 0.0], length=5.0, deg=0.0, z_order=12)

        for spine in ax_map.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        # Tooltip interattivo
        self.tooltip = ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#38BDF8", linewidth=1.5, alpha=0.94),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        # =====================================================================
        # 2. PANNELLO DESTRA ALTO: TELEMETRIA EGO & BADGE GROUND TRUTH
        # =====================================================================
        ax_telemetry.set_facecolor('#F8FAFC')
        for spine in ax_telemetry.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)

        ax_telemetry.text(0.04, 0.88, "EGO VEHICLE TELEMETRY & GROUND TRUTH METADATA",
                          transform=ax_telemetry.transAxes, fontsize=9.5, fontweight='bold', color=c_border)

        # Box velocità
        ax_telemetry.add_patch(Rectangle((0.03, 0.10), 0.28, 0.68, transform=ax_telemetry.transAxes,
                                         facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2))
        ax_telemetry.text(0.17, 0.60, "VELOCITÀ EGO", transform=ax_telemetry.transAxes,
                          fontsize=8.0, fontweight='bold', color='#1D4ED8', ha='center', va='center')
        ax_telemetry.text(0.14, 0.35, f"{self.ego_speed_kmh:.1f}", transform=ax_telemetry.transAxes,
                          fontsize=19, fontweight='bold', color='#0F172A', ha='center', va='center')
        ax_telemetry.text(0.24, 0.30, "km/h", transform=ax_telemetry.transAxes,
                          fontsize=8.0, color='#64748B', va='bottom')

        # Badge modalità Ground Truth attiva
        if self.gt_mode == 'REALE':
            badge_bg = '#DCFCE7'
            badge_edge = '#16A34A'
            badge_text_col = '#15803D'
            badge_title = "GT REALE nuScenes (Tutte le Zone)"
        elif self.gt_mode == 'REALE_POSITIVES':
            badge_bg = '#FEF3C7'
            badge_edge = '#D97706'
            badge_text_col = '#B45309'
            badge_title = "GT REALE (Solo Zone con Oggetto)"
        elif self.gt_mode == 'GEOMETRICA':
            badge_bg = '#E0F2FE'
            badge_edge = '#0284C7'
            badge_text_col = '#0369A1'
            badge_title = "★ GT GEOMETRICA (Fitting 3D)"
        elif self.gt_mode == 'SEMANTICA':
            badge_bg = '#FFF1F2'
            badge_edge = '#E11D48'
            badge_text_col = '#BE123C'
            badge_title = "★ GT SEMANTICA (Regole Naïve HD-Map)"
        else:  # NEURO_SIMB
            badge_bg = '#EDE9FE'
            badge_edge = '#7C3AED'
            badge_text_col = '#6D28D9'
            badge_title = "★ GT IBRIDA (Spazio 3D + HD-Map)"

        ax_telemetry.add_patch(FancyBboxPatch((0.35, 0.60), 0.59, 0.18, boxstyle="round,pad=0.02",
                                              transform=ax_telemetry.transAxes,
                                              facecolor=badge_bg, edgecolor=badge_edge,
                                              linewidth=1.2, zorder=2))
        ax_telemetry.text(0.645, 0.69, badge_title,
                          transform=ax_telemetry.transAxes,
                          fontsize=8.5, fontweight='bold', color=badge_text_col, ha='center', va='center')

        # Dati descrittivi telemetria
        total_occluded = sum(counts.values())
        total_occluders = sum(counts_occ.values())
        occ_status = "Visibili" if getattr(self, 'show_occluders', True) else "Nascosti"

        if self.gt_mode == "REALE":
            meta_lines = [
                f"Dataset: nuScenes ({self.scene_name})",
                f"Campione: {self.current_idx + 1} / {self.total_frames}",
                f"Occluders: {total_occluders} ({occ_status})",
                f"Occluded (dentro l'ombra): {total_occluded} oggetti",
                f"Mostra: Tutte le zone d'ombra (con e senza ostacolo)"
            ]
            meta_colors  = ['#334155', '#334155', '#B45309', '#7C3AED', '#15803D']
            meta_weights = ['normal', 'normal', 'bold', 'bold', 'bold']
        elif self.gt_mode == "REALE_POSITIVES":
            meta_lines = [
                f"Dataset: nuScenes ({self.scene_name})",
                f"Campione: {self.current_idx + 1} / {self.total_frames}",
                f"Occluders: {total_occluders} ({occ_status})",
                f"Occluded Reali Rilevati: {total_occluded} oggetti",
                f"Mostra: ESCLUSIVAMENTE zone con ostacoli reali"
            ]
            meta_colors  = ['#334155', '#334155', '#B45309', '#7C3AED', '#15803D']
            meta_weights = ['normal', 'normal', 'bold', 'bold', 'bold']
        elif self.gt_mode == "GEOMETRICA":
            flt_txt = "ATTIVO" if getattr(self, 'use_occluder_filter', True) else "DISATTIVATO"
            flt_col = '#15803D' if getattr(self, 'use_occluder_filter', True) else '#EA580C'
            meta_lines = [
                f"Dataset: nuScenes ({self.scene_name})",
                f"Campione: {self.current_idx + 1} / {self.total_frames}",
                f"Occluders: {total_occluders} ({occ_status})",
                f"Ostacoli Sintetici Previsti: {total_occluded}",
                f"Fitting Spaziale 3D (Filtro 'O': {flt_txt})"
            ]
            meta_colors  = ['#334155', '#334155', '#B45309', '#0F172A', flt_col]
            meta_weights = ['normal', 'normal', 'bold', 'bold', 'bold']
        elif self.gt_mode == "SEMANTICA":
            flt_txt = "ATTIVO" if getattr(self, 'use_occluder_filter', True) else "DISATTIVATO"
            flt_col = '#15803D' if getattr(self, 'use_occluder_filter', True) else '#EA580C'
            meta_lines = [
                f"Dataset: nuScenes ({self.scene_name})",
                f"Campione: {self.current_idx + 1} / {self.total_frames}",
                f"Occluders: {total_occluders} ({occ_status})",
                f"Ostacoli Sintetici Previsti: {total_occluded}",
                f"Affordance Semantica (Filtro 'O': {flt_txt})"
            ]
            meta_colors  = ['#334155', '#334155', '#B45309', '#0F172A', flt_col]
            meta_weights = ['normal', 'normal', 'bold', 'bold', 'bold']
        else:  # NEURO_SIMB
            flt_txt = "ATTIVO" if getattr(self, 'use_occluder_filter', True) else "DISATTIVATO"
            flt_col = '#15803D' if getattr(self, 'use_occluder_filter', True) else '#EA580C'
            meta_lines = [
                f"Dataset: nuScenes ({self.scene_name})",
                f"Campione: {self.current_idx + 1} / {self.total_frames}",
                f"Occluders: {total_occluders} ({occ_status})",
                f"Ostacoli Sintetici: {total_occluded} (Filtro 'O': {flt_txt})",
                f"Strategia: Ibrida (Spazio 3D + Mappa HD)"
            ]
            meta_colors  = ['#334155', '#334155', '#B45309', flt_col, '#7C3AED']
            meta_weights = ['normal', 'normal', 'bold', 'bold', 'bold']
        y_m = 0.48
        for idx_l, line in enumerate(meta_lines):
            ax_telemetry.text(0.35, y_m, line, transform=ax_telemetry.transAxes,
                              fontsize=7.8, color=meta_colors[idx_l], va='center',
                              fontweight=meta_weights[idx_l])
            y_m -= 0.095

        ax_telemetry.set_xlim(0, 1)
        ax_telemetry.set_ylim(0, 1)
        ax_telemetry.set_xticks([])
        ax_telemetry.set_yticks([])

        # =====================================================================
        # 3. PANNELLO DESTRA CENTRO: CONTEGGIO OSTACOLI GROUND TRUTH NELLE OMBRE
        # =====================================================================
        ax_objects.set_facecolor('#FFFFFF')
        for spine in ax_objects.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.0)
        ax_objects.spines['top'].set_visible(False)
        ax_objects.spines['right'].set_visible(False)

        cat_names = ["Cars", "Pedestrians", "Trucks", "Carts", "Barriers", "Cones"]
        cat_counts = [
            counts["Cars"],
            counts["Pedestrians"],
            counts["Trucks/Buses"],
            counts["Carts"],
            counts["Barriers"],
            counts["Cones"]
        ]
        cat_colors = [c_car, c_ped, c_truck, '#0284C7', c_barrier, '#EA580C']

        bars = ax_objects.bar(cat_names, cat_counts, width=0.45, color=cat_colors,
                              edgecolor=c_border, linewidth=1.1, zorder=3)
        max_c = max(max(cat_counts, default=1), 4)
        ax_objects.set_ylim(0, max_c + 2)
        ax_objects.set_ylabel(f"Numero Oggetti GT (entro {int(self.max_range)}m)", fontsize=8.5, fontweight='bold', color=c_border)
        ax_objects.set_title(f"Distribuzione Ostacoli Ground Truth ({self.gt_mode}) nelle Zone Occluse",
                             fontsize=10.0, fontweight='bold', pad=8, color=c_border)
        ax_objects.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_objects.tick_params(colors=c_border, labelsize=8.5)

        for bar, val in zip(bars, cat_counts):
            ax_objects.text(bar.get_x() + bar.get_width()/2.0, val + 0.3, str(val),
                            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_border)

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

        ax_legend.set_title("Simbologia & Legenda Ufficiale (nuScenes 3D & Sensori)",
                            fontsize=10.0, fontweight='bold', pad=9, color=c_border)

        columns = [
            (0.045, 0.095, [
                ('icon', 'car_ego', 'Ego Vehicle', 0.046),
                ('icon', 'car_blue', 'Car (Auto)', 0.046),
                ('icon', 'truck_purple', 'Truck / Trailer / Bus', 0.035),
                ('icon', 'construction_orange', 'Cantiere (Excavator)', 0.042),
                ('icon', 'barrier_hazard', 'Barrier (Barriera)', 0.044),
            ]),
            (0.375, 0.425, [
                ('icon', 'pedestrian_green', 'Pedestrian (VRU)', 0.055),
                ('icon', 'bicycle_orange', 'Bicycle (Bici)', 0.050),
                ('icon', 'motorcycle_amber', 'Motorcycle (Moto)', 0.050),
                ('icon', 'cart_trolley', 'Cart / Scala (Push.)', 0.050),
                ('icon', 'traffic_cone', 'Traffic Cone (Cono)', 0.048),
            ]),
            (0.705, 0.755, [
                ('rect', c_cone, 'Raycasting Occlusion', 0.55),
                ('dashed_rect', c_static_face, c_static_edge, 'Struttura Statica (Muro)', 0.45),
                ('rect', '#7D7D7D', 'Carreggiata (HD-Map)', 1.0),
                ('rect', '#DAE0E9', 'Marciapiede (Walkway)', 1.0),
                ('rect', '#FED7AA', 'Ombra con Oggetto Nascosto', 1.0),
            ]),
        ]

        for x_ic, x_tx, items in columns:
            start_y = 0.84 if len(items) > 5 else 0.82
            step_y = 0.138 if len(items) > 5 else 0.16
            for i, it in enumerate(items):
                cur_y = start_y - i * step_y
                if it[0] == 'icon':
                    oi = OffsetImage(self.icons[it[1]], zoom=it[3])
                    ab = AnnotationBbox(oi, (x_ic, cur_y), xycoords='axes fraction', frameon=False, zorder=5)
                    ax_legend.add_artist(ab)
                elif it[0] == 'rect':
                    r = Rectangle((x_ic - 0.016, cur_y - 0.020), 0.032, 0.040,
                                  transform=ax_legend.transAxes, facecolor=it[1], edgecolor=c_border,
                                  linewidth=0.8, alpha=it[3], zorder=5)
                    ax_legend.add_patch(r)
                elif it[0] == 'zebra':
                    r = Rectangle((x_ic - 0.016, cur_y - 0.020), 0.032, 0.040,
                                  transform=ax_legend.transAxes, facecolor=it[1], edgecolor=c_border,
                                  hatch='//', linewidth=0.8, alpha=it[3], zorder=5)
                    ax_legend.add_patch(r)
                elif it[0] == 'dashed_rect':
                    r = Rectangle((x_ic - 0.016, cur_y - 0.020), 0.032, 0.040,
                                  transform=ax_legend.transAxes, facecolor=it[1], edgecolor=it[2],
                                  linewidth=1.3, linestyle='--', alpha=it[4], zorder=5)
                    ax_legend.add_patch(r)
                elif it[0] == 'occluder_rect':
                    r = Rectangle((x_ic - 0.016, cur_y - 0.020), 0.032, 0.040,
                                  transform=ax_legend.transAxes, facecolor=it[1], edgecolor=it[2],
                                  linewidth=1.8, linestyle='solid', alpha=it[4], zorder=5)
                    ax_legend.add_patch(r)
                elif it[0] == 'occluded_rect':
                    r = Rectangle((x_ic - 0.016, cur_y - 0.020), 0.032, 0.040,
                                  transform=ax_legend.transAxes, facecolor=it[1], edgecolor=it[2],
                                  linewidth=1.8, linestyle=(0, (4, 2)), alpha=it[4], zorder=5)
                    ax_legend.add_patch(r)
                elif it[0] == 'circle':
                    c = Circle((x_ic, cur_y), 0.010, transform=ax_legend.transAxes,
                               facecolor=it[1], edgecolor='none', zorder=5)
                    ax_legend.add_patch(c)
                label = it[3] if it[0] in ('dashed_rect', 'occluder_rect', 'occluded_rect') else it[2]
                ax_legend.text(x_tx, cur_y, label, transform=ax_legend.transAxes,
                               fontsize=7.8, color=c_border, va='center', fontweight='medium')

        ax_legend.text(0.50, 0.04, "* Mostra ostacoli GT nelle zone d'ombra. Usa il Menu a tendina o premi [B] per cambiare modalità.",
                       transform=ax_legend.transAxes, fontsize=6.8, color='#64748B', ha='center', va='center', style='italic')

        # Pulsante toggle per mostrare/nascondere gli occludori
        ax_btn_occ = self.fig.add_axes([0.420, 0.938, 0.130, 0.036])
        if getattr(self, 'show_occluders', True):
            lbl_occ = "● Occlusori: ON"
            bg_occ = '#EFF6FF'
            hov_occ = '#DBEAFE'
            txt_occ = '#1D4ED8'
        else:
            lbl_occ = "○ Occlusori: OFF"
            bg_occ = '#F8FAFC'
            hov_occ = '#E2E8F0'
            txt_occ = '#64748B'

        self.btn_occ = Button(ax_btn_occ, lbl_occ, color=bg_occ, hovercolor=hov_occ)
        self.btn_occ.label.set_fontsize(8.0)
        self.btn_occ.label.set_fontweight('bold')
        self.btn_occ.label.set_color(txt_occ)
        self.btn_occ.on_clicked(self.toggle_occluders)
        self.ax_btn_occ = ax_btn_occ

        # Menu a tendina interattivo per la selezione della modalità Ground Truth
        ax_dd = self.fig.add_axes([0.565, 0.938, 0.280, 0.036])
        mode_labels = {
            "NEURO_SIMB": "★ GT Ibrida (Spazio+HDMap)",
            "GEOMETRICA": "★ GT Geometrica (Fitting 3D)",
            "SEMANTICA": "★ GT Semantica (Regole Naïve)",
            "REALE": "GT Reale nuScenes (Tutte le Zone)",
            "REALE_POSITIVES": "GT Reale (Solo Positivi)",
            "SINTETICA_HYBRID": "★ GT Ibrida (Spazio+HDMap)",
            "HYBRID": "★ GT Ibrida (Spazio+HDMap)"
        }
        curr_lbl = mode_labels.get(self.gt_mode, self.gt_mode)
        arrow = "▲" if getattr(self, 'dropdown_open', False) else "▼"
        self.btn_mode = Button(ax_dd, f"Modalità: {curr_lbl}  {arrow}", color='#F8FAFC', hovercolor='#E2E8F0')
        self.btn_mode.label.set_fontsize(8.0)
        self.btn_mode.label.set_fontweight('bold')
        self.btn_mode.label.set_color(c_border)
        self.btn_mode.on_clicked(self.toggle_dropdown)
        self.ax_dd = ax_dd

        # Casella interattiva per saltare direttamente a un frame
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
                      "[<- / ->]: Frame  |  [B / Menu]: Modalità GT  |  [V / Bottone]: Occlusori ON/OFF  |  [O]: Filtro Occludore  |  [Hover Mouse]: Dettagli  |  [S]: Salva HD",
                      fontsize=8.0, color='#64748B', ha='center', style='italic')

        self.fig.canvas.draw_idle()

    # =========================================================================
    # GESTIONE MENU A TENDINA (DROPDOWN)
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
            ("NEURO_SIMB", "1. ★ GT Ibrida (Spazio 3D + HD-Map)"),
            ("GEOMETRICA", "2. ★ GT Geometrica (Fitting 3D)"),
            ("SEMANTICA", "3. ★ GT Semantica (Regole Naïve)"),
            ("REALE", "4. GT Reale nuScenes (Tutte le Zone)"),
            ("REALE_POSITIVES", "5. GT Reale (Solo Zone con Oggetto)")
        ]
        
        y_offsets = [0.148, 0.111, 0.074, 0.037, 0.001]
        self.menu_buttons = []
        for (m_key, m_lbl), y_off in zip(modes, y_offsets):
            is_active = (self.gt_mode == m_key)
            prefix = " ●  " if is_active else " ○  "
            bg_col = '#EFF6FF' if is_active else '#FFFFFF'
            txt_col = '#1D4ED8' if is_active else '#334155'
            
            ax_item = self.fig.add_axes([0.567, 0.748 + y_off, 0.276, 0.035], zorder=101)
            btn = Button(ax_item, prefix + m_lbl, color=bg_col, hovercolor='#E0F2FE')
            btn.label.set_fontsize(7.8)
            btn.label.set_fontweight('bold' if is_active else 'normal')
            btn.label.set_color(txt_col)
            btn.label.set_ha('left')
            btn.label.set_x(0.035)
            btn.on_clicked(lambda ev, k=m_key: self.select_gt_mode(k))
            self.menu_buttons.append((ax_item, btn, m_key))

        self.fig.canvas.draw_idle()

    def _hide_dropdown_menu(self):
        if hasattr(self, 'menu_buttons') and self.menu_buttons:
            for item in self.menu_buttons:
                ax_item = item[0]
                try:
                    ax_item.remove()
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

    def toggle_occluders(self, event=None):
        self.show_occluders = not getattr(self, 'show_occluders', True)
        st = "VISIBILI" if self.show_occluders else "NASCOSTI"
        print(f"\n>>> [TOGGLE OCCLUSORI] Visualizzazione occludori: {st}")
        self.render()

    def select_gt_mode(self, mode_key):
        self.gt_mode = mode_key
        self.dropdown_open = False
        self._hide_dropdown_menu()
        print(f"\n>>> [DROPDOWN MENU] Selezionata modalità: {self.gt_mode}")
        self.render()

    # =========================================================================
    # EVENTI MOUSE & TASTIERA
    # =========================================================================
    def on_key(self, event):
        if event.key in ['right', 'd', 'D']:
            self.load_frame(self.current_idx + 1)
        elif event.key in ['left', 'a', 'A']:
            self.load_frame(self.current_idx - 1)
        elif event.key in ['v', 'V', 'h', 'H']:
            self.toggle_occluders()
        elif event.key in ['o', 'O']:
            self.use_occluder_filter = not getattr(self, 'use_occluder_filter', True)
            st = "ATTIVO" if self.use_occluder_filter else "DISATTIVATO"
            print(f"\n>>> [FILTRO COMPATIBILITA OCCLUDORE] Stato: {st}")
            self.render()
        elif event.key in ['b', 'B']:
            # Cicla tra le 5 modalità
            mode_cycle = ["NEURO_SIMB", "GEOMETRICA", "SEMANTICA", "REALE", "REALE_POSITIVES"]
            curr_idx = mode_cycle.index(self.gt_mode) if self.gt_mode in mode_cycle else 0
            self.gt_mode = mode_cycle[(curr_idx + 1) % len(mode_cycle)]
            print(f"\n>>> [GROUND TRUTH TOGGLE] Modalità attiva: {self.gt_mode}")
            self.render()
        elif event.key in ['s', 'S']:
            out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"gt_occlusioni_frame_{self.current_idx + 1}_{self.gt_mode.lower()}.png"
            fpath = os.path.join(out_dir, fname)
            self.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
            print(f"\n[SALVATAGGIO OK]: Immagine salvata in: {fpath} (300 DPI)")
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
            # Se il click è su uno dei pulsanti delle scelte del menu, seleziona direttamente la modalità
            for item in getattr(self, 'menu_buttons', []):
                ax_item, _, m_key = item
                if event.inaxes == ax_item:
                    self.select_gt_mode(m_key)
                    return
            # Se il click è sul pulsante di apertura/chiusura del menu, lascia gestire il toggle
            if event.inaxes == getattr(self, 'ax_dd', None):
                return
            # Se si clicca all'esterno, nasconde il menu e ritorna
            self._hide_dropdown_menu()
            return

        if event.inaxes is None or not self.drawn_occlusions:
            return
        if event.button != 1:  # Solo click sinistro
            return

        x_m, y_m = event.xdata, event.ydata
        if x_m is None or y_m is None:
            return

        pt = ShapelyPoint(x_m, y_m)
        clicked_item = None

        for item in self.drawn_occlusions:
            try:
                poly = ShapelyPolygon(item['polygon'])
                if poly.contains(pt):
                    clicked_item = item
                    break
            except Exception:
                pass

        if not hasattr(self, 'highlight_artists'):
            self.highlight_artists = []
            
        for art in self.highlight_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.highlight_artists.clear()

        if clicked_item is not None and clicked_item.get('has_occluded'):
            for box in clicked_item.get('occluded_boxes', []):
                corners_bev = box.corners_3d[:2, [0, 1, 5, 4]].T
                center_xy = np.mean(corners_bev, axis=0)
                
                # Cerchio evidenziatore sotto l'oggetto
                circle = Circle((center_xy[0], center_xy[1]), 2.8, 
                                facecolor='#FDE047', edgecolor='#EAB308', 
                                alpha=0.6, linewidth=2, zorder=7)
                event.inaxes.add_patch(circle)
                self.highlight_artists.append(circle)
                
        self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes is None or not self.drawn_occlusions:
            if self.tooltip and self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x_m, y_m = event.xdata, event.ydata
        if x_m is None or y_m is None:
            return

        pt = ShapelyPoint(x_m, y_m)
        hovered_item = None

        for item in self.drawn_occlusions:
            try:
                poly = ShapelyPolygon(item['polygon'])
                if poly.contains(pt):
                    hovered_item = item
                    break
            except Exception:
                pass

        if hovered_item != self.active_hovered:
            if self.active_hovered is not None:
                p = self.active_hovered['patch']
                p.set_facecolor(self.active_hovered['default_face'])
                p.set_edgecolor(self.active_hovered['default_edge'])
                p.set_alpha(self.active_hovered['default_alpha'])
                p.set_linewidth(0.9)
                p.set_zorder(self.active_hovered['default_z'])

            self.active_hovered = hovered_item

            if hovered_item is not None:
                p = hovered_item['patch']
                p.set_facecolor('#0284C7')
                p.set_edgecolor('#0369A1')
                p.set_alpha(0.85)
                p.set_linewidth(2.2)
                p.set_zorder(20)

                occ_data = hovered_item['data']
                src = occ_data.get('object_name', 'Sconosciuto')
                area = occ_data.get('area_sqm', 0.0)
                dist = occ_data.get('distance_m', 0.0)
                p_road = occ_data.get('road_fraction', 0.0)
                p_walk = occ_data.get('sidewalk_fraction', 0.0)
                p_cross = occ_data.get('crosswalk_fraction', 0.0)
                p_park = occ_data.get('carpark_fraction', 0.0)
                p_terr = occ_data.get('terrain_fraction', 0.0)

                parts = []
                if p_road >= 0.05:
                    parts.append(f"Carreggiata ({int(round(p_road * 100))}%)")
                if p_walk >= 0.05:
                    parts.append(f"Marciapiede ({int(round(p_walk * 100))}%)")
                if p_cross >= 0.05:
                    parts.append(f"Attraversamento ({int(round(p_cross * 100))}%)")
                if p_park >= 0.05:
                    parts.append(f"Parcheggio ({int(round(p_park * 100))}%)")
                if p_terr >= 0.05:
                    parts.append(f"Terreno/Verde ({int(round(p_terr * 100))}%)")
                label = ", ".join(parts) if parts else "Non Classificato"

                # Traduzioni amichevoli
                src_map = {
                    'vehicle.car': 'Automobile (vehicle.car)',
                    'vehicle.truck': 'Camion (vehicle.truck)',
                    'vehicle.bus': 'Autobus (vehicle.bus)',
                    'human.pedestrian.adult': 'Pedone (human.pedestrian)',
                    'static.manmade': 'Struttura Statica (Muro/Palazzo)',
                }
                src_fmt = src_map.get(src, src)

                occ_wlh = hovered_item.get('occ_wlh', None)
                if getattr(self, 'use_occluder_filter', True):
                    flt_info = get_occluder_filter_explanation(src, occ_wlh)
                else:
                    flt_info = "Disattivato (Tasto 'O')"

                tt_text = (
                    f"ZONA OCCLUSA (Raycasting)\n"
                    f"• Chi la genera:    {src_fmt}\n"
                    f"• Vincolo Occludore: {flt_info}\n"
                    f"• Dimensione:       Area {area:.1f} m²\n"
                    f"• Semantica:        {label}\n"
                    f"• Distanza da Ego:   {dist:.1f} m"
                )

                self.tooltip.set_text(tt_text)
                self.tooltip.xy = (x_m, y_m)
                self.tooltip.set_visible(True)
            else:
                self.tooltip.set_visible(False)

            self.fig.canvas.draw_idle()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore Tesi Ground Truth nelle Zone Occluse")
    parser.add_argument("frame", nargs="?", type=int, default=1, help="Numero del frame iniziale (1-404)")
    parser.add_argument("--mode", "-m",
                        choices=["NEURO_SIMB", "GEOMETRICA", "SEMANTICA", "REALE", "REALE_POSITIVES", "HYBRID", "GEOMETRIC", "SEMANTIC", "SINTETICA_HYBRID", "SINTETICA"],
                        default="NEURO_SIMB",
                        help="Modalità iniziale della Ground Truth ('NEURO_SIMB', 'GEOMETRICA', 'SEMANTICA', 'REALE', 'REALE_POSITIVES')")
    parser.add_argument("--save", action="store_true", help="Salva screenshot a 300 DPI ed esce")
    args = parser.parse_args()

    init_frame = max(1, min(404, args.frame))
    vis = GroundTruthOcclusionVisualizer(max_range=25.0, initial_mode=args.mode.upper())
    vis.load_frame(init_frame - 1, broadcast=True)

    if args.save:
        out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"fig_3_gt_occlusions_{vis.gt_mode.lower()}.png"
        fpath = os.path.join(out_dir, fname)
        vis.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"\n[SALVATA CON SUCCESSO]: Immagine Ground Truth salvata a 300 DPI in:\n  -> {fpath}")
        return

    # Timer sincronizzazione
    timer = vis.fig.canvas.new_timer(interval=150)
    timer.add_callback(vis.check_sync_file)
    timer.start()

    plt.show()


if __name__ == "__main__":
    main()
