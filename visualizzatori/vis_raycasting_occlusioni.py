# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - CALCOLO OCCLUSIONI RAYCASTING (NUSCENES BEV)
# File: visualizzatori/vis_raycasting_occlusioni.py
#
# Visualizzatore in stile paper accademico:
# - SINISTRA: Mappa BEV ad altissima fedeltà (Raggio 25m):
#             * Sfondo cartesiano, assi ortogonali e cerchi metrici (10m, 20m, 25m);
#             * HD-Map nuScenes (carreggiata stradale e marciapiedi);
#             * Nuvola di punti LiDAR reali nuScenes filtrati a 25m;
#             * Ego Vehicle e ostacoli con icone vettoriali orientate;
#             * Strutture statiche man-made (muri/edifici da clustering) in rosso;
#             * Coni d'ombra reali calcolati tramite Raycasting in grigio ardesia;
# - DESTRA IN ALTO: Blocco telemetria con VELOCITÀ EGO (km/h) e metadati scena;
# - DESTRA AL CENTRO: Istogramma conteggio ostacoli reali per categoria entro 25m;
# - DESTRA IN BASSO: Simbologia e legenda ufficiale (nuScenes 3D, sensori e layer);
# - CONTROLLI: Frecce DESTRA/SINISTRA per scorrere i frame, 'S' per salvare a 300 DPI.
# ==============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox

# Impostazione percorso root del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from dataset_adapter.factory_dataset import create_adapter
from raycaster.ray_caster import RayCaster

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")

class PaperStyleVisualizer:
    def __init__(self, max_range=25.0):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: DATASET NUSCENES & SENSORI (STILE PAPER SCIENTIFICO)")
        print("=" * 80)

        self.max_range = float(max_range)
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0

        # Setup tipografia accademica
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Raycasting Occlusions Visualizer")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None

        self.load_icons()
        self.load_frame(self.current_idx, broadcast=False)

        # Timer per sincronizzazione automatica bidirezionale tra visualizzatori
        self.sync_timer = self.fig.canvas.new_timer(interval=300)
        self.sync_timer.add_callback(self.check_sync_file)
        self.sync_timer.start()

    def load_icons(self):
        """Carica le icone vettoriali stilizzate e minimali per la vista BEV."""
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
        """Calcola la velocità istantanea dell'ego-vehicle in km/h dall'odometria nuScenes."""
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

        # Rileva pseudo-box statici (muri/edifici) con il clustering geometrico di ray_caster.py
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
                f.write(f"{idx},raycasting")
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
                    if source != "raycasting" and target != self.current_idx and 0 <= target < self.total_frames:
                        self.load_frame(target, broadcast=False)
        except Exception:
            pass

    # =========================================================================
    # RENDERING CON ICONE VETTORIALI STILIZZATE E MINIMALI (STILE PROVA 1)
    # =========================================================================
    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                         flip_h=False, halo_color=None, halo_radius=None):
        """Disegna un'icona raster/vettoriale trasparente ruotata e scalata preservando le proporzioni naturali."""
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Circle
        from PIL import Image

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
        self.fig.clf()
        self.fig.patch.set_facecolor('#FFFFFF')

        # Layout a griglia: identico a vis_ground_truth_occlusioni.py
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

        # Titolo visualizzatore unificato in alto a sinistra
        self.fig.text(0.03, 0.956,
                      f"Figure 2: Mappa BEV Raycasting & Calcolo Coni d'Ombra (Raggio {int(self.max_range)}m)",
                      fontsize=11.5, fontweight='bold', color='#0F172A', ha='left', va='center')

        # =====================================================================
        # 1. PANNELLO SINISTRO: MAPPA BEV VETTORIALE NUSCENES
        # =====================================================================
        c_map_bg = '#F8FAFC'      # Sfondo chiaro pulito
        c_grid = '#E2E8F0'        # Quadrettatura cartesiana
        c_border = '#0F172A'      # Assi e cerchi neri netti
        c_car = '#1D4ED8'         # Blu cobalto Auto
        c_ped = '#16A34A'         # Verde smeraldo Pedoni
        c_truck = '#9333EA'       # Viola Camion/Bus
        c_bike = '#EA580C'        # Arancione Bici
        c_moto = '#D97706'        # Ambra Moto
        c_barrier = '#F59E0B'     # Giallo/Ambra Barriera stradale
        c_cone = '#64748B'        # Coni d'ombra raycasting (Grigio ardesia)
        c_cone_vru = '#94A3B8'    # Coni d'ombra VRU (Grigio medio)
        c_cone_edge = '#475569'   # Bordo cono d'ombra (Grigio scuro)
        c_static_face = '#EF4444' # Strutture statiche (Rosso)
        c_static_edge = '#B91C1C' # Bordo tratteggiato strutture statiche (Rosso scuro)

        ax_map.set_facecolor(c_map_bg)

        # A. Quadrettatura metrica 2m x 2m (estesa su raggio max_range)
        grid_lim = self.max_range + 1.1
        for gx in np.arange(-grid_lim, grid_lim, 2.0):
            ax_map.axvline(gx, color=c_grid, linewidth=0.55, zorder=1)
        for gy in np.arange(-grid_lim, grid_lim, 2.0):
            ax_map.axhline(gy, color=c_grid, linewidth=0.55, zorder=1)

        # B. Assi ortogonali a croce passanti per l'ego-vehicle
        ax_map.axvline(0, color=c_border, linewidth=1.1, zorder=2)
        ax_map.axhline(0, color=c_border, linewidth=1.1, zorder=2)

        # C. Cerchi metrici concentrici con etichette standard
        circles = [10.0, 20.0, self.max_range] if self.max_range not in [10.0, 20.0] else [10.0, 20.0]
        for r in circles:
            c = Circle((0, 0), r, color=c_border, fill=False, linewidth=1.1, zorder=3)
            ax_map.add_patch(c)
            ax_map.text(0, -r + 1.1, f"{int(r)}m", color=c_border, fontsize=9.5,
                        ha='center', va='center', fontweight='bold', zorder=10,
                        bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))

        # D. Mappa Stradale Vettoriale HD-Map (nuScenes Semantic Map: Drivable Area & Walkway)
        smap = self.frame_data.get('semantic_map', None)
        if smap is not None:
            drivable = np.rot90(smap.get('drivable_area', np.zeros((200, 200))), 3)
            walkway = np.rot90(smap.get('walkway', np.zeros((200, 200))), 3)
            crossing = np.rot90(smap.get('ped_crossing', np.zeros((200, 200))), 3)

            h, w = drivable.shape
            map_rgba = np.zeros((h, w, 4), dtype=np.uint8)

            # 1. Marciapiedi ed aree pedonali autentiche da HD-Map in grigio chiaro (#DAE0E9)
            map_rgba[walkway == 1] = [218, 224, 233, 245]
            # 2. Carreggiata stradale asfaltata in grigio scuro solido ufficiale nuScenes (valore 125)
            map_rgba[drivable == 1] = [125, 125, 125, 255]
            # 3. Attraversamenti pedonali: Strisce pedonali zebrate ad alta visibilità
            if np.any(crossing):
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                stripe_diag = ((xx + yy) % 4) < 2
                border = crossing & (~ndimage.binary_erosion(crossing, structure=np.ones((3, 3))))
                map_rgba[crossing == 1] = [90, 90, 90, 255]  # fondo asfalto a contrasto
                map_rgba[(crossing == 1) & stripe_diag] = [255, 255, 255, 255]  # strisce zebrate bianche
                map_rgba[border == 1] = [255, 255, 255, 255]  # delimitazione perimetrale

            ax_map.imshow(map_rgba, extent=[-40.0, 40.0, -40.0, 40.0], origin='lower', zorder=2)

        # E. Punti LiDAR REALI nuScenes (filtrati entro il raggio di 25 metri)
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

        # F. Poligoni reali dei Coni d'Ombra (Raycasting Occlusions entro max_range)
        for occ in self.occlusions:
            poly_pts = occ.get("polygon_points_m", [])
            if len(poly_pts) < 3:
                continue
            poly_np = np.array(poly_pts)
            # Mostra l'ombra se almeno una sua porzione ricade entro il raggio visivo
            if np.min(np.hypot(poly_np[:, 0], poly_np[:, 1])) > self.max_range and occ.get("distance_m", 0.0) > self.max_range:
                continue
            poly_xy = poly_np[:, :2]
            area = occ.get("area_sqm", 0)
            f_col = c_cone if area > 3.0 else c_cone_vru
            mpl_occ = MplPolygon(poly_xy, closed=True, facecolor=f_col,
                                 edgecolor=c_cone_edge, linewidth=0.9, alpha=0.50, zorder=5)
            ax_map.add_patch(mpl_occ)
            from matplotlib.path import Path
            self.drawn_occlusions.append({
                'patch': mpl_occ,
                'poly_xy': poly_xy,
                'path': Path(poly_xy),
                'data': occ,
                'default_fc': f_col,
                'default_ec': c_cone_edge,
                'default_lw': 0.9,
                'default_alpha': 0.50,
                'default_z': 5
            })

        # F2. STRUTTURE STATICHE MAN-MADE (Muri / Edifici rilevati da clustering LiDAR in ray_caster.py)
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

        # G. OSTACOLI REALI NUSCENES DISEGNATI CON GRAFICA STILIZZATA AD ALTA FEDELTÀ (RAGGIO 25M)
        boxes = self.frame_data.get('boxes', [])
        counts = {"Cars": 0, "Pedestrians": 0, "Trucks/Buses": 0, "Bicycles": 0, "Motorcycles": 0,
                  "Barriers": 0, "Carts": 0, "Cones": 0}

        # Rilevamento intelligente dei camion accoppiati a semirimorchi (TIR articolati)
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
            if "barrier" in b_name:
                col = c_barrier
                cat = "Barriers"
            elif "pushable" in b_name or "pullable" in b_name:
                col = '#0284C7'
                cat = "Carts"
            elif "trafficcone" in b_name or "cone" in b_name:
                col = '#EA580C'
                cat = "Cones"
            elif "bicycle" in b_name:
                col = c_bike
                cat = "Bicycles"
            elif "motorcycle" in b_name:
                col = c_moto
                cat = "Motorcycles"
            elif "pedestrian" in b_name or "human" in b_name:
                col = c_ped
                cat = "Pedestrians"
            elif "car" in b_name or "emergency" in b_name:
                col = c_car
                cat = "Cars"
            elif "truck" in b_name or "trailer" in b_name or "bus" in b_name or "construction" in b_name:
                col = c_truck
                cat = "Trucks/Buses"
            else:
                continue

            # I 4 angoli corretti del rettangolo BEV sul piano XY (front-left, front-right, rear-right, rear-left)
            corners_bev = box.corners_3d[:2, [0, 1, 5, 4]].T
            center_xy = np.mean(corners_bev, axis=0)
            dist = np.linalg.norm(center_xy)

            # Filtro rigoroso: visualizza e conta solo gli ostacoli entro il raggio di analisi
            if dist <= self.max_range:
                counts[cat] += 1

                # Calcolo orientamento di marcia
                f_mid = (corners_bev[0] + corners_bev[1]) / 2.0
                r_mid = (corners_bev[2] + corners_bev[3]) / 2.0
                u_vec = f_mid - r_mid
                length = max(1.2, np.linalg.norm(u_vec))
                width = max(0.8, np.linalg.norm(corners_bev[1] - corners_bev[0]))
                deg = -np.degrees(np.arctan2(u_vec[0], u_vec[1]))

                if cat == "Pedestrians":
                    # Cartello stradale pedone (sempre eretto a testa in su, orientato verso il senso di marcia)
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'pedestrian_green', center=center_xy,
                                          length=2.8, deg=0.0, flip_h=flip,
                                          halo_color='#DCFCE7', halo_radius=1.5)
                elif cat == "Cars":
                    # Sagoma line-art auto dell'utente orientata lungo il vettore di marcia
                    c_l = min(4.8, max(3.8, length))
                    self.draw_icon_object(ax_map, 'car_blue', center=center_xy,
                                          length=c_l, deg=deg)
                elif cat == "Trucks/Buses":
                    # Camion / Mezzo da Cantiere / Rimorchio / Bus con misura massima vincolata
                    if "construction" in b_name:
                        flip = (u_vec[0] < 0)
                        # EXCAVATOR: misura massima rigorosa (larghezza max 4.5m, proporzioni naturali h/w=0.642)
                        max_w = 4.5
                        exc_w = min(max_w, max(3.2, length * 0.65))
                        exc_l = exc_w * (345.0 / 537.0)
                        self.draw_icon_object(ax_map, 'construction_orange', center=center_xy,
                                              length=exc_l, width=exc_w, deg=0.0, flip_h=flip)
                    elif "trailer" in b_name:
                        t_l = min(13.5, max(4.5, length))
                        t_w = min(2.6, max(1.8, width))
                        t_center = f_mid - (u_vec / length) * (t_l / 2.0)
                        self.draw_icon_object(ax_map, 'trailer_purple', center=t_center,
                                              length=t_l, width=t_w, deg=deg)
                    elif "truck" in b_name:
                        # Se il camion traina un semirimorchio (TIR), disegna la motrice frontale 'truck_cab_purple'
                        if box.token in coupled_truck_tokens:
                            cab_len = min(4.0, length * 0.65)
                            cab_center = f_mid - (u_vec / length) * (cab_len / 2.0 + 0.35)
                            self.draw_icon_object(ax_map, 'truck_cab_purple', center=cab_center,
                                                  length=cab_len, width=width * 1.05, deg=deg)
                        else:
                            # Altrimenti è un camion/camioncino autonomo: disegna il camion completo 'truck_purple'
                            if length <= 6.8:
                                tr_len = min(5.2, max(4.0, length))
                                tr_w = min(2.1, max(1.7, width))
                            else:
                                tr_len = min(7.8, max(5.5, length))
                                tr_w = min(2.5, max(2.0, width))
                            self.draw_icon_object(ax_map, 'truck_purple', center=center_xy,
                                                  length=tr_len, width=tr_w, deg=deg)
                    else:
                        eff_len = min(7.8, max(5.5, length))
                        eff_w = min(2.5, max(2.0, width))
                        self.draw_icon_object(ax_map, 'truck_purple', center=center_xy,
                                              length=eff_len, width=eff_w, deg=deg)
                elif cat == "Bicycles":
                    # Cartello stradale bicicletta
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'bicycle_orange', center=center_xy,
                                          length=2.6, deg=0.0, flip_h=flip,
                                          halo_color='#FED7AA', halo_radius=1.5)
                elif cat == "Motorcycles":
                    # Cartello stradale motociclista
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'motorcycle_amber', center=center_xy,
                                          length=2.6, deg=0.0, flip_h=flip,
                                          halo_color='#FDE68A', halo_radius=1.5)
                elif cat == "Barriers":
                    # Transenna da cantiere zebrata giallo/nera
                    self.draw_icon_object(ax_map, 'barrier_hazard', center=center_xy,
                                          length=2.2, deg=0.0)
                elif cat == "Carts":
                    # Carrello / scala con ruote da cantiere (movable_object.pushable_pullable)
                    self.draw_icon_object(ax_map, 'cart_trolley', center=center_xy,
                                          length=2.2, deg=0.0,
                                          halo_color='#E0F2FE', halo_radius=1.3)
                elif cat == "Cones":
                    # Cono stradale da cantiere (movable_object.trafficcone)
                    self.draw_icon_object(ax_map, 'traffic_cone', center=center_xy,
                                          length=1.5, deg=0.0)

        # H. EGO VEHICLE AL CENTRO
        self.draw_icon_object(ax_map, 'car_ego', center=[0.0, 0.0],
                              length=5.0, deg=0.0)

        # I. Rimozione overlay: la mappa BEV ora è totalmente libera da sovrapposizioni


        ax_map.set_xlim(-self.max_range, self.max_range)
        ax_map.set_ylim(-self.max_range, self.max_range)
        ax_map.set_aspect('equal')
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        # Tooltip interattivo per mostrare dettagli zona occlusa al passaggio del mouse
        self.tooltip = ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#38BDF8", linewidth=1.5, alpha=0.94),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        # =====================================================================
        # 2. PANNELLO DESTRA ALTO: TELEMETRIA EGO VEHICLE & METADATI SCENA
        # =====================================================================
        ax_telemetry.set_facecolor('#F8FAFC')
        for spine in ax_telemetry.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)

        ax_telemetry.text(0.04, 0.88, "EGO VEHICLE TELEMETRY & NUSCENES METADATA",
                          transform=ax_telemetry.transAxes, fontsize=9.5, fontweight='bold', color=c_border)

        # Box velocità evidenziata (allargato con spaziatura confortevole)
        ax_telemetry.add_patch(Rectangle((0.03, 0.10), 0.28, 0.68, transform=ax_telemetry.transAxes,
                                         facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2))
        ax_telemetry.text(0.17, 0.60, "VELOCITÀ EGO", transform=ax_telemetry.transAxes,
                          fontsize=8.5, fontweight='bold', color='#1D4ED8', ha='center', va='center')
        ax_telemetry.text(0.14, 0.35, f"{self.ego_speed_kmh:.1f}", transform=ax_telemetry.transAxes,
                          fontsize=20, fontweight='bold', color='#0F172A', ha='center', va='center')
        ax_telemetry.text(0.24, 0.30, "km/h", transform=ax_telemetry.transAxes,
                          fontsize=8.5, color='#64748B', va='bottom')

        # Dati descrittivi della scena
        meta_lines = [
            f"Dataset: nuScenes (v1.0-mini/trainval)",
            f"Scena: {self.scene_name}  ({self.location})",
            f"Campione: {self.current_idx + 1} / {self.total_frames}",
            f"Punti LiDAR (entro {int(self.max_range)}m): {len(pts_dists):,} pts ({len(pts):,} totali)",
            f"Ostacoli 3D (entro {int(self.max_range)}m): {sum(counts.values())} oggetti"
        ]
        y_m = 0.70
        for line in meta_lines:
            ax_telemetry.text(0.35, y_m, line, transform=ax_telemetry.transAxes,
                              fontsize=8.5, color='#334155', va='center')
            y_m -= 0.14

        ax_telemetry.set_xlim(0, 1)
        ax_telemetry.set_ylim(0, 1)
        ax_telemetry.set_xticks([])
        ax_telemetry.set_yticks([])

        # =====================================================================
        # 3. PANNELLO DESTRA CENTRO: CONTEGGIO OSTACOLI REALI NEL RAGGIO VISIVO
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
        max_c = max(max(cat_counts, default=1), 5)
        ax_objects.set_ylim(0, max_c + 2)
        ax_objects.set_ylabel(f"Numero Oggetti (entro {int(self.max_range)}m)", fontsize=9, fontweight='bold', color=c_border)
        ax_objects.set_title(f"Distribuzione Ostacoli Reali nel Raggio di Percezione ({int(self.max_range)}m)",
                             fontsize=10.5, fontweight='bold', pad=8, color=c_border)
        ax_objects.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_objects.tick_params(colors=c_border, labelsize=8.5)

        for bar, val in zip(bars, cat_counts):
            ax_objects.text(bar.get_x() + bar.get_width()/2.0, val + 0.3, str(val),
                            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_border)

        # =====================================================================
        # 4. PANNELLO DESTRA BASSO: LEGENDA UFFICIALE & SIMBOLOGIA ICONE
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
                            fontsize=10.5, fontweight='bold', pad=9, color=c_border)

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
                ('zebra', '#555555', 'Strisce Pedonali', 1.0),
                ('circle', '#0F172A', 'LiDAR Point Cloud', 1.0),
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
                elif it[0] == 'circle':
                    c = Circle((x_ic, cur_y), 0.010, transform=ax_legend.transAxes,
                               facecolor=it[1], edgecolor='none', zorder=5)
                    ax_legend.add_patch(c)
                label = it[3] if it[0] == 'dashed_rect' else it[2]
                ax_legend.text(x_tx, cur_y, label, transform=ax_legend.transAxes,
                               fontsize=7.8, color=c_border, va='center', fontweight='medium')

        ax_legend.text(0.50, 0.04, "* Coni e carrelli/scale non sono modellati come volumi occludenti (non generano coni d'ombra).",
                       transform=ax_legend.transAxes, fontsize=6.8, color='#64748B', ha='center', va='center', style='italic')

        # Casella interattiva per saltare direttamente a un frame
        ax_tb = self.fig.add_axes([0.915, 0.938, 0.055, 0.036])
        self.txt_frame = TextBox(ax_tb, 'Vai al Frame: ', initial=str(self.current_idx + 1),
                                 color='#F8FAFC', hovercolor='#E2E8F0')
        self.txt_frame.label.set_fontsize(8.5)
        self.txt_frame.label.set_fontweight('bold')
        self.txt_frame.label.set_color(c_border)
        self.txt_frame.text_disp.set_color('#1D4ED8')
        self.txt_frame.text_disp.set_fontweight('bold')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ax_tb = ax_tb

        # Dicitura in calce per i controlli interattivi (centrata)
        self.fig.text(0.50, 0.012,
                      "[<- / ->]: Frame  |  [Hover Mouse]: Dettagli Zona Occlusa  |  [S]: Salva HD 300 DPI",
                      fontsize=8.0, color='#64748B', ha='center', style='italic')

        self.fig.canvas.draw_idle()

    def get_occlusion_semantics(self, occ, poly_xy):
        """Calcola la reale composizione semantica (Carreggiata, Marciapiedi, Crossing, ecc.)
        intersecando direttamente il poligono dell'occlusione con la matrice HD-Map autentica nuScenes."""
        smap = self.frame_data.get('semantic_map', {})
        if not smap or len(poly_xy) < 3:
            return "N/A"

        import cv2
        drivable = np.rot90(smap.get('drivable_area', np.zeros((200, 200))), 3)
        walkway = np.rot90(smap.get('walkway', np.zeros((200, 200))), 3)
        crossing = np.rot90(smap.get('ped_crossing', np.zeros((200, 200))), 3)
        carpark = np.rot90(smap.get('carpark_area', np.zeros((200, 200))), 3)

        # Mappatura dei vertici metrici BEV [-40, 40] in coordinate raster [0, 199]
        pts_grid = ((np.array(poly_xy) + 40.0) / 80.0 * 200.0).astype(np.int32)
        poly_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts_grid], 1)

        total_px = int(np.sum(poly_mask == 1))
        if total_px == 0:
            return "Area Libera"

        is_poly = (poly_mask == 1)
        # Classificazione gerarchica disgiunta per evitare sovrapposizioni e garantire somma 100%
        m_cross = is_poly & (crossing == 1)
        m_walk = is_poly & (walkway == 1) & ~m_cross
        m_road = is_poly & (drivable == 1) & ~m_cross & ~m_walk
        m_park = is_poly & (carpark == 1) & ~m_cross & ~m_walk & ~m_road
        m_terr = is_poly & ~m_cross & ~m_walk & ~m_road & ~m_park

        n_cross = int(np.sum(m_cross))
        n_walk = int(np.sum(m_walk))
        n_road = int(np.sum(m_road))
        n_park = int(np.sum(m_park))
        n_terr = int(np.sum(m_terr))

        p_cross = n_cross / total_px
        p_walk = n_walk / total_px
        p_road = n_road / total_px
        p_park = n_park / total_px
        p_terr = n_terr / total_px

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

        return ", ".join(parts) if parts else "Terreno/Verde (100%)"

    def on_mouse_move(self, event):
        """Gestisce l'hover del mouse evidenziando la zona occlusa e mostrando il tooltip analitico."""
        if not hasattr(self, 'ax_map') or event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            if self.active_hovered is not None:
                patch = self.active_hovered['patch']
                patch.set_facecolor(self.active_hovered['default_fc'])
                patch.set_edgecolor(self.active_hovered['default_ec'])
                patch.set_linewidth(self.active_hovered['default_lw'])
                patch.set_alpha(self.active_hovered['default_alpha'])
                patch.set_zorder(self.active_hovered['default_z'])
                self.active_hovered = None
                if hasattr(self, 'tooltip') and self.tooltip is not None:
                    self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        found = None
        # Cerca il poligono occluso sotto il puntatore (in ordine inverso per selezionare il più in primo piano)
        for item in reversed(self.drawn_occlusions):
            if item['path'].contains_point((x, y)):
                found = item
                break

        if found is not None:
            if self.active_hovered != found:
                # Ripristina stile della zona precedentemente evidenziata
                if self.active_hovered is not None:
                    p_old = self.active_hovered['patch']
                    p_old.set_facecolor(self.active_hovered['default_fc'])
                    p_old.set_edgecolor(self.active_hovered['default_ec'])
                    p_old.set_linewidth(self.active_hovered['default_lw'])
                    p_old.set_alpha(self.active_hovered['default_alpha'])
                    p_old.set_zorder(self.active_hovered['default_z'])

                self.active_hovered = found
                # Evidenzia la nuova zona occlusa con blu/cyan elettrico brillante
                p_new = found['patch']
                p_new.set_facecolor('#0284C7')
                p_new.set_edgecolor('#38BDF8')
                p_new.set_linewidth(2.2)
                p_new.set_alpha(0.72)
                p_new.set_zorder(15)

            # Estrazione metadati analitici dell'occlusione
            occ = found['data']
            raw_obj = occ.get('object_name', 'Sconosciuto')
            obj_labels = {
                'vehicle.truck': 'Camion (vehicle.truck)',
                'vehicle.trailer': 'Rimorchio (vehicle.trailer)',
                'vehicle.construction': 'Mezzo Cantiere (vehicle.construction)',
                'vehicle.car': 'Automobile (vehicle.car)',
                'vehicle.bus': 'Autobus (vehicle.bus)',
                'vehicle.bicycle': 'Bicicletta (vehicle.bicycle)',
                'vehicle.motorcycle': 'Motocicletta (vehicle.motorcycle)',
                'static.manmade': 'Struttura Statica (Muro / Edificio)'
            }
            friendly_gen = obj_labels.get(raw_obj, raw_obj)
            if 'pedestrian' in raw_obj or 'human' in raw_obj:
                friendly_gen = f"Pedone VRU ({raw_obj.split('.')[-1].capitalize()})"

            area = occ.get('area_sqm', 0.0)
            bbox = occ.get('occlusion_bbox_m', [0, 0, 0, 0])
            dx = abs(bbox[2] - bbox[0])
            dy = abs(bbox[3] - bbox[1])
            dist = occ.get('distance_m', np.hypot(x, y))
            sem_str = self.get_occlusion_semantics(occ, found['poly_xy'])

            tooltip_text = (
                f"ZONA OCCLUSA (Raycasting)\n"
                f"• Chi la genera:  {friendly_gen}\n"
                f"• Dimensione:     Area {area:.1f} m²\n"
                f"• Larghezza:      {min(dx, dy):.1f} m (BBox: {dx:.1f}m × {dy:.1f}m)\n"
                f"• Semantica:      {sem_str}\n"
                f"• Distanza da Ego: {dist:.1f} m"
            )

            if hasattr(self, 'tooltip') and self.tooltip is not None:
                self.tooltip.xy = (x, y)
                # Offset dinamico per evitare che esca dai bordi dell'asse
                ox = -225 if x > 5.0 else 15
                oy = -105 if y > 10.0 else 15
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
                patch.set_zorder(self.active_hovered['default_z'])
                self.active_hovered = None
                if hasattr(self, 'tooltip') and self.tooltip is not None:
                    self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            target = (val - 1) if 1 <= val <= self.total_frames else val
            if 0 <= target < self.total_frames and target != self.current_idx:
                self.load_frame(target, broadcast=True)
        except ValueError:
            pass

    def on_key(self, event):
        # Ignora se l'utente sta digitando nella casella di testo
        if hasattr(self, 'ax_tb') and event.inaxes == self.ax_tb:
            return

        if event.key in ['right', 'd', ' ']:
            self.load_frame(self.current_idx + 1)
        elif event.key in ['left', 'a']:
            self.load_frame(self.current_idx - 1)
        elif event.key in ['s', 'S']:
            out_dir = os.path.join("documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "fig_2_raycasting_occlusions.png")
            # Nasconde temporaneamente la barra di ricerca e il tooltip per un'immagine paper pulitissima
            if hasattr(self, 'ax_tb') and self.ax_tb is not None:
                self.ax_tb.set_visible(False)
            if hasattr(self, 'tooltip') and self.tooltip is not None:
                self.tooltip.set_visible(False)
            self.fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
            if hasattr(self, 'ax_tb') and self.ax_tb is not None:
                self.ax_tb.set_visible(True)
                self.fig.canvas.draw_idle()
            print(f"\n[SALVATA CON SUCCESSO]: Immagine Zone Occluse Raycasting salvata a 300 DPI in:\n  -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore nuScenes Stile Paper Accademico")
    parser.add_argument("frame_pos", nargs="?", type=int, default=None, help="Numero del frame iniziale a cui saltare (1-404)")
    parser.add_argument("--save", action="store_true", help="Salva immediatamente a 300 DPI ed esce")
    parser.add_argument("--frame", type=int, default=None, help="Indice del frame iniziale")
    parser.add_argument("--range", "-r", type=float, default=25.0, help="Raggio di visualizzazione in metri (default: 25.0)")
    args = parser.parse_args()

    app = PaperStyleVisualizer(max_range=args.range)
    init_f = args.frame_pos if args.frame_pos is not None else args.frame
    if init_f is not None:
        target = (init_f - 1) if init_f >= 1 and args.frame_pos is not None else init_f
        app.load_frame(target)

    if args.save:
        out_dir = os.path.join("documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fig_2_raycasting_occlusions.png")
        app.fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"\n[SALVATA CON SUCCESSO]: Immagine Zone Occluse Raycasting salvata a 300 DPI in:\n  -> {out_path}")
    else:
        plt.show()
