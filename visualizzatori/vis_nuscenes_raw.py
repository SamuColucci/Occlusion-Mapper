# ==============================================================================
# VISUALIZZATORE UFFICIALE TESI - DATI GREZZI NUSCENES SENZA OCCLUSIONI (STILE PAPER)
# File: visualizzatori/vis_style1_raw_nuscenes.py
#
# Visualizzatore BEV per Dati Reali NuScenes PRIMA del calcolo delle occlusioni:
# - Mappa BEV ad altissima fedeltà entro 25m (quadrettatura, anelli metrici 10/20/25m);
# - Nuvola di punti LiDAR grezza (LIDAR_TOP nuScenes 32 raggi);
# - Ostacoli reali 3D annotati con sagome line-art e cartelli stradali ufficiali;
# - Mappa semantica HD-Map (carreggiata stradale, marciapiedi, attraversamenti);
# - NESSUNA ZONA OCCLUSA DISEGNATA (rappresentazione pura dello stato percettivo iniziale);
# - Telemetria ego vehicle, istogramma categorie e legenda completa;
# - Tasto [S] salva a 300 DPI in documentazione/immagini_tesi/fig_1_lidar_nuscenes_raw.png.
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox

# Impostazione percorso root del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_adapter.factory_dataset import create_adapter

class PaperRawVisualizer:
    def __init__(self):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: DATASET NUSCENES RAW (SENZA ZONE OCCLUSE)")
        print("=" * 80)

        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0
        self.max_range = 25.0

        # Setup tipografia accademica
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.load_icons()
        self.load_frame(self.current_idx)

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

    def load_frame(self, idx):
        self.current_idx = idx % self.total_frames
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

        self.render()

    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                         flip_h=False, halo_color=None, halo_radius=None):
        """Disegna un'icona raster/vettoriale trasparente ruotata e scalata preservando le proporzioni naturali."""
        from PIL import Image

        if halo_color and halo_radius:
            ax.add_patch(Circle(center, halo_radius, facecolor=halo_color, edgecolor='none', alpha=0.55, zorder=6))

        if icon_name not in self.icons:
            return

        pil_img = self.icons[icon_name]
        if flip_h:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        if deg != 0.0:
            pil_img = pil_img.rotate(deg, expand=True, resample=Image.BICUBIC)

        img_w, img_h = pil_img.size
        if width is None:
            width = length / (img_h / img_w) if img_h > 0 else length

        extent = [
            center[0] - width / 2.0,
            center[0] + width / 2.0,
            center[1] - length / 2.0,
            center[1] + length / 2.0
        ]
        ax.imshow(pil_img, extent=extent, zorder=8, interpolation='lanczos')

    def render(self):
        self.fig.clf()

        gs = GridSpec(3, 2, figure=self.fig,
                      width_ratios=[1.10, 1.0],
                      height_ratios=[0.24, 0.38, 0.38],
                      wspace=0.14, hspace=0.35,
                      left=0.03, right=0.97, top=0.94, bottom=0.06)

        ax_map = self.fig.add_subplot(gs[:, 0])
        ax_telemetry = self.fig.add_subplot(gs[0, 1])
        ax_objects = self.fig.add_subplot(gs[1, 1])
        ax_legend = self.fig.add_subplot(gs[2, 1])

        # Palette accademica
        c_bg = '#F8FAFC'
        c_grid = '#E2E8F0'
        c_border = '#0F172A'

        c_car = '#1D4ED8'
        c_ped = '#16A34A'
        c_truck = '#9333EA'
        c_bike = '#EA580C'
        c_moto = '#D97706'
        c_barrier = '#F59E0B'

        # =====================================================================
        # 1. PANNELLO SINISTRA: MAPPA BEV AD ALTA FEDELTÀ (RAGGIO 25M)
        # =====================================================================
        ax_map.set_facecolor(c_bg)

        # A. Quadrettatura metrica 2m x 2m (estesa su raggio 25m)
        for gx in np.arange(-26.0, 26.1, 2.0):
            ax_map.axvline(gx, color=c_grid, linewidth=0.55, zorder=1)
        for gy in np.arange(-26.0, 26.1, 2.0):
            ax_map.axhline(gy, color=c_grid, linewidth=0.55, zorder=1)

        # B. Assi ortogonali a croce passanti per l'ego-vehicle
        ax_map.axvline(0, color=c_border, linewidth=1.1, zorder=2)
        ax_map.axhline(0, color=c_border, linewidth=1.1, zorder=2)

        # C. Cerchi metrici concentrici (10m, 20m, 25m) con etichette standard
        for r in [10.0, 20.0, 25.0]:
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

            # 1. Marciapiedi ed aree pedonali in grigio chiaro (#DAE0E9)
            map_rgba[walkway == 1] = [218, 224, 233, 245]
            # 2. Carreggiata stradale in grigio scuro ufficiale nuScenes
            map_rgba[drivable == 1] = [125, 125, 125, 255]
            # 3. Attraversamenti pedonali in bianco
            map_rgba[crossing == 1] = [255, 255, 255, 255]

            ax_map.imshow(map_rgba, extent=[-40.0, 40.0, -40.0, 40.0], origin='lower', zorder=2)

        # E. Punti LiDAR REALI nuScenes (filtrati entro il raggio di 25 metri)
        pts = self.frame_data.get('points', None)
        pts_dists = []
        if pts is not None and len(pts) > 0:
            all_dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            mask = (all_dists <= 25.0)
            pts_bev = pts[mask]
            pts_dists = all_dists[mask]

            # Subcampionamento per nitidezza grafica
            if len(pts_bev) > 9000:
                idx_sub = np.random.choice(len(pts_bev), 9000, replace=False)
                pts_sub = pts_bev[idx_sub]
            else:
                pts_sub = pts_bev

            ax_map.scatter(pts_sub[:, 0], pts_sub[:, 1], s=1.1, c='#0F172A', alpha=0.60, zorder=4)

        # (F. NESSUNA OCCLUSIONE DISEGNATA IN QUESTA VERSIONE RAW)

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
            elif "truck" in b_name or "bus" in b_name or "trailer" in b_name or "construction" in b_name:
                col = c_truck
                cat = "Trucks/Buses"
            else:
                continue

            # I 4 angoli del rettangolo BEV sul piano XY
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
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'pedestrian_green', center=center_xy,
                                          length=2.8, deg=0.0, flip_h=flip,
                                          halo_color='#DCFCE7', halo_radius=1.5)
                elif cat == "Cars":
                    c_l = min(4.8, max(3.8, length))
                    self.draw_icon_object(ax_map, 'car_blue', center=center_xy,
                                          length=c_l, deg=deg)
                elif cat == "Trucks/Buses":
                    # Camion / Mezzo da Cantiere / Rimorchio / Bus con misura massima vincolata
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
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'bicycle_orange', center=center_xy,
                                          length=2.2, deg=0.0, flip_h=flip,
                                          halo_color='#FFEDD5', halo_radius=1.3)
                elif cat == "Motorcycles":
                    flip = (u_vec[0] < 0)
                    self.draw_icon_object(ax_map, 'motorcycle_amber', center=center_xy,
                                          length=2.4, deg=0.0, flip_h=flip,
                                          halo_color='#FEF3C7', halo_radius=1.3)
                elif cat == "Barriers":
                    self.draw_icon_object(ax_map, 'barrier_hazard', center=center_xy,
                                          length=max(2.5, length), deg=0.0)
                elif cat == "Carts":
                    self.draw_icon_object(ax_map, 'cart_trolley', center=center_xy,
                                          length=2.0, deg=0.0)
                elif cat == "Cones":
                    self.draw_icon_object(ax_map, 'traffic_cone', center=center_xy,
                                          length=1.4, deg=0.0)

        # H. EGO-VEHICLE (Posizionato all'origine [0,0] con sagoma line-art)
        self.draw_icon_object(ax_map, 'car_ego', center=[0.0, 0.0], length=4.8, deg=0.0)

        ax_map.set_xlim(-25.5, 25.5)
        ax_map.set_ylim(-25.5, 25.5)
        ax_map.set_aspect('equal')
        ax_map.set_title("Figure 1: nuScenes LiDAR Point Cloud & 3D Bounding Boxes (BEV Raggio 25m)",
                         fontsize=11.5, fontweight='bold', pad=10)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        # =====================================================================
        # 2. PANNELLO DESTRA ALTO: TELEMETRIA EGO VEHICLE & METADATI SCENA
        # =====================================================================
        ax_telemetry.set_facecolor('#F8FAFC')
        for spine in ax_telemetry.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)

        ax_telemetry.text(0.04, 0.86, "EGO VEHICLE TELEMETRY & NUSCENES METADATA",
                          transform=ax_telemetry.transAxes, fontsize=10, fontweight='bold', color=c_border)

        # Box velocità evidenziata (allargato con spaziatura confortevole)
        ax_telemetry.add_patch(Rectangle((0.03, 0.14), 0.30, 0.62, transform=ax_telemetry.transAxes,
                                         facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2))
        ax_telemetry.text(0.18, 0.56, "VELOCITÀ EGO", transform=ax_telemetry.transAxes,
                          fontsize=8.5, fontweight='bold', color='#1D4ED8', ha='center', va='center')
        ax_telemetry.text(0.15, 0.33, f"{self.ego_speed_kmh:.1f}", transform=ax_telemetry.transAxes,
                          fontsize=20, fontweight='bold', color='#0F172A', ha='center', va='center')
        ax_telemetry.text(0.26, 0.28, "km/h", transform=ax_telemetry.transAxes,
                          fontsize=8.5, color='#64748B', va='bottom')

        # Dati descrittivi della scena
        meta_lines = [
            f"Dataset: nuScenes (v1.0-mini/trainval)",
            f"Scena: {self.scene_name}  ({self.location})",
            f"Campione: {self.current_idx + 1} / {self.total_frames}",
            f"Punti LiDAR (entro 25m): {len(pts_dists):,} pts ({len(pts):,} totali)",
            f"Ostacoli 3D (entro 25m): {sum(counts.values())} oggetti"
        ]
        y_m = 0.65
        for line in meta_lines:
            ax_telemetry.text(0.36, y_m, line, transform=ax_telemetry.transAxes,
                              fontsize=8.5, color='#334155', va='center')
            y_m -= 0.13

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
        ax_objects.set_ylabel("Numero Oggetti (entro 25m)", fontsize=9, fontweight='bold', color=c_border)
        ax_objects.set_title("Distribuzione Ostacoli Reali nel Raggio di Percezione (25m)",
                             fontsize=10.5, fontweight='bold', pad=8, color=c_border)
        ax_objects.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_objects.tick_params(colors=c_border, labelsize=8.5)

        for bar, val in zip(bars, cat_counts):
            ax_objects.text(bar.get_x() + bar.get_width()/2.0, val + 0.3, str(val),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_border)

        # =====================================================================
        # 4. PANNELLO DESTRA IN BASSO: SIMBOLOGIA & LEGENDA UFFICIALE
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
                ('rect', '#7D7D7D', 'Carreggiata (HD-Map)', 1.0),
                ('rect', '#DAE0E9', 'Marciapiede (Walkway)', 1.0),
                ('circle', '#0F172A', 'LiDAR Point Cloud', 1.0),
            ]),
        ]

        for x_ic, x_tx, items in columns:
            start_y = 0.82
            step_y = 0.16
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
                elif it[0] == 'circle':
                    c = Circle((x_ic, cur_y), 0.010, transform=ax_legend.transAxes,
                               facecolor=it[1], edgecolor='none', zorder=5)
                    ax_legend.add_patch(c)
                ax_legend.text(x_tx, cur_y, it[2], transform=ax_legend.transAxes,
                               fontsize=7.8, color=c_border, va='center', fontweight='medium')

        ax_legend.text(0.50, 0.04, "* Rappresentazione grezza sensori LiDAR e annotazioni 3D ground truth nuScenes nel raggio di 25m.",
                       transform=ax_legend.transAxes, fontsize=6.8, color='#64748B', ha='center', va='center', style='italic')

        # Casella interattiva per saltare direttamente a un frame (in alto a destra sopra la telemetria)
        ax_tb = self.fig.add_axes([0.89, 0.942, 0.075, 0.034])
        self.txt_frame = TextBox(ax_tb, 'Vai al Frame: ', initial=str(self.current_idx + 1),
                                 color='#F8FAFC', hovercolor='#E2E8F0')
        self.txt_frame.label.set_fontsize(9.0)
        self.txt_frame.label.set_color('#1E293B')
        self.txt_frame.label.set_fontweight('bold')
        self.txt_frame.text_disp.set_color('#0F172A')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ax_tb = ax_tb

        # Dicitura in calce per i controlli interattivi
        self.fig.text(0.97, 0.015, f"[<- / ->]: Frame  |  [S]: Salva HD 300 DPI",
                      color='#64748B', fontsize=8.5, ha='right', style='italic')

        self.fig.canvas.draw_idle()

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            target = (val - 1) if 1 <= val <= self.total_frames else val
            if 0 <= target < self.total_frames and target != self.current_idx:
                self.load_frame(target)
        except ValueError:
            pass

    def on_key(self, event):
        if hasattr(self, 'ax_tb') and event.inaxes == self.ax_tb:
            return

        if event.key in ['right', 'd', ' ']:
            self.load_frame(self.current_idx + 1)
        elif event.key in ['left', 'a']:
            self.load_frame(self.current_idx - 1)
        elif event.key in ['s', 'S']:
            out_dir = os.path.join("documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "fig_1_lidar_nuscenes_raw.png")
            if hasattr(self, 'ax_tb') and self.ax_tb is not None:
                self.ax_tb.set_visible(False)
            self.fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
            if hasattr(self, 'ax_tb') and self.ax_tb is not None:
                self.ax_tb.set_visible(True)
                self.fig.canvas.draw_idle()
            print(f"\n[SALVATA CON SUCCESSO]: Immagine nuScenes Raw salvata a 300 DPI in:\n  -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizzatore nuScenes Raw Stile Paper (Senza Occlusioni)")
    parser.add_argument("frame_pos", nargs="?", type=int, default=None, help="Numero del frame iniziale a cui saltare (1-404)")
    parser.add_argument("--save", action="store_true", help="Salva immediatamente a 300 DPI ed esce")
    parser.add_argument("--frame", type=int, default=None, help="Indice del frame iniziale")
    args = parser.parse_args()

    app = PaperRawVisualizer()
    init_f = args.frame_pos if args.frame_pos is not None else args.frame
    if init_f is not None:
        target = (init_f - 1) if init_f >= 1 and args.frame_pos is not None else init_f
        app.load_frame(target)

    if args.save:
        out_dir = os.path.join("documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fig_1_lidar_nuscenes_raw.png")
        app.fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"\n[SALVATA CON SUCCESSO]: Immagine nuScenes Raw salvata a 300 DPI in:\n  -> {out_path}")
    else:
        plt.show()
