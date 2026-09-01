# Visualizzatore Comparativo Ground Truth per Zone d'Ombra (verify_runtime_ground_truth.py)
# Confronta per CIASCUNA ZONA D'OMBRA:
#   - Pannello Sinistro: Ground Truth Reale nuScenes (mostra solo gli ostacoli reali presenti DENTRO l'ombra)
#   - Pannello Destro: Ground Truth Neurosimbolica/Sintetica (mostra cosa assegna la logica stradale/varco all'ombra)
#   - HUD a Sinistra: Dettaglio 1-a-1 tra GT Reale vs GT Neurosimbolica con spiegazione delle regole fisiche

import sys
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from pyquaternion import Quaternion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, get_occlusion_ground_truth_target, rasterize_polygon
from ground_truth.ground_truth_extractor_synthetic import generate_synthetic_injected_gt, OBSTACLE_SPECS_M

GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4

CLASS_NAMES = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bicicletta", "Barriera"]
CLASS_COLORS = {
    "Auto": "#3B82F6",       # Blu Elettrico
    "Camion/Bus": "#EC4899", # Rosa Magenta
    "Pedone": "#10B981",     # Verde Smeraldo
    "Moto": "#F59E0B",       # Arancio
    "Bicicletta": "#8B5CF6", # Viola
    "Barriera": "#94A3B8"    # Grigio
}

class GroundTruthZoneVisualizer:
    def __init__(self, dataroot="./nuscenes"):
        print("\n" + "=" * 70)
        print("   CONFRONTO GROUND TRUTH REALE VS NEUROSIMBOLICA PER ZONA D'OMBRA")
        print("=" * 70)
        self.adapter = create_adapter("nuscenes", dataroot)
        self.nusc = self.adapter.nusc
        self.total_samples = self.adapter.get_num_samples()
        self.current_idx = 0
        self.current_occ_idx = 0
        self.max_distance = 25.0
        self.distance_options = [25.0, 20.0, 15.0, 30.0]

        self.fig, (self.ax_hud, self.ax_real, self.ax_syn) = plt.subplots(
            1, 3, figsize=(20, 9), facecolor='#0B0F19',
            gridspec_kw={'width_ratios': [1.1, 2.0, 2.0]}
        )
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.06)
        self.ax_hud.set_facecolor('#0F172A')
        self.ax_real.set_facecolor('#0B0F19')
        self.ax_syn.set_facecolor('#0B0F19')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        print("\nCONTROLLI: Frecce=Naviga, R=Raggio (25m/20m), Mouse=Ispeziona Ombra, Q/ESC=Esci\n")
        self.load_current_data()
        self.plot_current()
        plt.show()

    def load_current_data(self):
        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.sample_token = self.frame_data['sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)

        self.lidar_token = self.sample['data']['LIDAR_TOP']
        sd_record = self.nusc.get('sample_data', self.lidar_token)
        self.cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        self.pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        json_path = os.path.join("extracted_occlusions", f"{self.sample_token}.json")
        self.occlusions = []

        # Estrae maschere Ground Truth Reale 3D
        gt_real_masks = extract_ground_truth_masks(self.frame_data)
        self.gt_real_tensor = gt_real_masks

        # Informazioni semantiche di mappa
        semantic_map = self.frame_data.get("semantic_map", {})
        drivable_mask = semantic_map.get('drivable_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        walkway_mask = semantic_map.get('walkway', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
        ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                raw_data = json.load(f)
                for occ in raw_data.get("occlusions", []):
                    pts = occ.get("polygon_points_m", [])
                    if len(pts) >= 3:
                        pts_arr = np.array(pts)
                        poly_xy = np.column_stack([pts_arr[:, 1], pts_arr[:, 0]])
                        
                        area_sqm = float(occ.get("area_sqm", 0.0))
                        distance_m = float(occ.get("distance_m", 0.0))

                        # Calcolo Geometrico Rigoroso della Strettoia Fisica tramite Rettangolo Minimo Orientato (OBB)
                        poly_shapely = ShapelyPolygon(poly_xy)
                        if poly_shapely.is_valid and poly_shapely.area > 0.05:
                            min_rect = poly_shapely.minimum_rotated_rectangle
                            rect_coords = np.array(min_rect.exterior.coords)
                            side_a = float(np.linalg.norm(rect_coords[0] - rect_coords[1]))
                            side_b = float(np.linalg.norm(rect_coords[1] - rect_coords[2]))
                            
                            # La larghezza effettiva è la dimensione MINORE dell'OBB (la sezione trasversale più stretta)
                            obb_width = min(side_a, side_b)
                            obb_length = max(side_a, side_b)

                            # Calcolo dell'ampiezza angolare al punto di ingresso (distanza minima dall'auto)
                            dists = np.hypot(poly_xy[:, 0], poly_xy[:, 1])
                            angles = np.arctan2(poly_xy[:, 1], poly_xy[:, 0])
                            # Gestione corretta dell'apertura angolare (wrapping)
                            angle_diff = np.max(angles) - np.min(angles)
                            if angle_diff > np.pi:
                                angle_diff = 2 * np.pi - angle_diff
                            d_near = max(1.0, float(np.min(dists)))
                            angular_width = float(d_near * angle_diff)

                            # La vera larghezza è il collo di bottiglia reale
                            occ_w = min(obb_width, max(angular_width, obb_width * 0.7))
                            occ_depth = obb_length
                        else:
                            occ_w = 0.2
                            occ_depth = 0.5

                        # Calcolo rasterizzato pixel-perfect delle superfici della Mappa HD
                        occ_mask = rasterize_polygon(pts)
                        total_px = np.sum(occ_mask)

                        if total_px > 0:
                            carpark_mask = semantic_map.get('carpark_area', np.zeros((GRID_DIM, GRID_DIM), dtype=np.float32))
                            road_px = np.sum(occ_mask * drivable_mask) + np.sum(occ_mask * carpark_mask)
                            side_px = np.sum(occ_mask * walkway_mask)
                            cross_px = np.sum(occ_mask * ped_crossing_mask)
                            
                            road_f = float(road_px / total_px)
                            side_f = float(side_px / total_px)
                            cross_f = float(cross_px / total_px)
                            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
                        else:
                            road_f, side_f, cross_f, terr_f = 0.0, 0.0, 0.0, 1.0

                        # 1. GROUND TRUTH REALE NELL'OMBRA: cerca ostacoli 3D reali dentro questo poligono
                        gt_target = get_occlusion_ground_truth_target(self.gt_real_tensor, pts)
                        has_real_obj = np.sum(gt_target) > 0
                        real_classes = [CLASS_NAMES[i] for i, v in enumerate(gt_target) if v > 0.5]

                        # 2. GROUND TRUTH NEUROSIMBOLICA CON VINCOLI FISICI RIGOROSI (Dimensioni Bounding Box 3D Reali)
                        # - Auto: Larghezza OBB >= 1.8m, Lunghezza OBB >= 3.8m, Area >= 8.0m², Asfalto >= 25%
                        # - Camion/Bus: Larghezza OBB >= 2.4m, Lunghezza OBB >= 6.5m, Area >= 18.0m², Asfalto >= 30%
                        # - Moto: 0.9m <= Larghezza < 1.8m, Lunghezza >= 1.8m, Area >= 2.5m², Asfalto >= 25%
                        # - Pedone: Larghezza >= 0.5m, Area >= 0.8m², Marciapiede >= 15% o Strisce >= 8%
                        # - Bicicletta: Larghezza >= 0.8m, Lunghezza >= 1.4m, Area >= 1.8m², Marciapiede/Strisce >= 15%
                        # - Barriera: Terreno >= 55%, Area >= 3.0m² e non carrabile
                        neuro_classes = []
                        reasons = []

                        # Preserva ostacoli reali se già presenti nel dataset originale
                        if has_real_obj:
                            neuro_classes.extend(real_classes)
                            reasons.append(f"Reale: {', '.join(real_classes)}")

                        # Regola Auto: richiede sagoma 1.8m x 3.8m su asfalto
                        if road_f >= 0.25 and occ_w >= 1.8 and occ_depth >= 3.8 and area_sqm >= 8.0:
                            if "Auto" not in neuro_classes:
                                neuro_classes.append("Auto")
                                reasons.append(f"Auto (Strada {int(road_f*100)}%, Sezione={occ_w:.1f}x{occ_depth:.1f}m)")
                            
                            # Regola Camion/Bus: sagoma 2.4m x 6.5m
                            if road_f >= 0.30 and occ_w >= 2.4 and occ_depth >= 6.5 and area_sqm >= 18.0 and "Camion/Bus" not in neuro_classes:
                                neuro_classes.append("Camion/Bus")

                        # Regola Moto su Strada: richiede strada carrabile dominante (>= 50%) e no terreno/marciapiede
                        if road_f >= 0.50 and terr_f < 0.25 and side_f < 0.20 and (0.9 <= occ_w < 1.8) and occ_depth >= 1.8 and area_sqm >= 2.5:
                            if "Moto" not in neuro_classes and "Auto" not in neuro_classes:
                                neuro_classes.append("Moto")
                                reasons.append(f"Moto (Corsia Strada={int(road_f*100)}%, Varco={occ_w:.1f}m)")

                        # Regola Marciapiede / Strisce Pedonali -> Pedone e Bici
                        if (side_f >= 0.15 or cross_f >= 0.08) and occ_w >= 0.5 and area_sqm >= 0.8:
                            if "Pedone" not in neuro_classes:
                                neuro_classes.append("Pedone")
                                surf_str = "Marciapiede" if side_f >= 0.15 else "Strisce"
                                reasons.append(f"{surf_str} ({int(max(side_f, cross_f)*100)}%)")
                            if occ_w >= 0.8 and occ_depth >= 1.4 and area_sqm >= 1.8 and "Bicicletta" not in neuro_classes:
                                neuro_classes.append("Bicicletta")

                        # Regola Terreno / Bordo -> Barriera (solo se terreno dominante >= 55% e NON carrabile/pedonale)
                        if terr_f >= 0.55 and occ_w >= 1.0 and area_sqm >= 3.0 and road_f < 0.15 and side_f < 0.10:
                            if "Barriera" not in neuro_classes:
                                neuro_classes.append("Barriera")
                                reasons.append(f"Bordo/Terreno ({int(terr_f*100)}%)")

                        neuro_assigned_class = neuro_classes[0] if len(neuro_classes) > 0 else None
                        neuro_reason = " + ".join(reasons) if reasons else "Spazio Insufficiente / Ombra Vuota"

                        self.occlusions.append({
                            "polygon_pts": pts,
                            "polygon_xy": poly_xy,
                            "area_sqm": area_sqm,
                            "distance_m": distance_m,
                            "width_m": occ_w,
                            "depth_m": occ_depth,
                            "road_f": road_f,
                            "side_f": side_f,
                            "cross_f": cross_f,
                            "terr_f": terr_f,
                            "gt_real_target": gt_target,
                            "has_real_obj": has_real_obj,
                            "real_classes": real_classes,
                            "neuro_classes": neuro_classes,
                            "neuro_assigned_class": neuro_assigned_class,
                            "neuro_reason": neuro_reason
                        })

        self.current_occ_idx = 0

    def on_key(self, event):
        if event.key in ['right', 'up', 'pageup']:
            self.current_idx = (self.current_idx + 1) % self.total_samples
            self.load_current_data()
            self.plot_current()
        elif event.key in ['left', 'down', 'pagedown']:
            self.current_idx = (self.current_idx - 1) % self.total_samples
            self.load_current_data()
            self.plot_current()
        elif event.key in ['r', 'R']:
            curr_idx = self.distance_options.index(self.max_distance) if self.max_distance in self.distance_options else 0
            self.max_distance = self.distance_options[(curr_idx + 1) % len(self.distance_options)]
            print(f"\n[RAGGIO DISTANZA]: {self.max_distance:.0f}m")
            self.plot_current()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

    def on_mouse_move(self, event):
        if (event.inaxes not in [self.ax_real, self.ax_syn]) or not self.occlusions:
            return
        x_m, y_m = event.xdata, event.ydata
        if x_m is None or y_m is None:
            return
        pt = ShapelyPoint(x_m, y_m)
        for idx, occ in enumerate(self.occlusions):
            if occ["distance_m"] > self.max_distance:
                continue
            poly_pts = occ["polygon_xy"]
            if len(poly_pts) >= 3:
                try:
                    poly = ShapelyPolygon(poly_pts)
                    if poly.contains(pt):
                        if self.current_occ_idx != idx:
                            self.current_occ_idx = idx
                            self.plot_current()
                        break
                except Exception:
                    pass

    def draw_base_map(self, ax, title):
        ax.clear()
        ax.set_facecolor('#0B0F19')
        ax.set_xlim([-32, 32])
        ax.set_ylim([-32, 32])
        ax.set_aspect('equal')
        ax.grid(True, color='#1E293B', linestyle='--', alpha=0.5)

        for r in [10, 20, 25, 30]:
            circle = plt.Circle((0, 0), r, color='#334155', fill=False, linestyle=':', alpha=0.4)
            ax.add_patch(circle)

        roi_circle = plt.Circle((0, 0), self.max_distance, color='#38BDF8', fill=False, linestyle='--', linewidth=2.0, alpha=0.85, zorder=8)
        ax.add_patch(roi_circle)
        ax.text(0, self.max_distance + 0.8, f"Raggio: {self.max_distance:.0f}m", color='#38BDF8', fontsize=8, fontweight='bold', ha='center', zorder=9)

        ego_rect = Rectangle((-1.0, -2.0), 2.0, 4.0, color='#38BDF8', zorder=15)
        ax.add_patch(ego_rect)
        ax.plot([0, 0], [0, 2.5], color='#F43F5E', linewidth=2, zorder=16)

        if self.lidar_token:
            try:
                self.nusc.render_sample_data(self.lidar_token, ax=ax, underlay_map=True, verbose=False)
            except Exception:
                pass

        ax.set_title(title, fontsize=10, fontweight='bold', color='#38BDF8', pad=8)

    def plot_current(self):
        self.ax_hud.clear()
        self.ax_hud.axis('off')

        # 1. Disegna Mappa Sinistra: GROUND TRUTH REALE NELLE OMBRE
        self.draw_base_map(self.ax_real, f"GROUND TRUTH REALE 3D NELLE OMBRE | Raggio <={self.max_distance:.0f}m")
        
        real_occupied_count = 0
        real_classes_found = {c: 0 for c in CLASS_NAMES}

        for idx, occ in enumerate(self.occlusions):
            if occ["distance_m"] > self.max_distance:
                continue

            is_sel = (idx == self.current_occ_idx)
            has_obj = occ["has_real_obj"]
            
            if has_obj:
                real_occupied_count += 1
                top_c = occ["real_classes"][0]
                real_classes_found[top_c] += 1
                color = CLASS_COLORS[top_c]
                alpha = 0.85 if is_sel else 0.65
                edge_c = "#FFFFFF" if is_sel else color
                lw = 3.0 if is_sel else 1.8
            else:
                color = "#334155" # Grigio/Blu scuro = Ombra Vuota
                alpha = 0.65 if is_sel else 0.25
                edge_c = "#FFFFFF" if is_sel else "#475569"
                lw = 2.5 if is_sel else 1.0

            mpl_poly = MplPolygon(occ["polygon_xy"], closed=True, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha, zorder=10 if is_sel else (7 if has_obj else 5))
            self.ax_real.add_patch(mpl_poly)

            # Etichetta centrata sull'ombra se contiene un ostacolo reale
            if has_obj:
                cx, cy = np.mean(occ["polygon_xy"][:, 0]), np.mean(occ["polygon_xy"][:, 1])
                top_c = occ["real_classes"][0]
                self.ax_real.text(cx, cy, f"GT: {top_c}", color='#FFFFFF', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=12, bbox=dict(boxstyle='round,pad=0.2', facecolor=CLASS_COLORS[top_c], alpha=0.9, edgecolor='#FFFFFF'))

        # 2. Disegna Mappa Destra: GROUND TRUTH NEUROSIMBOLICA NELLE OMBRE
        self.draw_base_map(self.ax_syn, f"GROUND TRUTH NEUROSIMBOLICA (PLAUSIBILITÀ) | Raggio <={self.max_distance:.0f}m")
        
        neuro_assigned_count = 0
        neuro_classes_found = {c: 0 for c in CLASS_NAMES}

        for idx, occ in enumerate(self.occlusions):
            if occ["distance_m"] > self.max_distance:
                continue

            is_sel = (idx == self.current_occ_idx)
            assigned_c = occ["neuro_assigned_class"]

            if assigned_c:
                neuro_assigned_count += 1
                neuro_classes_found[assigned_c] += 1
                color = CLASS_COLORS[assigned_c]
                alpha = 0.85 if is_sel else 0.65
                edge_c = "#FFFFFF" if is_sel else color
                lw = 3.0 if is_sel else 1.8
            else:
                color = "#334155" # Grigio scuro = Ombra non idonea
                alpha = 0.65 if is_sel else 0.25
                edge_c = "#FFFFFF" if is_sel else "#475569"
                lw = 2.5 if is_sel else 1.0

            mpl_poly = MplPolygon(occ["polygon_xy"], closed=True, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha, zorder=10 if is_sel else (7 if assigned_c else 5))
            self.ax_syn.add_patch(mpl_poly)

            # Etichetta centrata con l'ostacolo plausibile assegnato
            if assigned_c:
                cx, cy = np.mean(occ["polygon_xy"][:, 0]), np.mean(occ["polygon_xy"][:, 1])
                self.ax_syn.text(cx, cy, f"NEURO: {assigned_c}", color='#FFFFFF', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=12, bbox=dict(boxstyle='round,pad=0.2', facecolor=CLASS_COLORS[assigned_c], alpha=0.9, edgecolor='#FFFFFF'))

        # 3. Rendering HUD Laterale con Confronto 1-a-1
        hud_text = []
        hud_text.append(("🎯 CONFRONTO GROUND TRUTH", "#38BDF8", 12, 'bold'))
        hud_text.append((f"Fotogramma: #{self.current_idx+1}/{self.total_samples}", "#F8FAFC", 9, 'bold'))
        hud_text.append((f"Raggio Attivo: <= {self.max_distance:.0f}m (Premi 'R')", "#38BDF8", 8, 'bold'))
        hud_text.append(("─" * 45, "#475569", 8, 'normal'))

        # SEZIONE 1: OSTACOLI REALI NELLE OMBRE
        hud_text.append((f"📦 GT REALE NELLE OMBRE (Totale: {real_occupied_count})", "#34D399", 10, 'bold'))
        for c_name in CLASS_NAMES:
            cnt = real_classes_found[c_name]
            p_col = CLASS_COLORS[c_name] if cnt > 0 else "#64748B"
            hud_text.append((f"  • {c_name:<11}: {cnt} ombre", p_col, 8, 'bold' if cnt > 0 else 'normal'))

        hud_text.append(("─" * 45, "#475569", 8, 'normal'))

        # SEZIONE 2: OSTACOLI NEUROSIMBOLICI VALORIZZATI
        hud_text.append((f"🧠 GT NEUROSIMBOLICA (Totale: {neuro_assigned_count})", "#06B6D4", 10, 'bold'))
        for c_name in CLASS_NAMES:
            cnt = neuro_classes_found[c_name]
            p_col = CLASS_COLORS[c_name] if cnt > 0 else "#64748B"
            hud_text.append((f"  • {c_name:<11}: {cnt} varchi plausibili", p_col, 8, 'bold' if cnt > 0 else 'normal'))

        hud_text.append(("─" * 45, "#475569", 8, 'normal'))

        # SEZIONE 3: OMBRA SELEZIONATA
        if self.occlusions and self.current_occ_idx < len(self.occlusions):
            sel = self.occlusions[self.current_occ_idx]
            hud_text.append((f"🔍 OMBRA #{self.current_occ_idx} (D={sel['distance_m']:.1f}m, W={sel['width_m']:.1f}m, L={sel['depth_m']:.1f}m)", "#FCD34D", 8.5, 'bold'))
            
            # Tipo di Superficie Fisso
            if sel['cross_f'] >= 0.10:
                dom_surf = "STRISCE PEDONALI"
            elif sel['side_f'] >= 0.30:
                dom_surf = "MARCIAPIEDE / PEDONALE"
            elif sel['road_f'] >= 0.40:
                dom_surf = "STRADA (Carrabile)"
            else:
                dom_surf = "TERRENO / BORDO STRADA"

            hud_text.append((f"• Tipo Superficie : {dom_surf}", "#38BDF8", 8, 'bold'))
            hud_text.append((f"• Ripartizione    : Strada={int(sel['road_f']*100)}%, Marc={int(sel['side_f']*100)}%, Terr={int(sel['terr_f']*100)}%", "#94A3B8", 7.5, 'normal'))

            # Ground truth reale
            if sel["has_real_obj"]:
                hud_text.append((f"• GT Reale       : {', '.join(sel['real_classes'])}", "#34D399", 8, 'bold'))
            else:
                hud_text.append(("• GT Reale       : VUOTA (Nessun ostacolo annotato)", "#94A3B8", 8, 'normal'))

            # Ground truth neurosimbolica
            if sel["neuro_assigned_class"]:
                hud_text.append((f"• GT Neurosimb.  : {', '.join(sel['neuro_classes'])}", CLASS_COLORS[sel['neuro_assigned_class']], 8, 'bold'))
            else:
                hud_text.append(("• GT Neurosimb.  : VUOTA (Spazio insufficiente)", "#94A3B8", 8, 'normal'))

            hud_text.append((f"• Motivo Regola  : {sel['neuro_reason']}", "#CBD5E1", 7.5, 'normal'))

        y_pos = 0.96
        for text_line, color, size, weight in hud_text:
            self.ax_hud.text(0.04, y_pos, text_line, color=color, fontsize=size, fontweight=weight, transform=self.ax_hud.transAxes, va='top')
            y_pos -= 0.034

        legend_elements = [
            Patch(facecolor='#334155', edgecolor='#475569', label='Ombra Vuota'),
            Patch(facecolor='#3B82F6', label='Auto'),
            Patch(facecolor='#10B981', label='Pedone'),
            Patch(facecolor='#EC4899', label='Camion'),
            Patch(facecolor='#F59E0B', label='Moto'),
            Patch(facecolor='#8B5CF6', label='Bici'),
            Patch(facecolor='#38BDF8', label='Ego Vehicle')
        ]
        self.fig.legend(handles=legend_elements, loc='upper right', ncol=7, fontsize=7.5, facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', framealpha=0.9)
        self.fig.canvas.draw()

if __name__ == "__main__":
    app = GroundTruthZoneVisualizer()
