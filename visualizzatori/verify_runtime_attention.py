# Visualizzatore Interattivo Ufficiale per il Modello Avanzato con SE-Attention + FiLM (verify_runtime_attention.py)
# Esegue l'INFERENZA NEURALE LIVE del modello vincente con Channel Attention e Modulazione FiLM
# Supporta il toggle dinamico tra:
#   1. Ground Truth Sintetica Neurosimbolica (Plausibilità Mappa HD + Regole Fisiche)
#   2. Ground Truth Reale 3D nuScenes (Annotazioni Umane)

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
from shapely.geometry import Polygon as ShapelyPolygon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nuscenes.nuscenes import NuScenes
from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
from architettura_neurale import AttentionPerZoneModel

GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4

CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"]
CAT_COLORS = {
    "Auto": "#38BDF8",       # Celeste / Blu chiaro
    "Camion/Bus": "#EC4899", # Rosa / Magenta
    "VRU (Pedoni/Bici)": "#10B981", # Verde Smeraldo
    "Barriera": "#F59E0B"    # Ambra / Giallo oro
}

class AttentionModelVisualizer:
    def __init__(self):
        print("\n" + "=" * 75)
        print("   INIZIALIZZAZIONE VISUALIZZATORE MODELLO AVANZATO (SE + FiLM)")
        print("=" * 75)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Dispositivo di Calcolo Attivo: {self.device}")

        # Inizializza nuScenes adapter
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        self.total_samples = self.adapter.get_num_samples()

        # Carica il Modello Avanzato con Attention e FiLM dai pesi ufficiali
        self.model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        ckpt_path = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        
        print(f"• Caricamento Pesi Modello: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        self.model.eval()

        self.current_idx = 0
        self.current_occ_idx = 0
        self.max_distance = 20.0
        self.gt_mode = "neuro" # "neuro" o "reale"
        self.inferred_occlusions = []

        # Setup Interfaccia Grafica Matplotlib a Tema Dark Moderno
        plt.rcParams['toolbar'] = 'none'
        self.fig = plt.figure(figsize=(17, 9.5), facecolor='#0B0F19')
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], left=0.03, right=0.97, bottom=0.05, top=0.93, wspace=0.12)
        
        self.ax_map = self.fig.add_subplot(self.gs[0, 0])
        self.ax_hud = self.fig.add_subplot(self.gs[0, 1])

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.load_frame_and_infer(self.current_idx)
        self.render_all()
        plt.show()

    def load_frame_and_infer(self, idx):
        self.current_idx = idx % self.total_samples
        self.frame_data = self.adapter.get_sample_data(self.current_idx)
        self.sample_token = self.frame_data['sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
        self.lidar_token = self.sample['data']['LIDAR_TOP']

        target_masks = extract_ground_truth_masks(self.frame_data)
        semantic_map = self.frame_data['semantic_map']
        self.drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
        self.walkway_mask = semantic_map.get('walkway', np.zeros_like(self.drivable_mask))
        self.ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(self.drivable_mask))

        input_channels = [
            self.frame_data.get('lidar_bev', np.zeros((GRID_DIM, GRID_DIM))),
            self.frame_data.get('occlusion_mask', np.zeros((GRID_DIM, GRID_DIM))),
            self.drivable_mask,
            self.walkway_mask,
            self.ped_crossing_mask
        ] + list(target_masks)

        self.input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

        occ_file = os.path.join("extracted_occlusions", f"{self.sample_token}.json")
        self.inferred_occlusions = []
        if not os.path.exists(occ_file):
            return

        with open(occ_file, "r") as f:
            occs = json.load(f)["occlusions"]

        for occ in occs:
            dist = float(occ.get("distance_m", 0.0))
            pts = occ.get("polygon_points_m", [])
            pts_np = np.array(pts)
            if len(pts_np) < 3:
                continue
            poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

            sp = ShapelyPolygon(poly_xy)
            if sp.is_valid and sp.area > 0.01:
                mrr = sp.minimum_rotated_rectangle
                mrr_coords = np.array(mrr.exterior.coords)[:-1]
                e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                obb_w, obb_l = min(e1, e2), max(e1, e2)
            else:
                obb_w, obb_l = 0.2, 0.5

            occ_mask = rasterize_polygon(pts)
            tot = np.sum(occ_mask)
            road_f = np.sum(occ_mask * self.drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * self.walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * self.ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = float(occ.get("area_sqm", 0.0))

            # Target Reale a 4 classi
            gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)
            gt_raw_4 = np.array([gt_raw_6[0], gt_raw_6[1], max(gt_raw_6[2], gt_raw_6[3], gt_raw_6[4]), gt_raw_6[5]], dtype=np.float32)

            # Target Sintetico a 4 classi
            gt_neuro_4 = np.zeros(4, dtype=np.float32)
            if np.sum(gt_raw_4) > 0:
                gt_neuro_4 = gt_raw_4.copy()
            else:
                if road_f >= 0.25 and obb_w >= 1.5 and obb_l >= 3.0 and area >= 5.0:
                    gt_neuro_4[0] = 1.0
                    if road_f >= 0.30 and obb_w >= 2.4 and obb_l >= 6.5 and area >= 18.0:
                        gt_neuro_4[1] = 1.0
                if (side_f >= 0.15 or cross_f >= 0.08 or (road_f >= 0.25 and obb_w < 1.8)) and obb_w >= 0.4 and area >= 0.6:
                    gt_neuro_4[2] = 1.0
                if terr_f >= 0.55 and obb_w >= 1.0 and area >= 3.0 and road_f < 0.15 and side_f < 0.10:
                    gt_neuro_4[3] = 1.0

            # Crop Patch BEV
            px_x = np.clip(((poly_xy[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
            px_y = np.clip(((GRID_RANGE - poly_xy[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
            xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
            ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
            if xmax <= xmin or ymax <= ymin:
                continue

            patch = self.input_tensor[:, ymin:ymax+1, xmin:xmax+1]
            patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device)
            scalars = torch.tensor([[area, dist, 2.0, 1.8, road_f, side_f, cross_f, 0.0, terr_f]], dtype=torch.float32).to(self.device)

            # Inferenza Live Modello Avanzato (SE-Attention + FiLM)
            with torch.no_grad():
                out_6 = torch.sigmoid(self.model(patch_res, scalars)).squeeze(0).cpu().numpy()

            probs_4 = np.array([out_6[0], out_6[1], max(out_6[2], out_6[3], out_6[4]), out_6[5]], dtype=np.float32)

            self.inferred_occlusions.append({
                "polygon_xy": poly_xy,
                "distance_m": dist,
                "width_m": obb_w,
                "length_m": obb_l,
                "area_sqm": area,
                "road_f": road_f,
                "side_f": side_f,
                "cross_f": cross_f,
                "terr_f": terr_f,
                "gt_raw": gt_raw_4,
                "gt_neuro": gt_neuro_4,
                "probs_6": out_6,
                "probs_4": probs_4
            })

        self.current_occ_idx = 0

    def render_all(self):
        self.ax_map.clear()
        self.ax_hud.clear()
        self.ax_hud.axis('off')

        gt_label = "SINTETICA NEUROSIMBOLICA" if self.gt_mode == "neuro" else "REALE 3D NUSCENES"
        self.ax_map.set_facecolor('#0B0F19')
        self.ax_map.set_xlim(-self.max_distance - 2, self.max_distance + 2)
        self.ax_map.set_ylim(-self.max_distance - 2, self.max_distance + 2)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title(f"MAPPA BEV LIVE: MODELLO AVANZATO (SE + FiLM) | GT: {gt_label}", color='#38BDF8', fontsize=11, fontweight='bold', pad=10)

        # Disegna nuScenes Underlay Map & LiDAR se disponibile
        if self.lidar_token:
            try:
                self.nusc.render_sample_data(self.lidar_token, ax=self.ax_map, underlay_map=True, verbose=False)
            except Exception:
                pass

        # Cerchio Raggio di Sicurezza
        circle = plt.Circle((0, 0), self.max_distance, color='#38BDF8', fill=False, linestyle='--', linewidth=1.5, alpha=0.8, zorder=6)
        self.ax_map.add_patch(circle)
        self.ax_map.text(0, self.max_distance + 0.6, f"Raggio: {self.max_distance:.0f}m", color='#38BDF8', fontsize=8, ha='center', zorder=7)

        # Ego Vehicle Icon
        ego_rect = plt.Rectangle((-1.0, -2.25), 2.0, 4.5, facecolor='#38BDF8', edgecolor='#FFFFFF', linewidth=1.5, alpha=0.9, zorder=20)
        self.ax_map.add_patch(ego_rect)

        # Statistiche Frame Corrente
        tp_cnt = {c: 0 for c in CATEGORIES}
        fp_cnt = {c: 0 for c in CATEGORIES}
        fn_cnt = {c: 0 for c in CATEGORIES}

        # Disegna Coni d'Ombra
        for idx, occ in enumerate(self.inferred_occlusions):
            if occ["distance_m"] > self.max_distance:
                continue

            target = occ["gt_neuro"] if self.gt_mode == "neuro" else occ["gt_raw"]
            probs = occ["probs_4"]
            preds = (probs >= 0.30).astype(int)

            # Determina stato visivo (TP, FP, FN, TN)
            is_tp = any((target[i] == 1 and preds[i] == 1) for i in range(4))
            is_fp = any((target[i] == 0 and preds[i] == 1) for i in range(4))
            is_fn = any((target[i] == 1 and preds[i] == 0) for i in range(4))
            is_sel = (idx == self.current_occ_idx)

            for i in range(4):
                c_name = CATEGORIES[i]
                if target[i] == 1 and preds[i] == 1: tp_cnt[c_name] += 1
                elif target[i] == 0 and preds[i] == 1: fp_cnt[c_name] += 1
                elif target[i] == 1 and preds[i] == 0: fn_cnt[c_name] += 1

            if is_tp:
                # Trova la classe principale con probabilità più alta
                top_c_idx = int(np.argmax(probs * preds))
                top_class = CATEGORIES[top_c_idx]
                color = CAT_COLORS[top_class]
                edge_c = '#FFFFFF' if is_sel else color
                alpha = 0.85 if is_sel else 0.65
                lw = 3.0 if is_sel else 1.8
                zorder = 15 if is_sel else 10
            elif is_fp:
                color = '#EF4444' # Rosso = Falso Positivo
                edge_c = '#FFFFFF' if is_sel else '#B91C1C'
                alpha = 0.85 if is_sel else 0.55
                lw = 2.5 if is_sel else 1.2
                zorder = 14 if is_sel else 8
            elif is_fn:
                color = '#F97316' # Arancio = Falso Negativo
                edge_c = '#FFFFFF' if is_sel else '#C2410C'
                alpha = 0.85 if is_sel else 0.60
                lw = 2.5 if is_sel else 1.2
                zorder = 14 if is_sel else 8
            else:
                color = '#334155' # Grigio scuro = Vero Negativo (Ombra soppressa)
                edge_c = '#FFFFFF' if is_sel else '#475569'
                alpha = 0.65 if is_sel else 0.25
                lw = 2.0 if is_sel else 0.8
                zorder = 12 if is_sel else 5

            mpl_poly = MplPolygon(occ["polygon_xy"], closed=True, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha, zorder=zorder)
            self.ax_map.add_patch(mpl_poly)

            # Etichetta centrata sull'ombra
            cx, cy = np.mean(occ["polygon_xy"][:, 0]), np.mean(occ["polygon_xy"][:, 1])
            if is_tp:
                self.ax_map.text(cx, cy, f"✓ {top_class}", color='#FFFFFF', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=16, bbox=dict(boxstyle='round,pad=0.2', facecolor=CAT_COLORS[top_class], alpha=0.9, edgecolor='#FFFFFF'))
            elif is_sel:
                self.ax_map.text(cx, cy, f"#{idx}", color='#FFFFFF', fontsize=8, fontweight='bold', ha='center', va='center', zorder=16)

        # 3. Costruzione Pannello HUD Laterale
        hud_lines = []
        hud_lines.append(("🧠 MODELLO AVANZATO (SE-ATTENTION + FiLM)", "#38BDF8", 12, 'bold'))
        hud_lines.append((f"Fotogramma: #{self.current_idx+1}/{self.total_samples} | Raggio: <={self.max_distance:.0f}m", "#F8FAFC", 9, 'bold'))
        hud_lines.append((f"Ground Truth Attiva: {gt_label} (Tasto 'M' o 'G')", "#FCD34D", 8.5, 'bold'))
        hud_lines.append(("─" * 48, "#475569", 8, 'normal'))

        # SEZIONE 1: RIEPILOGO STATO FRAME CORRENTE
        hud_lines.append(("📊 RIEPILOGO STATI NEL FOTOGRAMMA:", "#38BDF8", 9.5, 'bold'))
        tot_tp = sum(tp_cnt.values())
        tot_fp = sum(fp_cnt.values())
        tot_fn = sum(fn_cnt.values())

        hud_lines.append((f"  • Veri Positivi (TP)   : {tot_tp:<3d} 🟢 (Ostacoli correttamente rilevati)", "#34D399", 8.5, 'bold'))
        hud_lines.append((f"  • Falsi Allarmi (FP)  : {tot_fp:<3d} 🔴 (Ombre vuote allarmate)", "#F87171", 8.5, 'bold'))
        hud_lines.append((f"  • Pericoli Persi (FN) : {tot_fn:<3d} 🟠 (Ostacoli non intercettati)", "#FBBF24" if tot_fn>0 else "#94A3B8", 8.5, 'bold'))
        hud_lines.append(("─" * 48, "#475569", 8, 'normal'))

        # SEZIONE 2: METRICHE DETTAGLIATE PER CLASSE (CONTEGGI, RECALL, PRECISION, F1)
        hud_lines.append(("🏷️ METRICHE PER CLASSE NEL FRAME:", "#06B6D4", 9.5, 'bold'))
        for c_name in CATEGORIES:
            tp, fp, fn = tp_cnt[c_name], fp_cnt[c_name], fn_cnt[c_name]
            tot_scena = tp + fn
            p_cls = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            r_cls = (tp / tot_scena * 100) if tot_scena > 0 else (100.0 if fp == 0 else 0.0)
            f1_cls = (2 * p_cls * r_cls / (p_cls + r_cls)) if (p_cls + r_cls) > 0 else 0.0

            p_col = CAT_COLORS[c_name] if (tot_scena > 0 or fp > 0) else "#64748B"
            hud_lines.append((f"• {c_name:<18}: Trovati ({tp}/{tot_scena}) | FP={fp}", p_col, 8.5, 'bold' if tot_scena > 0 else 'normal'))
            if tot_scena > 0 or fp > 0:
                hud_lines.append((f"    └ Prec={p_cls:5.1f}% | Rec={r_cls:5.1f}% | F1={f1_cls:5.1f}%", "#94A3B8", 7.5, 'normal'))

        hud_lines.append(("─" * 48, "#475569", 8, 'normal'))

        # SEZIONE 3: DETTAGLIO ZONA SELEZIONATA
        if 0 <= self.current_occ_idx < len(self.inferred_occlusions):
            occ = self.inferred_occlusions[self.current_occ_idx]
            hud_lines.append((f"🔍 ZONA SELEZIONATA #{self.current_occ_idx} (D={occ['distance_m']:.1f}m, Area={occ['area_sqm']:.1f}m²):", "#FBBF24", 9, 'bold'))
            
            # Superficie Fissa
            if occ['cross_f'] >= 0.10:
                dom_surf = "STRISCE PEDONALI"
            elif occ['side_f'] >= 0.30:
                dom_surf = "MARCIAPIEDE / PEDONALE"
            elif occ['road_f'] >= 0.40:
                dom_surf = "STRADA (Carrabile)"
            else:
                dom_surf = "TERRENO / BORDO STRADA"

            hud_lines.append((f"  • Tipo Superficie : {dom_surf}", "#38BDF8", 8, 'bold'))
            hud_lines.append((f"  • Ripartizione    : Strada={int(occ['road_f']*100)}%, Marc={int(occ['side_f']*100)}%, Terr={int(occ['terr_f']*100)}%", "#94A3B8", 7.5, 'normal'))
            hud_lines.append((f"  • Sezione Varco   : Largh={occ['width_m']:.2f}m, Lungh={occ['length_m']:.2f}m", "#CBD5E1", 7.5, 'normal'))

            target = occ["gt_neuro"] if self.gt_mode == "neuro" else occ["gt_raw"]
            target_str = ", ".join([CATEGORIES[i] for i in range(4) if target[i] == 1]) or "Vuota / Nessun Pericolo"
            hud_lines.append((f"  • Ground Truth    : {target_str}", "#FFFFFF", 8, 'bold'))

            # Probabilità di Uscita della Rete
            hud_lines.append(("  • Predizioni Rete Neurale Live:", "#38BDF8", 8, 'bold'))
            probs = occ["probs_4"]
            for i, c_name in enumerate(CATEGORIES):
                bar = "█" * int(probs[i] * 12) + "░" * (12 - int(probs[i] * 12))
                is_active = probs[i] >= 0.30
                col = CAT_COLORS[c_name] if is_active else "#64748B"
                hud_lines.append((f"    - {c_name:<18}: {probs[i]*100:4.1f}% [{bar}]", col, 7.5, 'bold' if is_active else 'normal'))

        hud_lines.append(("─" * 48, "#475569", 8, 'normal'))
        hud_lines.append(("🎮 CONTROLLI TASTIERA:", "#94A3B8", 8.5, 'bold'))
        hud_lines.append(("  [Frecce SX/DX] : Cambia Fotogramma (#1 - #404)", "#E2E8F0", 7.5, 'normal'))
        hud_lines.append(("  [Tasto M o G]  : Switch Ground Truth (Sintetica ↔ Reale)", "#FCD34D", 7.5, 'bold'))
        hud_lines.append(("  [Tasto R]      : Cambia Raggio (20m ↔ 25m)", "#38BDF8", 7.5, 'normal'))
        hud_lines.append(("  [Click Mouse]  : Seleziona ed Ispeziona Cono d'Ombra", "#E2E8F0", 7.5, 'normal'))

        y_pos = 0.97
        for line_tuple in hud_lines:
            text, color, size, weight = line_tuple
            self.ax_hud.text(0.03, y_pos, text, color=color, fontsize=size, fontweight=weight, transform=self.ax_hud.transAxes, va='top')
            y_pos -= 0.029

        # Legenda in alto chiara e completa
        legend_handles = [
            Patch(facecolor='#38BDF8', label='Auto Rilevata'),
            Patch(facecolor='#10B981', label='VRU Rilevato'),
            Patch(facecolor='#EC4899', label='Camion Rilevato'),
            Patch(facecolor='#F59E0B', label='Barriera Rilevata'),
            Patch(facecolor='#EF4444', label='Falso Allarme (FP)'),
            Patch(facecolor='#F97316', label='Pericolo Perso (FN)'),
            Patch(facecolor='#334155', edgecolor='#475569', label='Ombra Vuota (TN)')
        ]
        self.fig.legend(handles=legend_handles, loc='upper right', ncol=7, fontsize=7.5, facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', framealpha=0.9)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ['right', 'd', ' ']:
            self.load_frame_and_infer(self.current_idx + 1)
            self.render_all()
        elif event.key in ['left', 'a']:
            self.load_frame_and_infer(self.current_idx - 1)
            self.render_all()
        elif event.key in ['m', 'M', 'g', 'G']:
            self.gt_mode = "reale" if self.gt_mode == "neuro" else "neuro"
            print(f"• Switch Ground Truth: {self.gt_mode.upper()}")
            self.render_all()
        elif event.key in ['r', 'R']:
            self.max_distance = 25.0 if self.max_distance == 20.0 else 20.0
            print(f"• Raggio di Sicurezza: {self.max_distance:.0f}m")
            self.render_all()
        elif event.key in ['up']:
            if self.inferred_occlusions:
                self.current_occ_idx = (self.current_occ_idx - 1) % len(self.inferred_occlusions)
                self.render_all()
        elif event.key in ['down']:
            if self.inferred_occlusions:
                self.current_occ_idx = (self.current_occ_idx + 1) % len(self.inferred_occlusions)
                self.render_all()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

    def on_click(self, event):
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return
        click_pt = ShapelyPolygon([[event.xdata-0.2, event.ydata-0.2], [event.xdata+0.2, event.ydata-0.2], [event.xdata+0.2, event.ydata+0.2], [event.xdata-0.2, event.ydata+0.2]])
        for idx, occ in enumerate(self.inferred_occlusions):
            if occ["distance_m"] > self.max_distance:
                continue
            sp = ShapelyPolygon(occ["polygon_xy"])
            if sp.intersects(click_pt):
                self.current_occ_idx = idx
                self.render_all()
                break

if __name__ == "__main__":
    app = AttentionModelVisualizer()
