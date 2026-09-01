import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, rasterize_polygon, get_occlusion_ground_truth_target
from architettura_neurale import AttentionPerZoneModel, PerZoneModel

GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
CATEGORIES = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)", "Barriera"]
CAT_COLORS = {
    "Auto": "#38BDF8",                   # Azzurro
    "Camion/Bus": "#C084FC",              # Viola
    "VRU (Pedoni/Bici)": "#34D399",       # Verde Smeraldo
    "Barriera": "#F87171"                 # Rosso Corallo
}

class DualModelVisualizer:
    def __init__(self):
        print("\n" + "=" * 75)
        print("   CONFRONTO AFFIANCATO: MODELLO BASE (CNN) VS MODELLO AVANZATO (ATTENTION)")
        print("=" * 75)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Dispositivo: {self.device}")

        # 1. Caricamento Modello Base (CNN)
        self.model_base = PerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        ckpt_b_path = os.path.join("pesi_modelli", "_prove", "per_zone_checkpoint_asl_standard.pth")
        if not os.path.exists(ckpt_b_path):
            ckpt_b_path = os.path.join("pesi_modelli", "per_zone_checkpoint_asl_standard.pth")
        if os.path.exists(ckpt_b_path):
            ckpt_b = torch.load(ckpt_b_path, map_location=self.device)
            self.model_base.load_state_dict(ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b)
        self.model_base.eval()

        # 2. Caricamento Modello Avanzato (SE-Attention + FiLM addestrato su Neuro GT)
        self.model_attn = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(self.device)
        ckpt_a_path = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        ckpt_a = torch.load(ckpt_a_path, map_location=self.device)
        self.model_attn.load_state_dict(ckpt_a["model_state_dict"] if "model_state_dict" in ckpt_a else ckpt_a)
        self.model_attn.eval()

        # 3. Adapter nuScenes
        self.adapter = create_adapter("nuscenes", "./nuscenes")
        self.num_frames = self.adapter.get_num_samples()

        # Stato applicazione
        self.current_frame_idx = 0
        self.gt_mode = "real"          # "real" (GT nuScenes Reale) oppure "neuro" (GT Sintetica)
        self.max_distance = 25.0       # 25m oppure 20m
        self.decision_threshold = 0.30
        self.current_occ_idx = 0

        self.current_frame_data = None
        self.inferred_occlusions = []

        # Setup Finestra Interattiva a Schermo Intero
        self.fig = plt.figure(figsize=(19, 9.8), facecolor='#0B0F19')
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 1.4], left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.15)

        self.ax_hud = self.fig.add_subplot(gs[0, 0])
        self.ax_base = self.fig.add_subplot(gs[0, 1])
        self.ax_attn = self.fig.add_subplot(gs[0, 2])

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.load_frame_data(self.current_frame_idx)
        self.render_scene()

    def load_frame_data(self, frame_idx):
        self.current_frame_data = self.adapter.get_sample_data(frame_idx)
        token = self.current_frame_data["sample_token"]
        target_masks = extract_ground_truth_masks(self.current_frame_data)

        semantic_map = self.current_frame_data['semantic_map']
        drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
        walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
        ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

        input_channels = [
            self.current_frame_data.get('lidar_bev', np.zeros((200, 200))),
            self.current_frame_data.get('occlusion_mask', np.zeros((200, 200))),
            drivable_mask,
            walkway_mask,
            ped_crossing_mask
        ] + list(target_masks)

        input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

        occ_file = os.path.join("extracted_occlusions", f"{token}.json")
        raw_occs = []
        if os.path.exists(occ_file):
            with open(occ_file) as f:
                raw_occs = json.load(f)["occlusions"]

        self.inferred_occlusions = []
        for occ in raw_occs:
            dist = occ.get("distance_m", 0.0)
            pts = occ.get("polygon_points_m", [])
            pts_np = np.array(pts)
            if len(pts_np) < 3: continue
            poly_xy = np.column_stack([pts_np[:, 1], pts_np[:, 0]])

            from shapely.geometry import Polygon as ShapelyPoly
            sp = ShapelyPoly(poly_xy)
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
            road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = occ.get("area_sqm", 0.0)

            # Target Reale a 4 classi
            gt_raw_6 = get_occlusion_ground_truth_target(target_masks, pts)
            gt_raw_4 = np.array([gt_raw_6[0], gt_raw_6[1], max(gt_raw_6[2], gt_raw_6[3], gt_raw_6[4]), gt_raw_6[5]], dtype=np.float32)

            # Target Sintetico a 4 classi
            gt_neuro_4 = np.zeros(4, dtype=np.float32)
            if np.sum(gt_raw_4) > 0:
                gt_neuro_4 = gt_raw_4.copy()
            else:
                if road_f >= 0.25 and obb_w >= 1.8 and obb_l >= 3.8 and area >= 8.0:
                    gt_neuro_4[0] = 1.0
                    if road_f >= 0.30 and obb_w >= 2.4 and obb_l >= 6.5 and area >= 18.0:
                        gt_neuro_4[1] = 1.0
                if (side_f >= 0.15 or cross_f >= 0.08 or (road_f >= 0.25 and obb_w < 1.8)) and obb_w >= 0.4 and area >= 0.6:
                    gt_neuro_4[2] = 1.0
                if terr_f >= 0.55 and obb_w >= 1.0 and area >= 3.0 and road_f < 0.15 and side_f < 0.10:
                    gt_neuro_4[3] = 1.0

            px_x = np.clip(((poly_xy[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
            px_y = np.clip(((40.0 - poly_xy[:, 1]) / 0.4).astype(int), 0, 199)
            xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
            ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
            if xmax <= xmin or ymax <= ymin: continue

            patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
            patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device)
            scalars = torch.tensor([[area, dist, 2.0, 1.8, road_f, side_f, cross_f, 0.0, terr_f]], dtype=torch.float32).to(self.device)

            # 1. Inferenza Modello Base
            with torch.no_grad():
                out_base_6 = torch.sigmoid(self.model_base(patch_res, scalars)).squeeze(0).cpu().numpy()
            probs_base_4 = np.array([out_base_6[0], out_base_6[1], max(out_base_6[2], out_base_6[3], out_base_6[4]), out_base_6[5]], dtype=np.float32)

            # 2. Inferenza Modello Avanzato (SE + FiLM)
            with torch.no_grad():
                out_attn_6 = torch.sigmoid(self.model_attn(patch_res, scalars)).squeeze(0).cpu().numpy()
            probs_attn_4 = np.array([out_attn_6[0], out_attn_6[1], max(out_attn_6[2], out_attn_6[3], out_attn_6[4]), out_attn_6[5]], dtype=np.float32)

            self.inferred_occlusions.append({
                "polygon_xy": poly_xy,
                "distance_m": dist,
                "area_sqm": area,
                "width_m": obb_w,
                "depth_m": obb_l,
                "road_f": road_f,
                "side_f": side_f,
                "cross_f": cross_f,
                "terr_f": terr_f,
                "gt_raw": gt_raw_4,
                "gt_neuro": gt_neuro_4,
                "probs_base": probs_base_4,
                "probs_attn": probs_attn_4
            })

    def render_map(self, ax, model_key, title_text):
        ax.clear()
        ax.set_facecolor('#0B0F19')
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.axis('off')

        # Sfondo semantico HD Map
        semantic = self.current_frame_data['semantic_map']
        drivable = np.maximum(semantic['drivable_area'], semantic.get('carpark_area', np.zeros((200, 200))))
        walkway = semantic.get('walkway', np.zeros((200, 200)))
        ped_cross = semantic.get('ped_crossing', np.zeros((200, 200)))

        extent = [-40.0, 40.0, -40.0, 40.0]
        ax.imshow(np.where(drivable > 0.5, 1.0, np.nan), cmap='gray', vmin=0, vmax=1.5, alpha=0.35, extent=extent, origin='upper', zorder=1)
        ax.imshow(np.where(walkway > 0.5, 1.0, np.nan), cmap='Blues', vmin=0, vmax=1.5, alpha=0.30, extent=extent, origin='upper', zorder=2)
        ax.imshow(np.where(ped_cross > 0.5, 1.0, np.nan), cmap='autumn', vmin=0, vmax=1.5, alpha=0.50, extent=extent, origin='upper', zorder=3)

        # Cerchio Raggio Attivo
        circ = plt.Circle((0, 0), self.max_distance, color='#38BDF8', fill=False, linestyle='--', linewidth=1.5, zorder=5)
        ax.add_patch(circ)
        ax.text(0, self.max_distance + 1.2, f"Raggio: {self.max_distance:.0f}m", color='#38BDF8', fontsize=7.5, ha='center', zorder=6)

        # Sagoma Ego-Veicolo
        ax.add_patch(MplPolygon(np.array([[-1.0, -2.0], [1.0, -2.0], [1.0, 2.0], [0.0, 2.7], [-1.0, 2.0]]), closed=True, facecolor='#0284C7', edgecolor='#FFFFFF', linewidth=1.2, zorder=10))

        # Disegno delle scatole 3D visibili
        for b in self.current_frame_data.get('boxes', []):
            cat = b.name.lower()
            c_name = "Auto" if "car" in cat else ("Camion/Bus" if ("truck" in cat or "bus" in cat) else ("VRU (Pedoni/Bici)" if ("pedestrian" in cat or "cycle" in cat) else "Barriera"))
            c = b.corners()
            c_xy = c[:2, [0, 1, 5, 4]].T
            color = CAT_COLORS.get(c_name, "#34D399")
            ax.add_patch(MplPolygon(c_xy, closed=True, facecolor=color, edgecolor='#FFFFFF', linewidth=1.0, alpha=0.85, zorder=8))

        # Poligoni d'Ombra e Predizioni del Modello
        for idx, occ in enumerate(self.inferred_occlusions):
            if occ["distance_m"] > self.max_distance: continue

            poly_np = occ["polygon_xy"]
            target = occ["gt_neuro"] if self.gt_mode == "neuro" else occ["gt_raw"]
            probs = occ[model_key]

            max_p = np.max(probs)
            pred_class_idx = np.argmax(probs)
            th_auto = 0.20 if (occ["road_f"] >= 0.35 and occ["width_m"] >= 1.8) else self.decision_threshold
            
            has_obj = np.sum(target) > 0
            pred_pos = any(probs[c] >= (th_auto if c==0 else self.decision_threshold) for c in range(4))

            top_cat = CATEGORIES[pred_class_idx]
            top_str = f"{top_cat} ({int(max_p*100)}%)"

            if has_obj and pred_pos:
                color = "#10B981"; status = "TP"
            elif not has_obj and pred_pos:
                color = "#EF4444"; status = "FP"
            elif has_obj and not pred_pos:
                color = "#F59E0B"; status = "FN"
            else:
                color = "#475569"; status = "TN"

            is_sel = (idx == self.current_occ_idx)
            ax.add_patch(MplPolygon(poly_np, closed=True, facecolor=color, edgecolor="#FFFFFF" if is_sel else color, linewidth=2.5 if is_sel else 1.0, alpha=0.80 if is_sel else 0.35, zorder=12 if is_sel else 6))

            cx, cy = np.mean(poly_np[:, 0]), np.mean(poly_np[:, 1])
            if status == "TP":
                ax.text(cx, cy, f"TP: {top_str}", color='#FFFFFF', fontsize=5.8, fontweight='bold', ha='center', va='center', zorder=14, bbox=dict(boxstyle='round,pad=0.2', facecolor=CAT_COLORS[top_cat], alpha=0.85, edgecolor='#FFFFFF'))
            elif status == "FP":
                ax.text(cx, cy, f"FP: {top_str}", color='#FFFFFF', fontsize=5.8, fontweight='bold', ha='center', va='center', zorder=14, bbox=dict(boxstyle='round,pad=0.2', facecolor='#DC2626', alpha=0.85, edgecolor='#FECACA'))
            elif status == "FN":
                real_c = [CATEGORIES[i] for i in range(4) if target[i] == 1]
                ax.text(cx, cy, f"FN: {real_c[0] if real_c else 'Obj'}", color='#FFFFFF', fontsize=5.8, fontweight='bold', ha='center', va='center', zorder=14, bbox=dict(boxstyle='round,pad=0.2', facecolor='#D97706', alpha=0.90, edgecolor='#FDE68A'))

        ax.set_title(title_text, color='#38BDF8', fontsize=11, fontweight='bold', pad=8)

    def render_hud(self):
        self.ax_hud.clear()
        self.ax_hud.set_facecolor('#0F172A')
        self.ax_hud.axis('off')

        gt_title = "GT Sintetica (Neurosimbolica)" if self.gt_mode == "neuro" else "GT nuScenes Reale (Grezza)"
        
        # Calcolo statistiche frame per entrambi i modelli
        active_occs = [occ for occ in self.inferred_occlusions if occ["distance_m"] <= self.max_distance]
        
        def compute_frame_stats(model_key):
            stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
            for occ in active_occs:
                target = occ["gt_neuro"] if self.gt_mode == "neuro" else occ["gt_raw"]
                probs = occ[model_key]
                th_auto = 0.20 if (occ["road_f"] >= 0.35 and occ["width_m"] >= 1.8) else self.decision_threshold
                for c in range(4):
                    t = target[c]
                    p = probs[c] >= (th_auto if c==0 else self.decision_threshold)
                    if t == 1 and p: stats[CATEGORIES[c]]["tp"] += 1
                    elif t == 0 and p: stats[CATEGORIES[c]]["fp"] += 1
                    elif t == 1 and not p: stats[CATEGORIES[c]]["fn"] += 1
            return stats

        st_base = compute_frame_stats("probs_base")
        st_attn = compute_frame_stats("probs_attn")

        hud_lines = []
        hud_lines.append(("⚖️ CONFRONTO DIRETTO 1-A-1", "#38BDF8", 12, 'bold'))
        hud_lines.append((f"Ground Truth (G): {gt_title}", "#34D399" if self.gt_mode=="neuro" else "#F59E0B", 9, 'bold'))
        hud_lines.append((f"Fotogramma: #{self.current_frame_idx+1}/{self.num_frames} | Raggio: <= {self.max_distance:.0f}m", "#CBD5E1", 8, 'normal'))
        hud_lines.append(("─" * 45, "#475569", 8, 'normal'))

        # Confronto Globale Frame
        hud_lines.append(("📊 METRICHE LIVE (MODELLO BASE vs AVANZATO):", "#38BDF8", 9, 'bold'))
        
        for c in CATEGORIES:
            tb, fb, nb = st_base[c]["tp"], st_base[c]["fp"], st_base[c]["fn"]
            ta, fa, na = st_attn[c]["tp"], st_attn[c]["fp"], st_attn[c]["fn"]
            n_gt = tb + nb
            color_c = CAT_COLORS[c]
            
            hud_lines.append((f"• {c:<21} (Reali GT: {n_gt})", color_c, 8.5, 'bold'))
            hud_lines.append((f"  └─ Base : TP={tb} | FP={fb:2d} | FN={nb}", "#94A3B8", 7.5, 'normal'))
            hud_lines.append((f"  └─ Attn : TP={ta} | FP={fa:2d} | FN={na}", "#34D399" if fa < fb else "#CBD5E1", 7.5, 'bold'))

        hud_lines.append(("─" * 45, "#475569", 8, 'normal'))
        
        tot_tp_b = sum(st_base[c]["tp"] for c in CATEGORIES)
        tot_fp_b = sum(st_base[c]["fp"] for c in CATEGORIES)
        tot_fn_b = sum(st_base[c]["fn"] for c in CATEGORIES)
        
        tot_tp_a = sum(st_attn[c]["tp"] for c in CATEGORIES)
        tot_fp_a = sum(st_attn[c]["fp"] for c in CATEGORIES)
        tot_fn_a = sum(st_attn[c]["fn"] for c in CATEGORIES)

        hud_lines.append(("🎯 RIEPILOGO TOTALE FOTOGRAMMA:", "#E2E8F0", 9, 'bold'))
        hud_lines.append((f"  • Base (CNN) : TP={tot_tp_b:2d} | FP={tot_fp_b:2d} | FN={tot_fn_b:2d}", "#94A3B8", 8, 'normal'))
        hud_lines.append((f"  • Attn (SE)  : TP={tot_tp_a:2d} | FP={tot_fp_a:2d} | FN={tot_fn_a:2d}", "#38BDF8", 8.5, 'bold'))
        hud_lines.append(("─" * 45, "#475569", 8, 'normal'))

        # Dettaglio Zona Selezionata
        if 0 <= self.current_occ_idx < len(self.inferred_occlusions):
            occ = self.inferred_occlusions[self.current_occ_idx]
            hud_lines.append((f"🔍 ZONA #{self.current_occ_idx} (D={occ['distance_m']:.1f}m, W={occ['width_m']:.1f}m):", "#FBBF24", 9, 'bold'))
            
            # Superficie Fissa
            if occ['cross_f'] >= 0.10:
                dom_surf = "STRISCE PEDONALI"
            elif occ['side_f'] >= 0.30:
                dom_surf = "MARCIAPIEDE / PEDONALE"
            elif occ['road_f'] >= 0.40:
                dom_surf = "STRADA (Carrabile)"
            else:
                dom_surf = "TERRENO / BORDO STRADA"

            hud_lines.append((f"  • Superficie : {dom_surf}", "#38BDF8", 8, 'bold'))
            hud_lines.append((f"  • Ripartiz.  : Strada={int(occ['road_f']*100)}%, Marc={int(occ['side_f']*100)}%, Terr={int(occ['terr_f']*100)}%", "#94A3B8", 7.5, 'normal'))

            target = occ["gt_neuro"] if self.gt_mode == "neuro" else occ["gt_raw"]
            real_str = ", ".join([CATEGORIES[i] for i in range(4) if target[i]==1]) or "Vuoto"
            hud_lines.append((f"  • Ground Truth: {real_str}", "#FFFFFF", 8, 'bold'))
            hud_lines.append((f"  • Base : Auto={occ['probs_base'][0]*100:.0f}%, VRU={occ['probs_base'][2]*100:.0f}%, Barr={occ['probs_base'][3]*100:.0f}%", "#94A3B8", 7.5, 'normal'))
            hud_lines.append((f"  • Attn : Auto={occ['probs_attn'][0]*100:.0f}%, VRU={occ['probs_attn'][2]*100:.0f}%, Barr={occ['probs_attn'][3]*100:.0f}%", "#38BDF8", 7.5, 'bold'))

        hud_lines.append(("─" * 45, "#475569", 8, 'normal'))
        hud_lines.append(("🎮 CONTROLLI: [M] Cambia GT | [R] Raggio | [Frecce] Frame", "#64748B", 7.5, 'normal'))

        y_pos = 0.97
        for text, color, size, weight in hud_lines:
            self.ax_hud.text(0.04, y_pos, text, color=color, fontsize=size, fontweight=weight, transform=self.ax_hud.transAxes, va='top', fontfamily='Arial')
            y_pos -= 0.033

    def render_scene(self):
        self.render_hud()
        self.render_map(self.ax_base, "probs_base", "1. MODELLO BASE (CNN SEMPLICE + ASL)")
        self.render_map(self.ax_attn, "probs_attn", "2. MODELLO AVANZATO (SE-ATTN + FiLM + ASL)")
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key in ['right', 'd', ' ']:
            self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
            self.load_frame_data(self.current_frame_idx)
            self.render_scene()
        elif event.key in ['left', 'a']:
            self.current_frame_idx = (self.current_frame_idx - 1) % self.num_frames
            self.load_frame_data(self.current_frame_idx)
            self.render_scene()
        elif event.key in ['m', 'M', 'g', 'G']:
            self.gt_mode = "neuro" if self.gt_mode == "real" else "real"
            self.render_scene()
        elif event.key in ['r', 'R']:
            self.max_distance = 20.0 if self.max_distance == 25.0 else 25.0
            self.render_scene()

    def on_click(self, event):
        if event.inaxes in [self.ax_base, self.ax_attn]:
            mx, my = event.xdata, event.ydata
            if mx is not None and my is not None:
                dists = [np.linalg.norm(np.mean(occ["polygon_xy"], axis=0) - np.array([mx, my])) for occ in self.inferred_occlusions]
                if dists:
                    min_idx = int(np.argmin(dists))
                    if dists[min_idx] < 8.0:
                        self.current_occ_idx = min_idx
                        self.render_scene()

if __name__ == "__main__":
    app = DualModelVisualizer()
    plt.show()
