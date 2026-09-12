# ==============================================================================
# VISUALIZZATORE UFFICIALE NUSCENES (NU-SCENES DEVKIT EXPLORER INTERATTIVO)
# File: visualizzatori/vis_official_nuscenes.py
#
# Permette di esplorare direttamente il dataset e i sensori nativi nuScenes
# tramite le routine grafiche ufficiali del nuScenes-devkit:
#
# - MODALITÀ 1 ([1]): Surround 360° Multi-Sensor + BEV HD-Map:
#   * Griglia con le 6 telecamere surround (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
#     CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT) con le bounding box 3D proiettate;
#   * Mappa Bird's-Eye View (BEV) ad alta risoluzione del LIDAR_TOP con sottofondo
#     vettoriale HD-Map (drivable area, corsie, attraversamenti) e annotazioni 3D;
#   * Riquadro metadati telemetria, velocità veicolo e statistiche ostacoli.
#
# - MODALITÀ 2 ([2]): Dedicated BEV + HD Map Vettoriale Fullscreen:
#   * Vista BEV ingrandita a tutto schermo con piano stradale vettoriale,
#     nuvola di punti LiDAR colorata per distanza, radar e box 3D orientati.
#
# - MODALITÀ 3 ([3]): Telecamere con Proiezione Depth Punti LiDAR:
#   * Routine ufficiale render_pointcloud_in_image, con i fasci LiDAR a 32 canali
#     proiettati sulle immagini RGB e colorati per profondità (da viola a giallo).
#
# Comandi interattivi da tastiera:
#   [<- / ->] oppure [A / D]: Frame precedente / successivo
#   [Up / Down]: Salto alla scena successiva / precedente
#   [1] / [2] / [3]: Cambio modalità di visualizzazione
#   [S]: Salva screenshot HD ad alta risoluzione in documentazione/immagini_tesi/
# ==============================================================================

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox

# Percorso root del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    raise ImportError("La libreria 'nuscenes-devkit' non risulta installata. Installa con: pip install nuscenes-devkit")


class NuScenesOfficialVisualizer:
    def __init__(self, dataroot="./nuscenes", version="v1.0-mini", initial_idx=0, mode=1):
        print("\n" + "=" * 80)
        print(f"   NUSCENES OFFICIAL VIEWER - DEVKIT EXPLORER (v1.0-mini)")
        print("=" * 80)
        print(f"Caricamento database nuScenes da: {dataroot} ...")

        self.dataroot = dataroot
        self.version = version
        self.mode = mode  # 1: Surround+BEV, 2: Solo BEV, 3: Proiezione LiDAR su Cam
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.all_samples = self.nusc.sample
        self.total_samples = len(self.all_samples)

        # Mappatura scene per navigazione rapida
        self.scene_to_indices = {}
        self.sample_to_scene = {}
        for idx, s in enumerate(self.all_samples):
            sc_tok = s['scene_token']
            sc_rec = self.nusc.get('scene', sc_tok)
            sc_name = sc_rec['name']
            if sc_name not in self.scene_to_indices:
                self.scene_to_indices[sc_name] = []
            self.scene_to_indices[sc_name].append(idx)
            self.sample_to_scene[idx] = sc_name

        self.scene_list = list(self.scene_to_indices.keys())
        self.current_idx = initial_idx % self.total_samples
        self.show_boxes = True  # Toggle per mostrare/nascondere bounding box 3D

        # Setup figura matplotlib
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#1E293B'
        plt.rcParams['axes.linewidth'] = 1.0

        self.fig = plt.figure(figsize=(19, 10), facecolor='#0F172A')
        self.fig.canvas.manager.set_window_title("nuScenes Official Visualizer - devkit Explorer")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Timer per sincronizzazione automatica bidirezionale tra visualizzatori
        self.sync_timer = self.fig.canvas.new_timer(interval=300)
        self.sync_timer.add_callback(self.check_sync_file)
        self.sync_timer.start()

        self.render(broadcast=False)

    def get_sample_info(self, idx):
        """Estrae metadati di scena, telemetria e categorie annotate."""
        sample = self.all_samples[idx]
        sc_name = self.sample_to_scene[idx]
        sc_rec = self.nusc.get('scene', sample['scene_token'])
        log_rec = self.nusc.get('log', sc_rec['log_token'])
        location = log_rec.get('location', 'N/A')

        # Calcolo velocità ego da odometria se possibile
        ego_speed = 0.0
        try:
            sd_curr = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            p_curr = self.nusc.get('ego_pose', sd_curr['ego_pose_token'])
            if idx + 1 < self.total_samples:
                s_next = self.all_samples[idx + 1]
                sd_next = self.nusc.get('sample_data', s_next['data']['LIDAR_TOP'])
                p_next = self.nusc.get('ego_pose', sd_next['ego_pose_token'])
                dt = abs(p_next['timestamp'] - p_curr['timestamp']) * 1e-6
                if dt > 1e-4:
                    dist = np.linalg.norm(np.array(p_next['translation'][:2]) - np.array(p_curr['translation'][:2]))
                    ego_speed = (dist / dt) * 3.6
        except Exception:
            ego_speed = 0.0

        # Conteggio annotazioni
        anns = [self.nusc.get('sample_annotation', a) for a in sample['anns']]
        cat_counts = {}
        for a in anns:
            cat = a['category_name'].split('.')[0]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            'sample': sample,
            'scene_name': sc_name,
            'location': location,
            'ego_speed': ego_speed,
            'anns_count': len(anns),
            'cat_counts': cat_counts,
            'timestamp': sample['timestamp']
        }

    def render(self, broadcast=True):
        """Esegue il rendering della visualizzazione in base alla modalità attiva."""
        self.fig.clf()
        if broadcast:
            self._write_sync(self.current_idx)

        info = self.get_sample_info(self.current_idx)
        sample = info['sample']

        # Disegna Header Superiore Informativo
        self._render_header(info)

        if self.mode == 1:
            self._render_mode_surround_bev(sample, info)
        elif self.mode == 2:
            self._render_mode_bev_fullscreen(sample, info)
        elif self.mode == 3:
            self._render_mode_lidar_projection(sample, info)

        # Casella interattiva per saltare a qualsiasi sample (in basso a destra, defilata e pulita)
        ax_tb = self.fig.add_axes([0.93, 0.007, 0.052, 0.024])
        self.txt_frame = TextBox(ax_tb, 'Vai a Sample: ', initial=str(self.current_idx + 1),
                                 color='#0F172A', hovercolor='#1E293B')
        self.txt_frame.label.set_color('#94A3B8')
        self.txt_frame.label.set_fontsize(8.5)
        self.txt_frame.label.set_fontweight('bold')
        self.txt_frame.text_disp.set_color('#38BDF8')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ax_tb = ax_tb

        self.fig.canvas.draw_idle()

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            target = (val - 1) if 1 <= val <= self.total_samples else val
            if 0 <= target < self.total_samples and target != self.current_idx:
                self.current_idx = target
                self.render(broadcast=True)
        except ValueError:
            pass

    def _write_sync(self, idx):
        try:
            os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
            with open(SYNC_FILE, "w") as f:
                f.write(f"{idx},official")
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
                    if source != "official" and target != self.current_idx and 0 <= target < self.total_samples:
                        self.current_idx = target
                        self.render(broadcast=False)
        except Exception:
            pass

    def _render_header(self, info):
        """Renderizza la barra di intestazione con stile dark high-tech."""
        ax_hdr = self.fig.add_axes([0.015, 0.945, 0.97, 0.045])
        ax_hdr.set_facecolor('#1E293B')
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])
        for spine in ax_hdr.spines.values():
            spine.set_color('#334155')

        # Titolo e metadati frame
        title_text = f"NUSCENES OFFICIAL VIEWER | Scena: {info['scene_name']} ({info['location']})"
        meta_text = (f"Sample: {self.current_idx + 1}/{self.total_samples}  |  "
                     f"Token: {info['sample']['token'][:10]}...  |  "
                     f"Velocità Ego: {info['ego_speed']:.1f} km/h  |  "
                     f"Annotazioni 3D: {info['anns_count']}")

        ax_hdr.text(0.012, 0.55, title_text, color='#38BDF8', fontsize=11, fontweight='bold', va='center')
        ax_hdr.text(0.012, 0.20, meta_text, color='#94A3B8', fontsize=8.5, va='center')

        # Comandi e modalità attiva
        ax_hdr.text(0.55, 0.58, "MODALITÀ ATTIVA (Premi 1, 2, 3 per cambiare):", color='#CBD5E1', fontsize=8.5, fontweight='bold', va='center')
        
        m1_col = '#38BDF8' if self.mode == 1 else '#64748B'
        m2_col = '#38BDF8' if self.mode == 2 else '#64748B'
        m3_col = '#38BDF8' if self.mode == 3 else '#64748B'
        m1_w = 'bold' if self.mode == 1 else 'normal'
        m2_w = 'bold' if self.mode == 2 else 'normal'
        m3_w = 'bold' if self.mode == 3 else 'normal'

        ax_hdr.text(0.55, 0.22, "[1] 360° Surround + BEV", color=m1_col, fontsize=8.5, fontweight=m1_w, va='center')
        ax_hdr.text(0.69, 0.22, "[2] Solo BEV HD-Map", color=m2_col, fontsize=8.5, fontweight=m2_w, va='center')
        ax_hdr.text(0.81, 0.22, "[3] Proiezione LiDAR", color=m3_col, fontsize=8.5, fontweight=m3_w, va='center')

        # Indicatore stato Bounding Box 3D
        box_txt = "[B] Box 3D: VISIBILI" if self.show_boxes else "[B] Box 3D: NASCOSTI"
        box_col = "#22C55E" if self.show_boxes else "#EF4444"
        ax_hdr.text(0.985, 0.50, box_txt, color=box_col, fontsize=9.0, fontweight='bold', ha='right', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#0F172A', edgecolor=box_col, linewidth=0.9))

    def _render_mode_surround_bev(self, sample, info):
        """MODALITÀ 1: 6 Camere Surround (3D boxes) + Official BEV HD-Map + Pannello Statistiche."""
        gs = GridSpec(2, 4, figure=self.fig, left=0.015, right=0.985, bottom=0.04, top=0.93,
                      wspace=0.10, hspace=0.18, width_ratios=[1.0, 1.0, 1.0, 1.35])

        # 6 Telecamere surround
        cam_configs = [
            ('CAM_FRONT_LEFT', 0, 0),
            ('CAM_FRONT', 0, 1),
            ('CAM_FRONT_RIGHT', 0, 2),
            ('CAM_BACK_LEFT', 1, 0),
            ('CAM_BACK', 1, 1),
            ('CAM_BACK_RIGHT', 1, 2)
        ]

        for cam_name, r, c in cam_configs:
            ax = self.fig.add_subplot(gs[r, c])
            ax.set_facecolor('#020617')
            try:
                sd_token = sample['data'][cam_name]
                self.nusc.explorer.render_sample_data(sd_token, ax=ax, with_anns=self.show_boxes, verbose=False)
                ax.set_title(cam_name, color='#E2E8F0', fontsize=9.5, fontweight='bold', pad=4)
            except Exception as e:
                ax.text(0.5, 0.5, f"Errore {cam_name}:\n{e}", color='red', ha='center', va='center')
            ax.axis('off')

        # BEV con HD Map e LiDAR Point Cloud
        ax_bev = self.fig.add_subplot(gs[:, 3])
        ax_bev.set_facecolor('#020617')
        try:
            lidar_token = sample['data']['LIDAR_TOP']
            self.nusc.explorer.render_sample_data(lidar_token, ax=ax_bev, with_anns=self.show_boxes, verbose=False,
                                                  underlay_map=True, axes_limit=38.0)
            self._overlay_crosswalk(ax_bev, sample, 38.0)
            ax_bev.set_title("Official LIDAR_TOP & Vector HD-Map (BEV)", color='#38BDF8',
                             fontsize=11, fontweight='bold', pad=6)
        except Exception as e:
            ax_bev.text(0.5, 0.5, f"Errore LIDAR_TOP:\n{e}", color='red', ha='center', va='center')

        # Barra di istruzioni inferiore
        self.fig.text(0.015, 0.012,
                      "[<- / ->] Frame  |  [Up / Down] Scena  |  [1/2/3] Modalità  |  [B] Bounding Box ON/OFF  |  [S] Salva HD",
                      color='#94A3B8', fontsize=8.5, ha='left')

    def _overlay_crosswalk(self, ax, sample, axes_limit):
        """Sovrappone le strisce pedonali vettoriali (ped_crossing) con pattern zebrato ad alta leggibilità."""
        try:
            lidar_token = sample['data']['LIDAR_TOP']
            sd = self.nusc.get('sample_data', lidar_token)
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            
            loc = log['location']
            if not hasattr(self, '_map_cache'):
                self._map_cache = {}
            if loc not in self._map_cache:
                from nuscenes.map_expansion.map_api import NuScenesMap
                self._map_cache[loc] = NuScenesMap(dataroot=self.dataroot, map_name=loc)
            nmap = self._map_cache[loc]
            
            from pyquaternion import Quaternion
            yaw_deg = np.degrees(Quaternion(pose['rotation']).yaw_pitch_roll[0])
            patch_box = (pose['translation'][0], pose['translation'][1], axes_limit * 2, axes_limit * 2)
            masks = nmap.get_map_mask(patch_box, yaw_deg, ['ped_crossing'], canvas_size=(200, 200))
            crossing = masks[0]
            
            if np.any(crossing):
                rgba = np.zeros((200, 200, 4), dtype=np.uint8)
                xx, yy = np.meshgrid(np.arange(200), np.arange(200))
                stripe_diag = ((xx + yy) % 4) < 2
                from scipy import ndimage
                border = crossing & (~ndimage.binary_erosion(crossing, structure=np.ones((3, 3))))
                
                rgba[crossing == 1] = [90, 90, 90, 255]
                rgba[(crossing == 1) & stripe_diag] = [255, 255, 255, 255]
                rgba[border == 1] = [255, 255, 255, 255]
                
                ax.imshow(rgba, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], origin='lower', zorder=1.5)
        except Exception:
            pass

    def _render_mode_bev_fullscreen(self, sample, info):
        """MODALITÀ 2: BEV HD Map a tutto schermo con Radar fuso e lista ostacoli."""
        gs = GridSpec(1, 4, figure=self.fig, left=0.02, right=0.98, bottom=0.05, top=0.90,
                      wspace=0.15, width_ratios=[2.5, 0.7, 0.7, 0.7])

        # Grande BEV centrale
        ax_bev = self.fig.add_subplot(gs[0, 0:3])
        ax_bev.set_facecolor('#020617')
        try:
            lidar_token = sample['data']['LIDAR_TOP']
            self.nusc.explorer.render_sample_data(lidar_token, ax=ax_bev, with_anns=self.show_boxes, verbose=False,
                                                  underlay_map=True, axes_limit=45.0)
            self._overlay_crosswalk(ax_bev, sample, 45.0)
            ax_bev.set_title(f"nuScenes Official LiDAR BEV + Vector HD Map Underlay (Range 45m)",
                             color='#38BDF8', fontsize=12, fontweight='bold', pad=8)
        except Exception as e:
            ax_bev.text(0.5, 0.5, f"Errore BEV:\n{e}", color='red', ha='center', va='center')

        # Pannello laterale categorie e ostacoli
        ax_side = self.fig.add_subplot(gs[0, 3])
        ax_side.set_facecolor('#1E293B')
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        for spine in ax_side.spines.values():
            spine.set_color('#334155')

        ax_side.text(0.5, 0.96, "CLASSI ANNOTATE", color='#F8FAFC', fontsize=11, fontweight='bold', ha='center')
        ax_side.axhline(0.93, color='#475569', linewidth=0.8)

        y_pos = 0.88
        for cat, cnt in sorted(info['cat_counts'].items(), key=lambda x: -x[1]):
            ax_side.text(0.08, y_pos, cat.capitalize(), color='#E2E8F0', fontsize=9.5, fontweight='bold')
            ax_side.text(0.88, y_pos, str(cnt), color='#38BDF8', fontsize=9.5, fontweight='bold', ha='right')
            y_pos -= 0.05

        ax_side.axhline(y_pos, color='#475569', linewidth=0.8)
        y_pos -= 0.04
        ax_side.text(0.08, y_pos, "Totale Oggetti:", color='#F8FAFC', fontsize=10, fontweight='bold')
        ax_side.text(0.88, y_pos, str(info['anns_count']), color='#22C55E', fontsize=10, fontweight='bold', ha='right')

        # Dettagli Radar
        y_pos -= 0.08
        ax_side.text(0.5, y_pos, "SENSOR SUITE", color='#F8FAFC', fontsize=10, fontweight='bold', ha='center')
        y_pos -= 0.04
        ax_side.text(0.08, y_pos, "LiDAR:", color='#94A3B8', fontsize=8.5)
        ax_side.text(0.88, y_pos, "32 beams (20 Hz)", color='#E2E8F0', fontsize=8.5, ha='right')
        y_pos -= 0.035
        ax_side.text(0.08, y_pos, "Telecamere:", color='#94A3B8', fontsize=8.5)
        ax_side.text(0.88, y_pos, "6x RGB (12 Hz)", color='#E2E8F0', fontsize=8.5, ha='right')
        y_pos -= 0.035
        ax_side.text(0.08, y_pos, "RADAR:", color='#94A3B8', fontsize=8.5)
        ax_side.text(0.88, y_pos, "5x 77GHz FMCW", color='#E2E8F0', fontsize=8.5, ha='right')

        self.fig.text(0.015, 0.012,
                      "[<- / ->] Frame  |  [Up / Down] Scena  |  [1] Surround  |  [2] BEV  |  [3] Proiezione LiDAR  |  [S] Salva HD",
                      color='#94A3B8', fontsize=8.5, ha='left')

    def _render_mode_lidar_projection(self, sample, info):
        """MODALITÀ 3: Proiezione ufficiale punti LiDAR su telecamere con scala di profondità."""
        gs = GridSpec(2, 3, figure=self.fig, left=0.015, right=0.985, bottom=0.05, top=0.93,
                      wspace=0.08, hspace=0.15)

        cams = [
            ('CAM_FRONT_LEFT', 0, 0),
            ('CAM_FRONT', 0, 1),
            ('CAM_FRONT_RIGHT', 0, 2),
            ('CAM_BACK_LEFT', 1, 0),
            ('CAM_BACK', 1, 1),
            ('CAM_BACK_RIGHT', 1, 2)
        ]

        for cam_name, r, c in cams:
            ax = self.fig.add_subplot(gs[r, c])
            ax.set_facecolor('#020617')
            try:
                self.nusc.explorer.render_pointcloud_in_image(sample['token'],
                                                              pointsensor_channel='LIDAR_TOP',
                                                              camera_channel=cam_name,
                                                              ax=ax, verbose=False, dot_size=5)
                ax.set_title(f"{cam_name} (LiDAR Depth Colormap)", color='#F8FAFC',
                             fontsize=9.5, fontweight='bold', pad=4)
            except Exception as e:
                ax.text(0.5, 0.5, f"Errore {cam_name}:\n{e}", color='red', ha='center', va='center')
            ax.axis('off')

        self.fig.text(0.015, 0.012,
                      "Scala Colore LiDAR: Viola/Blu (0-10m) -> Verde/Ciano (10-25m) -> Giallo/Rosso (>25m)  |  [S] Salva HD",
                      color='#94A3B8', fontsize=8.5, ha='left')

    def on_key(self, event):
        """Gestione degli eventi da tastiera per navigazione interattiva."""
        # Ignora se l'utente sta digitando nella casella di testo
        if hasattr(self, 'ax_tb') and event.inaxes == self.ax_tb:
            return

        if event.key in ['right', 'd', 'D']:
            self.current_idx = (self.current_idx + 1) % self.total_samples
            self.render()
        elif event.key in ['left', 'a', 'A']:
            self.current_idx = (self.current_idx - 1) % self.total_samples
            self.render()
        elif event.key in ['up', 'w', 'W']:
            curr_scene = self.sample_to_scene[self.current_idx]
            curr_s_idx = self.scene_list.index(curr_scene)
            next_scene = self.scene_list[(curr_s_idx + 1) % len(self.scene_list)]
            self.current_idx = self.scene_to_indices[next_scene][0]
            print(f">>> Passaggio a scena successiva: {next_scene} (Frame {self.current_idx})")
            self.render()
        elif event.key in ['down', 'x', 'X']:
            curr_scene = self.sample_to_scene[self.current_idx]
            curr_s_idx = self.scene_list.index(curr_scene)
            prev_scene = self.scene_list[(curr_s_idx - 1) % len(self.scene_list)]
            self.current_idx = self.scene_to_indices[prev_scene][0]
            print(f">>> Passaggio a scena precedente: {prev_scene} (Frame {self.current_idx})")
            self.render()
        elif event.key == '1':
            self.mode = 1
            print(">>> Modalità 1: Surround 360° Multi-Sensor + BEV HD-Map")
            self.render()
        elif event.key == '2':
            self.mode = 2
            print(">>> Modalità 2: Dedicated BEV + HD Map Vettoriale Fullscreen")
            self.render()
        elif event.key == '3':
            self.mode = 3
            print(">>> Modalità 3: Proiezione Ufficiale Depth LiDAR su Telecamere")
            self.render()
        elif event.key in ['b', 'B']:
            self.show_boxes = not self.show_boxes
            stato = "ATTIVI (Visibili)" if self.show_boxes else "DISATTIVATI (Nascosti)"
            print(f">>> [TOGGLE BOX] Bounding Box 3D: {stato}")
            self.render()
        elif event.key in ['s', 'S', 'p', 'P']:
            out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"fig_official_nuscenes_mode{self.mode}_sample{self.current_idx}.png")
            self.fig.savefig(out_path, dpi=300, facecolor=self.fig.get_facecolor(), bbox_inches='tight')
            print(f">>> [SALVATAGGIO] Figura HD 300 DPI salvata con successo in:\n    {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualizzatore Ufficiale nuScenes - Explorer Interattivo")
    parser.add_argument("sample_pos", nargs="?", type=int, default=None, help="Numero del campione iniziale a cui saltare (1-404)")
    parser.add_argument("--dataroot", "-d", type=str, default="./nuscenes", help="Percorso del dataset nuScenes")
    parser.add_argument("--version", "-v", type=str, default="v1.0-mini", help="Versione del dataset nuScenes")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Indice del sample iniziale (0-403)")
    parser.add_argument("--scene", type=str, default=None, help="Nome della scena specifica da caricare (es. scene-0061)")
    parser.add_argument("--mode", "-m", type=int, default=1, choices=[1, 2, 3],
                        help="Modalità iniziale: 1=Surround+BEV, 2=Solo BEV Map, 3=LiDAR su Cam")
    parser.add_argument("--hide_boxes", action="store_true", help="Avvia il visualizzatore nascondendo i box 3D (vista pulita)")
    parser.add_argument("--save", action="store_true", help="Salva l'immagine a 300 DPI ed esci (per script/tesi)")
    parser.add_argument("--out", type=str, default=None, help="Percorso di salvataggio personalizzato")
    args = parser.parse_args()

    init_s = args.sample_pos if args.sample_pos is not None else (args.sample if args.sample is not None else 0)
    target_idx = (init_s - 1) if init_s >= 1 and args.sample_pos is not None else init_s

    app = NuScenesOfficialVisualizer(dataroot=args.dataroot, version=args.version, initial_idx=target_idx, mode=args.mode)
    if args.hide_boxes:
        app.show_boxes = False
        app.render()

    if args.scene is not None:
        if args.scene in app.scene_to_indices:
            app.current_idx = app.scene_to_indices[args.scene][0]
            app.render()
        else:
            print(f"Attenzione: Scena '{args.scene}' non trovata. Scelte disponibili: {app.scene_list}")

    if args.save:
        out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
        os.makedirs(out_dir, exist_ok=True)
        out_path = args.out if args.out else os.path.join(out_dir, f"fig_official_nuscenes_mode{app.mode}.png")
        app.fig.savefig(out_path, dpi=300, facecolor=app.fig.get_facecolor(), bbox_inches='tight')
        print(f">>> [SALVATAGGIO DIRETTO] Immagine salvata in: {out_path}")
        return

    print("\nVisualizzatore Avviato:")
    print("  [<- / ->] oppure [A / D]: Scorrimento campioni nuScenes")
    print("  [Up / Down]: Salto alla scena successiva / precedente")
    print("  [1] / [2] / [3]: Cambio modalità visualizzazione")
    print("  [B]: Mostra / Nascondi i Bounding Box 3D (immagini pulite)")
    print("  [S]: Salva screenshot HD a 300 DPI in documentazione/immagini_tesi/\n")

    plt.show()


if __name__ == "__main__":
    main()
