# VISUALIZZATORE UFFICIALE TESI - VALUTAZIONE PRESTAZIONI & METRICHE (F1 / RECALL / PRECISION)
# File: visualizzatori/vis_valutazione_prestazioni.py
#
# Dashboard Scientifica di Valutazione Prestazioni (Figure 5 Tesi):
# - SINISTRA: Mappa BEV a 25m con classificazione visiva delle zone d'ombra:
#             * 🟢 VERDE (True Positive - TP): Pericolo anticipato confermato dalla GT!
#             * 🔴 ROSSO (False Positive - FP): Allarme preventivo non presente nella GT;
#             * 🟣 VIOLA TRATTEGGIATO (False Negative - FN): Pericolo della GT mancato;
#             * ⚪ GRIGIO (True Negative - TN): Spazio libero confermato.
# - DESTRA ALTO: Selettore Modello AI & Target GT (Pill buttons a selezione istantanea) + Telemetria;
# - DESTRA CENTRO: Scorecard F1-Score Globale & Istogramma Prestazioni per Classe;
# - DESTRA BASSO: Matrice di Confusione Compatta (TP, FP, FN) + Pulsante Confronto 1-a-1;
# - FINESTRA CONFRONTO 1-A-1: Apre un nuovo frame/finestra side-by-side con score attuale
#                            e un altro score/modello selezionabile direttamente dal menu.


import os
import sys
import time
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle, FancyBboxPatch
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
    rasterize_polygon,
    get_occlusion_ground_truth_target,
    extract_real_occlusion_ground_truth,
    clean_occlusion_polygon
)
from ground_truth.ground_truth_extractor_synthetic import compute_synthetic_ground_truth
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel

SYNC_FILE = os.path.join(ROOT_DIR, "scratch", "sync_frame.txt")
GRID_DIM = 200
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
CAT_NAMES_4 = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]

MODEL_CONFIGS = {
    "NEURO_SIMB": {
        "title": "Supervisione Ibrida: Rete Neurale con Vincoli Simbolici (Proposto)",
        "short": "★ Con Vincoli Simbolici",
        "macro": "NEURO_SIMB",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
        "badge_color": "#2563EB",
        "badge_bg": "#DBEAFE",
    },
    "PURE_NEURAL": {
        "title": "Supervisione Ibrida: Studio di Ablazione Senza Vincoli Simbolici",
        "short": "Senza Vincoli (Ablazione)",
        "macro": "NEURO_SIMB",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_pure_neural.pth"),
        "badge_color": "#475569",
        "badge_bg": "#F1F5F9",
    },
    "REAL_GT": {
        "title": "Supervisione Reale (nuScenes 3D): Rete Neurale con Vincoli Simbolici",
        "short": "Con Vincoli Simbolici",
        "macro": "REAL",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth"),
        "badge_color": "#0D9488",
        "badge_bg": "#CCFBF1",
    },
    "REAL_PURE_NEURAL": {
        "title": "Supervisione Reale: Studio di Ablazione Senza Vincoli Simbolici",
        "short": "Senza Vincoli (Ablazione)",
        "macro": "REAL",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_pure_neural.pth"),
        "badge_color": "#0F766E",
        "badge_bg": "#E6FFFA",
    },
    "POS_ONLY": {
        "title": "Supervisione Reale: Sole Zone Piene con Oggetto (Positives-Only)",
        "short": "Solo Zone Piene",
        "macro": "REAL",
        "ckpt": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_positive_only.pth"),
        "badge_color": "#D97706",
        "badge_bg": "#FEF3C7",
    },
    "BAYES": {
        "title": "Baseline Analitica: Probabilità Condizionata Bayesiana (No AI)",
        "short": "Modello Analitico (No AI)",
        "macro": "BAYES",
        "ckpt": None,
        "badge_color": "#0284C7",
        "badge_bg": "#E0F2FE",
    }
}
# Alias per retrocompatibilità
MODEL_CONFIGS["HYBRID"] = MODEL_CONFIGS["NEURO_SIMB"]
MODEL_CONFIGS["NO_NEURO"] = MODEL_CONFIGS["PURE_NEURAL"]
MODEL_CONFIGS["REAL_NO_NEURO"] = MODEL_CONFIGS["REAL_PURE_NEURAL"]

MACRO_CATEGORIES = {
    "NEURO_SIMB": {
        "label": "★ Addestramento Ibrido (Anticipazione)",
        "models": [("NEURO_SIMB", "★ Con Vincoli Simbolici (Full)"), ("PURE_NEURAL", "Senza Vincoli Simbolici (Puro)")],
        "color": "#1D4ED8",
        "bg": "#DBEAFE"
    },
    "REAL": {
        "label": "◆ Addestramento Reale (nuScenes)",
        "models": [("REAL_GT", "Con Vincoli Simbolici (Full)"), ("REAL_PURE_NEURAL", "Senza Vincoli Simbolici (Puro)"), ("POS_ONLY", "Solo Zone Piene")],
        "color": "#0D9488",
        "bg": "#CCFBF1"
    },
    "BAYES": {
        "label": "▲ Bayes",
        "models": [("BAYES", "Modello Analitico (No AI)")],
        "color": "#0284C7",
        "bg": "#E0F2FE"
    }
}


def percorso_ckpt(model_key, pesi_dir=None):
    """Restituisce il percorso assoluto del checkpoint del modello (None per la baseline BAYES)."""
    cfg = MODEL_CONFIGS.get(model_key)
    ckpt = cfg.get("ckpt") if cfg else None
    if not ckpt:
        return None
    filename = os.path.basename(ckpt)
    if pesi_dir and os.path.exists(os.path.join(pesi_dir, filename)):
        return os.path.join(pesi_dir, filename)
    if os.path.exists(os.path.join("D:/pesi_modelli", filename)):
        return os.path.join("D:/pesi_modelli", filename)
    if os.path.exists(os.path.join(ROOT_DIR, "pesi_modelli", filename)):
        return os.path.join(ROOT_DIR, "pesi_modelli", filename)
    return ckpt if os.path.isabs(ckpt) else os.path.join(ROOT_DIR, ckpt)


def modello_disponibile(model_key, pesi_dir=None):
    """True se i pesi del modello esistono su disco (BAYES è analitico: sempre disponibile)."""
    if model_key == "BAYES":
        return True
    p = percorso_ckpt(model_key, pesi_dir)
    return p is not None and os.path.exists(p)

GT_CONFIGS = {
    "NEURO_SIMB": {
        "name": "GT Neuro-Simbolica (Spazio + HD-Map)",
        "short": "GT Neuro-Simb. ★",
        "desc": "Spazio 3D + Semantica HD-Map",
        "color": "#2563EB",
        "bg": "#DBEAFE"
    },
    "REAL": {
        "name": "GT Reale nuScenes (3D)",
        "short": "GT Reale 3D",
        "desc": "Ostacoli fisici reali annotati",
        "color": "#7C3AED",
        "bg": "#EDE9FE"
    },
    "GEOMETRIC": {
        "name": "GT Sintetica Geometrica",
        "short": "GT Geometrica",
        "desc": "Fitting 3D volumetrico puro",
        "color": "#16A34A",
        "bg": "#DCFCE7"
    },
    "SEMANTIC": {
        "name": "GT Sintetica Semantica",
        "short": "GT Semantica",
        "desc": "Codice della strada puro",
        "color": "#D97706",
        "bg": "#FEF3C7"
    }
}
GT_CONFIGS["HYBRID"] = GT_CONFIGS["NEURO_SIMB"]


def to_macro_classes_4(arr_6):
    """Mappa le 6 classi elementari nelle 3 macro-classi standard da tesi [Auto, Camion/Bus, VRU]."""
    arr = np.array(arr_6, dtype=float)
    c0 = float(arr[0] > 0.5)
    c1 = float(arr[1] > 0.5)
    c2 = float(np.max(arr[2:5]) > 0.5) if len(arr) >= 5 else 0.0
    return np.array([c0, c1, c2], dtype=float)


class EvaluationDashboardVisualizer:
    def __init__(self, max_range=25.0, initial_model="NEURO_SIMB", initial_gt="NEURO_SIMB",
                 initial_threshold=0.50, cache_path=None, dataroot=None, occlusions_dir=None,
                 pesi_dir=None, headless=False):
        print("\n" + "=" * 80)
        print("   VISUALIZZATORE TESI: DASHBOARD PRESTAZIONI & METRICHE UFFICIALI (BEV 25M)")
        print("=" * 80)

        self.max_range = float(max_range)
        self.cache_path_forzato = cache_path
        self.headless = headless
        self.current_threshold = float(initial_threshold)
        self.eval_mode = "PER_CLASS"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Dispositivo di Calcolo: {self.device}")

        # Risoluzione percorsi dataset nuScenes, occlusioni e pesi
        if dataroot is None:
            dataroot = os.environ.get("NUSCENES_DATAROOT", "D:/nuScenes" if os.path.exists("D:/nuScenes") else "./nuscenes")
        if occlusions_dir is None:
            occlusions_dir = os.environ.get("OCCLUSIONS_OUTPUT", "D:/extracted_occlusions" if os.path.exists("D:/extracted_occlusions") else "extracted_occlusions")
        if pesi_dir is None:
            pesi_dir = os.environ.get("PESI_DIR", "D:/pesi_modelli" if os.path.exists("D:/pesi_modelli") else "pesi_modelli")

        self.dataroot = dataroot
        self.occlusions_dir = occlusions_dir
        self.pesi_dir = pesi_dir

        print(f"• Percorso nuScenes:   {self.dataroot}")
        print(f"• Percorso Occlusioni: {self.occlusions_dir}")
        print(f"• Percorso Pesi:       {self.pesi_dir}")

        # Inizializzazione dataset nuScenes
        self.adapter = create_adapter("nuscenes", self.dataroot)
        self.total_frames = self.adapter.get_num_samples()
        self.current_idx = 0
        self.current_model_key = "NEURO_SIMB" if initial_model.upper() in ["HYBRID", "NEURO_SIMB"] else initial_model.upper()
        self.current_gt_key = "NEURO_SIMB" if initial_gt.upper() in ["HYBRID", "NEURO_SIMB"] else initial_gt.upper()

        self.current_macro = "NEURO_SIMB"
        for m_k, m_info in MACRO_CATEGORIES.items():
            if any(self.current_model_key == child[0] for child in m_info["models"]):
                self.current_macro = m_k
                break

        # Split standard nuScenes trainval (train vs val inedito)
        from nuscenes.utils.splits import create_splits_scenes
        splits_dict = create_splits_scenes()
        prefix = "mini_" if getattr(self.adapter.nusc, 'version', '') == 'v1.0-mini' else ""
        self.train_scenes = set(splits_dict.get(f"{prefix}train", []))
        self.val_scenes = set(splits_dict.get(f"{prefix}val", []))

        self.train_indices = []
        self.val_indices = []
        for i in range(self.total_frames):
            s = self.adapter.all_samples[i]
            sc = self.adapter.nusc.get('scene', s['scene_token'])['name']
            if sc in self.val_scenes:
                self.val_indices.append(i)
            else:
                self.train_indices.append(i)

        # Caricamento cache di valutazione ufficiale (se non presente la genera in automatico)
        self.eval_cache = self._load_or_generate_eval_cache()

        # Se la cache è di validazione (o abbiamo split val), naviga nei frame inediti
        cache_split = (self.eval_cache.get("split", "") if self.eval_cache else "").lower()
        if cache_split == "val" or len(self.val_indices) > 0:
            self.nav_val_only = True
        else:
            self.nav_val_only = False

        # Se il frame iniziale 0 non è nella cache di validazione, posizionati sul primo frame valutato
        if self.eval_cache and "frames" in self.eval_cache and len(self.eval_cache["frames"]) > 0:
            eval_frame_keys = [int(k) for k in self.eval_cache["frames"].keys()]
            if self.current_idx not in eval_frame_keys:
                self.current_idx = min(eval_frame_keys)

        # Stato per finestra di confronto 1-a-1 e visualizzazione errori
        self.fig_cmp = None
        self.cmp_model_a_key = self.current_model_key
        self.cmp_model_b_key = "REAL_GT" if self.current_model_key != "REAL_GT" else "NEURO_SIMB"
        self.cmp_gt_key = self.current_gt_key
        self.cmp_txt_frame = None
        self.highlight_fn = False

        # Stato per finestra recap globale (Benchmark Validazione VAL)
        self.fig_recap = None
        self.recap_model_key = self.current_model_key
        self.recap_gt_key = self.current_gt_key
        self.recap_macro_key = self.current_macro
        self.recap_ui_buttons = []

        # Setup stile tipografico accademico
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.edgecolor'] = '#0F172A'
        plt.rcParams['axes.linewidth'] = 1.1

        self.fig = plt.figure(figsize=(18, 9.2), facecolor='#FFFFFF')
        self.fig.canvas.manager.set_window_title("nuScenes BEV - Official Performance & Metrics Dashboard")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.drawn_occlusions = []
        self.active_hovered = None
        self.tooltip = None
        self.ui_buttons = []
        self.inference_latency_ms = 0.0

        # Cache modelli neurali
        self.loaded_models = {}
        self.missing_reported = set()
        self._preload_model(self.current_model_key)
        self._preload_model(self.cmp_model_a_key)
        self._preload_model(self.cmp_model_b_key)

        self.load_icons()
        self.load_frame(self.current_idx, broadcast=False)

        # Timer sincronizzazione bidirezionale (non avviato in modalita' headless/export)
        self.sync_timer = None
        if not self.headless:
            self.sync_timer = self.fig.canvas.new_timer(interval=150)
            self.sync_timer.add_callback(self.check_sync_file)
            self.sync_timer.start()

    def _load_or_generate_eval_cache(self):
        """Carica la cache di valutazione JSON precalcolata (la piu' recente tra quelle presenti,
        oppure quella forzata con --cache); se non ne esiste nessuna, la genera invocando evaluate_final_official.py."""
        eval_dir = os.path.join(ROOT_DIR, "valutazione")
        cache_path = None

        if self.cache_path_forzato:
            # Cache forzata da riga di comando (--cache PATH)
            cache_path = os.path.abspath(self.cache_path_forzato)
            if not os.path.exists(cache_path):
                print(f"[ERRORE] Cache indicata con --cache non trovata: {cache_path}")
                return None
            print(f"• Cache di valutazione forzata da riga di comando: {cache_path}")
        else:
            # Tra le cache presenti sceglie quella modificata piu' di recente
            candidati = [os.path.join(eval_dir, n) for n in
                         ("cache_valutazione_ufficiale.json", "cache_valutazione_val.json", "cache_valutazione_all.json")]
            presenti = [p for p in candidati if os.path.exists(p)]
            if presenti:
                presenti.sort(key=os.path.getmtime, reverse=True)
                cache_path = presenti[0]
                print("• Cache di valutazione disponibili (dalla piu' recente):")
                for p in presenti:
                    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
                    print(f"   {'->' if p == cache_path else '  '} {os.path.basename(p)}  ({ts})")

        if cache_path is None:
            print("\n" + "=" * 80)
            print(" [CACHE NON TROVATA]: Generazione automatica cache con evaluate_final_official.py...")
            print("=" * 80)
            eval_script = os.path.join(eval_dir, "evaluate_final_official.py")
            try:
                import subprocess
                subprocess.run([sys.executable, eval_script, "--mode", "all", "--split", "all"], check=True)
                cache_path = os.path.join(eval_dir, "cache_valutazione_ufficiale.json")
            except Exception as e:
                print(f"[ERRORE GENERAZIONE CACHE]: {e}")
                return None

        if os.path.exists(cache_path):
            try:
                cache_mtime = os.path.getmtime(cache_path)
                # Verifica se qualche checkpoint in pesi_modelli è più recente della cache
                newer_ckpts = []
                for m_k, cfg in MODEL_CONFIGS.items():
                    p = percorso_ckpt(m_k, getattr(self, 'pesi_dir', None))
                    if p and os.path.exists(p) and os.path.getmtime(p) > cache_mtime:
                        newer_ckpts.append(os.path.basename(p))

                if newer_ckpts:
                    print("\n" + "!" * 80)
                    print(f" [AVVISO PESI AGGIORNATI]: I seguenti modelli sono più recenti della cache:")
                    for c in set(newer_ckpts):
                        print(f"   - {c}")
                    print(" Per ricalcolare il benchmark con i nuovi pesi esegui:")
                    print("   python valutazione/evaluate_final_official.py --mode all --split all")
                    print(f" Cache in uso: {os.path.basename(cache_path)}")
                    print("!" * 80 + "\n")

                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_frames = len(data.get("frames", {}))
                print(f"• [CACHE CARICATA]: {cache_path}  ({num_frames} fotogrammi memorizzati)")
                return data
            except Exception as e:
                print(f"[ATTENZIONE] Impossibile leggere la cache: {e}")
                return None
        return None

    def _preload_model(self, model_key):
        if model_key == "BAYES":
            return "BAYES"
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        # Pesi mancanti: il modello e' MANCANTE (nessun fallback su altri checkpoint)
        ckpt_path = percorso_ckpt(model_key, getattr(self, 'pesi_dir', None))
        if not modello_disponibile(model_key, getattr(self, 'pesi_dir', None)):
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

    def _cache_ha_modello(self, model_key, gt_key, idx=None):
        """True se la cache ufficiale contiene risultati per modello/GT (frame idx, o almeno un frame se idx=None)."""
        if not self.eval_cache or "frames" not in self.eval_cache:
            return False
        m_k = "NEURO_SIMB" if model_key in ["HYBRID", "NEURO_SIMB"] else model_key
        g_k = "NEURO_SIMB" if gt_key in ["HYBRID", "NEURO_SIMB"] else gt_key
        frames_cache = self.eval_cache["frames"]
        if idx is not None:
            frames_iter = [frames_cache.get(str(idx)) or {}]
        else:
            frames_iter = frames_cache.values()
        for f_data in frames_iter:
            if g_k in f_data.get("models", {}).get(m_k, {}):
                return True
        return False

    def modello_mancante(self, model_key, gt_key):
        """True se il modello non ha pesi su disco e nemmeno risultati in cache per il frame corrente."""
        if modello_disponibile(model_key, getattr(self, 'pesi_dir', None)):
            return False
        return not self._cache_ha_modello(model_key, gt_key, self.current_idx)

    def _disegna_avviso_mancante(self, ax, model_key):
        """Avviso centrato sulla BEV quando i pesi del modello non esistono."""
        ckpt_nome = os.path.basename(percorso_ckpt(model_key, getattr(self, 'pesi_dir', None)) or model_key)
        ax.text(0.5, 0.5, f"Pesi del modello mancanti:\n{ckpt_nome}",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                color='#475569', ha='center', va='center', zorder=40,
                bbox=dict(boxstyle="round,pad=0.6", facecolor='#F1F5F9',
                          edgecolor='#64748B', linewidth=1.2, alpha=0.95))

    def load_icons(self):
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
            'car_red': Image.open(os.path.join(icons_dir, "car_red.png")),
            'pedestrian_red': Image.open(os.path.join(icons_dir, "pedestrian_red.png")),
            'truck_red': Image.open(os.path.join(icons_dir, "truck_red.png")),
            'trailer_red': Image.open(os.path.join(icons_dir, "trailer_red.png")),
            'barrier_red': Image.open(os.path.join(icons_dir, "barrier_red.png")),
            'bicycle_red': Image.open(os.path.join(icons_dir, "bicycle_red.png")),
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

            if dt <= 0:
                return 0.0
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
                    if source != "perf_eval" and target_idx != self.current_idx and 0 <= target_idx < self.total_frames:
                        self.load_frame(target_idx, broadcast=False)
        except Exception:
            pass

    def _write_sync(self, idx):
        try:
            os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
            with open(SYNC_FILE, "w") as f:
                f.write(f"{idx},perf_eval")
        except Exception:
            pass

    def load_frame(self, idx, broadcast=True):
        self.current_idx = max(0, min(self.total_frames - 1, idx))
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

        try:
            rc = RayCaster(self.frame_data, verbose=False)
            self.static_boxes = rc.detect_static_manmade_boxes(self.frame_data.get('boxes', []))
        except Exception:
            self.static_boxes = []

        # Valutazione modello principale
        (self.inferred_occlusions, self.frame_stats,
         self.inference_latency_ms, self.f1_glob,
         self.rec_glob, self.prec_glob) = self.evaluate_frame(self.current_model_key, self.current_gt_key)

        self.render()

        # Se la finestra di confronto 1-a-1 è aperta, aggiornala in tempo reale
        if self.fig_cmp is not None and plt.fignum_exists(self.fig_cmp.number):
            self.render_comparison_window()

    def evaluate_frame(self, model_key, gt_key):
        """Restituisce le metriche e le zone inferite leggendo dalla cache ufficiale o calcolando come fallback."""
        model_k = "NEURO_SIMB" if model_key in ["HYBRID", "NEURO_SIMB"] else model_key
        gt_k = "NEURO_SIMB" if gt_key in ["HYBRID", "NEURO_SIMB"] else gt_key

        start_time = time.perf_counter()

        # 1. LETTURA DIRETTA DALLA CACHE UFFICIALE PRECALCOLATA
        if self.eval_cache and "frames" in self.eval_cache:
            frames_cache = self.eval_cache["frames"]
            f_data = frames_cache.get(str(self.current_idx))
            if f_data and "models" in f_data:
                m_cache = f_data["models"].get(model_k)
                if m_cache and gt_k in m_cache:
                    cached_eval = m_cache[gt_k]
                    th = getattr(self, 'current_threshold', 0.50)
                    cached_zones = cached_eval.get("zones", [])
                    cat_short = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]
                    frame_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0} for c in range(3)}

                    occ_file = os.path.join(self.occlusions_dir, f"{self.sample_token}.json")
                    occs = []
                    if os.path.exists(occ_file):
                        with open(occ_file, "r") as f:
                            d = json.load(f)
                            occs = d.get("occlusions", []) if isinstance(d, dict) else d

                    boxes_by_token = {b.token: b for b in self.frame_data.get('boxes', [])}
                    for sb in getattr(self, 'static_boxes', []):
                        boxes_by_token[sb.token] = sb

                    circle_25m = Point(0, 0).buffer(self.max_range - 0.4)
                    inferred = []
                    for cz in cached_zones:
                        occ_idx = cz["occ_idx"]
                        if occ_idx >= len(occs):
                            continue
                        occ = occs[occ_idx]
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

                        b_obj = boxes_by_token.get(occ.get("object_token"))
                        clean_polys = clean_occlusion_polygon(
                            poly_vis,
                            occluder_box=b_obj,
                            static_boxes=getattr(self, 'static_boxes', []),
                            min_area=0.25,
                            min_width=0.30
                        )
                        if not clean_polys:
                            continue

                        poly_main = clean_polys[0]
                        coords = np.array(poly_main.exterior.coords)[:-1]
                        if len(coords) < 3:
                            continue

                        pt = poly_main.representative_point()
                        cx_m, cy_m = float(pt.x), float(pt.y)

                        probs_3 = cz.get("probs_4", [0.0, 0.0, 0.0])[:3]
                        gt_b = cz.get("gt_target_4", [0, 0, 0])[:3]
                        pred_b = [1 if probs_3[c] >= th else 0 for c in range(3)]
                        pred_classes = [c for c in range(3) if pred_b[c] == 1]
                        gt_classes = [c for c in range(3) if gt_b[c] == 1]

                        mode = getattr(self, 'eval_mode', "PER_CLASS")
                        if mode == "ZONE_LEVEL":
                            is_pred_h = len(pred_classes) > 0
                            is_gt_h = len(gt_classes) > 0
                            if is_pred_h:
                                m = [c for c in pred_classes if c in gt_classes]
                                if m:
                                    verdict = "TP"
                                    verdict_label = f"TRUE POSITIVE (Pericolo Zona: {cat_short[m[0]]})"
                                    frame_stats[m[0]]["tp"] += 1
                                    frame_stats[m[0]]["gt_count"] += 1
                                    frame_stats[m[0]]["pred_count"] += 1
                                    z_tps = [m[0]]
                                    z_fps = []
                                    z_fns = []
                                else:
                                    verdict = "FP"
                                    verdict_label = f"FALSE POSITIVE ({cat_short[pred_classes[0]]})"
                                    frame_stats[pred_classes[0]]["fp"] += 1
                                    frame_stats[pred_classes[0]]["pred_count"] += 1
                                    z_tps = []
                                    z_fps = [pred_classes[0]]
                                    if is_gt_h:
                                        frame_stats[gt_classes[0]]["fn"] += 1
                                        frame_stats[gt_classes[0]]["gt_count"] += 1
                                        z_fns = [gt_classes[0]]
                                    else:
                                        z_fns = []
                            elif is_gt_h:
                                verdict = "FN"
                                verdict_label = f"FALSE NEGATIVE ({cat_short[gt_classes[0]]})"
                                frame_stats[gt_classes[0]]["fn"] += 1
                                frame_stats[gt_classes[0]]["gt_count"] += 1
                                z_tps = []
                                z_fps = []
                                z_fns = [gt_classes[0]]
                            else:
                                verdict = "TN"
                                verdict_label = "TRUE NEGATIVE (Zona Libera)"
                                z_tps = []
                                z_fps = []
                                z_fns = []
                        else:
                            for c in range(3):
                                p = pred_b[c]
                                g = gt_b[c]
                                if g == 1: frame_stats[c]["gt_count"] += 1
                                if p == 1: frame_stats[c]["pred_count"] += 1
                                if p == 1 and g == 1: frame_stats[c]["tp"] += 1
                                elif p == 1 and g == 0: frame_stats[c]["fp"] += 1
                                elif p == 0 and g == 1: frame_stats[c]["fn"] += 1

                            z_tps = [c for c in range(3) if pred_b[c] == 1 and gt_b[c] == 1]
                            z_fps = [c for c in range(3) if pred_b[c] == 1 and gt_b[c] == 0]
                            z_fns = [c for c in range(3) if pred_b[c] == 0 and gt_b[c] == 1]

                            if len(z_tps) > 0:
                                verdict = "TP"
                                verdict_label = f"TRUE POSITIVE ({cat_short[z_tps[0]]})"
                            elif len(z_fps) > 0:
                                verdict = "FP"
                                verdict_label = f"FALSE POSITIVE ({cat_short[z_fps[0]]})"
                            elif len(z_fns) > 0:
                                verdict = "FN"
                                verdict_label = f"FALSE NEGATIVE ({cat_short[z_fns[0]]})"
                            else:
                                verdict = "TN"
                                verdict_label = "TRUE NEGATIVE (Zona Libera)"

                        inferred.append({
                            "occ": occ,
                            "poly_xy": coords,
                            "center": (cx_m, cy_m),
                            "area": float(cz.get("area", occ.get("area_sqm", sp.area))),
                            "dist": float(cz.get("dist", occ.get("distance_m", np.hypot(cx_m, cy_m)))),
                            "road_f": float(cz.get("road_f", occ.get("road_fraction", 0.0))),
                            "side_f": float(cz.get("side_f", occ.get("sidewalk_fraction", 0.0))),
                            "cross_f": float(cz.get("cross_f", occ.get("crosswalk_fraction", 0.0))),
                            "roadside_f": float(cz.get("roadside_f", occ.get("roadside_fraction", 0.0))),
                            "min_d_road": float(cz.get("min_d_road", 0.0)),
                            "probs_4": cz["probs_4"],
                            "pred_binary": pred_b,
                            "gt_target_4": gt_b,
                            "pred_class": cz["pred_class"],
                            "pred_idx": cz["pred_idx"],
                            "risk_score": cz["risk_score"],
                            "verdict": verdict,
                            "verdict_label": verdict_label,
                            "zone_tps": z_tps,
                            "zone_fps": z_fps,
                            "zone_fns": z_fns
                        })

                    tot_tp = sum(frame_stats[c]["tp"] for c in range(3))
                    tot_fp = sum(frame_stats[c]["fp"] for c in range(3))
                    tot_fn = sum(frame_stats[c]["fn"] for c in range(3))
                    prec = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else 0.0
                    rec = (tot_tp / (tot_tp + tot_fn) * 100.0) if (tot_tp + tot_fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    latency = (time.perf_counter() - start_time) * 1000.0
                    return inferred, frame_stats, latency, f1, rec, prec

        # 2. FALLBACK: Calcolo dinamico sul modello
        model = self._preload_model(model_key)
        if model is None:
            return [], {c: {"tp":0,"fp":0,"fn":0,"gt_count":0,"pred_count":0} for c in range(3)}, 0.0, 0.0, 0.0, 0.0

        target_masks = extract_ground_truth_masks(self.frame_data)
        semantic_map = self.frame_data['semantic_map']
        drivable_mask = np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))
        walkway_mask = semantic_map.get('walkway', np.zeros_like(drivable_mask))
        ped_crossing_mask = semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))

        input_channels = [
            self.frame_data.get('lidar_bev', np.zeros((GRID_DIM, GRID_DIM))),
            self.frame_data.get('occlusion_mask', np.zeros((GRID_DIM, GRID_DIM))),
            drivable_mask,
            walkway_mask,
            ped_crossing_mask
        ] + list(target_masks)

        input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

        occ_file = os.path.join(self.occlusions_dir, f"{self.sample_token}.json")
        if not os.path.exists(occ_file):
            return [], {c: {"tp":0,"fp":0,"fn":0,"gt_count":0,"pred_count":0} for c in range(3)}, 0.0, 0.0, 0.0, 0.0

        with open(occ_file, "r") as f:
            data = json.load(f)
            occs = data.get("occlusions", []) if isinstance(data, dict) else data

        start_time = time.perf_counter()
        circle_25m = Point(0, 0).buffer(self.max_range - 0.4)

        bayes_occs = []
        if model_key == "BAYES":
            bayes_dir = os.path.join(ROOT_DIR, "extracted_occlusions_probabilities")
            matches = glob.glob(os.path.join(bayes_dir, f"*{self.sample_token}*.json"))
            if not matches:
                matches = glob.glob(os.path.join(bayes_dir, f"occlusion_sample_{self.current_idx:04d}_*.json"))
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        b_data = json.load(f)
                        bayes_occs = b_data.get("occlusions", [])
                except Exception:
                    bayes_occs = []

        import cv2
        dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * VOXEL_SIZE

        frame_stats = {
            c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0}
            for c in range(3)
        }
        thresholds = [0.28, 0.25, 0.25]
        cat_names_4 = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]
        inferred = []

        # Mappa rapida per estrarre le dimensioni 3D dell'occludore se noto
        boxes_by_token = {b.token: b for b in self.frame_data.get('boxes', [])}
        for sb in getattr(self, 'static_boxes', []):
            boxes_by_token[sb.token] = sb

        all_frame_boxes = self.frame_data.get('boxes', [])
        occluder_tokens = {occ.get('object_token') for occ in occs if occ.get('object_token')}

        for occ_idx, occ in enumerate(occs):
            dist = float(occ.get("distance_m", 0.0))
            if dist > 25.0:
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

            occ_mask = rasterize_polygon(pts)
            tot = np.sum(occ_mask)
            road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = float(occ.get("area_sqm", 0.0))

            min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
            roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

            if gt_key == "REAL":
                gt_raw_6, gt_target_4 = extract_real_occlusion_ground_truth(
                    all_frame_boxes, sp, occluder_tokens=occluder_tokens
                )
            else:
                gt_raw_6, _ = extract_real_occlusion_ground_truth(
                    all_frame_boxes, sp, occluder_tokens=occluder_tokens
                )
                if gt_key == "REAL":
                    gt_target_4 = to_macro_classes_4(gt_raw_6)
                else:
                    gt_hyb_6, _ = compute_synthetic_ground_truth(
                        sp=sp, road_f=road_f, side_f=side_f, cross_f=cross_f,
                        roadside_f=roadside_f, area=area, dist=dist, gt_raw_6=gt_raw_6, mode="hybrid",
                        occluder_name=occ_name, occluder_wlh=occ_wlh, use_occluder_filter=True
                    )
                    gt_target_4 = to_macro_classes_4(gt_hyb_6)

            if model_key == "BAYES":
                b_occ = bayes_occs[occ_idx] if occ_idx < len(bayes_occs) else {}
                b_probs = b_occ.get("estimated_probabilities", {})
                p_auto = float(b_probs.get("Auto", 0.0))
                p_camion = float(max(b_probs.get("Camion", 0.0), b_probs.get("Bus", 0.0), b_probs.get("Rimorchio", 0.0)))
                p_vru = float(max(b_probs.get("Pedone", 0.0), b_probs.get("Bicicletta", 0.0), b_probs.get("Moto", 0.0)))
                # ponytail: classe Barriera disattivata (mask=0 in build_compatibility_mask), esclusa da p4
                p4 = [p_auto, p_camion, p_vru]
            else:
                px_x = np.clip(((poly_xy[:, 0] + GRID_RANGE) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                px_y = np.clip(((GRID_RANGE - poly_xy[:, 1]) / VOXEL_SIZE).astype(int), 0, GRID_DIM - 1)
                xmin, xmax = max(0, np.min(px_x) - 2), min(GRID_DIM - 1, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(GRID_DIM - 1, np.max(px_y) + 2)
                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device)

                if sp.is_valid and sp.area > 0.01:
                    mrr = sp.minimum_rotated_rectangle
                    mrr_coords = np.array(mrr.exterior.coords)[:-1]
                    e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
                    e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
                    obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))
                else:
                    obb_w, obb_l = 0.5, 0.5

                scalars = torch.tensor([[area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f]], dtype=torch.float32).to(self.device)

                occluder_mask = AttentionPerZoneModel.build_compatibility_mask(
                    occ_name, occ_wlh, road_f=road_f, roadside_f=roadside_f, device=self.device
                )
                with torch.no_grad():
                    logits = model(patch_res, scalars, occluder_mask=occluder_mask)
                    out_6 = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                p4 = [
                    float(out_6[0]),
                    float(out_6[1]),
                    float(max(out_6[2], out_6[3], out_6[4]))
                ]
            pred_binary = [int(p4[c] >= thresholds[c]) for c in range(3)]

            mode = getattr(self, 'eval_mode', "PER_CLASS")
            pred_classes = [c for c in range(3) if pred_binary[c] == 1]
            gt_classes = [c for c in range(3) if gt_target_4[c] == 1]

            if mode == "ZONE_LEVEL":
                is_pred_h = len(pred_classes) > 0
                is_gt_h = len(gt_classes) > 0
                if is_pred_h:
                    m = [c for c in pred_classes if c in gt_classes]
                    if m:
                        verdict = "TP"
                        verdict_label = f"TRUE POSITIVE (Pericolo Zona: {CAT_NAMES_4[m[0]]})"
                        frame_stats[m[0]]["tp"] += 1
                        frame_stats[m[0]]["gt_count"] += 1
                        frame_stats[m[0]]["pred_count"] += 1
                        zone_tps = [m[0]]
                        zone_fps = []
                        zone_fns = []
                    else:
                        verdict = "FP"
                        verdict_label = f"FALSE POSITIVE ({CAT_NAMES_4[pred_classes[0]]})"
                        frame_stats[pred_classes[0]]["fp"] += 1
                        frame_stats[pred_classes[0]]["pred_count"] += 1
                        zone_tps = []
                        zone_fps = [pred_classes[0]]
                        if is_gt_h:
                            frame_stats[gt_classes[0]]["fn"] += 1
                            frame_stats[gt_classes[0]]["gt_count"] += 1
                            zone_fns = [gt_classes[0]]
                        else:
                            zone_fns = []
                elif is_gt_h:
                    verdict = "FN"
                    verdict_label = f"FALSE NEGATIVE ({CAT_NAMES_4[gt_classes[0]]})"
                    frame_stats[gt_classes[0]]["fn"] += 1
                    frame_stats[gt_classes[0]]["gt_count"] += 1
                    zone_tps = []
                    zone_fps = []
                    zone_fns = [gt_classes[0]]
                else:
                    verdict = "TN"
                    verdict_label = "TRUE NEGATIVE (Zona Libera)"
                    zone_tps = []
                    zone_fps = []
                    zone_fns = []
            else:
                zone_tps = [c for c in range(3) if pred_binary[c] == 1 and gt_target_4[c] == 1]
                zone_fps = [c for c in range(3) if pred_binary[c] == 1 and gt_target_4[c] == 0]
                zone_fns = [c for c in range(3) if pred_binary[c] == 0 and gt_target_4[c] == 1]

                for c in range(3):
                    if gt_target_4[c] == 1: frame_stats[c]["gt_count"] += 1
                    if pred_binary[c] == 1: frame_stats[c]["pred_count"] += 1
                    if pred_binary[c] == 1 and gt_target_4[c] == 1: frame_stats[c]["tp"] += 1
                    elif pred_binary[c] == 1 and gt_target_4[c] == 0: frame_stats[c]["fp"] += 1
                    elif pred_binary[c] == 0 and gt_target_4[c] == 1: frame_stats[c]["fn"] += 1

                if len(zone_tps) > 0 and len(zone_fns) == 0:
                    verdict = "TP"
                    verdict_label = f"TRUE POSITIVE ({CAT_NAMES_4[zone_tps[0]]})"
                elif len(zone_tps) > 0 and len(zone_fns) > 0:
                    verdict = "TP/FN"
                    verdict_label = f"PARZIALE: TP ({', '.join(CAT_NAMES_4[c] for c in zone_tps)}) + FN ({', '.join(CAT_NAMES_4[c] for c in zone_fns)})"
                elif len(zone_fns) > 0 and len(zone_fps) == 0:
                    verdict = "FN"
                    verdict_label = f"FALSE NEGATIVE ({CAT_NAMES_4[zone_fns[0]]})"
                elif len(zone_fns) > 0 and len(zone_fps) > 0:
                    verdict = "FN"
                    verdict_label = f"FALSE NEGATIVE ({CAT_NAMES_4[zone_fns[0]]})"
                elif len(zone_fps) > 0 and len(zone_fns) == 0:
                    verdict = "FP"
                    verdict_label = f"FALSE POSITIVE ({CAT_NAMES_4[zone_fps[0]]})"
                else:
                    verdict = "TN"
                    verdict_label = "TRUE NEGATIVE (Zona Libera)"

            max_class_idx = int(np.argmax(p4))
            max_prob = float(p4[max_class_idx])

            if poly_vis.geom_type == 'Polygon':
                poly_main = poly_vis
            elif poly_vis.geom_type in ['MultiPolygon', 'GeometryCollection']:
                polys = [g for g in poly_vis.geoms if g.geom_type == 'Polygon']
                poly_main = max(polys, key=lambda g: g.area) if polys else sp
            else:
                poly_main = sp

            pt = poly_main.representative_point()
            cx_m, cy_m = float(pt.x), float(pt.y)

            inferred.append({
                "occ": occ,
                "poly_xy": poly_xy,
                "center": (cx_m, cy_m),
                "area": area,
                "dist": dist,
                "road_f": road_f,
                "side_f": side_f,
                "cross_f": cross_f,
                "roadside_f": roadside_f,
                "min_d_road": min_d_road,
                "probs_4": p4,
                "pred_binary": pred_binary,
                "gt_target_4": gt_target_4,
                "pred_class": CAT_NAMES_4[max_class_idx],
                "pred_idx": max_class_idx,
                "risk_score": max_prob,
                "verdict": verdict,
                "verdict_label": verdict_label,
                "zone_tps": zone_tps,
                "zone_fps": zone_fps,
                "zone_fns": zone_fns
            })

        latency = (time.perf_counter() - start_time) * 1000.0
        tot_tp = sum(frame_stats[c]["tp"] for c in range(3))
        tot_fp = sum(frame_stats[c]["fp"] for c in range(3))
        tot_fn = sum(frame_stats[c]["fn"] for c in range(3))

        prec = (tot_tp / (tot_tp + tot_fp)) * 100.0 if (tot_tp + tot_fp) > 0 else 0.0
        rec = (tot_tp / (tot_tp + tot_fn)) * 100.0 if (tot_tp + tot_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return inferred, frame_stats, latency, f1, rec, prec

    def draw_icon_object(self, ax, icon_name, center, length, deg=0.0, width=None,
                         flip_h=False, halo_color=None, halo_radius=1.8):
        if halo_color is not None:
            c = Circle((center[0], center[1]), halo_radius,
                       facecolor=halo_color, edgecolor='none', alpha=0.85, zorder=7)
            ax.add_patch(c)

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

    def draw_bev_map(self, ax, inferred_occlusions, title=None, badge_text=None, badge_bg='#2563EB', is_interactive=False):
        """Disegna la mappa BEV accademica a 25m conforme allo stile di tesi."""
        c_map_bg = '#F8FAFC'
        c_grid = '#E2E8F0'
        c_border = '#0F172A'

        ax.set_facecolor(c_map_bg)
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-self.max_range, self.max_range)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.3)

        ticks = np.arange(-int(self.max_range), int(self.max_range) + 5, 5)
        for t in ticks:
            ax.axvline(t, color=c_grid, linewidth=0.6, zorder=1)
            ax.axhline(t, color=c_grid, linewidth=0.6, zorder=1)

        ax.axhline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)
        ax.axvline(0, color=c_border, linewidth=1.1, alpha=0.9, zorder=3)

        for r in [10.0, 20.0, self.max_range]:
            c = Circle((0, 0), r, color=c_border, fill=False, linewidth=1.1, zorder=3)
            ax.add_patch(c)
            ax.text(0, -r + 1.1, f"{int(r)}m", color=c_border, fontsize=9.0,
                    ha='center', va='center', fontweight='bold', zorder=10,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))

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

            ax.imshow(map_rgba, extent=[-40.0, 40.0, -40.0, 40.0], origin='lower', zorder=2)

        pts = self.frame_data.get('points', None)
        if pts is not None and len(pts) > 0:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            mask_r = dists <= self.max_range
            pts_r = pts[mask_r]
            if len(pts_r) > 0:
                ax.scatter(pts_r[:, 0], pts_r[:, 1], s=1.1, c='#0F172A',
                           alpha=0.68, zorder=3, edgecolors='none')

        drawn_list = []
        for occ_data in inferred_occlusions:
            poly_xy = occ_data["poly_xy"]
            verdict = occ_data["verdict"]
            cx, cy = occ_data["center"]
            dist_c = np.hypot(cx, cy)
            pred_cls = occ_data["pred_class"]
            risk = occ_data["risk_score"]
            gt_target_4 = occ_data.get("gt_target_4", [0, 0, 0])
            pred_binary = occ_data.get("pred_binary", [0, 0, 0])
            zone_fns = occ_data.get("zone_fns", [])
            short_names = ["Auto", "Camion", "VRU"]
            n_valid = min(len(gt_target_4), len(short_names))
            real_objs = [short_names[c] for c in range(n_valid) if gt_target_4[c] == 1]
            real_short = "+".join(real_objs) if real_objs else (short_names[zone_fns[0]] if len(zone_fns) > 0 and zone_fns[0] < len(short_names) else pred_cls)

            pred_objs = [short_names[c] for c in range(min(len(pred_binary), len(short_names))) if pred_binary[c] == 1]
            pred_short = "+".join(pred_objs) if pred_objs else "Nessuno"

            has_fn = len(zone_fns) > 0
            is_fn_mode = (self.highlight_fn and has_fn) or (verdict == "FN")

            if is_fn_mode:
                missed_classes = [short_names[c] for c in zone_fns]
                missed_str = "+".join(missed_classes) if missed_classes else (short_names[zone_fns[0]] if zone_fns else "FN")
                if self.highlight_fn:
                    # Evidenziazione speciale Falsi Negativi (colore vivo, target ring e alone)
                    fc, ec, alpha_fill, lw = '#F43F5E', '#9F1239', 0.65, 2.6
                    b_bg = '#881337'
                    b_lbl = f"⚠ FN: {missed_str}"
                else:
                    fc, ec, alpha_fill, lw = '#A78BFA', '#7C3AED', 0.40, 1.8
                    b_bg = '#6D28D9'
                    b_lbl = f"FN: {missed_str}"
                eff_cls = missed_classes[0] if missed_classes else (real_objs[0] if real_objs else pred_cls)
            elif verdict in ["TP", "TP/FN"]:
                if self.highlight_fn:
                    fc, ec, alpha_fill, lw = '#10B981', '#047857', 0.15, 0.9
                    b_bg, b_lbl = '#059669', "TP"
                else:
                    fc, ec, alpha_fill, lw = '#10B981', '#047857', 0.42, 1.6
                    b_bg, b_lbl = '#059669', "TP"
                eff_cls = pred_cls
            elif verdict == "FP":
                if self.highlight_fn:
                    fc, ec, alpha_fill, lw = '#EF4444', '#B91C1C', 0.12, 0.7
                    b_bg, b_lbl = '#DC2626', "FP"
                else:
                    fc, ec, alpha_fill, lw = '#EF4444', '#B91C1C', 0.38, 1.6
                    b_bg, b_lbl = '#DC2626', "FP"
                eff_cls = pred_cls
            else:
                fc, ec, alpha_fill, lw = '#E2E8F0', '#94A3B8', 0.05 if self.highlight_fn else 0.18, 0.5
                b_bg, b_lbl = '#64748B', "TN"
                eff_cls = pred_cls

            poly_patch = MplPolygon(poly_xy, closed=True, facecolor=fc, edgecolor=ec,
                                    linewidth=lw, alpha=alpha_fill,
                                    linestyle='--' if is_fn_mode else '-',
                                    zorder=10 if is_fn_mode else 5)
            ax.add_patch(poly_patch)

            if dist_c <= 24.8 and abs(cx) <= 24.8 and abs(cy) <= 24.8 and (verdict in ["TP", "FP", "FN", "TP/FN"] or is_fn_mode):
                if is_fn_mode:
                    # In modalità FN disegna l'icona dell'ostacolo MANCATO in rosso
                    if eff_cls == "Auto":
                        self.draw_icon_object(ax, 'car_red', center=[cx, cy], length=4.0, deg=0.0)
                        h_off = 2.4
                    elif eff_cls in ["Camion/Bus", "Camion"]:
                        self.draw_icon_object(ax, 'truck_red', center=[cx, cy], length=5.5, deg=0.0)
                        h_off = 3.1
                    elif "VRU" in eff_cls or eff_cls == "Pedone":
                        self.draw_icon_object(ax, 'pedestrian_red', center=[cx, cy], length=2.8, deg=0.0,
                                              halo_color='#FEE2E2', halo_radius=1.5)
                        h_off = 1.8
                    else:
                        self.draw_icon_object(ax, 'barrier_red', center=[cx, cy], length=2.2, deg=0.0)
                        h_off = 1.5

                    # Cerchio bersaglio attorno al pericolo non visto
                    ring = Circle((cx, cy), 2.2, facecolor='none', edgecolor='#E11D48',
                                  linewidth=2.4, linestyle='--', zorder=16)
                    ax.add_patch(ring)
                    glow = Circle((cx, cy), 2.8, facecolor='#FFE4E6', edgecolor='none',
                                  alpha=0.45, zorder=15)
                    ax.add_patch(glow)
                else:
                    if eff_cls == "Auto":
                        icon_k = 'car_red' if verdict == "FP" else 'car_blue'
                        self.draw_icon_object(ax, icon_k, center=[cx, cy], length=4.0, deg=0.0)
                        h_off = 2.4
                    elif eff_cls in ["Camion/Bus", "Camion"]:
                        icon_k = 'truck_red' if verdict == "FP" else 'truck_purple'
                        self.draw_icon_object(ax, icon_k, center=[cx, cy], length=5.5, deg=0.0)
                        h_off = 3.1
                    elif "VRU" in eff_cls or eff_cls == "Pedone":
                        icon_k = 'pedestrian_red' if verdict == "FP" else 'pedestrian_green'
                        self.draw_icon_object(ax, icon_k, center=[cx, cy], length=2.8, deg=0.0,
                                              halo_color='#FEE2E2' if verdict == "FP" else '#DCFCE7', halo_radius=1.5)
                        h_off = 1.8
                    else:
                        icon_k = 'barrier_red' if verdict == "FP" else 'barrier_hazard'
                        self.draw_icon_object(ax, icon_k, center=[cx, cy], length=2.2, deg=0.0)
                        h_off = 1.5

                lbl_text = b_lbl if is_fn_mode else f"{b_lbl} {int(risk*100)}%"
                lbl_x = float(np.clip(cx, -19.5, 19.5))
                lbl_y = float(np.clip(cy + h_off, -22.5, 22.0))
                ax.text(lbl_x, lbl_y, lbl_text,
                        fontsize=7.2 if is_fn_mode else 7.0,
                        fontweight='bold', color='#FFFFFF', linespacing=1.15,
                        ha='center', va='bottom', zorder=18 if is_fn_mode else 12, clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.20' if is_fn_mode else 'round,pad=0.15',
                                  facecolor=b_bg,
                                  edgecolor='#FFFFFF' if is_fn_mode else '#0F172A',
                                  linewidth=1.2 if is_fn_mode else 0.8,
                                  alpha=0.98 if is_fn_mode else 0.95))

            if is_interactive:
                from matplotlib.path import Path as MplPath
                drawn_list.append({
                    'patch': poly_patch,
                    'path': MplPath(poly_xy),
                    'meta': occ_data,
                    'default_fc': fc,
                    'default_ec': ec,
                    'default_lw': lw,
                    'default_alpha': alpha_fill,
                    'default_z': 5
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

        # Oggetti visibili LiDAR reali
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
                self.draw_icon_object(ax, 'car_blue', center=center_xy, length=min(4.8, max(3.8, length)), deg=deg)
            elif "truck" in b_name or "trailer" in b_name or "bus" in b_name or "construction" in b_name:
                if "construction" in b_name:
                    exc_w = min(4.5, max(3.2, length * 0.65))
                    self.draw_icon_object(ax, 'construction_orange', center=center_xy,
                                          length=exc_w * (345.0 / 537.0), width=exc_w, deg=0.0, flip_h=(u_vec[0] < 0))
                elif "trailer" in b_name:
                    self.draw_icon_object(ax, 'trailer_purple', center=f_mid - (u_vec / length) * (min(13.5, max(4.5, length)) / 2.0),
                                          length=min(13.5, max(4.5, length)), width=min(2.6, max(1.8, width)), deg=deg)
                elif "truck" in b_name:
                    if box.token in coupled_truck_tokens:
                        cab_len = min(4.0, length * 0.65)
                        self.draw_icon_object(ax, 'truck_cab_purple', center=f_mid - (u_vec / length) * (cab_len / 2.0 + 0.35),
                                              length=cab_len, width=width * 1.05, deg=deg)
                    else:
                        tr_len = min(5.2, max(4.0, length)) if length <= 6.8 else min(7.8, max(5.5, length))
                        tr_w = min(2.1, max(1.7, width)) if length <= 6.8 else min(2.5, max(2.0, width))
                        self.draw_icon_object(ax, 'truck_purple', center=center_xy, length=tr_len, width=tr_w, deg=deg)
                else:
                    self.draw_icon_object(ax, 'truck_purple', center=center_xy, length=min(7.8, max(5.5, length)), deg=deg)
            elif "pedestrian" in b_name or "human" in b_name:
                self.draw_icon_object(ax, 'pedestrian_green', center=center_xy, length=2.8, deg=0.0,
                                      flip_h=(u_vec[0] < 0), halo_color='#DCFCE7', halo_radius=1.5)
            elif "bicycle" in b_name:
                self.draw_icon_object(ax, 'bicycle_orange', center=center_xy, length=2.6, deg=0.0,
                                      flip_h=(u_vec[0] < 0), halo_color='#FED7AA', halo_radius=1.5)
            elif "motorcycle" in b_name:
                self.draw_icon_object(ax, 'motorcycle_amber', center=center_xy, length=2.6, deg=0.0,
                                      flip_h=(u_vec[0] < 0), halo_color='#FDE68A', halo_radius=1.5)
            elif "barrier" in b_name:
                self.draw_icon_object(ax, 'barrier_hazard', center=center_xy, length=2.2, deg=0.0)
            elif "pushable" in b_name or "pullable" in b_name:
                self.draw_icon_object(ax, 'cart_trolley', center=center_xy, length=2.2, deg=0.0)
            elif "trafficcone" in b_name or "cone" in b_name:
                self.draw_icon_object(ax, 'traffic_cone', center=center_xy, length=1.5, deg=0.0)

        # Ego Vehicle al centro
        self.draw_icon_object(ax, 'car_ego', center=[0.0, 0.0], length=5.0, deg=0.0)

        # Header badge opzionale per la mappa
        if title:
            ax.set_title(title, fontsize=10.0, fontweight='bold', pad=8, color=c_border)
        if badge_text:
            ax.text(0.04, 0.94, badge_text, transform=ax.transAxes,
                    fontsize=8.2, fontweight='bold', color='#FFFFFF', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_bg, edgecolor=c_border, linewidth=1.0, alpha=0.95))

        return drawn_list

    def render(self):
        """Renderizza la finestra grafica principale a 2 colonne in perfetto stile accademico tesi."""
        self.fig.clf()
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ui_buttons = []

        gs = GridSpec(3, 2, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.925,
                      width_ratios=[1.1, 0.9], height_ratios=[1.20, 1.05, 1.25],
                      wspace=0.14, hspace=0.28)

        ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map = ax_map
        ax_config = self.fig.add_subplot(gs[0, 1])
        ax_metrics = self.fig.add_subplot(gs[1, 1])
        ax_matrix = self.fig.add_subplot(gs[2, 1])

        c_border = '#0F172A'

        # Titolo visualizzatore pulito in alto a sinistra
        self.fig.text(0.025, 0.958,
                      f"Figure 5: Valutazione Prestazioni (BEV {int(self.max_range)}m)",
                      fontsize=9.8, fontweight='bold', color=c_border, ha='left', va='center')

        # Badge Split nuScenes (TRAIN vs VAL inedito)
        sample = self.adapter.all_samples[self.current_idx]
        sc_name = self.adapter.nusc.get('scene', sample['scene_token'])['name']
        is_val = sc_name in getattr(self, 'val_scenes', set())
        if self.nav_val_only and is_val and self.current_idx in self.val_indices:
            val_pos = self.val_indices.index(self.current_idx) + 1
            split_lbl = f"VAL: {val_pos}/{len(self.val_indices)} [{sc_name}]"
        else:
            split_lbl = f"SPLIT: VAL ★ [{sc_name}]" if is_val else f"SPLIT: TRAIN [{sc_name}]"
        split_bg = "#FEF3C7" if is_val else "#EFF6FF"
        split_fg = "#92400E" if is_val else "#1E40AF"
        split_edge = "#F59E0B" if is_val else "#3B82F6"
        # Badge Split ancorato a sinistra subito dopo il titolo
        self.fig.text(0.218, 0.957, split_lbl,
                      fontsize=7.2, fontweight='bold', color=split_fg, ha='left', va='center',
                      bbox=dict(boxstyle='round,pad=0.22', facecolor=split_bg, edgecolor=split_edge, linewidth=1.1))

        # Selettore Soglia Decisionale (0.50, 0.60, 0.70)
        th_val = getattr(self, 'current_threshold', 0.50)
        th_bg = '#EFF6FF' if th_val == 0.50 else ('#F0FDF4' if th_val == 0.60 else '#FEF3C7')
        th_fg = '#1D4ED8' if th_val == 0.50 else ('#15803D' if th_val == 0.60 else '#92400E')
        th_lbl = f'◆ SOGLIA: {th_val:.2f}'
        ax_th = self.fig.add_axes([0.318, 0.940, 0.098, 0.034])
        btn_th = Button(ax_th, th_lbl, color=th_bg, hovercolor='#DBEAFE')
        btn_th.label.set_fontsize(7.6); btn_th.label.set_fontweight('bold'); btn_th.label.set_color(th_fg)
        btn_th.on_clicked(self.toggle_threshold)
        self.ui_buttons.append((ax_th, btn_th))

        # Selettore Modalità: ALMENO 1 OSTACOLO vs TUTTI GLI OSTACOLI
        m_mode = getattr(self, 'eval_mode', "PER_CLASS")
        mode_bg = '#FEF3C7' if m_mode == "ZONE_LEVEL" else '#F1F5F9'
        mode_fg = '#92400E' if m_mode == "ZONE_LEVEL" else '#334155'
        mode_lbl = '★ ALMENO 1 OSTACOLO' if m_mode == "ZONE_LEVEL" else 'TUTTI GLI OSTACOLI'
        ax_mode = self.fig.add_axes([0.420, 0.940, 0.124, 0.034])
        btn_mode = Button(ax_mode, mode_lbl, color=mode_bg, hovercolor='#FDE68A' if m_mode == "ZONE_LEVEL" else '#E2E8F0')
        btn_mode.label.set_fontsize(6.7); btn_mode.label.set_fontweight('bold'); btn_mode.label.set_color(mode_fg)
        btn_mode.on_clicked(self.toggle_eval_mode)
        self.ui_buttons.append((ax_mode, btn_mode))

        # Filtro Navigazione: Tutti vs Solo VAL
        nav_bg = '#FEF3C7' if self.nav_val_only else '#F1F5F9'
        nav_fg = '#92400E' if self.nav_val_only else '#334155'
        val_count = len(self.val_indices)
        nav_lbl = f'★ SOLO VAL ({val_count})' if self.nav_val_only else f'TUTTI I FRAME ({self.total_frames})'
        ax_nav_flt = self.fig.add_axes([0.548, 0.940, 0.130, 0.034])
        btn_nav_flt = Button(ax_nav_flt, nav_lbl, color=nav_bg, hovercolor='#FDE68A' if self.nav_val_only else '#E2E8F0')
        btn_nav_flt.label.set_fontsize(7.3); btn_nav_flt.label.set_fontweight('bold'); btn_nav_flt.label.set_color(nav_fg)
        btn_nav_flt.on_clicked(self.toggle_nav_filter)
        self.ui_buttons.append((ax_nav_flt, btn_nav_flt))

        # Controlli salto frame e toggle FN in alto a destra
        fn_bg_main = '#E11D48' if self.highlight_fn else '#F8FAFC'
        fn_fg_main = '#FFFFFF' if self.highlight_fn else '#475569'
        fn_lbl_main = '⚠ FN: ON' if self.highlight_fn else '⚠ Evidenzia FN'
        ax_fn_main = self.fig.add_axes([0.684, 0.940, 0.116, 0.034])
        btn_fn_main = Button(ax_fn_main, fn_lbl_main, color=fn_bg_main,
                             hovercolor='#BE123C' if self.highlight_fn else '#E2E8F0')
        btn_fn_main.label.set_fontsize(7.5); btn_fn_main.label.set_fontweight('bold'); btn_fn_main.label.set_color(fn_fg_main)
        btn_fn_main.on_clicked(self.toggle_highlight_fn)
        self.ui_buttons.append((ax_fn_main, btn_fn_main))

        ax_prev = self.fig.add_axes([0.806, 0.940, 0.038, 0.034])
        btn_prev = Button(ax_prev, "◀ Prec", color='#F8FAFC', hovercolor='#E2E8F0')
        btn_prev.label.set_fontsize(7.6); btn_prev.label.set_fontweight('bold'); btn_prev.label.set_color(c_border)
        btn_prev.on_clicked(lambda ev: self.on_key_step(-1))
        self.ui_buttons.append((ax_prev, btn_prev))

        self.fig.text(0.848, 0.957, "Frame:", fontsize=7.6, fontweight='bold', color=c_border, ha='left', va='center')
        ax_tb = self.fig.add_axes([0.880, 0.940, 0.046, 0.034])
        self.txt_frame = TextBox(ax_tb, '', initial=str(self.current_idx + 1),
                                 color='#F8FAFC', hovercolor='#E2E8F0')
        self.txt_frame.text_disp.set_color('#1D4ED8'); self.txt_frame.text_disp.set_fontweight('bold')
        self.txt_frame.on_submit(self.on_jump_frame)
        self.ui_buttons.append((ax_tb, self.txt_frame))

        ax_next = self.fig.add_axes([0.932, 0.940, 0.046, 0.034])
        btn_next = Button(ax_next, "Succ ▶", color='#F8FAFC', hovercolor='#E2E8F0')
        btn_next.label.set_fontsize(7.6); btn_next.label.set_fontweight('bold'); btn_next.label.set_color(c_border)
        btn_next.on_clicked(lambda ev: self.on_key_step(1))
        self.ui_buttons.append((ax_next, btn_next))

        # 1. PANNELLO SINISTRO: MAPPA BEV AD ALTA FEDELTÀ (25M)

        self.drawn_occlusions = self.draw_bev_map(ax_map, self.inferred_occlusions, is_interactive=True)
        main_mancante = self.modello_mancante(self.current_model_key, self.current_gt_key)
        if main_mancante:
            self._disegna_avviso_mancante(ax_map, self.current_model_key)

        # Tooltip interattivo
        self.tooltip = ax_map.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3",
                      facecolor="#0F172A", edgecolor="#38BDF8", linewidth=1.5, alpha=0.95),
            fontsize=8.5, color="#F8FAFC", family='sans-serif', zorder=50,
            linespacing=1.35
        )
        self.tooltip.set_visible(False)

        # 2. PANNELLO DESTRA ALTO: SELETTORE MODELLO & GT (Pills a selezione istantanea)

        ax_config.set_facecolor('#F8FAFC')
        for spine in ax_config.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)
        ax_config.set_xlim(0, 1)
        ax_config.set_ylim(0, 1)
        ax_config.set_xticks([])
        ax_config.set_yticks([])

        ax_config.text(0.02, 0.91, "CONFIGURAZIONE MODELLO & GROUND TRUTH (GERARCHIA A 3 LIVELLI)",
                       transform=ax_config.transAxes, fontsize=8.4, fontweight='bold', color=c_border)

        bbox_btn = ax_config.get_position()

        # Livello 1: Selezione Supervisione / Addestramento
        ax_config.text(0.02, 0.76, "Supervisione:", transform=ax_config.transAxes,
                       fontsize=7.4, fontweight='bold', color='#475569', va='center')

        macro_items = [
            ("NEURO_SIMB", "★ Addestramento Ibrido (Anticipazione)", 0.152, 0.398, "#1D4ED8"),
            ("REAL", "◆ Addestramento Reale (nuScenes)", 0.558, 0.342, "#0D9488"),
            ("BAYES", "▲ Bayes", 0.908, 0.076, "#0284C7")
        ]

        for m_key, m_lbl, x_pos, w_btn, m_theme in macro_items:
            is_act = (self.current_macro == m_key)
            bg_c = m_theme if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'

            f_x = bbox_btn.x0 + x_pos * bbox_btn.width
            f_y = bbox_btn.y0 + 0.69 * bbox_btn.height
            f_w = w_btn * bbox_btn.width
            f_h = 0.16 * bbox_btn.height

            ax_btn = self.fig.add_axes([f_x, f_y, f_w, f_h])
            btn = Button(ax_btn, m_lbl, color=bg_c, hovercolor='#93C5FD' if not is_act else bg_c)
            btn.label.set_fontsize(6.7); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=m_key: self.select_macro(k))
            self.ui_buttons.append((ax_btn, btn))

        # Livello 2: Selezione Modello
        ax_config.text(0.02, 0.52, "Modello:", transform=ax_config.transAxes,
                       fontsize=7.4, fontweight='bold', color='#475569', va='center')

        if self.current_macro == "NEURO_SIMB":
            child_models = [
                ("NEURO_SIMB", "★ Con Vincoli Simbolici (Full)", 0.152, 0.410),
                ("PURE_NEURAL", "Senza Vincoli Simbolici (Puro)", 0.572, 0.412)
            ]
        elif self.current_macro == "REAL":
            child_models = [
                ("REAL_GT", "Con Vincoli Simbolici (Full)", 0.152, 0.335),
                ("REAL_PURE_NEURAL", "Senza Vincoli Simbolici (Puro)", 0.495, 0.335),
                ("POS_ONLY", "Solo Zone Piene", 0.838, 0.146)
            ]
        else:  # BAYES
            child_models = [
                ("BAYES", "Modello Analitico (No AI - Baseline Statistica)", 0.152, 0.832)
            ]

        for m_k, m_lbl, x_pos, w_btn in child_models:
            is_act = (self.current_model_key == m_k)
            bg_c = '#1E293B' if is_act else '#F1F5F9'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            if self.modello_mancante(m_k, self.current_gt_key):
                bg_c = '#64748B' if is_act else '#F8FAFC'
                tx_c = '#FFFFFF' if is_act else '#94A3B8'

            f_x = bbox_btn.x0 + x_pos * bbox_btn.width
            f_y = bbox_btn.y0 + 0.44 * bbox_btn.height
            f_w = w_btn * bbox_btn.width
            f_h = 0.16 * bbox_btn.height

            ax_btn = self.fig.add_axes([f_x, f_y, f_w, f_h])
            btn = Button(ax_btn, m_lbl, color=bg_c, hovercolor='#CBD5E1' if not is_act else bg_c)
            btn.label.set_fontsize(6.7 if len(child_models) > 2 else 7.2); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=m_k: self.select_model(k))
            self.ui_buttons.append((ax_btn, btn))

        # Livello 3: Selezione Target Ground Truth
        ax_config.text(0.02, 0.27, "Target GT:", transform=ax_config.transAxes,
                       fontsize=7.4, fontweight='bold', color='#475569', va='center')

        gts = [
            ("NEURO_SIMB", "★ GT Anticipazione (Ibrida)", 0.152, 0.410),
            ("REAL", "GT Rilevamento (Reale nuScenes)", 0.572, 0.412)
        ]

        for g_k, g_lbl, x_pos, w_btn in gts:
            is_act = (self.current_gt_key == g_k)
            bg_c = '#4338CA' if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'

            f_x = bbox_btn.x0 + x_pos * bbox_btn.width
            f_y = bbox_btn.y0 + 0.19 * bbox_btn.height
            f_w = w_btn * bbox_btn.width
            f_h = 0.16 * bbox_btn.height

            ax_btn = self.fig.add_axes([f_x, f_y, f_w, f_h])
            btn = Button(ax_btn, g_lbl, color=bg_c, hovercolor='#C7D2FE' if not is_act else bg_c)
            btn.label.set_fontsize(7.2); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=g_k: self.select_gt(k))
            self.ui_buttons.append((ax_btn, btn))

        # Livello 4: Telemetria e contesto scena
        ax_config.text(0.02, 0.05,
                       f"Scena: {self.scene_name}  |  Ego: {self.ego_speed_kmh:.1f} km/h  |  "
                       f"GPU: {self.inference_latency_ms:.1f} ms  |  Loc: {self.location}",
                       transform=ax_config.transAxes, fontsize=7.4, color='#64748B', va='center')

        # 3. PANNELLO DESTRA CENTRO: SCORECARD GLOBALE & ISTOGRAMMA METRICHE

        ax_metrics.set_facecolor('#FFFFFF')
        for spine in ax_metrics.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.0)
        ax_metrics.spines['top'].set_visible(False)
        ax_metrics.spines['right'].set_visible(False)

        cat_labels = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]
        prec_per_class, rec_per_class, f1_per_class = [], [], []

        for c in range(3):
            tp_c = self.frame_stats[c]["tp"]
            fp_c = self.frame_stats[c]["fp"]
            fn_c = self.frame_stats[c]["fn"]
            pr = (tp_c / (tp_c + fp_c)) * 100.0 if (tp_c + fp_c) > 0 else 0.0
            rc = (tp_c / (tp_c + fn_c)) * 100.0 if (tp_c + fn_c) > 0 else 0.0
            f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
            prec_per_class.append(pr)
            rec_per_class.append(rc)
            f1_per_class.append(f1)

        x_ind = np.arange(len(cat_labels))
        w_bar = 0.27

        bars_rec = ax_metrics.bar(x_ind - w_bar, rec_per_class, w_bar, label='Recall (%)',
                                  color='#0284C7', edgecolor=c_border, linewidth=1.0, zorder=3)
        bars_prec = ax_metrics.bar(x_ind, prec_per_class, w_bar, label='Precision (%)',
                                   color='#EA580C', edgecolor=c_border, linewidth=1.0, zorder=3)
        bars_f1 = ax_metrics.bar(x_ind + w_bar, f1_per_class, w_bar, label='F1-Score (%)',
                                 color='#10B981', edgecolor=c_border, linewidth=1.0, zorder=3)

        ax_metrics.set_ylim(0, 118)
        ax_metrics.set_xticks(x_ind)
        ax_metrics.set_xticklabels(cat_labels, fontsize=8.2, fontweight='bold', color=c_border)
        ax_metrics.set_ylabel("Score [%]", fontsize=8.5, fontweight='bold', color=c_border)

        # Scorecard badge F1 Frame in alto nel grafico
        f1_col = '#15803D' if self.f1_glob >= 70.0 else ('#B45309' if self.f1_glob >= 50.0 else '#B91C1C')
        score_txt = f"Scorecard Frame: F1={self.f1_glob:.1f}%  |  Recall={self.rec_glob:.1f}%  |  Precision={self.prec_glob:.1f}%"
        if main_mancante:
            # Nessuna metrica fittizia a 0%: il modello e' segnalato come mancante
            short_m = MODEL_CONFIGS.get(self.current_model_key, {}).get("short", self.current_model_key)
            score_txt = f"Scorecard Frame: {short_m} — PESI MANCANTI (metriche non disponibili)"
            f1_col = '#64748B'
        ax_metrics.set_title(score_txt, fontsize=9.2, fontweight='bold', pad=8, color=f1_col)
        ax_metrics.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_metrics.legend(loc='upper right', framealpha=0.92, fontsize=7.5, ncol=3)

        for bar in bars_rec:
            h = bar.get_height()
            if h > 0:
                ax_metrics.text(bar.get_x() + bar.get_width()/2.0, h + 2.0, f"{int(h)}%",
                                ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='#0284C7')
        for bar in bars_prec:
            h = bar.get_height()
            if h > 0:
                ax_metrics.text(bar.get_x() + bar.get_width()/2.0, h + 2.0, f"{int(h)}%",
                                ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='#EA580C')
        for bar in bars_f1:
            h = bar.get_height()
            if h > 0:
                ax_metrics.text(bar.get_x() + bar.get_width()/2.0, h + 2.0, f"{int(h)}%",
                                ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='#10B981')

        # 4. PANNELLO DESTRA BASSO: MATRICE DI CONFUSIONE & AZIONE CONFRONTO 1-A-1

        ax_matrix.set_facecolor('#FFFFFF')
        for spine in ax_matrix.spines.values():
            spine.set_color(c_border)
            spine.set_linewidth(1.1)
        ax_matrix.set_xlim(0, 1)
        ax_matrix.set_ylim(0, 1)
        ax_matrix.set_xticks([])
        ax_matrix.set_yticks([])
        ax_matrix.set_title("Matrice di Confusione Dettagliata per Categoria (TP / FP / FN)",
                            fontsize=9.5, fontweight='bold', pad=8, color=c_border)

        headers = ["Categoria", "TP (Veri)", "FP (Falsi)", "FN (Mancati)", "GT Presenti"]
        col_x = [0.03, 0.31, 0.51, 0.71, 0.88]

        ax_matrix.add_patch(Rectangle((0.02, 0.78), 0.96, 0.15, transform=ax_matrix.transAxes,
                                      facecolor='#F1F5F9', edgecolor='none', zorder=2))
        for x_h, h_txt in zip(col_x, headers):
            ax_matrix.text(x_h, 0.85, h_txt, transform=ax_matrix.transAxes,
                           fontsize=7.4, fontweight='bold', color=c_border, va='center')

        y_r = 0.66
        for c in range(len(cat_labels)):
            tp_c = self.frame_stats[c]["tp"]
            fp_c = self.frame_stats[c]["fp"]
            fn_c = self.frame_stats[c]["fn"]
            gt_c = self.frame_stats[c]["gt_count"]

            ax_matrix.text(col_x[0], y_r, cat_labels[c], transform=ax_matrix.transAxes,
                           fontsize=7.4, fontweight='bold', color='#1E293B', va='center')
            ax_matrix.text(col_x[1], y_r, f"{tp_c}", transform=ax_matrix.transAxes,
                           fontsize=7.8, fontweight='bold', color='#10B981', va='center')
            ax_matrix.text(col_x[2], y_r, f"{fp_c}", transform=ax_matrix.transAxes,
                           fontsize=7.8, fontweight='bold', color='#EF4444', va='center')
            ax_matrix.text(col_x[3], y_r, f"{fn_c}", transform=ax_matrix.transAxes,
                           fontsize=7.8, fontweight='bold', color='#8B5CF6', va='center')
            ax_matrix.text(col_x[4], y_r, f"{gt_c}", transform=ax_matrix.transAxes,
                           fontsize=7.8, fontweight='normal', color='#64748B', va='center')
            y_r -= 0.095

        # Simbologia Scientifica Ufficiale
        ax_matrix.add_patch(Rectangle((0.02, 0.22), 0.96, 0.11, transform=ax_matrix.transAxes,
                                      facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=0.8, zorder=2))
        sym_items = [
            ("● TP: True Positive", '#10B981', 0.06),
            ("● FP: False Positive", '#EF4444', 0.38),
            ("● FN: False Negative", '#7C3AED', 0.70)
        ]
        for s_lbl, s_col, s_x in sym_items:
            ax_matrix.text(s_x, 0.275, s_lbl, transform=ax_matrix.transAxes,
                           fontsize=7.5, fontweight='bold', color=s_col, va='center')

        # Pulsante Azione: Benchmark Globale VAL (Sostituisce il vecchio confronto 1-a-1)
        bbox_m = ax_matrix.get_position()
        ax_btn_recap = self.fig.add_axes([bbox_m.x0 + 0.04 * bbox_m.width,
                                          bbox_m.y0 + 0.020 * bbox_m.height,
                                          0.92 * bbox_m.width,
                                          0.145 * bbox_m.height])
        self.btn_recap = Button(ax_btn_recap, "★ APRI BENCHMARK GLOBALE (SPLIT VAL)",
                                color='#FEF3C7', hovercolor='#FDE68A')
        self.btn_recap.label.set_fontsize(7.9); self.btn_recap.label.set_fontweight('bold'); self.btn_recap.label.set_color('#92400E')
        self.btn_recap.on_clicked(self.open_global_recap_window)
        self.ui_buttons.append((ax_btn_recap, self.btn_recap))

        self.fig.text(0.50, 0.012,
                      "[<- / ->] Frame  |  [P]: Soglia  |  [Z]: Modo (Classi/Zona Safety)  |  [V]: Filtro VAL  |  [F]: Evidenzia FN  |  [M]: Modello  |  [T]: GT  |  [R]: Benchmark VAL  |  [S]: Salva HD",
                      fontsize=7.8, color='#64748B', ha='center', style='italic')

        self.fig.canvas.draw_idle()

    # GESTIONE SELEZIONI E PULSANTI

    def select_macro(self, macro_key):
        if macro_key in MACRO_CATEGORIES:
            self.current_macro = macro_key
            # Se il modello corrente non fa parte di questa macro, seleziona il primo modello della macro
            models_in_macro = [child[0] for child in MACRO_CATEGORIES[macro_key]["models"]]
            if self.current_model_key not in models_in_macro:
                self.select_model(models_in_macro[0])
            else:
                self.render()

    def select_model(self, model_key):
        if model_key in MODEL_CONFIGS and model_key != self.current_model_key:
            self.current_model_key = model_key
            for m_k, m_info in MACRO_CATEGORIES.items():
                if any(model_key == child[0] for child in m_info["models"]):
                    self.current_macro = m_k
                    break
            self.load_frame(self.current_idx, broadcast=False)

    def select_gt(self, gt_key):
        if gt_key in GT_CONFIGS and gt_key != self.current_gt_key:
            self.current_gt_key = gt_key
            self.load_frame(self.current_idx, broadcast=False)

    def toggle_eval_mode(self, event=None):
        self.eval_mode = "ZONE_LEVEL" if getattr(self, 'eval_mode', "PER_CLASS") == "PER_CLASS" else "PER_CLASS"
        desc = "ALMENO 1 OSTACOLO (Basta 1 Rilevamento nella Zona -> TP)" if self.eval_mode == "ZONE_LEVEL" else "TUTTI GLI OSTACOLI (Valutazione Separata di Ogni Oggetto)"
        print(f"\n>>> [CRITERIO VALUTAZIONE AGGIORNATO]: {desc}")
        (self.inferred_occlusions, self.frame_stats,
         self.inference_latency_ms, self.f1_glob,
         self.rec_glob, self.prec_glob) = self.evaluate_frame(self.current_model_key, self.current_gt_key)
        self.render()
        if self.fig_cmp is not None and plt.fignum_exists(self.fig_cmp.number):
            self.render_comparison_window()
        if self.fig_recap is not None and plt.fignum_exists(self.fig_recap.number):
            self.render_global_recap_window()

    def toggle_threshold(self, event=None):
        thresholds = [0.50, 0.60, 0.70]
        cur_th = getattr(self, 'current_threshold', 0.50)
        diffs = [abs(th - cur_th) for th in thresholds]
        cur_idx = diffs.index(min(diffs))
        self.current_threshold = thresholds[(cur_idx + 1) % len(thresholds)]
        print(f"\n>>> [SOGLIA DECISIONALE AGGIORNATA]: {self.current_threshold:.2f}")
        (self.inferred_occlusions, self.frame_stats,
         self.inference_latency_ms, self.f1_glob,
         self.rec_glob, self.prec_glob) = self.evaluate_frame(self.current_model_key, self.current_gt_key)
        self.render()
        if self.fig_cmp is not None and plt.fignum_exists(self.fig_cmp.number):
            self.render_comparison_window()
        if self.fig_recap is not None and plt.fignum_exists(self.fig_recap.number):
            self.render_global_recap_window()

    def toggle_nav_filter(self, event=None):
        self.nav_val_only = not self.nav_val_only
        st = (f"SOLO VAL ({len(self.val_indices)} Frame Inediti)" if self.nav_val_only
              else f"TUTTI I FRAME ({self.total_frames})")
        print(f"\n>>> [FILTRO SPLIT NAVIGAZIONE] Attivo: {st}")
        if self.nav_val_only and self.current_idx not in self.val_indices:
            next_vals = [i for i in self.val_indices if i >= self.current_idx]
            target = next_vals[0] if next_vals else self.val_indices[0]
            self.load_frame(target, broadcast=True)
        else:
            self.render()

    def on_key_step(self, step):
        if self.nav_val_only and len(self.val_indices) > 0:
            if self.current_idx in self.val_indices:
                curr_pos = self.val_indices.index(self.current_idx)
                target_pos = (curr_pos + step) % len(self.val_indices)
            else:
                if step > 0:
                    next_vals = [i for i in self.val_indices if i >= self.current_idx]
                    target_pos = self.val_indices.index(next_vals[0]) if next_vals else 0
                else:
                    prev_vals = [i for i in self.val_indices if i <= self.current_idx]
                    target_pos = self.val_indices.index(prev_vals[-1]) if prev_vals else len(self.val_indices) - 1
            target = self.val_indices[target_pos]
            self.load_frame(target, broadcast=True)
        else:
            target = self.current_idx + step
            if 0 <= target < self.total_frames:
                self.load_frame(target, broadcast=True)

    def on_jump_frame(self, text):
        try:
            val = int(text.strip())
            target = val - 1
            if 0 <= target < self.total_frames:
                self.load_frame(target, broadcast=True)
            else:
                if hasattr(self, 'txt_frame') and self.txt_frame is not None:
                    self.txt_frame.set_val(str(self.current_idx + 1))
                if hasattr(self, 'cmp_txt_frame') and self.cmp_txt_frame is not None:
                    self.cmp_txt_frame.set_val(str(self.current_idx + 1))
        except ValueError:
            if hasattr(self, 'txt_frame') and self.txt_frame is not None:
                self.txt_frame.set_val(str(self.current_idx + 1))
            if hasattr(self, 'cmp_txt_frame') and self.cmp_txt_frame is not None:
                self.cmp_txt_frame.set_val(str(self.current_idx + 1))

    # FINESTRA DEDICATA CONFRONTO 1-A-1 (SIDE-BY-SIDE TRA MODELLI / SCORE)

    def open_comparison_window(self, event=None):
        """Apre o porta in primo piano una nuova finestra per il confronto 1-a-1."""
        print("\n>>> Apertura Finestra di Confronto 1-a-1...")
        if self.fig_cmp is None or not plt.fignum_exists(self.fig_cmp.number):
            self.fig_cmp = plt.figure(num="nuScenes BEV - Confronto Scientifico 1-a-1",
                                      figsize=(18, 9.2), facecolor='#FFFFFF')
            self.fig_cmp.canvas.manager.set_window_title("nuScenes BEV - Confronto Scientifico 1-a-1")
            self.fig_cmp.canvas.mpl_connect('key_press_event', self.on_key)
            self.cmp_model_a_key = self.current_model_key
            if self.cmp_model_b_key == self.cmp_model_a_key:
                self.cmp_model_b_key = "REAL_GT" if self.cmp_model_a_key == "NEURO_SIMB" else "NEURO_SIMB"
        else:
            plt.figure(self.fig_cmp.number)

        self.render_comparison_window()
        if not self.headless:
            plt.show(block=False)

    def render_comparison_window(self):
        """Disegna il confronto side-by-side tra Modello Mappa A e Modello Mappa B."""
        if self.fig_cmp is None or not plt.fignum_exists(self.fig_cmp.number):
            return

        self.fig_cmp.clf()
        self.fig_cmp.patch.set_facecolor('#FFFFFF')
        self.cmp_ui_buttons = []

        c_border = '#0F172A'
        self.cmp_gt_key = self.current_gt_key
        gt_cfg = GT_CONFIGS.get(self.cmp_gt_key, GT_CONFIGS["HYBRID"])

        # Valutazione Modello Mappa A
        (occs_a, stats_a, lat_a,
         f1_a, rec_a, prec_a) = self.evaluate_frame(self.cmp_model_a_key, self.cmp_gt_key)

        # Valutazione Modello Mappa B
        (occs_b, stats_b, lat_b,
         f1_b, rec_b, prec_b) = self.evaluate_frame(self.cmp_model_b_key, self.cmp_gt_key)

        # Header superiore riga 1: Titolo pulito con GT target integrata
        th_val = getattr(self, 'current_threshold', 0.50)
        m_mode = getattr(self, 'eval_mode', "PER_CLASS")
        mode_txt = "ALMENO 1 OSTACOLO" if m_mode == "ZONE_LEVEL" else "TUTTI GLI OSTACOLI"

        self.fig_cmp.text(0.030, 0.966,
                          f"Figure 5b: Confronto 1-a-1 tra Modelli ({mode_txt})  |  Target GT: {gt_cfg['short']}",
                          fontsize=9.2, fontweight='bold', color=c_border, ha='left', va='center')

        # Badge Split nel confronto
        sample = self.adapter.all_samples[self.current_idx]
        sc_name = self.adapter.nusc.get('scene', sample['scene_token'])['name']
        is_val = sc_name in getattr(self, 'val_scenes', set())
        split_lbl = f"[{'VAL ★' if is_val else 'TRAIN'} - {sc_name}]"
        split_bg = "#FEF3C7" if is_val else "#EFF6FF"
        split_fg = "#92400E" if is_val else "#1E40AF"
        split_edge = "#F59E0B" if is_val else "#3B82F6"
        self.fig_cmp.text(0.335, 0.966, split_lbl,
                          fontsize=7.8, fontweight='bold', color=split_fg, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.20', facecolor=split_bg, edgecolor=split_edge, linewidth=1.0))

        # Pulsante Soglia nel confronto
        th_bg_cmp = '#EFF6FF' if th_val == 0.50 else ('#F0FDF4' if th_val == 0.60 else '#FEF3C7')
        th_fg_cmp = '#1D4ED8' if th_val == 0.50 else ('#15803D' if th_val == 0.60 else '#92400E')
        ax_cmp_th = self.fig_cmp.add_axes([0.410, 0.950, 0.088, 0.028])
        btn_cmp_th = Button(ax_cmp_th, f"◆ SOGLIA: {th_val:.2f}", color=th_bg_cmp, hovercolor='#DBEAFE')
        btn_cmp_th.label.set_fontsize(7.2); btn_cmp_th.label.set_fontweight('bold'); btn_cmp_th.label.set_color(th_fg_cmp)
        btn_cmp_th.on_clicked(self.toggle_threshold)
        self.cmp_ui_buttons.append((ax_cmp_th, btn_cmp_th))

        # Pulsante Modalità nel confronto
        mode_bg_cmp = '#FEF3C7' if m_mode == "ZONE_LEVEL" else '#F1F5F9'
        mode_fg_cmp = '#92400E' if m_mode == "ZONE_LEVEL" else '#334155'
        mode_cmp_lbl = f"★ {mode_txt}" if m_mode == "ZONE_LEVEL" else mode_txt
        ax_cmp_mode = self.fig_cmp.add_axes([0.504, 0.950, 0.112, 0.028])
        btn_cmp_mode = Button(ax_cmp_mode, mode_cmp_lbl, color=mode_bg_cmp, hovercolor='#FDE68A' if m_mode == "ZONE_LEVEL" else '#E2E8F0')
        btn_cmp_mode.label.set_fontsize(6.6); btn_cmp_mode.label.set_fontweight('bold'); btn_cmp_mode.label.set_color(mode_fg_cmp)
        btn_cmp_mode.on_clicked(self.toggle_eval_mode)
        self.cmp_ui_buttons.append((ax_cmp_mode, btn_cmp_mode))

        # Pulsante Evidenzia FN nel confronto
        fn_bg = '#E11D48' if self.highlight_fn else '#F1F5F9'
        fn_fg = '#FFFFFF' if self.highlight_fn else '#475569'
        fn_lbl = '⚠ FN: ON' if self.highlight_fn else '⚠ Evidenzia FN'
        ax_fn_btn = self.fig_cmp.add_axes([0.620, 0.950, 0.125, 0.028])
        btn_fn = Button(ax_fn_btn, fn_lbl, color=fn_bg, hovercolor='#BE123C' if self.highlight_fn else '#E2E8F0')
        btn_fn.label.set_fontsize(7.4); btn_fn.label.set_fontweight('bold'); btn_fn.label.set_color(fn_fg)
        btn_fn.on_clicked(self.toggle_highlight_fn)
        self.cmp_ui_buttons.append((ax_fn_btn, btn_fn))

        # Navigazione frame con TextBox salto rapido in alto a destra
        ax_prev = self.fig_cmp.add_axes([0.768, 0.950, 0.038, 0.028])
        btn_prev = Button(ax_prev, "◀ Prec", color='#F1F5F9', hovercolor='#E2E8F0')
        btn_prev.label.set_fontsize(7.8); btn_prev.label.set_fontweight('bold'); btn_prev.label.set_color('#1E293B')
        btn_prev.on_clicked(lambda ev: self.on_key_step(-1))
        self.cmp_ui_buttons.append((ax_prev, btn_prev))

        self.fig_cmp.text(0.812, 0.964, "Frame:", fontsize=8.0, fontweight='bold', color=c_border, ha='left', va='center')
        ax_tb = self.fig_cmp.add_axes([0.852, 0.950, 0.052, 0.028])
        self.cmp_txt_frame = TextBox(ax_tb, '', initial=str(self.current_idx + 1),
                                     color='#F8FAFC', hovercolor='#E2E8F0')
        self.cmp_txt_frame.text_disp.set_color('#1D4ED8')
        self.cmp_txt_frame.text_disp.set_fontweight('bold')
        self.cmp_txt_frame.on_submit(self.on_jump_frame)
        self.cmp_ui_buttons.append((ax_tb, self.cmp_txt_frame))

        ax_next = self.fig_cmp.add_axes([0.912, 0.950, 0.038, 0.028])
        btn_next = Button(ax_next, "Succ ▶", color='#F1F5F9', hovercolor='#E2E8F0')
        btn_next.label.set_fontsize(7.8); btn_next.label.set_fontweight('bold'); btn_next.label.set_color('#1E293B')
        btn_next.on_clicked(lambda ev: self.on_key_step(1))
        self.cmp_ui_buttons.append((ax_next, btn_next))

        # Header riga 2: Selettori Modelli per Mappa A e Mappa B

        # Mappa A (Sinistra, tema Blu #2563EB)
        self.fig_cmp.text(0.030, 0.914, "Mappa A (Modello):",
                          fontsize=8.2, fontweight='bold', color='#1E293B', ha='left', va='center')

        choices_a = [
            ("NEURO_SIMB", "★ NEURO", 0.130, 0.084),
            ("REAL_GT", "GT REALE", 0.220, 0.080),
            ("POS_ONLY", "SOLO POS.", 0.306, 0.084),
            ("BAYES", "BAYES", 0.396, 0.080),
        ]
        for m_k, m_lbl, x_pos, w_btn in choices_a:
            is_act = (self.cmp_model_a_key == m_k)
            bg_c = '#2563EB' if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            if self.modello_mancante(m_k, self.cmp_gt_key):
                bg_c = '#64748B' if is_act else '#F1F5F9'
                tx_c = '#FFFFFF' if is_act else '#94A3B8'

            ax_btn = self.fig_cmp.add_axes([x_pos, 0.900, w_btn, 0.027])
            btn = Button(ax_btn, m_lbl, color=bg_c, hovercolor='#93C5FD' if not is_act else bg_c)
            btn.label.set_fontsize(7.3); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=m_k: self.select_cmp_model_a(k))
            self.cmp_ui_buttons.append((ax_btn, btn))

        # Mappa B (Destra, tema Viola #7C3AED)
        self.fig_cmp.text(0.518, 0.914, "Mappa B (Modello):",
                          fontsize=8.2, fontweight='bold', color='#1E293B', ha='left', va='center')

        choices_b = [
            ("NEURO_SIMB", "★ NEURO", 0.618, 0.084),
            ("REAL_GT", "GT REALE", 0.708, 0.080),
            ("POS_ONLY", "SOLO POS.", 0.794, 0.084),
            ("BAYES", "BAYES", 0.884, 0.080),
        ]
        for m_k, m_lbl, x_pos, w_btn in choices_b:
            is_act = (self.cmp_model_b_key == m_k)
            bg_c = '#7C3AED' if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            if self.modello_mancante(m_k, self.cmp_gt_key):
                bg_c = '#64748B' if is_act else '#F1F5F9'
                tx_c = '#FFFFFF' if is_act else '#94A3B8'

            ax_btn = self.fig_cmp.add_axes([x_pos, 0.900, w_btn, 0.027])
            btn = Button(ax_btn, m_lbl, color=bg_c, hovercolor='#DDD6FE' if not is_act else bg_c)
            btn.label.set_fontsize(7.3); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=m_k: self.select_cmp_model_b(k))
            self.cmp_ui_buttons.append((ax_btn, btn))

        # Layout 2 colonne con sotto-card:
        # Colonna Sinistra (Mappa A) | Colonna Destra (Mappa B)
        gs = GridSpec(2, 2, figure=self.fig_cmp, left=0.03, right=0.97, bottom=0.075, top=0.845,
                      width_ratios=[1.0, 1.0], height_ratios=[1.7, 0.55], wspace=0.08, hspace=0.18)

        ax_map_a = self.fig_cmp.add_subplot(gs[0, 0])
        ax_map_b = self.fig_cmp.add_subplot(gs[0, 1])
        ax_card_a = self.fig_cmp.add_subplot(gs[1, 0])
        ax_card_b = self.fig_cmp.add_subplot(gs[1, 1])

        # Disegno Mappe BEV 25m
        cfg_a = MODEL_CONFIGS.get(self.cmp_model_a_key, MODEL_CONFIGS["NEURO_SIMB"])
        cfg_b = MODEL_CONFIGS.get(self.cmp_model_b_key, MODEL_CONFIGS["REAL_GT"])

        mancante_a = self.modello_mancante(self.cmp_model_a_key, self.cmp_gt_key)
        mancante_b = self.modello_mancante(self.cmp_model_b_key, self.cmp_gt_key)

        self.draw_bev_map(ax_map_a, occs_a,
                          title=f"MAPPA A: {cfg_a['title']}" + ("  [MANCANTE]" if mancante_a else ""),
                          badge_text="PESI MANCANTI" if mancante_a else f"F1: {f1_a:.1f}%  |  Rec: {rec_a:.1f}%",
                          badge_bg='#64748B' if mancante_a else '#2563EB')
        if mancante_a:
            self._disegna_avviso_mancante(ax_map_a, self.cmp_model_a_key)

        self.draw_bev_map(ax_map_b, occs_b,
                          title=f"MAPPA B: {cfg_b['title']}" + ("  [MANCANTE]" if mancante_b else ""),
                          badge_text="PESI MANCANTI" if mancante_b else f"F1: {f1_b:.1f}%  |  Rec: {rec_b:.1f}%",
                          badge_bg='#64748B' if mancante_b else '#7C3AED')
        if mancante_b:
            self._disegna_avviso_mancante(ax_map_b, self.cmp_model_b_key)

        # Card Metriche Mappa A
        self._render_cmp_card(ax_card_a, "MAPPA A", cfg_a["title"],
                              f1_a, rec_a, prec_a, stats_a, lat_a, '#2563EB', mancante=mancante_a)

        # Card Metriche Mappa B
        self._render_cmp_card(ax_card_b, "MAPPA B", cfg_b["title"],
                              f1_b, rec_b, prec_b, stats_b, lat_b, '#7C3AED', mancante=mancante_b)

        # Banner Delta Confronto Diretto in basso
        delta_f1 = f1_a - f1_b
        delta_rec = rec_a - rec_b
        delta_tp = sum(stats_a[c]['tp'] for c in range(3)) - sum(stats_b[c]['tp'] for c in range(3))
        delta_fn = sum(stats_a[c]['fn'] for c in range(3)) - sum(stats_b[c]['fn'] for c in range(3))

        if mancante_a or mancante_b:
            # Confronto non significativo se almeno un modello non ha pesi
            winner_txt = "Confronto non disponibile: pesi mancanti per " + \
                " e ".join(t for t, m in (("Mappa A", mancante_a), ("Mappa B", mancante_b)) if m)
            d_col = '#475569'
            bg_d = '#F1F5F9'
        elif delta_f1 > 0:
            winner_txt = f"Mappa A supera Mappa B (+{delta_f1:.1f}% F1)"
            d_col = '#15803D'
            bg_d = '#F0FDF4'
        elif delta_f1 < 0:
            winner_txt = f"Mappa B supera Mappa A ({delta_f1:.1f}% F1)"
            d_col = '#B91C1C'
            bg_d = '#FEF2F2'
        else:
            winner_txt = "Pari merito (F1 identico)"
            d_col = '#475569'
            bg_d = '#F8FAFC'

        banner_txt = (f"BILANCIO CONFRONTO 1-A-1:  "
                      f"Δ F1-Score: {delta_f1:+.1f}%  |  "
                      f"Δ Recall: {delta_rec:+.1f}%  |  "
                      f"Δ Pericoli Trovati (TP): {delta_tp:+d}  |  "
                      f"Δ Pericoli Mancati (FN): {delta_fn:+d}   [{winner_txt}]")

        self.fig_cmp.text(0.50, 0.025, banner_txt,
                          fontsize=9.0, fontweight='bold', color=d_col, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_d, edgecolor=d_col, linewidth=1.1))

        self.fig_cmp.canvas.draw_idle()

    def _render_cmp_card(self, ax, tag, model_title, f1, rec, prec, stats, latency, theme_color, mancante=False):
        """Renderizza la card riassuntiva di performance per un modello nella finestra di confronto."""
        c_border = '#0F172A'
        ax.set_facecolor('#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color(c_border); spine.set_linewidth(1.1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])

        # Modello senza pesi: card neutra grigia al posto di metriche fittizie a 0%
        if mancante:
            ax.add_patch(Rectangle((0.025, 0.10), 0.22, 0.80, transform=ax.transAxes,
                                    facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1.2, zorder=2))
            ax.text(0.135, 0.50, "MANCANTE", transform=ax.transAxes,
                    fontsize=12.0, fontweight='bold', color='#64748B', ha='center', va='center')
            ax.text(0.28, 0.72, f"{tag}: {model_title}", transform=ax.transAxes,
                    fontsize=8.2, fontweight='bold', color='#64748B')
            ax.text(0.28, 0.42, "PESI MANCANTI: metriche non disponibili per questo modello",
                    transform=ax.transAxes, fontsize=8.0, color='#475569')
            return

        # Box F1 grande
        f1_col = '#15803D' if f1 >= 70.0 else ('#B45309' if f1 >= 50.0 else '#B91C1C')
        f1_bg = '#DCFCE7' if f1 >= 70.0 else ('#FEF3C7' if f1 >= 50.0 else '#FEE2E2')
        ax.add_patch(Rectangle((0.025, 0.10), 0.22, 0.80, transform=ax.transAxes,
                                facecolor=f1_bg, edgecolor=f1_col, linewidth=1.2, zorder=2))
        ax.text(0.135, 0.68, "F1-SCORE", transform=ax.transAxes,
                fontsize=7.8, fontweight='bold', color=f1_col, ha='center')
        ax.text(0.135, 0.38, f"{f1:.1f}%", transform=ax.transAxes,
                fontsize=16.0, fontweight='bold', color=f1_col, ha='center')

        # Dati riassuntivi
        tot_tp = sum(stats[c]["tp"] for c in range(3))
        tot_fp = sum(stats[c]["fp"] for c in range(3))
        tot_fn = sum(stats[c]["fn"] for c in range(3))

        ax.text(0.28, 0.78, f"{tag}: {model_title}", transform=ax.transAxes,
                fontsize=8.2, fontweight='bold', color=theme_color)
        ax.text(0.28, 0.52, f"Recall: {rec:.1f}% ({tot_tp} trovati)   |   Precision: {prec:.1f}% (FP: {tot_fp})",
                transform=ax.transAxes, fontsize=8.0, color='#1E293B')
        ax.text(0.28, 0.26, f"Pericoli Mancati (FN): {tot_fn}   |   Latenza Inferenza GPU: {latency:.1f} ms",
                transform=ax.transAxes, fontsize=7.8, color='#64748B')

    def select_cmp_model_a(self, model_key):
        if model_key in MODEL_CONFIGS and model_key != self.cmp_model_a_key:
            self._preload_model(model_key)
            self.cmp_model_a_key = model_key
            self.render_comparison_window()

    def select_cmp_model_b(self, model_key):
        if model_key in MODEL_CONFIGS and model_key != self.cmp_model_b_key:
            self._preload_model(model_key)
            self.cmp_model_b_key = model_key
            self.render_comparison_window()

    # FINESTRA DEDICATA RECAP GLOBALE: TRAIN vs VAL (BENCHMARK GENERALIZZAZIONE)

    # FINESTRA DEDICATA BENCHMARK GLOBALE: SPLIT DI VALIDAZIONE (VAL)

    def open_global_recap_window(self, event=None):
        """Apre o porta in primo piano la finestra di benchmark generale sullo split di validazione (VAL)."""
        print("\n>>> Apertura Finestra Benchmark Globale (Split VAL)...")
        if self.fig_recap is None or not plt.fignum_exists(self.fig_recap.number):
            self.fig_recap = plt.figure(num="nuScenes BEV - Benchmark Globale VAL",
                                        figsize=(18, 9.4), facecolor='#FFFFFF')
            self.fig_recap.canvas.manager.set_window_title("nuScenes BEV - Benchmark Globale (Split VAL Inedito)")
            self.fig_recap.canvas.mpl_connect('key_press_event', self.on_key)
            self.recap_model_key = self.current_model_key
            self.recap_gt_key = self.current_gt_key
            self.recap_macro_key = self.current_macro
        else:
            plt.figure(self.fig_recap.number)

        self.render_global_recap_window()
        if not self.headless:
            plt.show(block=False)

    def select_recap_macro(self, macro_key):
        if macro_key in MACRO_CATEGORIES:
            self.recap_macro_key = macro_key
            models_in_macro = [child[0] for child in MACRO_CATEGORIES[macro_key]["models"]]
            if self.recap_model_key not in models_in_macro:
                self.recap_model_key = models_in_macro[0]
            self.render_global_recap_window()

    def select_recap_model(self, model_key):
        if model_key != self.recap_model_key:
            self.recap_model_key = model_key
            for m_k, m_info in MACRO_CATEGORIES.items():
                if any(model_key == child[0] for child in m_info["models"]):
                    self.recap_macro_key = m_k
                    break
            self.render_global_recap_window()

    def select_recap_gt(self, gt_key):
        if gt_key != self.recap_gt_key:
            self.recap_gt_key = gt_key
            self.render_global_recap_window()

    def compute_split_aggregates(self, model_key, gt_key):
        """Aggrega metriche globali per classe su TRAIN e VAL dalla cache precalcolata.
        Restituisce anche quanti frame di ciascuno split sono effettivamente presenti in cache per modello/GT."""
        m_k = "NEURO_SIMB" if model_key in ["HYBRID", "NEURO_SIMB"] else model_key
        g_k = "NEURO_SIMB" if gt_key in ["HYBRID", "NEURO_SIMB"] else gt_key

        def empty_stats():
            return {c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0} for c in range(3)}

        train_stats = empty_stats()
        val_stats = empty_stats()
        n_train, n_val = 0, 0
        val_set = set(self.val_indices)
        cache_split = (self.eval_cache.get("split", "") if self.eval_cache else "").lower()
        th = getattr(self, 'current_threshold', 0.50)
        mode = getattr(self, 'eval_mode', "PER_CLASS")
        use_cached_st = (abs(th - 0.50) < 1e-4 and mode == "PER_CLASS")

        if self.eval_cache and "frames" in self.eval_cache:
            fc = self.eval_cache["frames"]
            for idx_str, f_data in fc.items():
                idx = int(idx_str)
                is_val = (cache_split == "val") or (idx in val_set)

                m_data = f_data.get("models", {}).get(m_k, {})
                if g_k not in m_data:
                    continue
                if is_val:
                    n_val += 1
                else:
                    n_train += 1
                curr_dict = val_stats if is_val else train_stats
                g_data = m_data.get(g_k, {})

                if use_cached_st and "stats" in g_data:
                    st = g_data.get("stats", {})
                    for c in range(3):
                        c_st = st.get(str(c), {})
                        curr_dict[c]["tp"] += c_st.get("tp", 0)
                        curr_dict[c]["fp"] += c_st.get("fp", 0)
                        curr_dict[c]["fn"] += c_st.get("fn", 0)
                        curr_dict[c]["gt_count"] += c_st.get("gt_count", 0)
                        curr_dict[c]["pred_count"] += c_st.get("pred_count", 0)
                elif mode == "ZONE_LEVEL":
                    for z in g_data.get("zones", []):
                        p_arr = z.get("probs_4", [0.0, 0.0, 0.0])[:3]
                        gt_arr = z.get("gt_target_4", [0, 0, 0])[:3]
                        pred_classes = [c for c in range(3) if p_arr[c] >= th]
                        gt_classes = [c for c in range(3) if gt_arr[c] == 1]
                        is_pred_h = len(pred_classes) > 0
                        is_gt_h = len(gt_classes) > 0

                        if is_pred_h:
                            m = [c for c in pred_classes if c in gt_classes]
                            if m:
                                curr_dict[m[0]]["tp"] += 1
                                curr_dict[m[0]]["gt_count"] += 1
                                curr_dict[m[0]]["pred_count"] += 1
                            else:
                                curr_dict[pred_classes[0]]["fp"] += 1
                                curr_dict[pred_classes[0]]["pred_count"] += 1
                                if is_gt_h:
                                    curr_dict[gt_classes[0]]["fn"] += 1
                                    curr_dict[gt_classes[0]]["gt_count"] += 1
                        elif is_gt_h:
                            curr_dict[gt_classes[0]]["fn"] += 1
                            curr_dict[gt_classes[0]]["gt_count"] += 1
                else:
                    for z in g_data.get("zones", []):
                        p_arr = z.get("probs_4", [0.0, 0.0, 0.0])
                        gt_arr = z.get("gt_target_4", [0, 0, 0])
                        for c in range(3):
                            p = 1 if p_arr[c] >= th else 0
                            g = gt_arr[c]
                            if g == 1: curr_dict[c]["gt_count"] += 1
                            if p == 1: curr_dict[c]["pred_count"] += 1
                            if p == 1 and g == 1: curr_dict[c]["tp"] += 1
                            elif p == 1 and g == 0: curr_dict[c]["fp"] += 1
                            elif p == 0 and g == 1: curr_dict[c]["fn"] += 1

        def calc_metrics(st_dict):
            t_tp = sum(st_dict[c]["tp"] for c in range(3))
            t_fp = sum(st_dict[c]["fp"] for c in range(3))
            t_fn = sum(st_dict[c]["fn"] for c in range(3))
            t_gt = sum(st_dict[c]["gt_count"] for c in range(3))
            prec = (t_tp / (t_tp + t_fp) * 100.0) if (t_tp + t_fp) > 0 else 0.0
            rec = (t_tp / (t_tp + t_fn) * 100.0) if (t_tp + t_fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class = {}
            for c in range(3):
                tp = st_dict[c]["tp"]
                fp = st_dict[c]["fp"]
                fn = st_dict[c]["fn"]
                gt = st_dict[c]["gt_count"]
                p = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
                r = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
                f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                per_class[c] = {"p": p, "r": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "gt": gt}

            return {"tp": t_tp, "fp": t_fp, "fn": t_fn, "gt": t_gt, "prec": prec, "rec": rec, "f1": f1, "per_class": per_class}

        tr_res, val_res = calc_metrics(train_stats), calc_metrics(val_stats)
        tr_res["n_frames"] = n_train
        val_res["n_frames"] = n_val
        return tr_res, val_res

    def render_global_recap_window(self):
        """Renderizza la finestra di benchmark globale focalizzata interamente sullo split di validazione (VAL)."""
        if self.fig_recap is None or not plt.fignum_exists(self.fig_recap.number):
            return

        self.fig_recap.clf()
        self.fig_recap.patch.set_facecolor('#FFFFFF')
        self.recap_ui_buttons = []

        c_border = '#0F172A'

        # Titolo e Badge Superiore
        th_val = getattr(self, 'current_threshold', 0.50)
        m_mode = getattr(self, 'eval_mode', "PER_CLASS")
        mode_txt = "ALMENO 1 OSTACOLO" if m_mode == "ZONE_LEVEL" else "TUTTI GLI OSTACOLI"
        self.fig_recap.text(0.030, 0.965,
                            f"Figure 5c: Benchmark Globale VAL ({mode_txt})",
                            fontsize=9.6, fontweight='bold', color=c_border, ha='left', va='center')

        # Pulsante Soglia anche in fig_recap
        th_bg_r = '#EFF6FF' if th_val == 0.50 else ('#F0FDF4' if th_val == 0.60 else '#FEF3C7')
        th_fg_r = '#1D4ED8' if th_val == 0.50 else ('#15803D' if th_val == 0.60 else '#92400E')
        ax_r_th = self.fig_recap.add_axes([0.365, 0.950, 0.095, 0.030])
        btn_r_th = Button(ax_r_th, f"◆ SOGLIA: {th_val:.2f}", color=th_bg_r, hovercolor='#DBEAFE')
        btn_r_th.label.set_fontsize(7.2); btn_r_th.label.set_fontweight('bold'); btn_r_th.label.set_color(th_fg_r)
        btn_r_th.on_clicked(self.toggle_threshold)
        self.recap_ui_buttons.append((ax_r_th, btn_r_th))

        # Pulsante Modalità anche in fig_recap
        mode_bg_r = '#FEF3C7' if m_mode == "ZONE_LEVEL" else '#F1F5F9'
        mode_fg_r = '#92400E' if m_mode == "ZONE_LEVEL" else '#334155'
        mode_r_lbl = f"★ {mode_txt}" if m_mode == "ZONE_LEVEL" else mode_txt
        ax_r_mode = self.fig_recap.add_axes([0.468, 0.950, 0.170, 0.030])
        btn_r_mode = Button(ax_r_mode, mode_r_lbl, color=mode_bg_r, hovercolor='#FDE68A' if m_mode == "ZONE_LEVEL" else '#E2E8F0')
        btn_r_mode.label.set_fontsize(6.8); btn_r_mode.label.set_fontweight('bold'); btn_r_mode.label.set_color(mode_fg_r)
        btn_r_mode.on_clicked(self.toggle_eval_mode)
        self.recap_ui_buttons.append((ax_r_mode, btn_r_mode))

        n_val_disp = len(self.val_indices) if hasattr(self, 'val_indices') and len(self.val_indices) > 0 else 6019
        self.fig_recap.text(0.650, 0.965, f"[SPLIT VAL: {n_val_disp} FRAME INEDITI — ZERO DATA LEAKAGE]",
                            fontsize=7.8, fontweight='bold', color='#92400E', ha='left', va='center',
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.1))

        # Riga 1: Supervisione / Addestramento & Target GT
        self.fig_recap.text(0.025, 0.922, "Supervisione:", fontsize=7.6, fontweight='bold', color='#475569', va='center')

        macro_items = [
            ("NEURO_SIMB", "★ Addestramento Ibrido (Anticipazione)", 0.110, 0.230, "#1D4ED8"),
            ("REAL", "◆ Addestramento Reale (nuScenes)", 0.346, 0.194, "#0D9488"),
            ("BAYES", "▲ Bayes", 0.546, 0.058, "#0284C7")
        ]
        for m_key, m_lbl, x_pos, w_btn, m_theme in macro_items:
            is_act = (self.recap_macro_key == m_key)
            bg_c = m_theme if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            ax_m = self.fig_recap.add_axes([x_pos, 0.908, w_btn, 0.027])
            btn_m = Button(ax_m, m_lbl, color=bg_c, hovercolor='#93C5FD' if not is_act else bg_c)
            btn_m.label.set_fontsize(6.7); btn_m.label.set_fontweight('bold'); btn_m.label.set_color(tx_c)
            btn_m.on_clicked(lambda ev, k=m_key: self.select_recap_macro(k))
            self.recap_ui_buttons.append((ax_m, btn_m))

        self.fig_recap.text(0.612, 0.922, "Target GT:", fontsize=7.6, fontweight='bold', color='#475569', va='center')
        gts = [
            ("NEURO_SIMB", "★ GT Anticipazione (Ibrida)", 0.672, 0.160),
            ("REAL", "GT Rilevamento (Reale nuScenes)", 0.838, 0.152)
        ]
        for g_k, g_lbl, x_pos, w_btn in gts:
            is_act = (self.recap_gt_key == g_k)
            bg_c = '#4338CA' if is_act else '#E2E8F0'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            ax_g = self.fig_recap.add_axes([x_pos, 0.908, w_btn, 0.027])
            btn_g = Button(ax_g, g_lbl, color=bg_c, hovercolor='#C7D2FE' if not is_act else bg_c)
            btn_g.label.set_fontsize(7.0); btn_g.label.set_fontweight('bold'); btn_g.label.set_color(tx_c)
            btn_g.on_clicked(lambda ev, k=g_k: self.select_recap_gt(k))
            self.recap_ui_buttons.append((ax_g, btn_g))

        # Riga 2: Modello nella macro-famiglia attiva
        self.fig_recap.text(0.030, 0.872, "Modello:", fontsize=7.8, fontweight='bold', color='#475569', va='center')

        if self.recap_macro_key == "NEURO_SIMB":
            child_models = [
                ("NEURO_SIMB", "★ Con Vincoli Simbolici (Full)", 0.118, 0.200),
                ("PURE_NEURAL", "Senza Vincoli Simbolici (Puro)", 0.328, 0.200)
            ]
        elif self.recap_macro_key == "REAL":
            child_models = [
                ("REAL_GT", "Con Vincoli Simbolici (Full)", 0.110, 0.170),
                ("REAL_PURE_NEURAL", "Senza Vincoli Simbolici (Puro)", 0.288, 0.170),
                ("POS_ONLY", "Solo Zone Piene", 0.466, 0.138)
            ]
        else:  # BAYES
            child_models = [
                ("BAYES", "Modello Analitico (No AI)", 0.110, 0.220)
            ]

        for m_k, m_lbl, x_pos, w_btn in child_models:
            is_act = (self.recap_model_key == m_k)
            bg_c = '#1E293B' if is_act else '#F1F5F9'
            tx_c = '#FFFFFF' if is_act else '#1E293B'
            if not self._cache_ha_modello(m_k, self.recap_gt_key):
                bg_c = '#64748B' if is_act else '#F8FAFC'
                tx_c = '#FFFFFF' if is_act else '#94A3B8'

            ax_btn = self.fig_recap.add_axes([x_pos, 0.858, w_btn, 0.027])
            btn = Button(ax_btn, m_lbl, color=bg_c, hovercolor='#CBD5E1' if not is_act else bg_c)
            btn.label.set_fontsize(6.7 if len(child_models) > 2 else 7.2); btn.label.set_fontweight('bold'); btn.label.set_color(tx_c)
            btn.on_clicked(lambda ev, k=m_k: self.select_recap_model(k))
            self.recap_ui_buttons.append((ax_btn, btn))

        # Calcolo aggregato per VAL
        tr_res, val_res = self.compute_split_aggregates(self.recap_model_key, self.recap_gt_key)
        recap_mancante = not self._cache_ha_modello(self.recap_model_key, self.recap_gt_key)

        # Layout 2 Colonne su VAL (Sinistra: KPI Card + Istogramma | Destra: Card per Classe + Tabella)
        gs = GridSpec(2, 2, figure=self.fig_recap, left=0.03, right=0.97, bottom=0.08, top=0.825,
                      width_ratios=[1.0, 1.25], height_ratios=[0.55, 1.45], wspace=0.10, hspace=0.22)

        ax_kpi = self.fig_recap.add_subplot(gs[0, 0])
        ax_bars = self.fig_recap.add_subplot(gs[1, 0])
        ax_cards = self.fig_recap.add_subplot(gs[0, 1])
        ax_table = self.fig_recap.add_subplot(gs[1, 1])

        cat_names = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]

        # 1. PANNELLO SINISTRO ALTO: KPI CARD GLOBALE

        ax_kpi.set_facecolor('#FFFFFF')
        for sp in ax_kpi.spines.values():
            sp.set_color(c_border); sp.set_linewidth(1.1)
        ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
        ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

        num_frames = val_res.get("n_frames", 0)
        tot_val = len(self.val_indices)

        if recap_mancante:
            ax_kpi.add_patch(Rectangle((0.02, 0.10), 0.22, 0.80, transform=ax_kpi.transAxes,
                                       facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1.3, zorder=2))
            ax_kpi.text(0.13, 0.50, "N/D", transform=ax_kpi.transAxes,
                        fontsize=17.0, fontweight='bold', color='#64748B', ha='center', va='center')
            ax_kpi.text(0.27, 0.65, "MODELLO MANCANTE (Pesi non disponibili o non valutato)",
                        transform=ax_kpi.transAxes, fontsize=8.6, fontweight='bold', color='#64748B')
            ax_kpi.text(0.27, 0.35, "Esegui il benchmark per popolare la cache: python valutazione/evaluate_final_official.py",
                        transform=ax_kpi.transAxes, fontsize=7.6, color='#64748B', style='italic')
        else:
            f1_val = val_res["f1"]
            f1_col = '#15803D' if f1_val >= 60.0 else ('#B45309' if f1_val >= 30.0 else '#B91C1C')
            bg_card = '#F0FDF4' if f1_val >= 60.0 else ('#FEF3C7' if f1_val >= 30.0 else '#FEF2F2')

            ax_kpi.add_patch(Rectangle((0.02, 0.10), 0.22, 0.80, transform=ax_kpi.transAxes,
                                       facecolor=bg_card, edgecolor=f1_col, linewidth=1.4, zorder=2))
            ax_kpi.text(0.13, 0.70, "F1-SCORE MEDIO", transform=ax_kpi.transAxes,
                        fontsize=7.8, fontweight='bold', color=f1_col, ha='center')
            ax_kpi.text(0.13, 0.36, f"{f1_val:.1f}%", transform=ax_kpi.transAxes,
                        fontsize=17.5, fontweight='bold', color=f1_col, ha='center')

            ax_kpi.text(0.27, 0.80, f"SPLIT DI VALIDAZIONE NUSCENES ({num_frames} di {tot_val} Fotogrammi Valutati)",
                        transform=ax_kpi.transAxes, fontsize=8.4, fontweight='bold', color='#1E40AF')
            ax_kpi.text(0.27, 0.56, f"Recall Globale: {val_res['rec']:.1f}%   |   Precision Globale: {val_res['prec']:.1f}%",
                        transform=ax_kpi.transAxes, fontsize=8.0, fontweight='bold', color='#1E293B')
            ax_kpi.text(0.27, 0.32, f"GT Totali (TP+FN): {val_res['tp'] + val_res['fn']}   |   TP: {val_res['tp']}   |   FP: {val_res['fp']}   |   FN: {val_res['fn']}",
                        transform=ax_kpi.transAxes, fontsize=7.4, color='#334155')
            ax_kpi.text(0.27, 0.12, "Condizione: ★ DATI MAI VISTI IN TRAINING (GENERALIZZAZIONE SENZA LEAKAGE)",
                        transform=ax_kpi.transAxes, fontsize=7.1, color='#15803D', style='italic')

        # 2. PANNELLO SINISTRO BASSO: ISTOGRAMMA METRICHE PER CLASSE & MEDIA

        ax_bars.set_facecolor('#FFFFFF')
        for sp in ax_bars.spines.values():
            sp.set_color(c_border); sp.set_linewidth(1.0)
        ax_bars.spines['top'].set_visible(False)
        ax_bars.spines['right'].set_visible(False)

        bar_groups = ["Auto", "Camion/Bus", "VRU", "MEDIA GLOB."]
        rec_list = [val_res["per_class"][0]["r"], val_res["per_class"][1]["r"], val_res["per_class"][2]["r"], val_res["rec"]]
        prec_list = [val_res["per_class"][0]["p"], val_res["per_class"][1]["p"], val_res["per_class"][2]["p"], val_res["prec"]]
        f1_list = [val_res["per_class"][0]["f1"], val_res["per_class"][1]["f1"], val_res["per_class"][2]["f1"], val_res["f1"]]

        x_ind = np.arange(len(bar_groups))
        w_b = 0.26

        bars_r = ax_bars.bar(x_ind - w_b, rec_list, w_b, label='Recall (%)', color='#0284C7', edgecolor=c_border, linewidth=0.9, zorder=3)
        bars_p = ax_bars.bar(x_ind, prec_list, w_b, label='Precision (%)', color='#EA580C', edgecolor=c_border, linewidth=0.9, zorder=3)
        bars_f = ax_bars.bar(x_ind + w_b, f1_list, w_b, label='F1-Score (%)', color='#10B981', edgecolor=c_border, linewidth=0.9, zorder=3)

        ax_bars.set_ylim(0, 115)
        ax_bars.set_xticks(x_ind)
        ax_bars.set_xticklabels(bar_groups, fontsize=8.0, fontweight='bold', color=c_border)
        ax_bars.set_ylabel("Score [%]", fontsize=8.2, fontweight='bold', color=c_border)
        ax_bars.set_title(f"Istogramma Metriche per Macro-Classe & Media ({mode_txt} | Soglia {th_val:.2f})", fontsize=8.8, fontweight='bold', pad=8, color=c_border)
        ax_bars.grid(axis='y', color='#E2E8F0', linestyle='-', linewidth=0.7, zorder=1)
        ax_bars.legend(loc='upper right', framealpha=0.92, fontsize=7.4, ncol=3)

        for bar_seq in [bars_r, bars_p, bars_f]:
            for bar in bar_seq:
                h = bar.get_height()
                if h > 0:
                    ax_bars.annotate(f"{h:.1f}%",
                                     xy=(bar.get_x() + bar.get_width() / 2, h),
                                     xytext=(0, 2.5), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=6.8, fontweight='bold',
                                     color='#1E293B', clip_on=False)

        # 3. PANNELLO DESTRO ALTO: CARD ANALITICHE DELLE 3 CLASSI

        ax_cards.set_facecolor('#FFFFFF')
        for sp in ax_cards.spines.values():
            sp.set_color(c_border); sp.set_linewidth(1.1)
        ax_cards.set_xlim(0, 1); ax_cards.set_ylim(0, 1)
        ax_cards.set_xticks([]); ax_cards.set_yticks([])

        c_palette = [
            ("Auto (Veicoli Leggeri)", '#2563EB', '#EFF6FF', val_res["per_class"][0]),
            ("Camion / Bus (Mezzi Pesanti)", '#7C3AED', '#F5F3FF', val_res["per_class"][1]),
            ("VRU (Pedoni & Bici)", '#16A34A', '#F0FDF4', val_res["per_class"][2])
        ]

        w_card = 0.315
        gap_card = 0.02
        for idx_c, (c_title, c_accent, c_bg, c_res) in enumerate(c_palette):
            x_c0 = 0.015 + idx_c * (w_card + gap_card)
            ax_cards.add_patch(Rectangle((x_c0, 0.08), w_card, 0.84, transform=ax_cards.transAxes,
                                         facecolor=c_bg, edgecolor=c_accent, linewidth=1.1, zorder=2))
            ax_cards.text(x_c0 + w_card / 2.0, 0.76, c_title, transform=ax_cards.transAxes,
                          fontsize=7.3, fontweight='bold', color=c_accent, ha='center')
            ax_cards.text(x_c0 + w_card / 2.0, 0.50, f"F1: {c_res['f1']:.1f}%", transform=ax_cards.transAxes,
                          fontsize=11.5, fontweight='bold', color='#0F172A', ha='center')
            ax_cards.text(x_c0 + w_card / 2.0, 0.28, f"P: {c_res['p']:.1f}%  |  R: {c_res['r']:.1f}%", transform=ax_cards.transAxes,
                          fontsize=7.2, color='#334155', ha='center')
            gt_tot_c = c_res['tp'] + c_res['fn']
            ax_cards.text(x_c0 + w_card / 2.0, 0.12, f"GT: {gt_tot_c}  (TP={c_res['tp']}, FN={c_res['fn']})", transform=ax_cards.transAxes,
                          fontsize=6.8, color='#64748B', ha='center')

        # 4. PANNELLO DESTRO BASSO: TABELLA PRESTAZIONI DETTAGLIATE

        ax_table.set_facecolor('#FFFFFF')
        for sp in ax_table.spines.values():
            sp.set_color(c_border); sp.set_linewidth(1.1)
        ax_table.set_xlim(0, 1); ax_table.set_ylim(0, 1)
        ax_table.set_xticks([]); ax_table.set_yticks([])
        ax_table.set_title(f"Tabella Prestazioni Dettagliate ({mode_txt} | Soglia {th_val:.2f}) - Validazione ({num_frames} Frame Inediti)",
                           fontsize=9.0, fontweight='bold', pad=8, color=c_border)

        headers = ["Categoria", "GT Tot", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"]
        col_x = [0.03, 0.24, 0.33, 0.42, 0.51, 0.60, 0.74, 0.87]

        ax_table.add_patch(Rectangle((0.02, 0.85), 0.96, 0.12, transform=ax_table.transAxes,
                                     facecolor='#F1F5F9', edgecolor='none', zorder=2))
        for x_h, h_txt in zip(col_x, headers):
            ax_table.text(x_h, 0.91, h_txt, transform=ax_table.transAxes,
                          fontsize=7.3, fontweight='bold', color=c_border, va='center')

        y_r = 0.73
        for c in range(3):
            c_data = val_res["per_class"][c]
            gt_c_tot = c_data['tp'] + c_data['fn']
            ax_table.text(col_x[0], y_r, cat_names[c], transform=ax_table.transAxes,
                          fontsize=7.4, fontweight='bold', color='#1E293B', va='center')
            ax_table.text(col_x[1], y_r, f"{gt_c_tot}", transform=ax_table.transAxes,
                          fontsize=7.4, fontweight='bold', color='#334155', va='center')
            ax_table.text(col_x[2], y_r, f"{c_data['tp']}", transform=ax_table.transAxes,
                          fontsize=7.4, fontweight='bold', color='#10B981', va='center')
            ax_table.text(col_x[3], y_r, f"{c_data['fp']}", transform=ax_table.transAxes,
                          fontsize=7.4, fontweight='bold', color='#EF4444', va='center')
            ax_table.text(col_x[4], y_r, f"{c_data['fn']}", transform=ax_table.transAxes,
                          fontsize=7.4, fontweight='bold', color='#8B5CF6', va='center')
            ax_table.text(col_x[5], y_r, f"{c_data['p']:.1f}%", transform=ax_table.transAxes,
                          fontsize=7.4, color='#0284C7', va='center')
            rec_str = f"{c_data['r']:.1f}%" if gt_c_tot > 0 else "0.0% (N/A)"
            ax_table.text(col_x[6], y_r, rec_str, transform=ax_table.transAxes,
                          fontsize=7.4, color='#0284C7' if gt_c_tot > 0 else '#94A3B8', va='center')
            ax_table.text(col_x[7], y_r, f"{c_data['f1']:.1f}%", transform=ax_table.transAxes,
                          fontsize=7.8, fontweight='bold', color='#0F172A', va='center')
            y_r -= 0.135

        # Media Globale
        res_gt_tot = val_res['tp'] + val_res['fn']
        ax_table.add_patch(Rectangle((0.02, 0.08), 0.96, 0.12, transform=ax_table.transAxes,
                                     facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.1, zorder=2))
        ax_table.text(col_x[0], 0.14, "MEDIA GLOBALE", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#92400E', va='center')
        ax_table.text(col_x[1], 0.14, f"{res_gt_tot}", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#334155', va='center')
        ax_table.text(col_x[2], 0.14, f"{val_res['tp']}", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#10B981', va='center')
        ax_table.text(col_x[3], 0.14, f"{val_res['fp']}", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#EF4444', va='center')
        ax_table.text(col_x[4], 0.14, f"{val_res['fn']}", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#8B5CF6', va='center')
        ax_table.text(col_x[5], 0.14, f"{val_res['prec']:.1f}%", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#0284C7', va='center')
        ax_table.text(col_x[6], 0.14, f"{val_res['rec']:.1f}%", transform=ax_table.transAxes,
                      fontsize=7.8, fontweight='bold', color='#0284C7', va='center')
        ax_table.text(col_x[7], 0.14, f"{val_res['f1']:.1f}%", transform=ax_table.transAxes,
                      fontsize=8.5, fontweight='bold', color='#92400E', va='center')

        # Banner Inferiore: Bilancio e Analisi Scientifica di Validazione
        if recap_mancante:
            banner = "BILANCIO VALIDAZIONE: Modello MANCANTE (nessun risultato in cache per il modello/GT selezionati)."
            b_col = '#475569'; b_bg = '#F1F5F9'
        elif val_res['f1'] >= 70.0:
            banner = f"BILANCIO SCIENTIFICO VALIDAZIONE:  F1={val_res['f1']:.1f}%  |  Recall={val_res['rec']:.1f}%  |  Precision={val_res['prec']:.1f}%    [Prestazioni Eccellenti: Elevata capacità di anticipazione senza overfitting]"
            b_col = '#15803D'; b_bg = '#F0FDF4'
        elif val_res['f1'] >= 50.0:
            banner = f"BILANCIO SCIENTIFICO VALIDAZIONE:  F1={val_res['f1']:.1f}%  |  Recall={val_res['rec']:.1f}%  |  Precision={val_res['prec']:.1f}%    [Prestazioni Moderate / Baseline Analitica: Solido comportamento globale]"
            b_col = '#0369A1'; b_bg = '#F0F9FF'
        else:
            banner = f"BILANCIO SCIENTIFICO VALIDAZIONE:  F1={val_res['f1']:.1f}%  |  Recall={val_res['rec']:.1f}%  |  Precision={val_res['prec']:.1f}%    [Studio Comparativo / Ablazione: Valutazione su Ground Truth Reale 3D]"
            b_col = '#B45309'; b_bg = '#FFFBEB'

        self.fig_recap.text(0.50, 0.025, banner,
                            fontsize=9.0, fontweight='bold', color=b_col, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.35', facecolor=b_bg, edgecolor=b_col, linewidth=1.1))

        self.fig_recap.canvas.draw_idle()

    def toggle_highlight_fn(self, event=None):
        self.highlight_fn = not self.highlight_fn
        print(f"\n>>> Evidenziazione Falsi Negativi (FN): {'ATTIVA (Modalita Diagnostica Errore)' if self.highlight_fn else 'DISATTIVATA'}")
        self.render()
        if self.fig_cmp is not None and plt.fignum_exists(self.fig_cmp.number):
            self.render_comparison_window()
        if self.fig_recap is not None and plt.fignum_exists(self.fig_recap.number):
            self.render_global_recap_window()

    # GESTIONE EVENTI (MOUSE, TASTIERA, TOOLTIP)

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.on_key_step(1)
        elif event.key in ['left', 'a']:
            self.on_key_step(-1)
        elif event.key in ['v', 'V']:
            self.toggle_nav_filter()
        elif event.key in ['p', 'P']:
            self.toggle_threshold()
        elif event.key in ['z', 'Z']:
            self.toggle_eval_mode()
        elif event.key in ['r', 'R']:
            self.open_global_recap_window()
        elif event.key in ['f', 'F']:
            self.toggle_highlight_fn()
        elif event.key in ['m', 'M']:
            cycle = ["NEURO_SIMB", "REAL_GT", "POS_ONLY"]
            cur_i = cycle.index(self.current_model_key) if self.current_model_key in cycle else 0
            self.select_model(cycle[(cur_i + 1) % len(cycle)])
        elif event.key in ['t', 'T']:
            cycle_gt = ["NEURO_SIMB", "REAL", "GEOMETRIC", "SEMANTIC"]
            cur_i = cycle_gt.index(self.current_gt_key) if self.current_gt_key in cycle_gt else 0
            self.select_gt(cycle_gt[(cur_i + 1) % len(cycle_gt)])
        elif event.key in ['c', 'C']:
            self.open_comparison_window()
        elif event.key in ['s', 'S']:
            # Salva la figura della finestra in cui e' stato premuto il tasto
            fig_evento = event.canvas.figure if event.canvas is not None else self.fig
            out_dir = os.path.join(ROOT_DIR, "documentazione", "immagini_tesi")
            fpath = self._salva_figura(fig_evento, out_dir, self._nome_file_figura(fig_evento))
            print(f"\n[SALVATA CON SUCCESSO]: Immagine salvata in:\n  -> {fpath}")

    def _nome_file_figura(self, fig):
        """Nome file descrittivo in base alla finestra (dashboard, confronto 1-a-1, recap)."""
        n = self.current_idx + 1
        if self.fig_cmp is not None and fig is self.fig_cmp:
            return (f"fig_5b_confronto_{self.cmp_model_a_key.lower()}_vs_{self.cmp_model_b_key.lower()}"
                    f"_gt_{self.cmp_gt_key.lower()}_sample{n}.png")
        if self.fig_recap is not None and fig is self.fig_recap:
            return f"fig_5c_recap_{self.recap_model_key.lower()}_gt_{self.recap_gt_key.lower()}.png"
        return f"fig_5_evaluation_metrics_{self.current_model_key.lower()}_vs_{self.current_gt_key.lower()}_sample{n}.png"

    def _salva_figura(self, fig, out_dir, nome):
        """Salva una figura a 300 DPI con sfondo bianco e bbox stretto; restituisce il percorso."""
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, nome)
        fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        return fpath

    def esporta_batch(self, out_dir):
        """Export headless: recap per ogni modello/GT disponibile in cache e dashboard del frame iniziale."""
        out_dir = os.path.abspath(out_dir)
        modelli = ["NEURO_SIMB", "REAL_GT", "POS_ONLY"]
        gts = ["REAL", "NEURO_SIMB", "GEOMETRIC", "SEMANTIC"]
        gt_iniziale = self.current_gt_key
        salvati = []
        print(f"\n>>> [EXPORT BATCH] Cartella di destinazione: {out_dir}")

        for m_k in modelli:
            if not any(self._cache_ha_modello(m_k, g) for g in gts):
                print(f"[EXPORT] Modello {m_k} assente dalla cache di valutazione: saltato.")
                continue

            # Recap globale per ogni GT
            for g_k in gts:
                if not self._cache_ha_modello(m_k, g_k):
                    print(f"[EXPORT] Nessun risultato in cache per {m_k} / GT {g_k}: recap saltato.")
                    continue
                self.recap_model_key = m_k
                self.recap_gt_key = g_k
                if self.fig_recap is None or not plt.fignum_exists(self.fig_recap.number):
                    self.fig_recap = plt.figure(num="nuScenes BEV - Recap Globale TRAIN vs VAL",
                                                figsize=(18, 9.4), facecolor='#FFFFFF')
                self.render_global_recap_window()
                fpath = self._salva_figura(self.fig_recap, out_dir, self._nome_file_figura(self.fig_recap))
                salvati.append(fpath)
                print(f"[EXPORT] Recap salvato: {fpath}")

            # Dashboard principale sul frame iniziale con la GT iniziale
            self.current_model_key = m_k
            self.current_gt_key = gt_iniziale
            self._preload_model(m_k)
            self.load_frame(self.current_idx, broadcast=False)
            nome = f"fig_5_dashboard_{m_k.lower()}_gt_{gt_iniziale.lower()}_sample{self.current_idx + 1}.png"
            fpath = self._salva_figura(self.fig, out_dir, nome)
            salvati.append(fpath)
            print(f"[EXPORT] Dashboard salvata: {fpath}")

        print(f"\n[EXPORT COMPLETATO]: {len(salvati)} immagini salvate in {out_dir}")
        return salvati

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
                p.set_facecolor('#F59E0B')
                p.set_edgecolor('#B45309')
                p.set_linewidth(2.2)
                p.set_alpha(0.85)

            meta = hit['meta']
            verdict_str = meta['verdict_label']
            probs = meta['probs_4']
            gt4 = meta['gt_target_4']
            pred_bin = meta.get('pred_binary', [0, 0, 0])
            cat_names = ["Auto", "Camion/Bus", "VRU (Pedoni/Bici)"]

            gt_pres = [cat_names[c] for c in range(min(len(cat_names), len(gt4))) if gt4[c] == 1]
            gt_str = ", ".join(gt_pres) if gt_pres else "Nessuno (Ombra Libera)"

            pred_pres = [cat_names[c] for c in range(min(len(cat_names), len(pred_bin))) if pred_bin[c] == 1]
            if pred_pres:
                pred_str = ", ".join(pred_pres)
            else:
                max_c = cat_names[int(np.argmax(probs))]
                pred_str = f"Nessuno (Sotto Soglia | max: {max_c} {int(max(probs)*100)}%)"

            rf_pct = int(round(meta['road_f'] * 100))
            sf_pct = int(round(meta['side_f'] * 100))
            cf_pct = int(round(meta['cross_f'] * 100))
            tf_pct = max(0, 100 - (rf_pct + sf_pct + cf_pct))

            roadside_line = ""
            if 'min_d_road' in meta and meta['min_d_road'] < 90.0:
                roadside_line = f"• Vicinanza Carreggiata: d={meta['min_d_road']:.1f}m dal cordolo (Accosto: {int(meta.get('roadside_f', 0.0)*100)}%)\n"

            zone_fns = meta.get('zone_fns', [])
            is_fn = (meta['verdict'] in ['FN', 'TP/FN']) or (len(zone_fns) > 0)
            header_title = "⚠ DIAGNOSTICA ERRORE: FALSO NEGATIVO (FN)" if is_fn else "VALUTAZIONE SCIENTIFICA ZONA D'OMBRA"

            fn_line = ""
            if len(zone_fns) > 0:
                fn_str = ", ".join(cat_names[c] for c in zone_fns)
                fn_line = f"• OSTACOLI MANCATI (FN): {fn_str}\n"

            barr_str = f" | Barr: {int(probs[3]*100)}%" if len(probs) > 3 else ""
            p_auto = int(probs[0]*100) if len(probs) > 0 else 0
            p_cam = int(probs[1]*100) if len(probs) > 1 else 0
            p_vru = int(probs[2]*100) if len(probs) > 2 else 0

            criterio_str = "ALMENO 1 OSTACOLO (Basta 1 per evitare collisione)" if getattr(self, 'eval_mode', "PER_CLASS") == "ZONE_LEVEL" else "TUTTI GLI OSTACOLI (Valutazione individuale)"

            tooltip_text = (
                f"{header_title}\n"
                f"• Criterio:              {criterio_str}\n"
                f"• Verdetto:              {verdict_str}\n"
                f"• Ostacolo Reale:        {gt_str}\n"
                f"• Ostacolo Predetto:     {pred_str}\n"
                f"{fn_line}"
                f"• Modello: {self.current_model_key}  |  Target GT: {self.current_gt_key}\n"
                f"• Probabilità AI:        Auto: {p_auto}% | Camion: {p_cam}% | VRU: {p_vru}%{barr_str}\n"
                f"• Composizione Suolo:    Strada: {rf_pct}% | Marciapiede: {sf_pct}% | Strisce: {cf_pct}% | Terreno/Verde: {tf_pct}%\n"
                f"{roadside_line}"
                f"• Distanza da Ego:       {meta['dist']:.1f} m | Area: {meta['area']:.1f} m²"
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
    parser = argparse.ArgumentParser(description="Dashboard Ufficiale Valutazione Prestazioni & Metriche Tesi")
    parser.add_argument("frame", nargs="?", type=int, default=None, help="Numero del frame iniziale (1-N, N = frame del dataset; default: primo frame di validazione)")
    parser.add_argument("--threshold", type=float, default=0.50, choices=[0.50, 0.60, 0.70],
                        help="Soglia decisionale (default: 0.50; opzioni: 0.50, 0.60, 0.70)")
    parser.add_argument("--model", choices=["HYBRID", "NEURO_SIMB", "PURE_NEURAL", "REAL_GT", "REAL_PURE_NEURAL", "POS_ONLY", "BAYES"], default="NEURO_SIMB", help="Modello iniziale")
    parser.add_argument("--gt", choices=["HYBRID", "NEURO_SIMB", "REAL", "GEOMETRIC", "SEMANTIC"], default="NEURO_SIMB", help="Target GT iniziale")
    parser.add_argument("--range", type=float, default=25.0, help="Raggio operativo BEV in metri (default: 25.0)")
    parser.add_argument("--save", type=str, default=None, help="Salva l'immagine della dashboard ad alta risoluzione (300 DPI) ed esce")
    parser.add_argument("--cache", type=str, default=None,
                        help="Forza il file di cache di valutazione JSON da caricare (default: il piu' recente in valutazione/)")
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Percorso dataset nuScenes (default: D:/nuScenes o env NUSCENES_DATAROOT o ./nuscenes)")
    parser.add_argument("--occlusions_dir", type=str, default=None,
                        help="Percorso JSON occlusioni (default: D:/extracted_occlusions o env OCCLUSIONS_OUTPUT o extracted_occlusions)")
    parser.add_argument("--pesi_dir", type=str, default=None,
                        help="Percorso pesi modelli (default: D:/pesi_modelli o env PESI_DIR o pesi_modelli)")
    parser.add_argument("--esporta", type=str, default=None, metavar="DIR",
                        help="Export headless: salva recap (per modello/GT) e dashboard del frame iniziale in DIR ed esce")
    args = parser.parse_args()

    # Modalita' senza finestre: backend non interattivo prima di creare le figure
    headless = bool(args.esporta or args.save)
    if headless:
        plt.switch_backend("Agg")

    vis = EvaluationDashboardVisualizer(max_range=args.range, initial_model=args.model, initial_gt=args.gt,
                                        initial_threshold=args.threshold,
                                        cache_path=args.cache, dataroot=args.dataroot,
                                        occlusions_dir=args.occlusions_dir, pesi_dir=args.pesi_dir,
                                        headless=headless)
    if args.frame is not None and 1 <= args.frame <= vis.total_frames:
        vis.load_frame(args.frame - 1, broadcast=False)
    elif args.esporta and vis.eval_cache and vis.eval_cache.get("frames"):
        # Senza frame esplicito l'export parte dal primo frame valutato in cache (metriche calibrate)
        primo = min(int(k) for k in vis.eval_cache["frames"].keys())
        if primo != vis.current_idx:
            print(f"• [EXPORT] Frame iniziale non presente in cache: uso il primo frame valutato ({primo + 1})")
            vis.load_frame(primo, broadcast=False)

    if args.esporta:
        vis.esporta_batch(args.esporta)
    if args.save:
        out_p = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        vis.fig.savefig(out_p, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"[OK] Dashboard salvata in: {out_p}")
    if not headless:
        plt.show()


if __name__ == "__main__":
    main()
