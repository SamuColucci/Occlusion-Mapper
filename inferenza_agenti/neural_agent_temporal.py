# Agente Neurale Temporale Per-Zone (neural_agent_temporal.py)
# Estende neural_agent.py con Aggiornamento Bayesiano Ricorsivo con Memoria Temporale:
#
# Quando un oggetto tracciato entra in una zona d'ombra, la probabilità di quella zona
# NON viene calcolata solo dal modello statico, ma viene aggiornata frame per frame con:
#
#   Frame t=0 (primo frame nell'ombra, ingresso rilevato):   P = min(p_model + RAMP_UP,  P_MAX_ENTRY)
#   Frame t=1,2 (traiettoria confermata nella zona):         P = min(p_prev  + RAMP_STEP, P_PEAK)
#   Frame t=3,4,5 (permanenza prolungata, incertezza sale):  P = max(p_prev  * DECAY,     P_PRIOR)
#   Frame t>6 (alta incertezza, reset alla prior HD):        P -> lerp verso P_PRIOR
#
# Le probabilità temporali aggiornate vengono salvate in: extracted_occlusions_temporal/
# Il confronto con i risultati statici è visibile dalla dashboard (Opzione 8).

# Import dei moduli di sistema per la manipolazione dei percorsi
import os
import sys
# Import di glob per la ricerca di file su disco
import glob
# Import di json per la serializzazione e lettura delle predizioni
import json
# Import di numpy per le operazioni algebriche vettoriali
import numpy as np
# Import di torch per la gestione del calcolo e delle funzioni neurali
import torch
import torch.nn.functional as F

# Aggiunge la cartella radice del progetto al sys.path per l'importazione dei pacchetti interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'agente neurale base PerZoneOcclusionAgent e della funzione di suddivisione in sub-zone
from inferenza_agenti.neural_agent import PerZoneOcclusionAgent, subdivide_occlusion_into_subzones
# Import del caricatore del dataset OcclusionDatasetNeural
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetNeural

# ── Parametri della Curva Temporale Ramp-Up → Picco → Decadimento → Prior ──────
P_MAX_ENTRY = 0.94   # Probabilità al primo frame di ingresso nell'ombra
P_PEAK      = 0.99   # Probabilità massima dopo traiettoria confermata (t=1-2)
RAMP_UP     = 0.30   # Delta di salita al frame di ingresso rispetto al modello statico
RAMP_STEP   = 0.05   # Delta di salita per ogni frame di permanenza (fase di Ramp)
DECAY       = 0.87   # Fattore di decadimento per frame di permanenza prolungata (> t=2)
P_PRIOR     = 0.40   # Probabilità a priori di ritorno dopo lunga assenza (> t=7)
T_PEAK      = 2      # Numero di frame in cui si raggiunge il picco (ramp-up)
T_DECAY     = 5      # Frame totali prima di inizio decadimento aggressivo
T_RESET     = 7      # Frame totali prima del ritorno alla prior
# ────────────────────────────────────────────────────────────────────────────────

def apply_temporal_update(p_model: float, t_in_shadow: int, p_prev: float | None) -> float:
    """
    Aggiorna la probabilità di una zona d'ombra considerando la memoria temporale.
    
    Args:
        p_model:      Probabilità istantanea del modello neurale statico (0.0-1.0).
        t_in_shadow:  Numero di frame in cui l'oggetto è rimasto nell'ombra (0 = primo frame).
        p_prev:       Probabilità del frame precedente (None se è il primo frame).
    Returns:
        float: Probabilità aggiornata con la memoria temporale.
    """
    # Nessun oggetto tracciato → usa la probabilità del modello statico puro
    if t_in_shadow < 0 or p_prev is None:
        return p_model
    
    if t_in_shadow == 0:
        # Frame di ingresso nell'ombra: ramp-up immediato dalla prior del modello
        p_updated = min(p_model + RAMP_UP, P_MAX_ENTRY)
    elif t_in_shadow <= T_PEAK:
        # Traiettoria confermata nei frame 1-2: salita verso il picco
        p_updated = min(p_prev + RAMP_STEP, P_PEAK)
    elif t_in_shadow <= T_DECAY:
        # Permanenza prolungata nei frame 3-5: decadimento moderato
        p_updated = max(p_prev * DECAY, P_PRIOR)
    else:
        # Alta incertezza (> T_RESET frame): convergenza lineare verso P_PRIOR
        alpha = min(1.0, (t_in_shadow - T_DECAY) / (T_RESET - T_DECAY + 1))
        p_updated = (1.0 - alpha) * p_prev + alpha * P_PRIOR
    
    # Restituisce il valore arrotondato a 4 cifre decimali
    return round(float(p_updated), 4)


class TemporalPerZoneAgent(PerZoneOcclusionAgent):
    """
    Agente Neurale Per-Zone con Memoria Temporale (Recursive Bayesian Update).
    Estende PerZoneOcclusionAgent aggiungendo il tracking delle zone d'ombra tra frame successivi.
    """

    def process_dataset(self, dataset, out_dir="extracted_occlusions_temporal"):
        # Crea la cartella di destinazione per le predizioni temporali salvate
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nInizio generazione predizioni Agente Temporale Per-Zone per {len(dataset)} campioni...")
        print(f"  Output: {os.path.abspath(out_dir)}")
        print(f"  Parametri Curva Temporale: P_ENTRY={P_MAX_ENTRY}, P_PEAK={P_PEAK}, DECAY={DECAY}, P_PRIOR={P_PRIOR}\n")

        # Dizionario di Memoria Temporale: {scene_token -> {occ_id -> {'t': int, 'p_prev': dict}}}
        # ponytail: occ_id = hash geometrico del centroide (stabile tra frame consecutivi nella stessa scena)
        scene_memory: dict = {}

        # Indice delle scene per processarle in ordine temporale corretto
        scene_indices = dataset.adapter.get_scene_indices()

        # Ordine di processing: per ogni scena processa i frame in ordine di timestamp
        all_frame_indices = []
        for scene_token, frame_idxs in scene_indices.items():
            for fi in frame_idxs:
                all_frame_indices.append((scene_token, fi))
        
        # Fallback se get_scene_indices non è implementato: ordine sequenziale
        if not all_frame_indices:
            all_frame_indices = [('', i) for i in range(len(dataset))]

        processed = 0
        for scene_token, idx in all_frame_indices:
            input_tensor, _ = dataset[idx]
            frame_data = dataset.adapter.get_sample_data(idx)
            sample_token = frame_data['sample_token']
            lidar_token = frame_data['lidar_token']
            s_token = frame_data.get('scene_token', scene_token)

            json_base_path = os.path.join("extracted_occlusions", f"{sample_token}.json")
            if not os.path.exists(json_base_path):
                processed += 1
                continue

            with open(json_base_path, "r") as f:
                raw_data = json.load(f)
                raw_occlusions = raw_data.get("occlusions", [])

            # Memoria della scena corrente
            if s_token not in scene_memory:
                scene_memory[s_token] = {}
            mem = scene_memory[s_token]

            per_zone_occlusions = []
            active_occ_ids = set()

            for raw_occ in raw_occlusions:
                pts = raw_occ.get("polygon_points_m", [])
                if len(pts) < 3:
                    continue

                occ = subdivide_occlusion_into_subzones(raw_occ, frame_data["semantic_map"])
                
                pts_np = np.array(pts)
                y_ahead = pts_np[:, 0]
                x_right = pts_np[:, 1]
                px_x = np.clip(((x_right + 40.0) / 0.4).astype(int), 0, 199)
                px_y = np.clip(((40.0 - y_ahead) / 0.4).astype(int), 0, 199)

                xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
                ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # ID stabile dell'ombra: usa il centroide arrotondato a 2m
                cx = round(float(np.mean(x_right)), 0)
                cy = round(float(np.mean(y_ahead)), 0)
                occ_id = f"{cx:.0f}_{cy:.0f}"
                active_occ_ids.add(occ_id)

                # Inferenza statica tramite la classe base
                patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
                patch_resized = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).to(self.device, dtype=torch.float32)

                area_sqm = float(occ.get("area_sqm", 0.0))
                distance_m = float(occ.get("distance_m", 0.0))
                occ_w = float(occ.get("occluder_width_m", 2.0)) if "occluder_width_m" in occ else 2.0
                occ_h = float(occ.get("occluder_height_m", 1.8)) if "occluder_height_m" in occ else 1.8

                sub_zones_raw = occ.get("sub_zones", [])
                processed_sub_zones = []

                if sub_zones_raw:
                    global_probs_acc = {k: 0.0 for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Bus", "Rimorchio", "Barriera"]}
                    for sz in sub_zones_raw:
                        surf_name = sz.get("surface", "driveable_surface")
                        frac = float(sz.get("area_fraction", 1.0))
                        sz_area = float(sz.get("area_sqm", area_sqm * frac))
                        sz_w = float(sz.get("occlusion_width_m", occ_w))
                        r_flag = 1.0 if surf_name == "driveable_surface" else 0.0
                        s_flag = 1.0 if surf_name == "sidewalk" else 0.0
                        c_flag = 1.0 if surf_name == "ped_crossing" else 0.0
                        p_flag = 1.0 if surf_name == "other_flat" else 0.0
                        t_flag = 1.0 if surf_name == "terrain" else 0.0
                        sz_scalars = torch.tensor([[sz_area, distance_m, sz_w, occ_h, r_flag, s_flag, c_flag, p_flag, t_flag]], dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            sz_logits = self.model(patch_resized, sz_scalars)
                            sz_probs_raw = torch.sigmoid(sz_logits).squeeze(0).cpu().numpy()
                        p_auto, p_camion, p_ped, p_moto, p_bici, p_barr = (float(sz_probs_raw[i]) for i in range(6))
                        sz_probs = {"Auto": p_auto, "Pedone": p_ped, "Camion": p_camion, "Bicicletta": p_bici, "Moto": p_moto, "Bus": p_camion * 0.6, "Rimorchio": p_camion * 0.4, "Barriera": p_barr, "Cono": 0.0, "Altro": 0.0}
                        for k in global_probs_acc:
                            global_probs_acc[k] = max(global_probs_acc[k], sz_probs.get(k, 0.0))
                        new_sz = dict(sz)
                        new_sz["estimated_probabilities"] = {k: round(v, 4) for k, v in sz_probs.items()}
                        processed_sub_zones.append(new_sz)
                    static_probs = {k: round(v, 4) for k, v in global_probs_acc.items()}
                else:
                    road_f = float(occ.get("road_fraction", 0.0))
                    side_f = float(occ.get("sidewalk_fraction", 0.0))
                    cross_f = float(occ.get("crosswalk_fraction", 0.0))
                    park_f = float(occ.get("carpark_fraction", 0.0))
                    terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f + park_f))
                    scalars = torch.tensor([[area_sqm, distance_m, occ_w, occ_h, road_f, side_f, cross_f, park_f, terr_f]], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        logits = self.model(patch_resized, scalars)
                        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    static_probs = {"Auto": round(float(probs[0]), 4), "Camion": round(float(probs[1]), 4), "Pedone": round(float(probs[2]), 4), "Moto": round(float(probs[3]), 4), "Bicicletta": round(float(probs[4]), 4), "Bus": round(float(probs[1]) * 0.6, 4), "Rimorchio": round(float(probs[1]) * 0.4, 4), "Barriera": round(float(probs[5]), 4), "Cono": 0.0, "Altro": 0.0}

                # ── Aggiornamento Temporale Ricorsivo ─────────────────────────────
                occ_mem = mem.get(occ_id, {'t': 0, 'p_prev': None})
                t_in = occ_mem['t']
                p_prev_dict = occ_mem['p_prev']

                temporal_probs = {}
                for cls, p_static in static_probs.items():
                    p_prev_cls = p_prev_dict[cls] if p_prev_dict and cls in p_prev_dict else None
                    temporal_probs[cls] = apply_temporal_update(p_static, t_in, p_prev_cls)

                # Aggiorna la memoria: incrementa t e salva le probabilità attuali
                mem[occ_id] = {'t': t_in + 1, 'p_prev': temporal_probs}
                # ─────────────────────────────────────────────────────────────────

                new_occ = dict(occ)
                new_occ["estimated_probabilities"] = temporal_probs
                new_occ["estimated_probabilities_static"] = static_probs  # Salvato per confronto
                new_occ["t_in_shadow"] = t_in
                new_occ["sub_zones"] = processed_sub_zones
                new_occ["per_zone_predicted"] = True
                new_occ["temporal_update_applied"] = True
                per_zone_occlusions.append(new_occ)

            # Pulizia della memoria: rimuovi l'ombra che non esiste più nel frame corrente
            for old_id in list(mem.keys()):
                if old_id not in active_occ_ids:
                    del mem[old_id]

            # Salva il file JSON contenente le predizioni aggiornate temporalmente
            out_path = os.path.join(out_dir, f"occlusion_per_zone_{idx:04d}_{sample_token}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "sample_token": sample_token,
                    "lidar_token": lidar_token,
                    "scene_token": s_token,
                    "model_checkpoint_used": getattr(self, "checkpoint_used", "N/A"),
                    "temporal_update": True,
                    "num_occlusions": len(per_zone_occlusions),
                    "occlusions": per_zone_occlusions
                }, f, indent=4)

            processed += 1
            if processed % 50 == 0 or processed == len(all_frame_indices):
                print(f"  Progresso Inferenza Temporale: {processed}/{len(all_frame_indices)} campioni elaborati...")

        print(f"Predizioni Agente Temporale completate e salvate in: {os.path.abspath(out_dir)}\n")


# Blocco principale di esecuzione se eseguito direttamente
if __name__ == "__main__":
    dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot="./nuscenes")
    agent = TemporalPerZoneAgent()
    agent.process_dataset(dataset)
