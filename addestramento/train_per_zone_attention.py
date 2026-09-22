# Script Ufficiale di Addestramento Unificato per AttentionPerZoneModel (addestramento/train_per_zone_attention.py)
# Addestra l'architettura neurale con Channel Attention (Squeeze-and-Excitation) e condizionamento FiLM
# supportando tutte le 4 modalità di supervisione dello studio di ablazione della tesi:
# 1. 'geometric'      : GT Sintetica Neurosimbolica con vincoli 3D (Spatially-Constrained / Physical-Aware)
# 2. 'semantic'       : GT Sintetica Neurosimbolica permissiva (Semantic-Affordance / Naïve Prior)
# 3. 'real'           : GT Reale nuScenes completa (18.682 zone, inclusi i vuoti)
# 4. 'positives_only' : GT Reale nuScenes addestrata ESCLUSIVAMENTE sulle zone con ostacoli reali
# 5. 'all'            : Esegue l'addestramento sequenziale completo di tutte le configurazioni.

import os
import sys
import gc
import time
import json
import random
import torch
from multiprocessing import Pool
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Aggiunge la directory radice del progetto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_adapter.factory_dataset import create_adapter
from raycaster.ray_caster import RayCaster
from ground_truth.ground_truth_extractor import (
    extract_ground_truth_masks,
    rasterize_polygon,
    get_occlusion_ground_truth_target,
    clean_occlusion_polygon
)
from ground_truth.ground_truth_extractor_synthetic import compute_synthetic_ground_truth
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel
from architettura_neurale.loss_functions import AsymmetricLoss
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPoly


# Numero di fotogrammi accorpati in ciascun blocco della cache su disco. L'intero dataset
# non entrerebbe in memoria, quindi viene prodotto e consumato un blocco per volta
FOTOGRAMMI_PER_BLOCCO = 1000

# Popolati dal processo padre prima di distribuire il lavoro: i processi figli li
# ereditano con il fork, senza ricaricare i metadati ne' ricostruire le mappe vettoriali
adapter = None
GT_MODE = None


# Costruisce gli 11 canali BEV del fotogramma e ne ritaglia una patch per ogni zona d'ombra.
# I fotogrammi sono indipendenti, quindi possono essere elaborati in parallelo
def genera_campioni_fotogramma(idx: int):
    gt_mode = GT_MODE
    patches_list, scalars_list, masks_list, targets_list = [], [], [], []

    frame_data = adapter.get_sample_data(idx)
    token = frame_data["sample_token"]

    occ_file = os.path.join("extracted_occlusions", f"{token}.json")
    if not os.path.exists(occ_file):
        return [], [], [], []
    with open(occ_file) as f:
        occs = json.load(f)["occlusions"]

    # Maschera circolare metrica a 25m per tagliare rigorosamente TUTTI gli 11 canali di input
    r_grid = np.arange(200)
    gy, gx = np.meshgrid(r_grid, r_grid)
    mask_25m = (np.sqrt((gx - 99.5)**2 + (gy - 99.5)**2) * 0.4 <= 25.0).astype(np.float32)
    circle_25m = ShapelyPoint(0, 0).buffer(25.0)

    # 1. Canale 0: LiDAR Density BEV entro 25m
    lidar_bev = np.zeros((200, 200), dtype=np.float32)
    points = frame_data.get("points", np.zeros((0, 3)))
    if len(points) > 0:
        for px_m, py_m in points[:, :2]:
            px = int((px_m + 40.0) / 0.4)
            py = int((40.0 - py_m) / 0.4)
            if 0 <= px < 200 and 0 <= py < 200:
                lidar_bev[py, px] = 1.0
    lidar_bev = lidar_bev * mask_25m

    # Rilevamento pseudo-box statici (muri/edifici) per pulizia fisica delle ombre
    boxes_all = frame_data.get('boxes', [])
    boxes_by_token = {b.token: b for b in boxes_all}
    try:
        rc = RayCaster(frame_data, verbose=False)
        static_boxes = rc.detect_static_manmade_boxes(boxes_all)
    except Exception:
        static_boxes = []

    # 2. Canale 1: Ombre Raycasting cumulative entro 25m (pulite da occludori e muri)
    occlusion_mask = np.zeros((200, 200), dtype=np.float32)
    for occ in occs:
        pts = occ.get("polygon_points_m", [])
        if len(pts) >= 3:
            sp_occ = ShapelyPoly(pts)
            if not sp_occ.is_valid:
                sp_occ = sp_occ.buffer(0)
            sp_occ_25 = sp_occ.intersection(circle_25m)
            if not sp_occ_25.is_empty and sp_occ_25.area >= 0.1:
                b_occ = boxes_by_token.get(occ.get("object_token"))
                clean_polys_c1 = clean_occlusion_polygon(
                    sp_occ_25,
                    occluder_box=b_occ,
                    static_boxes=static_boxes,
                    min_area=0.25,
                    min_width=0.30
                )
                for sg in clean_polys_c1:
                    coords = np.array(sg.exterior.coords)[:-1]
                    m = rasterize_polygon(coords)
                    occlusion_mask = np.maximum(occlusion_mask, m)
    occlusion_mask = occlusion_mask * mask_25m

    # 3. Canali 2, 3, 4: HD Map entro 25m
    semantic_map = frame_data['semantic_map']
    drivable_mask = (np.maximum(semantic_map['drivable_area'], semantic_map.get('carpark_area', np.zeros_like(semantic_map['drivable_area'])))) * mask_25m
    walkway_mask = (semantic_map.get('walkway', np.zeros_like(drivable_mask))) * mask_25m
    ped_crossing_mask = (semantic_map.get('ped_crossing', np.zeros_like(drivable_mask))) * mask_25m

    # 4. Canali 5-10: Ostacoli visibili nuScenes entro 25m
    target_masks = extract_ground_truth_masks(frame_data) * mask_25m

    input_channels = [
        lidar_bev,
        occlusion_mask,
        drivable_mask,
        walkway_mask,
        ped_crossing_mask
    ] + list(target_masks)

    input_tensor = torch.tensor(np.stack(input_channels, axis=0), dtype=torch.float32)

    import cv2
    dt_road_map = cv2.distanceTransform((1 - drivable_mask).astype(np.uint8), cv2.DIST_L2, 3) * 0.4

    for occ in occs:
        pts = occ.get("polygon_points_m", [])
        pts_np = np.array(pts)
        if len(pts_np) < 3:
            continue
        poly_xy = pts_np[:, :2]

        sp = ShapelyPoly(poly_xy)
        if not sp.is_valid or sp.area <= 0.01:
            sp = sp.buffer(0)
        if not sp.is_valid or sp.area <= 0.01:
            continue

        # Vincolo rigoroso: l'ombra viene ritagliata esattamente entro i 25 metri
        sp_25m = sp.intersection(circle_25m)
        if sp_25m.is_empty or sp_25m.area < 0.1:
            continue

        occ_name = occ.get("object_name", None)
        occ_tok = occ.get("object_token", None)
        b_obj = boxes_by_token.get(occ_tok, None)
        occ_wlh = b_obj.wlh if (b_obj is not None and hasattr(b_obj, 'wlh')) else None

        clean_polys = clean_occlusion_polygon(
            sp_25m,
            occluder_box=b_obj,
            static_boxes=static_boxes,
            min_area=0.25,
            min_width=0.30
        )
        for sp_sub in clean_polys:
            if sp_sub.area < 0.1:
                continue

            vis_coords = np.array(sp_sub.exterior.coords)[:-1]
            if len(vis_coords) < 3:
                continue

            # Feature semantiche calcolate STRETTAMENTE sulla maschera entro 25m
            occ_mask = rasterize_polygon(vis_coords)
            tot = np.sum(occ_mask)
            road_f = np.sum(occ_mask * drivable_mask) / tot if tot > 0 else 0
            side_f = np.sum(occ_mask * walkway_mask) / tot if tot > 0 else 0
            cross_f = np.sum(occ_mask * ped_crossing_mask) / tot if tot > 0 else 0
            terr_f = max(0.0, 1.0 - (road_f + side_f + cross_f))
            area = float(sp_sub.area)
            dist = float(np.hypot(sp_sub.centroid.x, sp_sub.centroid.y))

            # Calcolo prossimità al bordo strada (accosto / sosta entro 2.5m)
            min_d_road = float(np.min(dt_road_map[occ_mask > 0])) if tot > 0 else 99.0
            roadside_f = max(0.0, 1.0 - (min_d_road / 2.5))

            # Step 1: Estrae la Ground Truth Reale nuScenes a 6 classi entro i 25m
            gt_raw_6 = get_occlusion_ground_truth_target(target_masks, vis_coords)

            # Se positives_only e la zona e vuota, la saltiamo
            if gt_mode == "positives_only" and np.sum(gt_raw_6) == 0:
                continue

            # Step 2: Calcolo del target in base alla modalita (applicato al poligono 25m)
            if gt_mode in ["real", "positives_only"]:
                gt_target = np.array(gt_raw_6, dtype=np.float32)
            else:
                gt_target, _ = compute_synthetic_ground_truth(
                    sp=sp_sub,
                    sp_sample=sp_sub,
                    road_f=road_f,
                    side_f=side_f,
                    cross_f=cross_f,
                    roadside_f=roadside_f,
                    area=area,
                    dist=dist,
                    gt_raw_6=gt_raw_6,
                    mode=gt_mode,
                    occluder_name=occ_name,
                    occluder_wlh=occ_wlh,
                    use_occluder_filter=True
                )

            # Ritaglio patch calcolato strettamente sui pixel entro 25m
            px_x = np.clip(((vis_coords[:, 0] + 40.0) / 0.4).astype(int), 0, 199)
            px_y = np.clip(((40.0 - vis_coords[:, 1]) / 0.4).astype(int), 0, 199)
            xmin, xmax = max(0, np.min(px_x) - 2), min(199, np.max(px_x) + 2)
            ymin, ymax = max(0, np.min(px_y) - 2), min(199, np.max(px_y) + 2)
            if xmax <= xmin or ymax <= ymin:
                continue

            patch = input_tensor[:, ymin:ymax+1, xmin:xmax+1]
            patch_res = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

            # Calcolo larghezza e lunghezza tramite OBB sul poligono 25m
            mrr = sp_sub.minimum_rotated_rectangle
            mrr_coords = np.array(mrr.exterior.coords)[:-1]
            e1 = np.linalg.norm(mrr_coords[0] - mrr_coords[1])
            e2 = np.linalg.norm(mrr_coords[1] - mrr_coords[2])
            obb_w, obb_l = float(min(e1, e2)), float(max(e1, e2))

            scalars = torch.tensor([area, dist, obb_w, obb_l, road_f, side_f, cross_f, roadside_f, terr_f], dtype=torch.float32)
            target = torch.tensor(gt_target, dtype=torch.float32)
            occ_mask_t = AttentionPerZoneModel.build_compatibility_mask(
                occ_name, occ_wlh, road_f=road_f, roadside_f=roadside_f
            )

            patches_list.append(patch_res)
            scalars_list.append(scalars)
            masks_list.append(occ_mask_t)
            targets_list.append(target)

    return patches_list, scalars_list, masks_list, targets_list


# Elenco ordinato dei blocchi di cache presenti sul disco
def elenco_blocchi(cartella: str):
    if not os.path.isdir(cartella):
        return []
    return sorted(os.path.join(cartella, f) for f in os.listdir(cartella) if f.endswith(".pth"))


# Carica un blocco riportando le patch a float32: su disco sono in float16 per dimezzare
# spazio e tempo di lettura, che sono il vero collo di bottiglia dell'addestramento
def carica_blocco(percorso: str, gt_mode: str):
    dati = torch.load(percorso)
    patches = dati["patches"].float()
    scalars, masks, targets = dati["scalars"], dati["masks"], dati["targets"]

    # positives_only riusa i blocchi della GT reale tenendo solo le zone con ostacoli
    if gt_mode == "positives_only":
        tieni = (targets.sum(dim=1) > 0)
        patches, scalars, masks, targets = patches[tieni], scalars[tieni], masks[tieni], targets[tieni]

    return patches, scalars, masks, targets


# Scrive un blocco in modo atomico, cosi' un'interruzione non lascia un file troncato
def scrivi_blocco(cartella: str, numero: int, buffer) -> int:
    percorso = os.path.join(cartella, f"blocco_{numero:04d}.pth")
    contenuto = {
        "patches": torch.stack(buffer[0]).half(),
        "scalars": torch.stack(buffer[1]),
        "masks": torch.stack(buffer[2]),
        "targets": torch.stack(buffer[3]),
    }
    tmp = percorso + ".parziale"
    torch.save(contenuto, tmp)
    os.replace(tmp, percorso)
    return len(buffer[0])


# Indici dei fotogrammi appartenenti allo split nuScenes richiesto
def seleziona_indici(split: str):
    from nuscenes.utils.splits import create_splits_scenes
    splits_dict = create_splits_scenes()
    train_scenes = set(splits_dict.get('train', []))
    val_scenes = set(splits_dict.get('val', []))

    indici = []
    for idx in range(adapter.get_num_samples()):
        sc_name = adapter.nusc.get('scene', adapter.all_samples[idx]['scene_token'])['name']
        if split == "train" and sc_name not in train_scenes:
            continue
        elif split == "val" and sc_name not in val_scenes:
            continue
        indici.append(idx)
    return indici


# Genera l'intero dataset distribuendolo su piu' processi e scrivendolo a blocchi
def genera_blocchi(cartella: str, gt_mode: str, split: str, workers: int) -> None:
    global adapter, GT_MODE

    print(f"• Generazione dataset [{gt_mode}] (split: {split.upper()}) su {workers} processi...")
    adapter = create_adapter("nuscenes", "./nuscenes")
    indici = seleziona_indici(split)
    print(f"• Fotogrammi dello split: {len(indici)}")

    # Le mappe vengono costruite adesso, mentre il processo e' ancora unico, cosi' i figli le ereditano
    adapter.precarica_mappe()

    # positives_only e' un sottoinsieme della GT reale: si generano i blocchi completi
    # e il filtro viene applicato al caricamento, evitando di duplicare la generazione
    GT_MODE = "real" if gt_mode == "positives_only" else gt_mode

    # Senza questo il garbage collector attraverserebbe i metadati nei figli, duplicandoli
    gc.freeze()

    os.makedirs(cartella, exist_ok=True)
    buffer = ([], [], [], [])
    n_blocco = totale_zone = fotogrammi_nel_blocco = 0

    with Pool(processes=workers) as pool:
        for n, risultato in enumerate(pool.imap(genera_campioni_fotogramma, indici, chunksize=8), start=1):
            for destinazione, prodotte in zip(buffer, risultato):
                destinazione.extend(prodotte)
            fotogrammi_nel_blocco += 1

            if fotogrammi_nel_blocco >= FOTOGRAMMI_PER_BLOCCO and buffer[0]:
                totale_zone += scrivi_blocco(cartella, n_blocco, buffer)
                buffer = ([], [], [], [])
                fotogrammi_nel_blocco = 0
                n_blocco += 1

            if n % 500 == 0 or n == len(indici):
                print(f"  fotogrammi {n}/{len(indici)} | zone accumulate: {totale_zone + len(buffer[0])}")

    if buffer[0]:
        totale_zone += scrivi_blocco(cartella, n_blocco, buffer)

    print(f"• Cache scritta in {cartella}: {len(elenco_blocchi(cartella))} blocchi, {totale_zone} zone")


def train_attention_neuro(epochs=5, batch_size=64, lr=1e-3, gt_mode="geometric", checkpoint_path=None, split="train", force_extract=False, workers=8):
    """
    Funzione di addestramento universale.
    Supporta:
      - gt_mode: 'geometric', 'semantic', 'hybrid', 'real', 'positives_only', 'all'.
      - split: 'train' (700 scene train, default), 'val' (150 scene val), 'all' (tutte).
    """
    if gt_mode == "all":
        print("\n" + "=" * 80)
        print(f"   AVVIO ADDESTRAMENTO SEQUENZIALE DI TUTTI I MODELLI (--mode all, split: {split.upper()})")
        print("=" * 80)
        modes_to_train = ["geometric", "semantic", "hybrid", "real", "positives_only"]
        for m in modes_to_train:
            train_attention_neuro(epochs=epochs, batch_size=batch_size, lr=lr, gt_mode=m, split=split, force_extract=force_extract, workers=workers)
        print("\n[COMPLETATO] Addestramento sequenziale di tutti i modelli terminato con successo!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mappatura automatica dei percorsi checkpoint
    if checkpoint_path is None:
        ckpt_map = {
            "geometric": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_geometric.pth"),
            "semantic": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_semantic.pth"),
            "hybrid": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth"),
            "real": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_real_gt.pth"),
            "positives_only": os.path.join("pesi_modelli", "per_zone_checkpoint_attention_positive_only.pth"),
        }
        checkpoint_path = ckpt_map.get(gt_mode, os.path.join("pesi_modelli", f"per_zone_checkpoint_attention_{gt_mode}.pth"))

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"\n===========================================================================")
    print(f"   ADDESTRAMENTO ATTENTION-PER-ZONE: [{gt_mode.upper()}] (SPLIT: {split.upper()})")
    print(f"===========================================================================")
    print(f"• Dispositivo Selezionato: {device}")
    print(f"• Modalità Supervisione: {gt_mode.upper()}")
    print(f"• Split Dataset: {split.upper()} (Ufficiale nuScenes)")
    print(f"• Checkpoint di Output: {checkpoint_path}")

    # Gestione Cache differenziata per modalità e split
    split_suffix = f"_{split}" if split != "all" else ""
    if gt_mode in ["real", "positives_only"]:
        cache_path = os.path.join("addestramento", f"cached_dataset_per_zone{split_suffix}.pth")
    else:
        cache_path = os.path.join("addestramento", f"cached_dataset_neuro_{gt_mode}{split_suffix}.pth")

    # La cache e' suddivisa in blocchi su disco: l'intero dataset non entrerebbe in memoria
    cartella_blocchi = cache_path.replace(".pth", "_blocchi")
    if force_extract or not elenco_blocchi(cartella_blocchi):
        genera_blocchi(cartella_blocchi, gt_mode, split, workers)
    else:
        print(f"• Blocchi gia' presenti: {cartella_blocchi}")

    blocchi = elenco_blocchi(cartella_blocchi)
    if not blocchi:
        raise RuntimeError(f"Nessun blocco di dati prodotto in {cartella_blocchi}")

    print(f"• Blocchi di dati: {len(blocchi)} | Dimensione batch: {batch_size}")

    # Inizializzazione del Modello AttentionPerZoneModel
    model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)

    # Funzione di costo Asymmetric Loss Bilanciata (gamma_pos=0.5 per compromesso ideale Recall/Precision)
    # Evita l'ipersensibilità e abbatte i falsi positivi mantenendo altissima la copertura
    if gt_mode in ["real", "positives_only"]:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.5, 2.5, 2.5, 2.0, 2.0, 2.5])
    elif gt_mode == "hybrid":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.2, 2.2, 2.0, 1.8, 1.8, 3.5])
    else:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nInizio Addestramento [{gt_mode.upper()}] per {epochs} Epoche...\n")
    start_t = time.time()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_campioni = 0

        # Ad ogni epoca si attraversano TUTTI i blocchi, in ordine rimescolato. Esaurire un
        # blocco per piu' epoche prima di passare al successivo farebbe dimenticare alla rete
        # quanto appreso dai blocchi precedenti (catastrophic forgetting)
        ordine = list(blocchi)
        random.Random(epoch).shuffle(ordine)

        for percorso in ordine:
            patches, scalars, masks, targets = carica_blocco(percorso, gt_mode)
            dataloader = DataLoader(TensorDataset(patches, scalars, masks, targets),
                                    batch_size=batch_size, shuffle=True, drop_last=False)
            for p_b, s_b, m_b, t_b in dataloader:
                p_b, s_b, m_b, t_b = p_b.to(device), s_b.to(device), m_b.to(device), t_b.to(device)
                optimizer.zero_grad()
                logits = model(p_b, s_b, occluder_mask=m_b)
                loss = criterion(logits, t_b, s_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(p_b)
                n_campioni += len(p_b)

        scheduler.step()
        avg_loss = epoch_loss / n_campioni
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoca {epoch+1:2d}/{epochs}] ASL Loss: {avg_loss:.5f} | LR: {cur_lr:.6f}")

    elapsed = time.time() - start_t
    print(f"\nAddestramento [{gt_mode.upper()}] completato in {elapsed:.2f} secondi.")

    ckpt_dict = {
        "model_state_dict": model.state_dict(),
        "architecture": f"AttentionPerZoneModel_{gt_mode}",
        "epochs": epochs,
        "final_loss": avg_loss,
        "gt_mode": gt_mode,
        "split": split
    }
    torch.save(ckpt_dict, checkpoint_path)
    print(f"[OK] Pesi salvati con successo in: {checkpoint_path}")

    # Se geometric, aggiorna anche il default del repository
    if gt_mode == "geometric":
        default_ckpt = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        torch.save(ckpt_dict, default_ckpt)
        print(f"[OK] Pesi standard aggiornati in: {default_ckpt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Addestramento AttentionPerZoneModel su GT Sintetica o Reale")
    parser.add_argument("--mode", "--gt_mode", dest="mode", type=str, default="hybrid",
                        choices=["geometric", "semantic", "hybrid", "real", "positives_only", "all"],
                        help="Modalità: 'geometric', 'semantic', 'hybrid', 'real', 'positives_only' o 'all' per addestrare tutti")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Numero di epoche (default: 5). Le 20 originali erano tarate sul dataset mini: "
                             "con il trainval una sola epoca espone la rete a decine di volte piu' esempi")
    parser.add_argument("--workers", type=int, default=8,
                        help="Processi paralleli per la generazione del dataset (default: 8)")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch (default: 64)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                        help="Split nuScenes: 'train' (700 scene, default), 'val' (150 scene) o 'all' (850 scene)")
    parser.add_argument("--force", "--force_extract", dest="force", action="store_true", help="Forza la rigenerazione della cache dataset da zero")
    args = parser.parse_args()

    train_attention_neuro(epochs=args.epochs, batch_size=args.batch_size, gt_mode=args.mode, split=args.split,
                          force_extract=args.force, workers=args.workers)
