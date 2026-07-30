import os
import sys
import glob
import json
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from per_zone_model import PerZoneModel
from dataset_generator_per_zone import OcclusionDatasetPerZone
from ground_truth_extractor import get_occlusion_ground_truth_target
from ground_truth_extractor_synthetic import generate_synthetic_injected_gt

def run_synthetic_evaluation(threshold=0.30):
    print("\n" + "=" * 95)
    print(f"   STRESS TEST SU GROUND TRUTH SINTETICA VEROSIMILE A 6 CLASSI (SOGLIA = {threshold*100:.0f}%)")
    print("   (Valuta i 6 modelli quando più zone d'ombra contengono ostacoli verosimili distribuiti)")
    print("=" * 95 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Caricamento Dataset ed iniezione ostacoli sintetici verosimili su NuScenes...")
    ds = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
    
    categories = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bicicletta", "Barriera"]
    
    def load_model(ckpt_path):
        m = PerZoneModel(num_classes=6).to(device)
        full_path = os.path.join("pesi_modelli", ckpt_path) if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("pesi_modelli", ckpt_path)) else ckpt_path
        if os.path.exists(full_path):
            try:
                m.load_state_dict(torch.load(full_path, map_location=device)['model_state_dict'])
                m.eval()
                return m, True
            except Exception:
                pass
        return m, False

    model_bce, has_bce = load_model("per_zone_checkpoint.pth")
    model_bce_sem, has_bce_sem = load_model("per_zone_checkpoint_semantica.pth")
    model_focal, has_focal = load_model("per_zone_checkpoint_focal.pth")
    model_focal_sem, has_focal_sem = load_model("per_zone_checkpoint_focal_semantica.pth")
    model_surr, has_surr = load_model("per_zone_checkpoint_surrounding.pth")

    # Cache Agente Bayesiano
    b_files = glob.glob(os.path.join("extracted_occlusions_probabilities", "*.json"))
    b_cache = {}
    for fpath in b_files:
        tok = os.path.basename(fpath).split("_")[-1].replace(".json", "")
        with open(fpath, "r") as f:
            b_cache[tok] = json.load(f).get("occlusions", [])

    # Contatori TP, FP, FN per la GT Sintetica
    bce_tp, bce_fp, bce_fn = np.zeros(6), np.zeros(6), np.zeros(6)
    bce_sem_tp, bce_sem_fp, bce_sem_fn = np.zeros(6), np.zeros(6), np.zeros(6)
    focal_tp, focal_fp, focal_fn = np.zeros(6), np.zeros(6), np.zeros(6)
    focal_sem_tp, focal_sem_fp, focal_sem_fn = np.zeros(6), np.zeros(6), np.zeros(6)
    surr_tp, surr_fp, surr_fn = np.zeros(6), np.zeros(6), np.zeros(6)
    bayes_tp, bayes_fp, bayes_fn = np.zeros(6), np.zeros(6), np.zeros(6)

    bce_viol, bce_sem_viol, focal_viol, focal_sem_viol, surr_viol, bayes_viol = 0, 0, 0, 0, 0, 0
    non_driveable_zones = 0
    total_injected_obstacles = 0

    synthetic_gt_cache = {}
    total_frames = len(ds.base_dataset)
    for idx in range(total_frames):
        fd = ds.base_dataset.adapter.get_sample_data(idx)
        tok = fd['sample_token']
        syn_gt_masks, injected_cnt = generate_synthetic_injected_gt(fd, injection_rate=0.25, seed=42)
        synthetic_gt_cache[tok] = torch.tensor(syn_gt_masks, dtype=torch.float32)
        total_injected_obstacles += injected_cnt

    print(f"Iniezione completata: Aggiunti {total_injected_obstacles} ostacoli sintetici verosimili sulle 6 classi.")
    print("Valutazione automatica dei 6 modelli sulla Ground Truth Sintetica in corso...\n")

    with torch.no_grad():
        for sample in ds.samples:
            sample_token = sample['sample_token']
            raw_pts = sample['polygon_points']
            syn_gt_tensor = synthetic_gt_cache.get(sample_token, sample['target'])
            
            target_vec = get_occlusion_ground_truth_target(syn_gt_tensor, raw_pts)
            
            patch_tensor = sample['patch'].unsqueeze(0).to(device, dtype=torch.float32)
            scalar_tensor = sample['scalars'].unsqueeze(0).to(device, dtype=torch.float32)
            
            sc_np = sample['scalars'].numpy()
            is_non_driveable = (sc_np[5] + sc_np[8]) > 0.5
            if is_non_driveable:
                non_driveable_zones += 1

            def eval_net(model, tp_arr, fp_arr, fn_arr):
                logits = model(patch_tensor, scalar_tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                pred = (probs >= threshold).astype(np.float32)
                for c in range(6):
                    if target_vec[c] == 1.0 and pred[c] == 1.0: tp_arr[c] += 1
                    elif target_vec[c] == 0.0 and pred[c] == 1.0: fp_arr[c] += 1
                    elif target_vec[c] == 1.0 and pred[c] == 0.0: fn_arr[c] += 1
                viol = 1 if is_non_driveable and (pred[0] == 1 or pred[1] == 1 or pred[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0 else 0
                return viol

            if has_bce: bce_viol += eval_net(model_bce, bce_tp, bce_fp, bce_fn)
            if has_bce_sem: bce_sem_viol += eval_net(model_bce_sem, bce_sem_tp, bce_sem_fp, bce_sem_fn)
            if has_focal: focal_viol += eval_net(model_focal, focal_tp, focal_fp, focal_fn)
            if has_focal_sem: focal_sem_viol += eval_net(model_focal_sem, focal_sem_tp, focal_sem_fp, focal_sem_fn)
            if has_surr: surr_viol += eval_net(model_surr, surr_tp, surr_fp, surr_fn)

            # Agente Bayesiano
            b_occs = b_cache.get(sample_token, [])
            b_dict = {}
            for b_occ in b_occs:
                b_pts = b_occ.get("polygon_points_m", [])
                if len(b_pts) >= 3 and len(raw_pts) >= 3:
                    if np.linalg.norm(np.mean(b_pts, axis=0) - np.mean(raw_pts, axis=0)) < 0.5:
                        b_dict = b_occ.get("estimated_probabilities", {})
                        break
            b_probs = np.array([
                b_dict.get("Auto", 0.0),
                max(b_dict.get("Camion", 0.0), b_dict.get("Bus", 0.0)),
                b_dict.get("Pedone", 0.0),
                b_dict.get("Moto", 0.0),
                b_dict.get("Bicicletta", 0.0),
                b_dict.get("Barriera", 0.0)
            ])
            pred_b = (b_probs >= threshold).astype(np.float32)
            for c in range(6):
                if target_vec[c] == 1.0 and pred_b[c] == 1.0: bayes_tp[c] += 1
                elif target_vec[c] == 0.0 and pred_b[c] == 1.0: bayes_fp[c] += 1
                elif target_vec[c] == 1.0 and pred_b[c] == 0.0: bayes_fn[c] += 1
            if is_non_driveable and (pred_b[0] == 1 or pred_b[1] == 1 or pred_b[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0:
                bayes_viol += 1

    def print_metric_table(title, tp_arr, fp_arr, fn_arr, viol_cnt):
        print("\n" + "=" * 95)
        print(f"  {title.upper()} (Sintetica GT - Soglia = {threshold*100:.0f}%)")
        print("=" * 95)
        print(f" {'Classe':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'IoU':<12} {'TP/FP/FN'}")
        print("-" * 95)
        prec_l, rec_l, f1_l = [], [], []
        for c in range(6):
            tp, fp, fn = tp_arr[c], fp_arr[c], fn_arr[c]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec_l.append(prec); rec_l.append(rec); f1_l.append(f1)
            print(f" {categories[c]:<14} {prec*100:6.1f}%       {rec*100:6.1f}%       {f1*100:6.1f}%       {iou*100:6.1f}%       {int(tp)}/{int(fp)}/{int(fn)}")
        print("-" * 95)
        coherence_rate = (1.0 - (viol_cnt / max(1, non_driveable_zones))) * 100
        print(f" {'MEDIA MACRO':<14} {np.mean(prec_l)*100:6.1f}%       {np.mean(rec_l)*100:6.1f}%       {np.mean(f1_l)*100:6.1f}%")
        print(f"  TASSO DI COERENZA SEMANTICO TERRENO/MARCIAPIEDE: {coherence_rate:.1f}% ({viol_cnt} violazioni su {non_driveable_zones} zone)\n")
        print("=" * 95)

    print_metric_table("1. Agente Bayesiano Dinamico (Prior HD)", bayes_tp, bayes_fp, bayes_fn, bayes_viol)
    if has_bce:
        print_metric_table("2. Agente Neurale Baseline (BCE Standard)", bce_tp, bce_fp, bce_fn, bce_viol)
    if has_bce_sem:
        print_metric_table("3. Agente Neurale BCE + Penalizzazione Semantica", bce_sem_tp, bce_sem_fp, bce_sem_fn, bce_sem_viol)
    if has_focal:
        print_metric_table("4. Agente Neurale Focal Loss Standard", focal_tp, focal_fp, focal_fn, focal_viol)
    if has_focal_sem:
        print_metric_table("5. Agente Neurale Focal Loss + Penalizzazione Semantica", focal_sem_tp, focal_sem_fp, focal_sem_fn, focal_sem_viol)
    if has_surr:
        print_metric_table("6. Agente Neurale Contesto Esterno (Ring Semantics)", surr_tp, surr_fp, surr_fn, surr_viol)

if __name__ == "__main__":
    run_synthetic_evaluation(threshold=0.30)
