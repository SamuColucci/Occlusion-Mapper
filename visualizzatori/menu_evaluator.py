# Menu Interattivo di Valutazione e Confronto Modelli (visualizzatori/menu_evaluator.py)
# Permette di consultare gli score dei 6 approcci del progetto e di effettuare confronti diretti affiancati

import os
import sys
import glob
import json
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from per_zone_model import PerZoneModel
from dataset_generator_per_zone import OcclusionDatasetPerZone

MODEL_NAMES = [
    "1. Agente Bayesiano Dinamico (Prior HD)",
    "2. Agente Neurale Baseline (BCE Standard)",
    "3. Agente Neurale BCE + Penalizzazione Semantica",
    "4. Agente Neurale Focal Loss Standard",
    "5. Agente Neurale Focal Loss + Penalizzazione (Picco Macro F1: 32.3%)",
    "6. Agente Neurale Contesto Esterno (Ring Semantics)",
    "7. Agente Neurale Asymmetric Loss - CVPR 2021 (Picco Pedoni Recall: 98.8%)"
]

MODEL_CKPTS = [
    None,  # Bayes
    os.path.join("pesi_modelli", "per_zone_checkpoint.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_semantica.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_focal.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_focal_semantica.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_surrounding.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_asl.pth")
]

CATEGORIES = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bicicletta", "Barriera"]

class ProjectEvaluatorMenu:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.cached_results = {}
        self.non_driveable_count = 0

    def load_dataset_if_needed(self):
        if self.dataset is None:
            print("\nInizializzazione Dataset nuScenes ed estrazione feature (18.682 coni d'ombra)...")
            self.dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
            
            # Conta zone non carrabili per coerenza semantica
            self.non_driveable_count = sum(
                1 for s in self.dataset.samples 
                if (s['scalars'].numpy()[5] + s['scalars'].numpy()[8]) > 0.5
            )

    def evaluate_model_index(self, idx, threshold=0.30):
        if idx in self.cached_results:
            return self.cached_results[idx]

        self.load_dataset_if_needed()
        print(f"\nCalcolo metriche in corso per: {MODEL_NAMES[idx]}...")

        tp_arr, fp_arr, fn_arr = np.zeros(6), np.zeros(6), np.zeros(6)
        viol_count = 0

        if idx == 0:
            # Agente Bayesiano
            b_files = glob.glob(os.path.join("extracted_occlusions_probabilities", "*.json"))
            b_cache = {}
            for fpath in b_files:
                tok = os.path.basename(fpath).split("_")[-1].replace(".json", "")
                with open(fpath, "r") as f:
                    b_cache[tok] = json.load(f).get("occlusions", [])

            for sample in self.dataset.samples:
                target_vec = sample['target'].numpy()
                sample_token = sample['sample_token']
                raw_pts = sample['polygon_points']
                sc_np = sample['scalars'].numpy()
                is_non_driveable = (sc_np[5] + sc_np[8]) > 0.5

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
                pred = (b_probs >= threshold).astype(np.float32)

                for c in range(6):
                    if target_vec[c] == 1.0 and pred[c] == 1.0: tp_arr[c] += 1
                    elif target_vec[c] == 0.0 and pred[c] == 1.0: fp_arr[c] += 1
                    elif target_vec[c] == 1.0 and pred[c] == 0.0: fn_arr[c] += 1

                if is_non_driveable and (pred[0] == 1 or pred[1] == 1 or pred[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0:
                    viol_count += 1
        else:
            # Agente Neurale
            ckpt_path = MODEL_CKPTS[idx]
            if not os.path.exists(ckpt_path):
                print(f"[WARNING] Checkpoint '{ckpt_path}' non trovato sul disco! Esegui prima lo script di addestramento.")
                return None

            model = PerZoneModel(num_classes=6).to(self.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
            model.eval()

            with torch.no_grad():
                for sample in self.dataset.samples:
                    target_vec = sample['target'].numpy()
                    patch_tensor = sample['patch'].unsqueeze(0).to(self.device, dtype=torch.float32)
                    scalar_tensor = sample['scalars'].unsqueeze(0).to(self.device, dtype=torch.float32)
                    sc_np = sample['scalars'].numpy()
                    is_non_driveable = (sc_np[5] + sc_np[8]) > 0.5

                    logits = model(patch_tensor, scalar_tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    pred = (probs >= threshold).astype(np.float32)

                    for c in range(6):
                        if target_vec[c] == 1.0 and pred[c] == 1.0: tp_arr[c] += 1
                        elif target_vec[c] == 0.0 and pred[c] == 1.0: fp_arr[c] += 1
                        elif target_vec[c] == 1.0 and pred[c] == 0.0: fn_arr[c] += 1

                    if is_non_driveable and (pred[0] == 1 or pred[1] == 1 or pred[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0:
                        viol_count += 1

        prec_l, rec_l, f1_l, iou_l = [], [], [], []
        per_class = []

        for c in range(6):
            tp, fp, fn = tp_arr[c], fp_arr[c], fn_arr[c]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            prec_l.append(prec); rec_l.append(rec); f1_l.append(f1); iou_l.append(iou)
            per_class.append({'prec': prec, 'rec': rec, 'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn})

        coherence_rate = (1.0 - (viol_count / max(1, self.non_driveable_count))) * 100

        result = {
            'name': MODEL_NAMES[idx],
            'per_class': per_class,
            'macro_prec': np.mean(prec_l),
            'macro_rec': np.mean(rec_l),
            'macro_f1': np.mean(f1_l),
            'coherence': coherence_rate,
            'viol_count': viol_count
        }

        self.cached_results[idx] = result
        return result

    def print_single_model_table(self, idx):
        res = self.evaluate_model_index(idx)
        if res is None:
            return

        print("\n" + "=" * 90)
        print(f"  {res['name'].upper()} (Soglia Decisionale = 30%)")
        print("=" * 90)
        print(f" {'Classe':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'IoU':<12} {'TP/FP/FN'}")
        print("-" * 90)
        for c in range(6):
            pc = res['per_class'][c]
            print(f" {CATEGORIES[c]:<14} {pc['prec']*100:6.1f}%       {pc['rec']*100:6.1f}%       {pc['f1']*100:6.1f}%       {pc['iou']*100:6.1f}%       {int(pc['tp'])}/{int(pc['fp'])}/{int(pc['fn'])}")
        print("-" * 90)
        print(f" {'MEDIA MACRO':<14} {res['macro_prec']*100:6.1f}%       {res['macro_rec']*100:6.1f}%       {res['macro_f1']*100:6.1f}%")
        print(f"  TASSO DI COERENZA SEMANTICO TERRENO/MARCIAPIEDE: {res['coherence']:.1f}% ({res['viol_count']} violazioni su {self.non_driveable_count} zone)\n")
        print("=" * 90)
        input("\nPremi INVIO per tornare al menu...")

    def compare_two_models(self):
        print("\n" + "=" * 60)
        print("     CONFRONTO DIRETTO ED AFFIANCATO TRA 2 MODELLI")
        print("=" * 60)
        for i, name in enumerate(MODEL_NAMES):
            print(f"  {i+1}. {name}")
        print("-" * 60)

        try:
            m1_idx = int(input("Seleziona il PRIMO Modello (1-7): ")) - 1
            m2_idx = int(input("Seleziona il SECONDO Modello (1-7): ")) - 1
            if not (0 <= m1_idx < 7 and 0 <= m2_idx < 7):
                print("[ERROR] Indice modello non valido.")
                input("\nPremi INVIO per tornare al menu...")
                return
        except ValueError:
            print("[ERROR] Inserire un numero compreso tra 1 e 7.")
            input("\nPremi INVIO per tornare al menu...")
            return

        res1 = self.evaluate_model_index(m1_idx)
        res2 = self.evaluate_model_index(m2_idx)

        if res1 is None or res2 is None:
            input("\nPremi INVIO per tornare al menu...")
            return

        print("\n" + "=" * 95)
        print(f"  CONFRONTO: [M1] {res1['name']}  VS  [M2] {res2['name']}")
        print("=" * 95)
        print(f" {'Classe':<12} | {'F1 (M1)':<9} {'F1 (M2)':<9} {'Delta F1':<10} | {'Prec (M1)':<9} {'Prec (M2)':<9} | {'Rec (M1)':<9} {'Rec (M2)':<9}")
        print("-" * 95)

        for c in range(6):
            pc1 = res1['per_class'][c]
            pc2 = res2['per_class'][c]
            df1 = (pc2['f1'] - pc1['f1']) * 100
            d_str = f"{df1:+5.1f}%" if df1 != 0 else "  0.0%"
            print(f" {CATEGORIES[c]:<12} | {pc1['f1']*100:6.1f}%   {pc2['f1']*100:6.1f}%   {d_str:<10} | {pc1['prec']*100:6.1f}%   {pc2['prec']*100:6.1f}% | {pc1['rec']*100:6.1f}%   {pc2['rec']*100:6.1f}%")

        print("-" * 95)
        d_macro_f1 = (res2['macro_f1'] - res1['macro_f1']) * 100
        df1_macro_str = f"{d_macro_f1:+5.1f}%" if d_macro_f1 != 0 else "  0.0%"
        print(f" {'MEDIA MACRO':<12} | {res1['macro_f1']*100:6.1f}%   {res2['macro_f1']*100:6.1f}%   {df1_macro_str:<10} | {res1['macro_prec']*100:6.1f}%   {res2['macro_prec']*100:6.1f}% | {res1['macro_rec']*100:6.1f}%   {res2['macro_rec']*100:6.1f}%")
        print("-" * 95)
        print(f" COERENZA  | {res1['coherence']:6.1f}%   {res2['coherence']:6.1f}%   {(res2['coherence']-res1['coherence']):+5.1f}%")
        print("=" * 95 + "\n")
        input("\nPremi INVIO per tornare al menu...")

    def run_menu(self):
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 75)
            print("        MENU INTERATTIVO VALUTAZIONE E CONFRONTO METRICHE")
            print("=" * 75)
            print(" Seleziona l'opzione desiderata:")
            print(" " + "-" * 71)
            print("  1. Mostra Score: 1. Agente Bayesiano Dinamico (Prior HD)")
            print("  2. Mostra Score: 2. Agente Neurale Baseline (BCE Standard)")
            print("  3. Mostra Score: 3. Agente Neurale BCE + Penalizzazione Semantica")
            print("  4. Mostra Score: 4. Agente Neurale Focal Loss Standard")
            print("  5. Mostra Score: 5. Agente Neurale Focal + Penalizzazione (Picco Macro F1: 32.3%)")
            print("  6. Mostra Score: 6. Agente Neurale Contesto Esterno (Ring Semantics)")
            print("  7. Mostra Score: 7. Agente Neurale Asymmetric Loss (Picco Pedoni Recall: 98.8%)")
            print(" " + "-" * 71)
            print("  8. CONFRONTO DIRETTO AFFIANCATO TRA 2 MODELLI A SCELTA")
            print("  9. ESCI")
            print("=" * 75)

            choice = input(" Inserisci il numero dell'opzione (1-9): ").strip()
            if choice in [str(i) for i in range(1, 8)]:
                self.print_single_model_table(int(choice) - 1)
            elif choice == "8":
                self.compare_two_models()
            elif choice == "9":
                print("\nUscita dal menu di valutazione. Arrivederci!\n")
                break
            else:
                input("\n[ERROR] Opzione non valida. Premi INVIO per riprovare...")

if __name__ == "__main__":
    menu = ProjectEvaluatorMenu()
    menu.run_menu()
