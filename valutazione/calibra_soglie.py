# Calibrazione delle soglie di decisione per modello (valutazione/calibra_soglie.py)
# Sceglie, per ogni modello e per ciascuna delle 4 macro-classi [Auto, Camion/Bus, VRU, Barriera],
# la soglia che massimizza l'F1 rispetto alla GT su cui il modello e' stato addestrato.
# La taratura usa la cache di TRAIN: lo split di validazione resta inedito per la valutazione finale.

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel

CATEGORIE = ["Auto", "Camion/Bus", "VRU", "Barriera"]

# Modello -> (checkpoint, cartella dei blocchi con la GT di addestramento)
MODELLI = {
    "hybrid": ("per_zone_checkpoint_attention_neuro_hybrid.pth", "cached_dataset_neuro_hybrid_train_blocchi"),
    "real": ("per_zone_checkpoint_attention_real_gt.pth", "cached_dataset_per_zone_train_blocchi"),
    "positives_only": ("per_zone_checkpoint_attention_positive_only.pth", "cached_dataset_per_zone_train_blocchi"),
}

# Vincoli sull'area identici a quelli di evaluate_final_official.py
AREA_MINIMA = [3.5, 8.0, 0.0, 0.0]


# Riduce le 6 classi della rete alle 4 macro-classi usate in valutazione (VRU = massimo delle classi 2-4)
def a_macro_classi(x: np.ndarray) -> np.ndarray:
    return np.stack([x[:, 0], x[:, 1], x[:, 2:5].max(axis=1), x[:, 5]], axis=1)


# Soglia che massimizza l'F1 su una griglia regolare, applicando lo stesso vincolo d'area della valutazione
def miglior_soglia(prob: np.ndarray, vero: np.ndarray, area_ok: np.ndarray):
    griglia = np.round(np.arange(0.05, 0.96, 0.01), 2)
    migliore = (0.5, 0.0, 0.0, 0.0)
    for s in griglia:
        pred = (prob >= s) & area_ok
        tp = int((pred & vero).sum())
        n_pred, n_veri = int(pred.sum()), int(vero.sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_veri if n_veri else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > migliore[1]:
            migliore = (float(s), f1, prec, rec)
    return migliore


def main():
    parser = argparse.ArgumentParser(description="Calibrazione delle soglie per modello sulla cache di train")
    parser.add_argument("--cache_dir", type=str, default="addestramento", help="Cartella dei blocchi della cache")
    parser.add_argument("--pesi_dir", type=str, default="pesi_modelli", help="Cartella dei checkpoint")
    parser.add_argument("--blocchi", type=int, default=6, help="Numero di blocchi usati, distribuiti su tutta la cache")
    parser.add_argument("--output", type=str, default=os.path.join("valutazione", "soglie_calibrate.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risultato = {}

    for chiave, (file_ckpt, cartella) in MODELLI.items():
        ckpt_path = os.path.join(args.pesi_dir, file_ckpt)
        if not os.path.exists(ckpt_path):
            print(f"[ATTENZIONE] Checkpoint non trovato per [{chiave}]: {ckpt_path}. Salto.")
            continue

        percorso = os.path.join(args.cache_dir, cartella)
        tutti = sorted(f for f in os.listdir(percorso) if f.endswith(".pth"))
        scelti = [tutti[i] for i in np.linspace(0, len(tutti) - 1, min(args.blocchi, len(tutti))).astype(int)]

        model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model.eval()

        prob_l, vero_l, area_l = [], [], []
        for nome in scelti:
            dati = torch.load(os.path.join(percorso, nome))
            targets = dati["targets"]
            # positives_only si addestra solo sulle zone con ostacoli, ma viene valutato su tutte:
            # le soglie si calibrano quindi sull'insieme completo delle zone
            with torch.no_grad():
                for i in range(0, len(targets), 1024):
                    logits = model(dati["patches"][i:i + 1024].float().to(device),
                                   dati["scalars"][i:i + 1024].to(device),
                                   occluder_mask=dati["masks"][i:i + 1024].to(device))
                    prob_l.append(torch.sigmoid(logits).cpu().numpy())
            vero_l.append(targets.numpy())
            area_l.append(dati["scalars"][:, 0].numpy())
            print(f"  [{chiave}] letto {nome} ({len(targets)} zone)", flush=True)

        prob = a_macro_classi(np.concatenate(prob_l))
        vero = a_macro_classi(np.concatenate(vero_l)) > 0.5
        area = np.concatenate(area_l)

        soglie = []
        print(f"\n=== [{chiave}] {len(area)} zone da {len(scelti)} blocchi")
        for c, nome_c in enumerate(CATEGORIE):
            s, f1, prec, rec = miglior_soglia(prob[:, c], vero[:, c], area >= AREA_MINIMA[c])
            soglie.append(s)
            print(f"  {nome_c:12s} soglia {s:.2f} | F1 {f1 * 100:5.1f}% | Precision {prec * 100:5.1f}% | Recall {rec * 100:5.1f}%")
        risultato[chiave] = soglie

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(risultato, f, indent=4)
    print(f"\n[OK] Soglie salvate in: {args.output}")


if __name__ == "__main__":
    main()
