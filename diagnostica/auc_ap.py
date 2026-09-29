# AUC / Average Precision per classe, indipendenti dalla soglia, sulle zone della cache di train
import sys, torch, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
sys.path.insert(0, ".")
from architettura_neurale.attention_per_zone_model import AttentionPerZoneModel
C = "/media/vrlab/Extreme Pro/occlusion_mapper_cache/"
CACHE = {"hybrid": C + "cached_dataset_neuro_hybrid_train_blocchi", "real": C + "cached_dataset_per_zone_train_blocchi"}
BLOCCHI = ["blocco_0002.pth", "blocco_0013.pth", "blocco_0024.pth"]
# modello -> GT su cui e' stato addestrato
MODELLI = {"hybrid": ("pesi_modelli/per_zone_checkpoint_attention_neuro_hybrid.pth", "hybrid"),
           "real": ("pesi_modelli/per_zone_checkpoint_attention_real_gt.pth", "real"),
           "positives_only": ("pesi_modelli/per_zone_checkpoint_attention_positive_only.pth", "real")}
CLASSI = ["c0 Auto", "c1 Camion/Bus", "c2 VRU-a", "c3 VRU-b", "c4 VRU-c", "c5 Barriera"]
SOGLIE_EVAL = [0.28, 0.25, 0.25, 0.25, 0.25, 0.26]
dev = "cuda"

dati = {}
for k, cart in CACHE.items():
    parti = [torch.load(f"{cart}/{b}") for b in BLOCCHI]
    dati[k] = {c: torch.cat([p[c] for p in parti]) for c in ["patches", "scalars", "masks", "targets"]}
    print(f"cache {k}: {len(dati[k]['targets'])} zone", flush=True)

for mk, (ck, gt) in MODELLI.items():
    net = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"]); net.eval()
    for gk in ["hybrid", "real"]:
        d = dati[gk]
        with torch.no_grad():
            pr = torch.cat([torch.sigmoid(net(d["patches"][i:i+1024].float().to(dev), d["scalars"][i:i+1024].to(dev),
                            occluder_mask=d["masks"][i:i+1024].to(dev))).cpu() for i in range(0, len(d["targets"]), 1024)]).numpy()
        T = (d["targets"].numpy() > 0.5); M = d["masks"].numpy() > 0.5
        tag = " (GT di addestramento)" if gk == gt else ""
        print(f"\n=== modello {mk} vs GT {gk}{tag}")
        print(f"  {'classe':15s} {'n zone':>7s} {'prev.':>6s} {'AUC':>6s} {'AP':>6s} {'F1@soglia eval':>15s} {'F1 max':>7s} {'soglia F1max':>12s}")
        for c in range(6):
            m = M[:, c]; y = T[m, c]; p = pr[m, c]
            if y.sum() == 0 or y.all():
                print(f"  {CLASSI[c]:15s} {m.sum():7d}  classe senza positivi/negativi"); continue
            pp = p >= SOGLIE_EVAL[c]; tp = (pp & y).sum()
            f1e = 2 * tp / max(pp.sum() + y.sum(), 1)
            P, R, th = precision_recall_curve(y, p); f1 = 2 * P * R / np.maximum(P + R, 1e-9); i = np.nanargmax(f1[:-1])
            print(f"  {CLASSI[c]:15s} {m.sum():7d} {y.mean():6.3f} {roc_auc_score(y, p):6.3f} {average_precision_score(y, p):6.3f} {f1e:15.3f} {f1[i]:7.3f} {th[i]:12.3f}")
