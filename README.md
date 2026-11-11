# Occlusion-Mapper: Stima Neuro-Simbolica per la Percezione delle Zone Occluse nella Guida Autonoma

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![nuScenes](https://img.shields.io/badge/dataset-nuScenes-green.svg)](https://www.nuscenes.org/)

Repository ufficiale del progetto di Tesi dedicato alla stima, modellazione probabilistica e inferenza degli ostacoli nascosti nelle zone cieche sensoriali (*Beyond Line-of-Sight*) per veicoli a guida autonoma su dataset **nuScenes**.

---

## 🎯 Il Problema e la Soluzione

I sensori fisici di bordo (LiDAR e telecamere) operano rigorosamente in **linea di vista diretta** (*Line-of-Sight*): ostacoli come veicoli in sosta o edifici schermano completamente la visuale retrostante, generando **coni d'ombra sensoriali**. I sistemi di percezione convenzionali non segnalano alcun pericolo finché l'ostacolo non si rende visibile, riducendo a zero il margine di reazione.

<p align="center">
  <img src="Figure/raycasting_coni_ombra.png" alt="Raycasting LiDAR e Coni d'Ombra" width="85%" />
</p>

**Occlusion-Mapper** supera questo limite integrando:
1. **Geometria dell'Ombra (LiDAR 3D)**: raycasting 3D in coordinate polari per identificare con esattezza le aree occluse.
2. **Semantica Stradale (Mappe HD)**: informazione topologica sul piano stradale (corsia, marciapiede, strisce pedonali).
3. **Apprendimento Ibrido Neuro-Simbolico**: una rete convoluzionale guidata da attenzione su canali (*Channel Attention*) e modulazione scalare (*FiLM*), vincolata da regole fisiche deterministiche (che azzerano a priori predizioni incompatibili con la fisica del mondo reale).

---

## 🔄 Architettura della Pipeline End-to-End

Flusso di elaborazione suddiviso nei 5 macro-moduli funzionali:

<p align="center">
  <img src="Figure/pipeline_architettura_cvpr.png" alt="Pipeline Architettura Completa" width="95%" />
</p>

```
[ Nuvola LiDAR 3D + HD-Map nuScenes ]
                 │
                 ▼
 1. Space Carving & Raycasting 3D (Bird's-Eye-View 200x200 a 0.4 m/pixel)
                 │
                 ▼
 2. Decomposizione Poligonale & Bounding Box Orientati (Shapely + OBB)
                 │
                 ▼
 3. Estrazione Feature Multimodali per Zona (Patch BEV 11 canali 64x64 + 9 scalari)
                 │
                 ▼
 4. Rete Neurale Attention-per-Zone (Channel Attention + Modulazione FiLM)
                 │
                 ▼
 5. Mascheramento Neuro-Simbolico & Verdetto di Rischio (Auto, Camion/Bus, VRU)
```

<p align="center">
  <img src="Figure/patch_11_canali_bev.png" alt="Patch Multimodale 11 Canali BEV" width="85%" />
</p>

---

## 📊 Risultati Sperimentali e Spiegabilità (XAI)

### 1. Inferenza Neurale e Mappa del Rischio
Stima probabilistica per zona e visualizzazione intuitiva delle probabilità di classe:

<p align="center">
  <img src="Figure/fig_4_neural_inference_neuro_simb.png" alt="Inferenza Neurale Mappa del Rischio" width="85%" />
</p>

### 2. Confronto Sperimentale: Neuro-Simbolico vs Pura Rete Neurale
L'applicazione dei vincoli deterministici elimina sistematicamente i falsi allarmi fisicamente impossibili:

<p align="center">
  <img src="Figure/fig_5b_confronto_neuro_simb_vs_pure_neural.png" alt="Confronto Neuro-Simbolico vs Pure Neural" width="95%" />
</p>

### 3. Spiegabilità con Grad-CAM e Valori di Shapley (SHAP)
Verifica della focalizzazione visiva dei filtri convoluzionali e dell'attribuzione di importanza dei 9 descrittori scalari topologici:

<p align="center">
  <img src="Figure/grad_cam_explainability.png" alt="Grad-CAM Spatial Explainability" width="48%" />
  <img src="Figure/shap_feature_importance.png" alt="SHAP Feature Importance" width="48%" />
</p>

## ⚙️ Installazione

### Metodo 1: Con Conda (Consigliato per PC Laboratorio / Server GPU / Linux)
```bash
# Creazione e attivazione ambiente da file environment.yml (Python 3.10)
conda env create -f environment.yml
conda activate occlusion-mapper
```

> **GPU meno recenti o driver NVIDIA < 580** (es. Quadro P4000, driver 565): la versione di PyTorch installata di default è compilata per CUDA 13 e non si avvia (`ImportError ... libtorch_cuda.so`). Sostituiscila con la build per CUDA 12.6:
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
> pip install "numpy<2"        # richiesto da nuscenes-devkit
> ```

### Metodo 2: Con Python venv (Standard / Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset nuScenes e Pesi dei Modelli
* **Dataset**: gli script leggono nuScenes da `./nuscenes`, una cartella dataroot standard con `v1.0-trainval/`, `samples/LIDAR_TOP/` e `maps/` comprensiva della **Map Expansion v1.3** (`maps/expansion/*.json`). Se il dataset sta su un hard disk esterno, basta un collegamento simbolico:
  ```bash
  ln -s /media/$USER/ExtremePro/nuscenes nuscenes     # adatta il percorso al punto di montaggio del disco
  ```
  Per un test rapido si può usare anche `v1.0-mini`: l'adapter la riconosce in automatico se `v1.0-trainval` non è presente.
* **Pesi**: i checkpoint `.pth` stanno nella cartella `pesi_modelli/` del progetto. Non sono versionati su git, quindi vanno copiati a mano su ogni PC.

### Test Rapido dell'Installazione
```bash
python smoke_test.py
```
Lo smoke test verifica librerie, lettura del dataset, raycasting LiDAR su 2 frame, caricamento dei pesi, forward pass sulla rete e agente neurale.

---

## 🚀 Comandi Principali di Esecuzione

I comandi presuppongono l'ambiente attivo (`conda activate occlusion-mapper`) e la cartella del progetto come cartella corrente. Con venv su Windows, sostituisci `python` con `.\.venv\Scripts\python.exe`.

### A. Run Completa da Zero (Pipeline End-to-End)
Esegue l'intero flusso dall'estrazione geometrica alla valutazione finale:

```bash
# 1. Estrazione varchi occlusi via raycasting 3D (8 worker paralleli) -> extracted_occlusions/
python estrazione_zone_occluse.py --workers 8

# 2. Addestramento modello proposto (Ibrido Neuro-Simbolico) -> pesi_modelli/per_zone_checkpoint_attention_neuro_hybrid.pth
python addestramento/train_per_zone_attention.py --mode hybrid --split train --workers 8

# 3. Calibrazione delle soglie decisionali (F1 sul train) -> valutazione/soglie_calibrate.json
python valutazione/calibra_soglie.py

# 4. Valutazione ufficiale con le soglie calibrate (split validazione: 6.019 frame inediti)
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```

> Senza `--soglie` la valutazione usa le soglie di default dello script, non quelle calibrate.

---

### B. Run Rapida: Solo Modello Ibrido Proposto
```bash
python addestramento/train_per_zone_attention.py --mode hybrid
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```

---

### C. Studio di Ablazione & Altri Modelli

| `--mode` | Modello | Checkpoint in `pesi_modelli/` |
|---|---|---|
| `hybrid` | **Proposto**: Attention Neuro-Simbolico su GT ibrida spazio-semantica | `per_zone_checkpoint_attention_neuro_hybrid.pth` |
| `pure_neural` | Ablazione: stesso modello senza vincoli neuro-simbolici | `per_zone_checkpoint_attention_pure_neural.pth` |
| `real` | Baseline su GT reale 3D (box annotati) | `per_zone_checkpoint_attention_real_gt.pth` |
| `real_pure_neural` | Baseline GT reale senza vincoli simbolici | `per_zone_checkpoint_attention_real_pure_neural.pth` |
| `positives_only` | Baseline GT reale con le sole zone che contengono ostacoli | `per_zone_checkpoint_attention_positive_only.pth` |
| `bayes` | Baseline analitica bayesiana, senza rete (solo valutazione) | — |

```bash
# Addestramento di una variante (o di tutte con --mode all)
python addestramento/train_per_zone_attention.py --mode pure_neural
python addestramento/train_per_zone_attention.py --mode all

# Valutazione comparativa di tutti i modelli insieme
python valutazione/evaluate_final_official.py --mode all --split val --soglie valutazione/soglie_calibrate.json
```

Opzioni utili del training: `--epochs N`, `--batch_size 64`, `--force` (rigenera la cache del dataset), `--cache_dir` (cartella della cache a blocchi, default `addestramento/`), `--pesi_dir` (default `pesi_modelli/`).

---

### D. File Generati in Automatico al Termine della Valutazione
* **`documentazione/RISULTATI_VALUTAZIONE_<SPLIT>.md`**: report con tabelle comparative di Precision, Recall e F1-Score per categoria semantica (Auto, Camion/Bus, VRU) e raggio di distanza (10m, 15m, 20m, 25m).
* **`documentazione/TABELLA_RISULTATI_COMPLETA.md`**: tabella riassuntiva ufficiale, aggiornata a ogni valutazione.
* **`valutazione/cache_valutazione_<split>.json`**: cache con i verdetti (TP, FP, FN) e le probabilità per ciascun fotogramma. Viene usata dalla dashboard dei visualizzatori.

---

### E. Spiegabilità (XAI)
```bash
# Grad-CAM e SHAP sul modello ibrido -> Figure/ e documentazione/immagini_tesi/
python diagnostica/genera_xai_shap_gradcam.py
```

---

## 🖥️ Suite dei Visualizzatori Interattivi

È possibile esplorare un fotogramma specificandone il numero (es. `17`):

```bash
# 1. Raycasting e Coni d'Ombra LiDAR
python visualizzatori/vis_raycasting_occlusioni.py 17

# 2. Input Multimodali alla Rete (11 canali BEV + 9 scalari)
python visualizzatori/vis_input_rete_neurale.py 17

# 3. Inferenza Neurale Live e Zone di Rischio (--model NEURO_SIMB | PURE_NEURAL | REAL_GT | REAL_PURE_NEURAL | POS_ONLY)
python visualizzatori/vis_inferenza_neurale.py 17

# 4. Confronto Ground Truth (Reale vs Sintetica Ibrida)
python visualizzatori/vis_ground_truth_occlusioni.py 17

# 5. Dashboard Valutazione Prestazioni (Verdetti TP, FP, FN; richiede una valutazione già eseguita)
python visualizzatori/vis_valutazione_prestazioni.py 17

# 6. Dati Grezzi nuScenes prima del raycasting (LiDAR, box 3D annotati e mappa HD in BEV)
python visualizzatori/vis_nuscenes_raw.py 17
```

**Comandi nei visualizzatori**:
* `->` / `<-`: Avanza / retrocede di un fotogramma
* `Click sinistro`: Ispeziona le statistiche di una zona d'ombra
* `T`: Mostra/nasconde i layer topologici ausiliari
* `S`: Salva screenshot in alta risoluzione (con `--save` il salvataggio è immediato)

> Nota: diversi visualizzatori limitano il numero di frame iniziale a 1-404, l'intervallo della `v1.0-mini`.

---

## 📁 Struttura del Progetto

```text
Occlusion-Mapper/
├── architettura_neurale/       # Modello AttentionPerZone, blocchi SE, FiLM, Asymmetric Loss
├── addestramento/              # Script di training GPU con cache a blocchi e multiprocessing
├── ground_truth/               # Generatori di Ground Truth (Reale 3D e Ibrida Spazio-Semantica)
├── raycaster/                  # Raycasting polar-grid 3D e space carving LiDAR
├── dataset_adapter/            # Adapter nuScenes (auto-detect trainval/mini) e cache mappe HD
├── inferenza_agenti/           # Agente neurale per-zone e baseline bayesiana
├── valutazione/                # Benchmark ufficiale e calibrazione soglie
├── visualizzatori/             # Suite di visualizzatori grafici interattivi
├── diagnostica/                # Spiegabilità XAI (Grad-CAM, SHAP) e metriche AUC/AP
├── documentazione/             # Guide, comandi rapidi e immagini per la tesi
├── Figure/                     # Figure usate nel README e nella tesi
├── nuscenes                    # Dataset nuScenes (collegamento al disco esterno, non versionato)
├── pesi_modelli/               # Checkpoint .pth dei modelli addestrati (non versionato)
├── environment.yml             # Ambiente Conda (Python 3.10)
├── requirements.txt            # Dipendenze Python per venv
├── smoke_test.py               # Test rapido dell'installazione
├── COMANDI_RAPIDI.md           # Cheat-sheet sintetico con comandi da copiare ed eseguire
└── estrazione_zone_occluse.py  # Script batch per l'estrazione parallela dei varchi occlusi
```
