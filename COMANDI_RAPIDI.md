# Comandi Rapidi (Cheat-Sheet da Copiare e Incollare)

> **Cos'è Occlusion-Mapper**: Sistema per la stima e l'anticipazione di ostacoli nascosti nei punti ciechi del LiDAR (coni d'ombra Beyond-LoS) per la guida autonoma, che unisce raycasting 3D geometrico, semantica da mappe HD e una rete neurale ad attenzione con vincoli logico-fisici deterministici.

Tutti i comandi vanno eseguiti dalla cartella del progetto con l'ambiente attivo. Con venv su Windows, sostituisci `python` con `.\.venv\Scripts\python.exe`.

---

## 1. Setup Ambiente

### Opzione A: Con Conda (Consigliato in Laboratorio / Server GPU)
```bash
conda env create -f environment.yml     # Python 3.10
conda activate occlusion-mapper
```
*Se l'ambiente esiste già:* `conda env update -n occlusion-mapper -f environment.yml`

*GPU meno recenti o driver NVIDIA < 580 (es. Quadro P4000), se `import torch` fallisce:*
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "numpy<2"
```

### Opzione B: Con venv standard (Windows / Standalone)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset e pesi
Il dataset sta sul disco esterno ed è collegato al progetto come `./nuscenes`. I pesi stanno in `./pesi_modelli/`, nel progetto.
```bash
ln -s /media/$USER/ExtremePro/nuscenes nuscenes   # solo la prima volta (adatta il percorso del disco)
ls nuscenes/v1.0-trainval | head -3                # se dà errore, il disco non è collegato o il percorso è cambiato
```

---

## ⚡ Test Rapido di Funzionamento (Smoke Test in ~10 secondi)
Verifica librerie, dataset nuScenes, raycasting LiDAR, pesi PyTorch e agente neurale:
```bash
python smoke_test.py
```

---

## 2. Run Completa da Zero (Pipeline End-to-End)

```bash
# 1. Estrazione varchi occlusi da nuScenes via raycasting 3D (8 worker)
python estrazione_zone_occluse.py --workers 8

# 2. Addestramento modello proposto (Ibrido Neuro-Simbolico)
python addestramento/train_per_zone_attention.py --mode hybrid --split train --workers 8

# 3. Calibrazione delle soglie decisionali -> valutazione/soglie_calibrate.json
python valutazione/calibra_soglie.py

# 4. Valutazione ufficiale sui 6.019 frame di validazione, con soglie calibrate
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```

---

## 3. Run Rapida: Solo Modello Ibrido Proposto

```bash
python addestramento/train_per_zone_attention.py --mode hybrid
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```

---

## 4. Run Altri Modelli e Studio di Ablazione

```bash
# Modello puro neurale (senza vincoli neuro-simbolici)
python addestramento/train_per_zone_attention.py --mode pure_neural

# Baseline su GT Reale 3D: completa / senza vincoli simbolici / solo zone con ostacoli
python addestramento/train_per_zone_attention.py --mode real
python addestramento/train_per_zone_attention.py --mode real_pure_neural
python addestramento/train_per_zone_attention.py --mode positives_only

# Tutte le varianti in sequenza
python addestramento/train_per_zone_attention.py --mode all
```
Opzioni: `--epochs N`, `--batch_size 64`, `--force` (rigenera la cache), `--cache_dir <cartella>`, `--pesi_dir <cartella>`.

---

## 5. Valutazione e Benchmark Ufficiale

```bash
# Benchmark comparativo di tutti i modelli (incluso il baseline bayesiano)
python valutazione/evaluate_final_official.py --mode all --split val --soglie valutazione/soglie_calibrate.json

# Checkpoint personalizzato
python valutazione/evaluate_final_official.py --mode hybrid --ckpt percorso/modello.pth
```

### File Generati Automaticamente al Termine della Valutazione:
* **`documentazione/RISULTATI_VALUTAZIONE_<SPLIT>.md`**: tabelle con Precision, Recall e F1-Score a 10m, 15m, 20m e 25m per modello e categoria (Auto, Camion/Bus, VRU).
* **`documentazione/TABELLA_RISULTATI_COMPLETA.md`**: tabella riassuntiva aggiornata.
* **`valutazione/cache_valutazione_<split>.json`**: verdetti per zona (TP, FP, FN) e probabilità stimate per ogni fotogramma (usata dalla dashboard).

---

## 6. Visualizzatori Interattivi

Sostituisci `17` con il numero di frame desiderato (alcuni visualizzatori accettano solo valori da 1 a 404):

```bash
# Mappa BEV raycasting e coni d'ombra
python visualizzatori/vis_raycasting_occlusioni.py 17

# Input multimodali alla rete neurale (canali BEV + scalari FiLM)
python visualizzatori/vis_input_rete_neurale.py 17

# Inferenza neurale e zone di rischio (--model NEURO_SIMB | PURE_NEURAL | REAL_GT | REAL_PURE_NEURAL | POS_ONLY)
python visualizzatori/vis_inferenza_neurale.py 17

# Confronto visivo GT Reale vs GT Sintetica Ibrida
python visualizzatori/vis_ground_truth_occlusioni.py 17

# Dashboard analitica verdetti di valutazione (TP, FP, FN)
python visualizzatori/vis_valutazione_prestazioni.py 17

# Dati grezzi nuScenes prima del raycasting (LiDAR, box 3D, mappa HD)
python visualizzatori/vis_nuscenes_raw.py 17
```
Aggiungi `--save` per salvare subito la figura a 300 DPI in `documentazione/immagini_tesi/`.

---

## 7. Spiegabilità (XAI)

```bash
python diagnostica/genera_xai_shap_gradcam.py    # Grad-CAM + SHAP -> Figure/ e documentazione/immagini_tesi/
```
