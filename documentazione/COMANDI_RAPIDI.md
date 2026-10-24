# Comandi Rapidi (Cheat-Sheet da Copiare e Incollare)

> **Cos'è Occlusion-Mapper**: Sistema per la stima e l'anticipazione di ostacoli nascosti nei punti ciechi del LiDAR (coni d'ombra Beyond-LoS) per la guida autonoma, che unisce raycasting 3D geometrico, semantica da mappe HD e una rete neurale ad attenzione con vincoli logico-fisici deterministici.

---

## 1. Setup Ambiente

### Opzione A: Con Conda (Consigliato in Laboratorio / Server GPU)
```bash
# Creazione ambiente da file environment.yml
conda env create -f environment.yml

# Attivazione
conda activate occlusion-mapper
```
*Oppure creazione manuale:*
```bash
conda create -n occlusion-mapper python=3.10 -y
conda activate occlusion-mapper
pip install -r requirements.txt
```

### Opzione B: Con venv standard (Windows / Standalone)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


---

## 2. Run Completa da Zero (Pipeline End-to-End)

Esegue l'intero flusso dall'estrazione geometrica alla valutazione finale:

```powershell
# 1. Estrazione varchi occlusi da nuScenes via raycasting 3D (8 worker)
.\.venv\Scripts\python.exe estrazione_zone_occluse.py --workers 8

# 2. Addestramento modello proposto (Ibrido Neuro-Simbolico)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode hybrid --split train --workers 8

# 3. Calibrazione fine delle soglie decisionali ottimali
.\.venv\Scripts\python.exe valutazione/calibra_soglie.py

# 4. Valutazione ufficiale sui 6.019 frame di validazione
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode hybrid --split val
```

---

## 3. Run Rapida: Solo Modello Ibrido Proposto

Per addestrare e valutare unicamente l'architettura principale:

```powershell
# Addestramento modello ibrido
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode hybrid

# Valutazione modello ibrido su split val (150 scene mai viste)
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode hybrid --split val
```

---

## 4. Run Altri Modelli e Studio di Ablazione

```powershell
# Modello puro neurale (senza vincoli neuro-simbolici)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode pure_neural

# Modello baseline addestrato solo su GT Reale 3D
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode real

# Addestramento di tutte le varianti
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode all
```

---

## 5. Valutazione e Benchmark Ufficiale

```powershell
# Benchmark comparativo di tutti i modelli insieme
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode all --split val
```

### File Generati Automaticamente al Termine della Valutazione:
* **`documentazione/RISULTATI_VALUTAZIONE_VAL.md`**: Report con le tabelle di riepilogo Markdown contenenti Precision, Recall e F1-Score a 10m, 15m, 20m e 25m per ciascun modello e classe semantica.
* **`valutazione/cache_valutazione_val.json`**: Cache strutturata contenente verdetti per zona (TP, FP, FN) e probabilità stimate per ciascuno dei 6.019 fotogrammi.

---

## 6. Visualizzatori Interattivi

Sostituire `17` con qualsiasi numero di frame desiderato (es. 1-404 su mini):

```powershell
# Mappa BEV raycasting e coni d'ombra
.\.venv\Scripts\python.exe visualizzatori/vis_raycasting_occlusioni.py 17

# Input multimodali alla rete neurale (canali BEV + scalari FiLM)
.\.venv\Scripts\python.exe visualizzatori/vis_input_rete_neurale.py 17

# Mappa inferenza neurale e zone di rischio stimate
.\.venv\Scripts\python.exe visualizzatori/vis_inferenza_neurale.py 17

# Confronto visivo GT Reale vs GT Sintetica Ibrida
.\.venv\Scripts\python.exe visualizzatori/vis_ground_truth_occlusioni.py 17

# Dashboard analitica verdetti di valutazione (TP, FP, FN)
.\.venv\Scripts\python.exe visualizzatori/vis_valutazione_prestazioni.py 17

# Vista sincrona nuScenes (LiDAR + 6 camere surround)
.\.venv\Scripts\python.exe visualizzatori/vis_nuscenes_raw.py 17
```
