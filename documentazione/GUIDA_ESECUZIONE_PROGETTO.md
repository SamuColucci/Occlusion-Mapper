# Guida Completa all'Esecuzione del Progetto: Occlusion-Mapper

Benvenuto in **Occlusion-Mapper**. Questa guida è pensata per spiegare in modo chiaro e accessibile il funzionamento del sistema, la struttura della pipeline e tutti i comandi necessari per riprodurre gli esperimenti, eseguire l'addestramento, calcolare le valutazioni e avviare i visualizzatori grafici.

---

## 1. Cos'è Occlusion-Mapper? (Spiegazione Intuitiva)

### Il Problema dei Punti Ciechi nei Veicoli Autonomi
I sensori a bordo delle auto a guida autonoma (telecamere e LiDAR) funzionano in **linea di vista diretta** (*Line-of-Sight*): vedono solo ciò che non è coperto da ostacoli. Se un furgone è parcheggiato lungo la strada, il sensore non può vedere cosa c'è dietro. Un pedone o un ciclista che sbuca all'improvviso da quel varco cieco costringe il veicolo a frenate d'emergenza o causa collisioni.

I sistemi attuali hanno due limiti opposti:
* **I rilevatori 3D classici** ignorano l'area nascosta finché l'ostacolo non è visibile (tempo di reazione pari a zero).
* **Le regole conservative (es. Mobileye RSS)** assumono il caso peggiore ovunque (frenate continue e traffico bloccato).

### La Soluzione Proposta
**Occlusion-Mapper** stima in tempo reale il rischio all'interno dei varchi occlusi combinando:
1. **Geometria dell'ombra (LiDAR 3D)**: individua con precisione matematica i volumi d'ombra proiettati dagli ostacoli.
2. **Semantica Stradale (Mappe HD)**: analizza la superficie su cui cade l'ombra (es. asfalto carrabile, marciapiede, strisce pedonali).
3. **Rete Neurale con Vincoli Fisici (Neuro-Simbolica)**: una rete convoluzionale guidata dall'attenzione (*AttentionPerZoneModel*) stima la probabilità che dietro l'angolo vi sia un'auto, un mezzo pesante o un pedone/ciclista (VRU), bloccando a priori predizioni fisicamente impossibili (es. un camion non può nascondersi dietro un'utilitaria bassa; un'auto non transita sul marciapiede).

---

## 2. La Pipeline di Elaborazione in 5 Fasi

La pipeline trasforma la nuvola di punti grezza del LiDAR in un verdetto di sicurezza:

```
[ Nuvola LiDAR 3D + Mappa HD ]
              │
              ▼
  Fase 1: Space Carving & Raycasting 3D
  Simulazione dei raggi laser nel piano polare e proiezione Bird's-Eye-View (BEV 200x200).
              │
              ▼
  Fase 2: Decomposizione Poligonale & Bounding Box (OBB)
  Isolamento dei singoli varchi d'ombra geometrici e calcolo delle dimensioni fisiche minime.
              │
              ▼
  Fase 3: Estrazione Feature Multimodali per Zona
  Ritaglio di una patch BEV a 11 canali (64x64) e calcolo di 9 descrittori topologici scalari.
              │
              ▼
  Fase 4: Rete Neurale Ibrida Attention-per-Zone
  Estrazione feature con Channel Attention (SE-Net) e modulazione FiLM dai 9 descrittori.
              │
              ▼
  Fase 5: Inferenza Neuro-Simbolica & Valutazione
  Applicazione del mascheramento logico deterministico e stima del rischio per zona d'ombra.
```

---

## 3. Requisiti e Configurazione Ambiente

### Requisiti di Sistema
* **Sistema Operativo**: Linux (Ubuntu 20.04+) o Windows 10/11
* **Python**: 3.10 (definito in `environment.yml`)
* **GPU**: NVIDIA con supporto CUDA (consigliata per l'addestramento; inferenza e visualizzatori girano anche su CPU)
* **Dataset**: nuScenes `v1.0-trainval` con la **Map Expansion v1.3** per il benchmark completo; `v1.0-mini` per i test rapidi

### Installazione con Conda (consigliata)
```bash
conda env create -f environment.yml
conda activate occlusion-mapper
```
Se l'ambiente esiste già, aggiornalo con `conda env update -n occlusion-mapper -f environment.yml`.

**GPU meno recenti o driver NVIDIA < 580** (es. Quadro P4000 con driver 565): `environment.yml` installa l'ultima versione di PyTorch, compilata per CUDA 13, che su questi sistemi non si avvia (`ImportError ... libtorch_cuda.so`). Sostituiscila con la build per CUDA 12.6:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "numpy<2"        # nuscenes-devkit richiede numpy < 2
```

### Installazione con venv (Windows / standalone)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset e Pesi dei Modelli
* **Dataset**: tutti gli script leggono nuScenes da `./nuscenes`. Alcuni accettano anche `--dataroot` o la variabile `NUSCENES_DATAROOT`, ma lo smoke test e diversi visualizzatori usano solo `./nuscenes`. La cartella deve avere la struttura standard di nuScenes: `v1.0-trainval/`, `samples/LIDAR_TOP/` e `maps/` con `maps/expansion/*.json` della Map Expansion v1.3. Se il dataset sta su un hard disk esterno, collegalo una volta con:
  ```bash
  ln -s /media/$USER/ExtremePro/nuscenes nuscenes
  ```
  Se il disco viene montato con un altro nome, ricrea il collegamento con `ln -sfn <nuovo percorso>/nuscenes nuscenes`.
* **Pesi**: i checkpoint `.pth` stanno in `pesi_modelli/` dentro il progetto. Sono esclusi da git e vanno copiati a mano quando si clona il repository su un altro PC. I modelli senza checkpoint vengono segnalati e non valutati da `evaluate_final_official.py`.

### Verifica dell'Installazione
```bash
python smoke_test.py
```
Il test controlla, in ordine: librerie, caricamento del dataset, raycasting su 2 frame, caricamento del checkpoint `neuro_hybrid`, forward pass della rete e agente neurale. Deve terminare con `SMOKE TEST SUPERATO CON SUCCESSO` e senza avvisi `Checkpoint ... non trovato`.

---

## 4. Guida ai Comandi di Esecuzione

I comandi presuppongono l'ambiente attivo e la cartella del progetto come cartella corrente. Su Windows con venv, sostituisci `python` con `.\.venv\Scripts\python.exe`.

### A. Run Completa da Zero (Pipeline End-to-End)
Per rieseguire l'intero processo (estrazione zone, addestramento del modello ibrido, calibrazione soglie e benchmark):

```bash
# Passo 1: Estrazione varchi occlusi via raycasting 3D (multiprocessing su 8 core) -> extracted_occlusions/
python estrazione_zone_occluse.py --workers 8

# Passo 2: Addestramento del modello proposto (Ibrido Neuro-Simbolico)
python addestramento/train_per_zone_attention.py --mode hybrid --split train --workers 8

# Passo 3: Calibrazione delle soglie decisionali (F1 ottimizzato su train) -> valutazione/soglie_calibrate.json
python valutazione/calibra_soglie.py

# Passo 4: Valutazione ufficiale sui 6.019 fotogrammi inediti di validazione, con le soglie calibrate
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```

Note:
* L'estrazione salta i frame già elaborati; usa `--rigenera` per ricalcolarli tutti.
* Il training salva il dataset di addestramento in una cache a blocchi (`--cache_dir`, default `addestramento/`) e la riusa nelle esecuzioni successive; `--force` la rigenera. Le patch sono compresse senza perdita, quindi ogni cache occupa circa 1-2 GB invece di ~40 GB.
* Senza `--soglie` la valutazione usa le soglie di default dello script, non quelle calibrate.

---

### B. Addestramento dei Modelli per Studio Comparativo (Ablazione)

| `--mode` | Descrizione | Checkpoint generato in `pesi_modelli/` |
|---|---|---|
| `hybrid` | **Modello proposto**: Ibrido Neuro-Simbolico su GT spazio-semantica | `per_zone_checkpoint_attention_neuro_hybrid.pth` |
| `pure_neural` | Ablazione: senza vincoli fisici neuro-simbolici | `per_zone_checkpoint_attention_pure_neural.pth` |
| `real` | Baseline: box 3D annotati da annotatori umani | `per_zone_checkpoint_attention_real_gt.pth` |
| `real_pure_neural` | Baseline reale senza vincoli simbolici | `per_zone_checkpoint_attention_real_pure_neural.pth` |
| `positives_only` | Baseline reale addestrata sulle sole zone con ostacoli | `per_zone_checkpoint_attention_positive_only.pth` |
| `all` | Tutte le varianti in sequenza | — |

```bash
python addestramento/train_per_zone_attention.py --mode hybrid
python addestramento/train_per_zone_attention.py --mode all
```
Altre opzioni: `--epochs N` (default automatico), `--batch_size 64`, `--split train|val|all`, `--pesi_dir`, `--dataroot`.

---

### C. Valutazione e Benchmark Ufficiale
Lo script di valutazione confronta ciascun modello sia con la **Ground Truth Reale** (oggetti fisicamente nascosti) sia con la **Ground Truth Ibrida** (affordance potenziale di transito). Con `--mode all` include anche la baseline analitica **bayesiana** (`bayes`), che non usa una rete neurale:

```bash
# Solo il Modello Ibrido sullo split di validazione (150 scene mai viste)
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json

# Probabilita' della baseline bayesiana (una volta, prima di --mode all) -> extracted_occlusions_probabilities/
python inferenza_agenti/bayes_agent.py

# Valutazione comparativa completa (tutti i modelli)
python valutazione/evaluate_final_official.py --mode all --split val --soglie valutazione/soglie_calibrate.json
```
Per valutare un checkpoint diverso da quello predefinito: `--ckpt percorso/modello.pth`.

---

### D. Spiegabilità (XAI)
```bash
python diagnostica/genera_xai_shap_gradcam.py
```
Genera le figure Grad-CAM (attivazioni spaziali) e SHAP (importanza dei 9 descrittori scalari) per il modello ibrido, in `Figure/` e `documentazione/immagini_tesi/`.

---

## 5. File Generati in Automatico dopo la Valutazione

Al termine di `evaluate_final_official.py` vengono generati o aggiornati:

1. **`documentazione/RISULTATI_VALUTAZIONE_<SPLIT>.md`** (es. `RISULTATI_VALUTAZIONE_VAL.md`):
   * Tabelle comparative con Precision, Recall e F1-Score per ciascun modello, divise per categoria semantica (Auto, Camion/Bus, VRU) e per raggio di distanza (10m, 15m, 20m, 25m).
2. **`documentazione/TABELLA_RISULTATI_COMPLETA.md`**:
   * Tabella riassuntiva ufficiale dei risultati, aggiornata a ogni valutazione.
3. **`valutazione/cache_valutazione_<split>.json`**:
   * Predizioni, verdetti (True Positive, False Positive, False Negative) e punteggi di confidenza per ogni zona occlusa. Questa cache alimenta la dashboard dei visualizzatori senza ricalcoli.

---

## 6. Suite dei Visualizzatori Grafici Interattivi

I visualizzatori permettono di ispezionare visivamente ciò che accade in ogni frame:

```bash
# 1. Raycasting e Coni d'Ombra (Mappa BEV con varchi d'ombra e strutture urbane)
python visualizzatori/vis_raycasting_occlusioni.py 17

# 2. Input Multimodali Rete Neurale (canali BEV e 9 descrittori scalari FiLM)
python visualizzatori/vis_input_rete_neurale.py 17

# 3. Inferenza Neurale Live (probabilità stimate per ogni zona d'ombra)
#    --model NEURO_SIMB | PURE_NEURAL | REAL_GT | REAL_PURE_NEURAL | POS_ONLY
python visualizzatori/vis_inferenza_neurale.py 17

# 4. Ground Truth Duale a Confronto (GT Reale 3D vs GT Sintetica Ibrida)
python visualizzatori/vis_ground_truth_occlusioni.py 17

# 5. Dashboard di Valutazione Prestazioni (verdetti TP, FP, FN; richiede una valutazione già eseguita)
python visualizzatori/vis_valutazione_prestazioni.py 17

# 6. Dati Grezzi nuScenes prima del raycasting (LiDAR, box 3D annotati, mappa HD)
python visualizzatori/vis_nuscenes_raw.py 17
```

Con `--save` la figura del frame viene salvata subito a 300 DPI in `documentazione/immagini_tesi/`. Alcuni visualizzatori limitano il frame iniziale all'intervallo 1-404 (quello della `v1.0-mini`).

### Controlli da Tastiera nei Visualizzatori:
* `Freccia Destra (->)` / `Freccia Sinistra (<-)`: avanza o retrocede di un fotogramma.
* `Click Sinistro` su una zona: mostra nel pannello laterale i dettagli analitici (probabilità, dimensioni, classi predette).
* `T`: attiva/disattiva i livelli grafici ausiliari (overlay semantici della mappa HD).
* `S`: salva uno screenshot ad alta risoluzione del fotogramma corrente.
