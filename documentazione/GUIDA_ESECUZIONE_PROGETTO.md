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
* **Sistema Operativo**: Windows 10/11 o Linux (Ubuntu 20.04+)
* **Python**: 3.11
* **GPU**: NVIDIA con supporto CUDA (consigliata per addestramento veloce; l'inferenza e i visualizzatori girano anche su CPU)
* **Dataset**: nuScenes (`v1.0-mini` per test rapidi, `v1.0-trainval` per il benchmark completo)

### Installazione Dipendenze
Apri un terminale (PowerShell o Bash) nella cartella radice del progetto:

```powershell
# 1. Creazione ambiente virtuale (tramite venv o conda)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Installazione librerie richieste
pip install -r requirements.txt
```

---

## 4. Guida ai Comandi di Esecuzione

Tutti i comandi sottostanti utilizzano l'interprete dell'ambiente virtuale (`.\.venv\Scripts\python.exe` su Windows o `python` se l'ambiente è attivato).

### A. Run Completa da Zero (Pipeline End-to-End)
Se desideri rieseguire l'intero processo da zero (estrazione zone, addestramento del modello ibrido, calibrazione soglie e benchmark):

```powershell
# Passo 1: Estrazione varchi occlusi via raycasting 3D (multiprocessing su 8 core)
.\.venv\Scripts\python.exe estrazione_zone_occluse.py --workers 8

# Passo 2: Addestramento del modello proposto (Ibrido Neuro-Simbolico)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode hybrid --split train --workers 8

# Passo 3: Calibrazione fine delle soglie decisionali (F1 ottimizzato su train)
.\.venv\Scripts\python.exe valutazione/calibra_soglie.py

# Passo 4: Valutazione ufficiale sui 6.019 fotogrammi inediti di validazione
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode hybrid --split val
```

---

### B. Addestramento dei Modelli per Studio Comparativo (Ablazione)
Per confrontare le prestazioni del modello ibrido con le varianti di controllo descritte nella tesi:

```powershell
# 1. Modello Proposto (Ibrido Neuro-Simbolico su GT Spazio-Semantica)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode hybrid

# 2. Modello Ablation Puro (Senza Vincoli Fisici Neuro-Simbolici)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode pure_neural

# 3. Baseline Reale (Addestrato esclusivamente sui box 3D annotati da annotatori umani)
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode real

# 4. Addestramento di tutte le varianti in sequenza
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py --mode all
```

---

### C. Valutazione e Benchmark Ufficiale
Lo script di valutazione confronta automaticamente ciascun modello sia con la **Ground Truth Reale** (oggetti fisicamente nascosti) sia con la **Ground Truth Ibrida** (affordance potenziale di transito):

```powershell
# Valutazione solo del Modello Ibrido su split di validazione (150 scene mai viste)
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode hybrid --split val

# Valutazione comparativa completa (Tutti i modelli insieme per generare la tabella comparativa)
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py --mode all --split val
```

---

## 5. File Generati in Automatico dopo la Valutazione

Al termine dell'esecuzione di `evaluate_final_official.py`, il sistema genera automaticamente due file fondamentali:

1. **`documentazione/RISULTATI_VALUTAZIONE_VAL.md`** (o `RISULTATI_VALUTAZIONE_ALL.md`):
   * Contiene le tabelle comparative formattate in Markdown con Precision, Recall e F1-Score per ciascun modello, disaggregate per categoria semantica (Auto, Camion/Bus, VRU) e per raggio di distanza (10m, 15m, 20m, 25m).
2. **`valutazione/cache_valutazione_val.json`**:
   * File di cache contenente le predizioni, i verdetti di sicurezza (True Positive, False Positive, False Negative) e i punteggi di confidenza per ogni singola zona occlusa analizzata. Questa cache alimenta i visualizzatori grafici senza richiedere ricalcoli.

---

## 6. Suite dei Visualizzatori Grafici Interattivi

I visualizzatori permettono di ispezionare visivamente ciò che accade in ogni frame:

```powershell
# 1. Raycasting e Coni d'Ombra (Mappa BEV con varchi d'ombra e strutture urbane)
.\.venv\Scripts\python.exe visualizzatori/vis_raycasting_occlusioni.py 17

# 2. Input Multimodali Rete Neurale (Mostra i canali BEV e i 9 descrittori scalari FiLM)
.\.venv\Scripts\python.exe visualizzatori/vis_input_rete_neurale.py 17

# 3. Inferenza Neurale Live (Mappa delle probabilità stimate per ogni zona d'ombra)
.\.venv\Scripts\python.exe visualizzatori/vis_inferenza_neurale.py 17

# 4. Ground Truth Duale a Confronto (GT Reale 3D vs GT Sintetica Ibrida)
.\.venv\Scripts\python.exe visualizzatori/vis_ground_truth_occlusioni.py 17

# 5. Dashboard di Valutazione Prestazioni (Verdetti TP, FP, FN con box comparativi)
.\.venv\Scripts\python.exe visualizzatori/vis_valutazione_prestazioni.py 17

# 6. Ispezione Dati Grezzi nuScenes (LiDAR 3D + telecamere surround sincronizzate)
.\.venv\Scripts\python.exe visualizzatori/vis_nuscenes_raw.py 17
```

### Controlli da Tastiera nei Visualizzatori:
* `Freccia Destra (->)` / `Freccia Sinistra (<-)`: Avanza o retrocede di un fotogramma.
* `Click Sinistro` su una zona: Mostra nel pannello laterale i dettagli analitici (probabilità, dimensioni, classi predette).
* `T`: Attiva/disattiva livelli grafici ausiliari (overlay semantici della mappa HD).
* `S`: Salva uno screenshot ad alta risoluzione del fotogramma corrente.
