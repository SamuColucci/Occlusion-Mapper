# Occlusion-Mapper: Stima Neuro-Simbolica e Apprendimento Neurale per la Percezione delle Zone Occluse nella Guida Autonoma

Repository ufficiale del progetto di Tesi di Laurea dedicato alla stima, modellazione probabilistica e inferenza neurale degli ostacoli nascosti nelle zone cieche sensoriali (occlusioni LiDAR Beyond-LoS) per veicoli a guida autonoma su dataset **nuScenes**.

---

## Panoramica della Pipeline

I sensori fisici di bordo (LiDAR e telecamere) sono strettamente limitati alla linea di vista (*Line-of-Sight*, LoS). **Occlusion-Mapper** stima la presenza, la plausibilità spaziale e la classe di potenziali ostacoli nascosti all'interno delle regioni d'ombra generate da altri veicoli o elementi urbani tramite una pipeline modulare in quattro fasi:

```
[ Nuvola di Punti LiDAR 3D + HD-Map nuScenes ]
                 │
                 ▼
 1. Space Carving & Raycasting 3D Parallelizzato (Bird's-Eye-View a 11 canali)
                 │
                 ▼
 2. Decomposizione Poligonale & Pulizia Geometrica Vettoriale (Shapely + OBB)
                 │
                 ▼
 3. Modello Neurale Attention-per-Zone (AttentionPerZoneModel)
    ├── Channel Attention (Squeeze-and-Excitation su 11 canali 64x64)
    ├── Condizionamento Semantico FiLM (9 Descrittori Scalari HD-Map)
    ├── Maschere Logico-Simboliche Determistiche (Altezza/Dimensioni Occludente ISO 26262)
    └── Funzione di Costo Asymmetric Loss (ASL) Bilanciata
                 │
                 ▼
 4. Valutazione Empirica Duale (GT Reale nuScenes 3D vs. GT Sintetica Ibrida Beyond-LoS)
```

1. **Space Carving e Raycasting LiDAR 3D**: Simulazione del fascio LiDAR 3D in coordinate polari per identificare i coni d'ombra generati da veicoli e ostacoli. Le aree occluse vengono proiettate in coordinate *Bird's-Eye-View* (BEV $200 \times 200$, risoluzione $0.4\,\text{m}$, raggio $25\,\text{m}$) ed estratte a livello di singola zona (*patch-per-zone* $64 \times 64$).
2. **Descrittori Geometrici e Semantici (HD-Map)**: Calcolo di $9$ descrittori scalari per ciascuna zona d'ombra (area, distanza dal veicolo ego, larghezza/lunghezza minima OBB, frazione di asfalto carrabile, marciapiede, attraversamento pedonale, banchina e terreno).
3. **Architettura Neurale Ibrida (`AttentionPerZoneModel`)**:
   - Backbone convoluzionale residuo su **11 canali BEV** (layer semantici HD-map, densità e altezza LiDAR, mappe d'ombra dinamiche e statiche).
   - Meccanismo di **Channel Attention** (*Squeeze-and-Excitation*) a livello di input e in ciascun blocco residuo.
   - Modulazione multimodale **FiLM** (*Feature-wise Linear Modulation*) generata da una rete ausiliaria MLP operante sui 9 scalari topologici.
   - **Logit Masking Neuro-Simbolico**: vincoli fisici differenziabili che azzerano a priori le classi impossibili (es. un camion non può nascondersi dietro una berlina bassa con altezza $< 1.50\,\text{m}$; i veicoli non possono trovarsi sul marciapiede).
   - Funzione di costo **Asymmetric Loss (ASL)** con disaccoppiamento dei gradienti ($\gamma_{\text{neg}}=4.0, \gamma_{\text{pos}}=0.5, \text{clip}=0.05$) per contrastare il marcato sbilanciamento di classe.
4. **Valutazione Comparativa Duale**:
   - **GT Reale nuScenes (3D Box)**: Valutazione su ostacoli fisici annotati effettivamente nascosti nell'ombra (capacità di intercettazione reale).
   - **GT Sintetica Ibrida Spazio-Semantica**: Valutazione della capacità di anticipare la transitabilità e il rischio potenziale (*affordance Beyond-LoS*) per pianificazione difensiva.

---

## Struttura del Repository

```text
Occlusion-Mapper/
├── architettura_neurale/       # AttentionPerZoneModel, ResidualSEBlock, FiLM, AsymmetricLoss
├── pesi_modelli/               # Checkpoint addestrati (.pth) per le varie modalità
├── addestramento/              # Script di training GPU parallelizzati (supporto automatico mini/trainval)
├── inferenza_agenti/           # Agenti di inferenza runtime (Neurale e Bayesiano)
├── ground_truth/               # Estrattori GT Reale 3D e GT Sintetica (Geometrica, Semantica, Ibrida)
├── raycaster/                  # Raycasting LiDAR 3D polar-grid e space carving
├── dataset_adapter/            # Adapter nuScenes con precaricamento mappe in RAM e auto-detect versione
├── valutazione/                # Benchmark ufficiale, calibrazione soglie e calcolo metriche (split val e all)
├── visualizzatori/             # Suite di visualizzatori interattivi grafici runtime
├── documentazione/             # Report di valutazione ufficiali, grafici e diagrammi della tesi
├── diagnostica/                # Controlli indipendenti dalla soglia (AUC / Average Precision)
├── estrazione_zone_occluse.py  # Script batch per l'estrazione ultra-rapida delle zone d'ombra
├── verify_complete_pipeline.py # Script di diagnostica e collaudo (7/7 test automatici)
└── requirements.txt            # Dipendenze Python
```

---

## Prestazioni di Calcolo e Ottimizzazione

Grazie alla completa parallelizzazione multiprocessing (compatibile con l'architettura `spawn` di Windows e `fork` di Linux) e al caching in RAM delle mappe vettoriali HD-Map:

| Fase della Pipeline | `v1.0-mini` (404 frame) | `v1.0-trainval` (34.149 frame) |
| :--- | :---: | :---: |
| **Estrazione Raycasting & Zone** | 57 secondi | ~50 minuti |
| **Generazione Dataset & Caching Patch** (per modalità, 28.130 frame di train) | ~60 secondi | ~1 h 10 – 1 h 20 |
| **Addestramento GPU CUDA** (20 epoche mini, 5 trainval) | 150 secondi | 10 – 58 minuti |
| **Calibrazione Soglie** (3 modelli) | — | ~7 minuti |
| **Valutazione Ufficiale** (split val, 6.019 frame, 3 modelli) | ~90 secondi | ~1 h 35 |

> I tempi su `v1.0-trainval` sono stati misurati con 8 worker su CPU a 16 core, GPU Quadro P4000 e dataset su SSD esterno collegato in USB 2.0. La generazione della cache è limitata dalla CPU; la lettura della cache durante l'addestramento dalla velocità del disco.
>
> Su `v1.0-trainval` la cache di una modalità occupa **~42 GB** (~506.000 zone, ~90 KB per zona in float16): con `--cache_dir` e `--pesi_dir` cache e checkpoint possono essere salvati su un disco esterno.

---

## Installazione ed Esecuzione

### 1. Configurazione Ambiente
Il progetto viene eseguito in un ambiente **conda** dedicato:
```bash
# Creazione e attivazione dell'ambiente
conda create -n occlusion-mapper python=3.11
conda activate occlusion-mapper

# Installazione dipendenze
pip install -r requirements.txt
```

> Se nel `PATH` sono presenti altri interpreti Python (ad esempio di altri ambienti conda), `python` e `conda run` possono risolversi su quello sbagliato. In tal caso usare esplicitamente l'interprete dell'ambiente: `"$CONDA_PREFIX/bin/python" <script>.py`.

### 2. Collaudo del Sistema (7 Test Diagnostici)
Per verificare che l'accelerazione CUDA, i moduli di raycasting, la ground truth, i modelli neurali e i visualizzatori siano operativi:
```bash
python verify_complete_pipeline.py
```

### 3. Estrazione delle Zone d'Ombra LiDAR
Estrae le geometrie dei coni d'ombra in parallelo per tutte le scene:
```bash
python estrazione_zone_occluse.py --workers 8
```

### 4. Addestramento dei Modelli
L'addestramento rileva automaticamente il dataset presente (`v1.0-mini` o `v1.0-trainval`), configurando le epoche ottimali (20 epoche su mini, 5 su trainval):
```bash
# Addestramento modello Neuro-Simbolico Ibrido (raccomandato per la tesi)
python addestramento/train_per_zone_attention.py --mode hybrid --split train

# Addestramento con supervisione Solo Positivi (Positives-Only)
python addestramento/train_per_zone_attention.py --mode positives_only --split train

# Addestramento con supervisione Reale nuScenes completa
python addestramento/train_per_zone_attention.py --mode real --split train

# Addestramento sequenziale di tutte le ablazioni sperimentali
python addestramento/train_per_zone_attention.py --mode all --split train

# Cache e checkpoint su un disco esterno (consigliato su trainval: ~42 GB di cache per modalità)
python addestramento/train_per_zone_attention.py --mode hybrid --split train \
    --cache_dir "/percorso/disco/cache" --pesi_dir "/percorso/disco/pesi"
```

> La cache viene considerata completa se la sua cartella contiene almeno un blocco: se una generazione viene interrotta, eliminare la cartella `*_blocchi` della modalità (oppure usare `--force`) prima di rilanciare.

### 5. Calibrazione delle Soglie
Le soglie di decisione fisse (0.25 – 0.28) erano state tarate sui modelli `v1.0-mini`. I modelli addestrati su `v1.0-trainval` producono probabilità più alte: con quelle soglie quasi ogni zona compatibile risulta positiva e le predizioni coincidono con la maschera di compatibilità. Le soglie vanno quindi calibrate per modello, sulla cache di **train**, così che lo split di validazione resti inedito:
```bash
python valutazione/calibra_soglie.py --cache_dir "/percorso/disco/cache" --pesi_dir pesi_modelli
# -> valutazione/soglie_calibrate.json
```

### 6. Valutazione Ufficiale e Benchmark
Calcola le metriche complete (TP, FP, FN, Precision, Recall, F1-Score) sui dati di validazione inediti:
```bash
# Valutazione di tutti i modelli disponibili con le soglie calibrate (report Markdown + cache per i visualizzatori)
python valutazione/evaluate_final_official.py --mode all --split val --soglie valutazione/soglie_calibrate.json

# Valutazione di un singolo modello
python valutazione/evaluate_final_official.py --mode hybrid --split val --soglie valutazione/soglie_calibrate.json
```
I modelli senza checkpoint (e la baseline Bayesiana, se mancano le probabilità precalcolate in `extracted_occlusions_probabilities/`) vengono saltati automaticamente.

---

## Visualizzatori Interattivi Runtime

Tutti i visualizzatori si trovano in `visualizzatori/` e supportano la navigazione interattiva frame per frame. I modelli di cui non esiste il checkpoint vengono indicati come **MANCANTI**, senza ripiegare sui pesi di un altro modello:

```bash
# 1. Dashboard Valutazione Prestazioni (Figure 5, 5b, 5c - Recap TRAIN vs VAL, KPI Cards, Matrice di Confusione)
python visualizzatori/vis_valutazione_prestazioni.py

# 2. Inferenza Neurale Live su GPU (Confronto modelli dinamico, classificazione con icone orientate)
python visualizzatori/vis_inferenza_neurale.py

# 3. Input Multimodali Rete Neurale (Ispezione 11 Canali BEV + Pesi Attention SE + Sottorete FiLM)
python visualizzatori/vis_input_rete_neurale.py

# 4. Raycasting Occlusioni e Geometrie 3D (Simulazione LiDAR, coni d'ombra e clustering manmade)
python visualizzatori/vis_raycasting_occlusioni.py

# 5. Confronto Ground Truth (Reale 3D vs. Sintetica Ibrida con Bounding Box 3D)
python visualizzatori/vis_ground_truth_occlusioni.py

# 6. Agente Probabilistico Bayesiano Condizionato (Baseline analitica e memoria temporale)
python visualizzatori/vis_probabilita_bayes.py
```

---

## Risultati Sperimentali e Benchmark della Tesi

La validazione ufficiale è condotta su **`v1.0-trainval`**: addestramento sulle 700 scene di train (28.130 frame, ~506.000 zone d'ombra) e valutazione sulle **150 scene di validazione inedite** (6.019 frame), con raggio di analisi di 25 metri e soglie calibrate per modello sulla cache di train.

### 1. Tabella Comparativa Finale a 25 Metri (Split VAL)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | **F1-Score (%)** |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Hybrid)** | **GT Reale nuScenes** (Ostacoli fisici) | 1.6% | 49.1% | **3.1%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Geometrica** (Fitting 3D) | 63.1% | 54.3% | **58.3%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Semantica** (Regole Naïve) | 39.5% | 55.0% | **46.0%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Ibrida** (Spazio-Semantica) | **84.5%** | **83.5%** | **84.0%** |

### 2. Dettaglio per Categoria Semantica del Modello Ibrido (su GT Ibrida a 25m)

| Categoria Semantica | Precision (%) | Recall (%) | **F1-Score (%)** |
| :--- | :---: | :---: | :---: |
| **VRU (Pedoni / Ciclisti)** | **92.0%** | 84.2% | **88.0%** |
| **Auto** | 81.4% | **91.7%** | **86.2%** |
| **Barriere** | 69.2% | 73.6% | **71.3%** |
| **Camion / Bus** | 39.9% | 66.7% | **49.9%** |
| **MEDIA GLOBALE** | **84.5%** | **83.5%** | **84.0%** |

### 3. Confronto tra Modelli (Ablation Study a 25m)

| Modello | F1 su GT Ibrida | F1 su GT Reale nuScenes | Precision / Recall su GT Reale |
| :--- | :---: | :---: | :---: |
| **★ Neuro-Simbolico (Hybrid)** | **84.0%** | 3.1% | 1.6% / 49.1% |
| **Supervisione Reale nuScenes (`REAL_GT`)** | 15.3% | 10.0% | 6.1% / 27.1% |
| **Positives-Only (`POS_ONLY`)** | 18.9% | **10.5%** | 6.3% / 32.6% |

I modelli *Geometrico*, *Semantico* e la *Baseline Bayesiana* non sono stati rivalutati su `v1.0-trainval`.

Soglie calibrate (Auto, Camion/Bus, VRU, Barriera): Hybrid `[0.63, 0.64, 0.69, 0.72]`, Real `[0.60, 0.58, 0.58, 0.62]`, Positives-Only `[0.78, 0.66, 0.74, 0.82]` (`valutazione/soglie_calibrate.json`). Sulla cache di train l'AUC per classe è 0.90 – 0.95 per Hybrid, 0.83 – 0.92 per Real e 0.50 – 0.82 per Positives-Only rispetto alla rispettiva GT di addestramento (`diagnostica/auc_ap_risultati.txt`).

> **Conclusioni Metodologiche per la Tesi**:
> - Il modello **Hybrid** raggiunge un **F1 dell'84.0%** sulla GT Ibrida con Precision e Recall bilanciate; il miglioramento più marcato rispetto a `v1.0-mini` riguarda le classi rare (Barriere da 31.7% a 71.3%, Camion/Bus da 14.0% a 49.9%).
> - Ogni modello ottiene i risultati migliori sulla GT su cui è stato addestrato e con cui sono state calibrate le sue soglie: Hybrid sulla GT Ibrida, Real e Positives-Only sulla GT Reale. Il confronto fra modelli va quindi letto per target di valutazione.
> - La **bassa precisione formale sulla GT Reale** (F1 3.1% – 10.5%) riflette l'*observation bias* di nuScenes: un'area occlusa sul marciapiede è pericolosa a priori per un veicolo autonomo difensivo, anche se nessun pedone è transitato esattamente in quel decimo di secondo del log.

### 4. Risultati Precedenti su `v1.0-mini` (Riferimento)

Valutazione su `mini_val` (81 fotogrammi) con soglie fisse 0.25 – 0.28:

| Modello | Precision su GT Ibrida | Recall su GT Ibrida | **F1-Score Globale** | **F1-Score VRU (Pedoni)** |
| :--- | :---: | :---: | :---: | :---: |
| **★ Neuro-Simbolico (Hybrid)** | **50.8%** | **97.0%** | **66.6%** | **83.6%** |
| **Positives-Only (`POS_ONLY`)** | 44.2% | 91.9% | 59.7% | 81.5% |
| **Fitting Geometrico Puro** | 46.6% | 77.0% | 58.1% | 62.6% |
| **Semantico Naïve Puro** | 30.7% | 82.0% | 44.7% | 43.4% |
| **Supervisione Reale nuScenes (`REAL_GT`)** | 53.0% | 44.4% | 48.3% | 56.4% |
| **Baseline Bayesiana Analitica (`BAYES`)** | 62.0% | 45.0% | 52.1% | 56.6% |

> Il confronto fra `v1.0-mini` e `v1.0-trainval` non è omogeneo: oltre al dataset (circa 75 volte più grande in validazione) sono cambiate le soglie, fisse su mini e calibrate per modello su trainval. Non è quindi possibile attribuire il miglioramento al solo aumento dei dati. I valori di Recall molto alti su mini erano in parte dovuti a soglie basse che classificavano positive quasi tutte le zone compatibili.

---

## Requisiti di Sistema
* Python 3.11 (ambiente conda `occlusion-mapper`)
* PyTorch $\ge$ 2.0 (supporto CUDA raccomandato)
* nuScenes devkit (`nuscenes-devkit`)
* Shapely, OpenCV, NumPy, Matplotlib, SciPy

---

## Citazione e Crediti
Progetto di Tesi di Laurea in Informatica sviluppato presso il **Dipartimento di Informatica**, **Università degli Studi di Salerno**.  
Candidato: *Samuele Colucci*  
Dataset: *nuScenes by Motional* (Holger Caesar et al.).
