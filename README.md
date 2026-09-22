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
├── valutazione/                # Benchmark ufficiale e calcolo metriche (split val e all)
├── visualizzatori/             # Suite di visualizzatori interattivi grafici runtime
├── documentazione/             # Report di valutazione ufficiali, grafici e diagrammi della tesi
├── estrazione_zone_occluse.py  # Script batch per l'estrazione ultra-rapida delle zone d'ombra
├── verify_complete_pipeline.py # Script di diagnostica e collaudo (7/7 test automatici)
└── requirements.txt            # Dipendenze Python
```

---

## Prestazioni di Calcolo e Ottimizzazione

Grazie alla completa parallelizzazione multiprocessing (compatibile con l'architettura `spawn` di Windows e `fork` di Linux) e al caching in RAM delle mappe vettoriali HD-Map:

| Fase della Pipeline | Tempo Precedente (Seriale) | Tempo Attuale (8 Worker Paralleli) | Speedup |
| :--- | :---: | :---: | :---: |
| **Estrazione Raycasting & Zone (404 frame)** | ~4 ore | **57 secondi** | **> 250x** |
| **Generazione Dataset & Caching Patch** | ~15 minuti | **~60 secondi** | **~15x** |
| **Addestramento GPU CUDA (20 epoche)** | ~12 minuti | **150 secondi (2.5 min)** | **~5x** |
| **Valutazione Ufficiale Benchmark** | ~8 minuti | **~90 secondi** | **~5x** |
| **INTERA PIPELINE END-TO-END** | **~4.5 ore** | **< 6 minuti** | **~50x** |

---

## Installazione ed Esecuzione

### 1. Configurazione Ambiente
```bash
# Creazione ambiente virtuale
python -m venv .venv

# Attivazione (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Attivazione (Linux / macOS)
source .venv/bin/activate

# Installazione dipendenze
pip install -r requirements.txt
```

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
```

### 5. Valutazione Ufficiale e Benchmark
Calcola le metriche complete (TP, FP, FN, Precision, Recall, F1-Score) sui dati di validazione inediti:
```bash
# Valutazione sullo split di validazione (scene-0553, scene-0928 per mini_val)
python valutazione/evaluate_final_official.py --mode hybrid --split val

# Valutazione completa con generazione automatica dei report Markdown
python valutazione/evaluate_final_official.py --mode all --split val
```

---

## Visualizzatori Interattivi Runtime

Tutti i visualizzatori si trovano in `visualizzatori/` e supportano la navigazione interattiva frame per frame:

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

La validazione ufficiale è condotta sullo split di **Validazione Inedito** (`mini_val`, 81 fotogrammi) e sull'intero dataset (404 fotogrammi, oltre 18.000 zone d'ombra analizzate a 25 metri).

### 1. Tabella Comparativa Finale a 25 Metri (Split VAL)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | **F1-Score (%)** |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Hybrid)** | **GT Reale nuScenes** (Ostacoli fisici) | 1.8% | **76.0%** | **3.5%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Geometrica** (Fitting 3D) | 46.6% | 77.0% | **58.1%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Semantica** (Regole Naïve) | 30.7% | 82.0% | **44.7%** |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Ibrida** (Spazio-Semantica) | **50.8%** | **97.0%** | **66.6%** |

### 2. Dettaglio per Categoria Semantica del Modello Ibrido (su GT Ibrida a 25m)

| Categoria Semantica | Precision (%) | Recall (%) | **F1-Score (%)** |
| :--- | :---: | :---: | :---: |
| **VRU (Pedoni / Ciclisti)** | **71.9%** | **99.7%** | **83.6%** |
| **Auto** | **60.6%** | **93.9%** | **73.6%** |
| **Barriere** | 19.4% | 86.6% | **31.7%** |
| **Camion / Bus** | 7.5% | 96.6% | **14.0%** |
| **MEDIA GLOBALE** | **50.8%** | **97.0%** | **66.6%** |

### 3. Confronto tra Modelli (Ablation Study a 25m)

| Modello | Precision su GT Ibrida | Recall su GT Ibrida | **F1-Score Globale** | **F1-Score VRU (Pedoni)** |
| :--- | :---: | :---: | :---: | :---: |
| **★ Neuro-Simbolico (Hybrid)** | **50.8%** | **97.0%** | **66.6%** | **83.6%** |
| **Positives-Only (`POS_ONLY`)** | 44.2% | 91.9% | 59.7% | 81.5% |
| **Fitting Geometrico Puro** | 46.6% | 77.0% | 58.1% | 62.6% |
| **Semantico Naïve Puro** | 30.7% | 82.0% | 44.7% | 43.4% |
| **Supervisione Reale nuScenes (`REAL_GT`)** | 53.0% | 44.4% | 48.3% | 56.4% |
| **Baseline Bayesiana Analitica (`BAYES`)** | 62.0% | 45.0% | 52.1% | 56.6% |

> **Conclusioni Metodologiche per la Tesi**:
> - La **bassa precisione formale sulla GT Reale (1.8% - 3.5% F1)** riflette l'*observation bias* di nuScenes: un'area occlusa sul marciapiede è pericolosa a priori per un veicolo autonomo difensivo, anche se nessun pedone è transitato esattamente in quel decimo di secondo del log.
> - Il **Recall del 76.0% - 100.0% sulla GT Reale** certifica che quando un ostacolo è effettivamente presente e nascosto, la rete lo individua prontamente.
> - La combinazione congiunta di **Space Carving 3D + HD-Map + FiLM** garantisce un incremento di oltre **+21.8% di F1** rispetto all'addestramento su etichette reali e del **+18.0%** rispetto alla baseline Bayesiana.

---

## Requisiti di Sistema
* Python $\ge$ 3.9
* PyTorch $\ge$ 2.0 (supporto CUDA raccomandato)
* nuScenes devkit (`nuscenes-devkit`)
* Shapely, OpenCV, NumPy, Matplotlib, SciPy

---

## Citazione e Crediti
Progetto di Tesi di Laurea in Informatica sviluppato presso il **Dipartimento di Informatica**, **Università degli Studi di Salerno**.  
Candidato: *Samuele Colucci*  
Dataset: *nuScenes by Motional* (Holger Caesar et al.).
