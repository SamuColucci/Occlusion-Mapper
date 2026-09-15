# Occlusion-Mapper: Stima Neuro-Simbolica e Apprendimento Neurale per la Percezione delle Zone Occluse nella Guida Autonoma

Repository ufficiale del progetto di Tesi di Laurea dedicato alla stima, modellazione probabilistica e inferenza neurale degli ostacoli nascosti nelle zone cieche sensoriali (occlusioni LiDAR) per veicoli a guida autonoma su dataset **nuScenes**.

---

## Panoramica della Pipeline

I sensori fisici di bordo (LiDAR e telecamere) sono limitati alla linea di vista (*Line-of-Sight*, LoS). **Occlusion-Mapper** stima la presenza e la classe di ostacoli nascosti all'interno delle regioni d'ombra generate da altri veicoli o elementi urbani tramite una pipeline modulare in quattro fasi:

```
[ Nuvola di Punti LiDAR 3D + HD-Map ]
                 │
                 ▼
 1. Space Carving & Raycasting 3D (Bird's-Eye-View a 11 canali)
                 │
                 ▼
 2. Decomposizione Poligonale & Filtri di Ammissibilità Fisica
                 │
                 ▼
 3. Modello Neurale Attention-per-Zone (AttentionPerZoneModel)
    ├── Channel Attention (Squeeze-and-Excitation)
    ├── Condizionamento Semantico FiLM (HD-Map vettoriale)
    └── Maschere Logiche Determistiche (Altezza/Dimensioni Occludente)
                 │
                 ▼
 4. Valutazione Empirica Duale (GT Reale 3D vs. Affordance Geometrica)
```

1. **Raycasting LiDAR e Decomposizione Geometrica**: Simulazione del fascio LiDAR 3D in coordinate polari per identificare i coni d'ombra generati da veicoli e ostacoli. Le aree occluse vengono proiettate in coordinate *Bird's-Eye-View* (BEV) e ritagliate a livello di singola zona (*patch-per-zone*).
2. **Descrittori Geometrici e Semantici (HD-Map)**: Calcolo dei descrittori scalari per ciascuna zona d'ombra (area, distanza dal veicolo ego, ingombro OBB, frazione di asfalto carrabile, marciapiede, attraversamento pedonale, banchina e terreno).
3. **Inferenza Neurale Neuro-Simbolica (`AttentionPerZoneModel`)**:
   - Backbone convoluzionale residuo su **11 canali BEV** (layer semantici HD-map, densità e altezza LiDAR, mappe d'ombra dinamiche e statiche).
   - Meccanismo di **Channel Attention** (*Squeeze-and-Excitation*) per pesare dinamicamente la rilevanza dei canali.
   - Modulazione multimodale **FiLM** (*Feature-wise Linear Modulation*) condizionata dai vettori scalari della mappa HD.
   - **Maschere Logico-Simboliche di Compatibilità**: vincoli fisici differenziabili che azzerano a priori le classi impossibili (es. un camion non può nascondersi dietro un veicolo con altezza $< 1.50\,\text{m}$, un'auto non può trovarsi sul marciapiede).
   - Funzione di costo **Asymmetric Loss (ASL)** con disaccoppiamento dei gradienti per contrastare l'estremo sbilanciamento delle classi.
4. **Valutazione Comparativa Multi-Benchmark**:
   - **1. GT Reale nuScenes (3D Box)**: Rilevamento stretto di ostacoli fisici effettivamente presenti e nascosti nell'ombra (escludendo il veicolo occludente stesso).
   - **2. GT Ibrida / Geometrica (Affordance)**: Anticipazione preventiva dei varchi carrabili e delle traiettorie pedonali plausibili per pianificazione difensiva.

---

## Struttura del Repository

```text
Occlusion-Mapper/
├── architettura_neurale/       # AttentionPerZoneModel, FiLM, Channel Attention, ASL Loss
├── pesi_modelli/               # Checkpoint addestrati (.pth) per le varie modalità
├── addestramento/              # Script di training GPU (split official train/val)
├── inferenza_agenti/           # Agenti di inferenza runtime (Neurale e Bayesiano)
├── ground_truth/               # Estrattori GT Reale 3D e GT Sintetica Neurosimbolica
├── raycaster/                  # Raycasting LiDAR 3D polar-grid e space carving
├── dataset_adapter/            # Adapter nuScenes e parser mappe HD vettoriali
├── valutazione/                # Benchmark ufficiale e calcolo metriche (404 frame)
├── visualizzatori/             # Suite di visualizzatori interattivi runtime
├── documentazione/             # Report di valutazione, immagini per tesi, tabelle
├── estrazione_zone_occluse.py  # Script batch per l'estrazione delle zone d'ombra
├── verify_complete_pipeline.py # Script di diagnostica e test di integrità
└── requirements.txt            # Dipendenze Python
```

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

### 2. Estrazione delle Zone d'Ombra LiDAR
Estrae le geometrie dei coni d'ombra per tutte le scene del dataset:
```bash
python estrazione_zone_occluse.py
```

### 3. Addestramento dei Modelli
L'addestramento supporta gli split ufficiali nuScenes (`train`, `val`, `all`) e diverse modalità di supervisione:
```bash
# Addestramento modello con supervisione Solo Positivi (Positives-Only)
python addestramento/train_per_zone_attention.py --gt-mode positives_only --split train

# Addestramento modello con supervisione Sintetica Ibrida
python addestramento/train_per_zone_attention.py --gt-mode hybrid --split train

# Addestramento con supervisione Reale nuScenes completa
python addestramento/train_per_zone_attention.py --gt-mode real --split train
```

### 4. Valutazione Ufficiale e Benchmark
Calcola le metriche complete (TP, FP, FN, TN, Precision, Recall, F1, Specificity) sui 404 frame (split `all`, `train` o `val`) con soglie decisionali calibrate:
```bash
# Valutazione completa su tutti i 404 frame
python valutazione/evaluate_final_official.py --split all

# Valutazione sullo split di validazione ufficiale nuScenes (scene-0103, scene-0916)
python valutazione/evaluate_final_official.py --split val
```

---

## Visualizzatori Interattivi Runtime

Tutti i visualizzatori grafici si trovano nella cartella `visualizzatori/`. Possono essere eseguiti singolarmente o sincronizzati in parallelo con il viewer ufficiale nuScenes.

### Avvio Sincronizzato con nuScenes Explorer
```bash
# Sintassi: python visualizzatori/avvia_entrambi.py [frame] [--FLAG]
python visualizzatori/avvia_entrambi.py 14 --eval         # Dashboard Valutazione Prestazioni
python visualizzatori/avvia_entrambi.py 14 --neural       # Inferenza Neurale Live
python visualizzatori/avvia_entrambi.py 14 --inputs       # Ispezione 11 Canali BEV + FiLM
python visualizzatori/avvia_entrambi.py 14 --bayes        # Probabilità Bayesiana
python visualizzatori/avvia_entrambi.py 14 --gt           # Ispezione Ground Truth
```

### Avvio Singolo dei Moduli
```bash
# 1. Dashboard Valutazione Prestazioni (Figure 5c Tesi - Recap TRAIN vs VAL, KPI Cards, Metriche)
python visualizzatori/vis_valutazione_prestazioni.py [frame]

# 2. Inferenza Neurale Live su GPU (Confronto modelli: NEURO_SIMB, POS_ONLY, REAL_GT)
# Comandi: M (cambia modello), G (cambia GT di confronto), Freccia Dx/Sx (frame)
python visualizzatori/vis_inferenza_neurale.py [frame]

# 3. Input Multimodali Rete Neurale (11 canali BEV + Scalari HD-Map + Sottorete FiLM)
# Comandi: Spazio/T (toggle ausiliaria), Z/X (scorri zone), Click BEV (ispeziona)
python visualizzatori/vis_input_rete_neurale.py [frame]

# 4. Raycasting Occlusioni e Geometrie 3D
python visualizzatori/vis_raycasting_occlusioni.py [frame]

# 5. Confronto Ground Truth (Reale 3D vs. Sintetica Ibrida)
python visualizzatori/vis_ground_truth_occlusioni.py [frame]

# 6. Agente Probabilistico Bayesiano Condizionato
python visualizzatori/vis_probabilita_bayes.py [frame]
```

---

## Risultati Sperimentali

La validazione ufficiale è condotta sui **404 fotogrammi** del dataset nuScenes (oltre 18.000 zone d'ombra valutate), ripartiti secondo lo split ufficiale in **322 frame di Train** (8 scene) e **82 frame di Validation** (2 scene: `scene-0103` e `scene-0916`).

### 1. Benchmark su Ground Truth Reale nuScenes (Rilevamento Ostacoli Fisici 3D Nascosti)
Valuta la capacità di individuare gli ostacoli fisici reali celati nell'ombra (escludendo rigorosamente il veicolo occludente visibile):

| Modello | Split | Recall Globale | Precision | F1-Score | Recall Auto | Recall Camion | Recall VRU | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`POS_ONLY`** | **VAL** | **83.3%** | **8.4%** | **15.2%** | 80.0% | 50.0% | **83.3%** | Neuro-simbolico su campioni puliti |
| **`POS_ONLY`** | **TRAIN** | **78.4%** | **7.6%** | **13.9%** | 77.0% | 50.0% | **81.0%** | Zero overfitting |
| **`NEURO_SIMB`** | **VAL** | 83.3% | 2.1% | 4.1% | 85.0% | 50.0% | 66.7% | Bias allarmista da etichette sintetiche |
| **`REAL_GT`** | **VAL** | 75.0% | 8.8% | 15.8% | 75.0% | 0.0% | 66.7% | Sopprime classi a bassa prevalenza |

> *Nota metodologica*: Nella GT Reale 3D la prevalenza degli ostacoli nascosti è solo del **1.5%** (489 ostacoli su oltre 32.000 valutazioni classe-zona). In questo scenario di sicurezza preventiva, il modello `POS_ONLY` intercetta **l'83.3% dei pedoni/ciclisti (VRU)** e l'**80% delle auto** con tempi di inferenza di soli **1.2 ms per zona**.

---

### 2. Benchmark su Ground Truth Geometrica / Affordance (Anticipazione del Rischio Preventivo)
Valuta la capacità del modello di anticipare la transitabilità e il rischio spaziale secondo la mappa HD:

| Modello | Split | Recall | Precision | F1-Score | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`POS_ONLY`** | **VAL** | **77.8%** | **49.8%** | **60.9%** | **87.2%** |
| **`POS_ONLY`** | **TRAIN** | **89.5%** | **53.9%** | **67.3%** | **88.6%** |
| **`NEURO_SIMB`** | **VAL** | 99.8% | 34.2% | 50.9% | 41.5% |

---

## Requisiti di Sistema
* Python $\ge$ 3.9
* PyTorch $\ge$ 2.0 (supporto CUDA consigliato)
* nuScenes devkit (`nuscenes-devkit`)
* Shapely, OpenCV, NumPy, Matplotlib, SciPy

---

## Citazione e Crediti
Progetto sviluppato presso il **Dipartimento di Informatica**, **Università degli Studi di Salerno**.  
Candidato: *Samuele Colucci*  
Dataset: *nuScenes by Motional* (Holger Caesar et al.).
