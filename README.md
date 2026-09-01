# Occlusion-Mapper: Stima Probabilistica e Apprendimento Neurale per la Percezione delle Zone Occluse nella Guida Autonoma

Repository ufficiale del progetto di Tesi di Laurea dedicato alla stima, modellazione probabilistica e inferenza neurale degli ostacoli nascosti nelle zone cieche sensoriali (occlusioni LiDAR) per veicoli a guida autonoma su dataset nuScenes.

---

## Panoramica della Pipeline

Il sistema implementa una pipeline modulare in quattro fasi:

1. **Raycasting LiDAR e Decomposizione Geometrica**: Simulazione del fascio LiDAR 3D in coordinate polar-grid per identificare i coni d'ombra generati da ostacoli ed elementi urbani. Le zone d'ombra vengono proiettate in coordinate Bird's Eye View (BEV) e scomposte in sotto-zone disgiunte in base all'ingombro geometrico (Oriented Bounding Box, OBB).
2. **Stima Probabilistica Condizionata (Agente Bayesiano)**: Calcolo analitico della probabilita a priori condizionata $P(C_k \mid \text{Mappa HD}, W_{\text{sub}}, \text{Area})$ combinando i vincoli di transitabilita del codice della strada con il tracciamento della memoria storica tra fotogrammi consecutivi.
3. **Apprendimento Neurale Multimodale (AttentionPerZoneModel)**: Modello neurale basato su un backbone convoluzionale residuo con meccanismo di Channel Attention (Squeeze-and-Excitation) per pesare dinamicamente gli 11 canali BEV, modulazione semantica FiLM (Feature-wise Linear Modulation) guidata dai vettori scalari della mappa HD, Layer Normalization e funzione di costo Asymmetric Loss (ASL) con disaccoppiamento dei gradienti.
4. **Validazione a Doppia Ground Truth**: Valutazione comparativa condotta sia sulla Ground Truth Reale nuScenes (annotazioni 3D visibili) sia sulla Ground Truth Sintetica Neurosimbolica (che include i varchi plausibili e le affordance stradali).

---

## Struttura del Progetto

```text
Occlusion-Mapper/
|-- architettura_neurale/       # Definizione AttentionPerZoneModel e loss functions (ASL)
|-- pesi_modelli/               # Pesi addestrati del modello finale (.pth)
|-- addestramento/              # Script di training su GPU
|-- inferenza_agenti/           # Agenti di inferenza a bordo (Neurale e Bayesiano)
|-- ground_truth/               # Estrattori Ground Truth Reale e Sintetica Neurosimbolica
|-- raycaster/                  # Modulo di Raycasting LiDAR 3D polar-grid
|-- dataset_adapter/            # Adapter per il dataset nuScenes e mappe HD
|-- valutazione/                # Script di benchmark e calcolo metriche ufficiali
|-- visualizzatori/             # Visualizzatori interattivi grafici runtime
|-- documentazione/             # Note teoriche, tabelle e guide di esecuzione
|-- estrazione_zone_occluse.py  # Script batch per l'estrazione dei coni d'ombra
|-- verify_complete_pipeline.py # Script di diagnostica e verifica completa del sistema
`-- requirements.txt            # Dipendenze Python
```

---

## Istruzioni di Installazione ed Esecuzione

### 1. Configurazione Ambiente
```bash
python -m venv .venv
source .venv/bin/activate  # Su Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Estrazione delle Zone d'Ombra LiDAR
```bash
python estrazione_zone_occluse.py
```

### 3. Addestramento del Modello
L'addestramento ufficiale della rete neurale viene eseguito sulla **Ground Truth Sintetica Neurosimbolica**:
```bash
python addestramento/train_per_zone_attention.py
```

Per scopi di ricerca e confronto empirico (Ablation Study), e disponibile anche lo script di addestramento su sola **Ground Truth Reale nuScenes**:
```bash
python addestramento/train_per_zone_real_gt.py
```

### 4. Valutazione e Benchmark su Dataset Completo
* **Valutazione del Modello Ufficiale** (su entrambe le Ground Truth a 20m e 25m):
  ```bash
  python valutazione/evaluate_final_official.py
  ```
* **Confronto Comparativo di Ablazione** (Supervisione Reale vs Neurosimbolica):
  ```bash
  python valutazione/evaluate_training_supervision_ablation.py
  ```

### 5. Visualizzatori Runtime Interattivi
```bash
# Visualizzatore del modello neurale finale con switch Ground Truth in tempo reale (tasto M/G)
python visualizzatori/verify_runtime_attention.py

# Visualizzatore della probabilità condizionata Bayesiana
python visualizzatori/verify_runtime_bayes.py

# Visualizzatore comparativo Ground Truth Reale vs Sintetica
python visualizzatori/verify_runtime_ground_truth.py

# Visualizzatore comparativo Modello Base vs Modello Avanzato
python visualizzatori/verify_runtime_model_comparison.py
```

---

## Risultati Sperimentali

La validazione e stata condotta su tutti i 404 fotogrammi del dataset nuScenes (oltre 18.000 zone d'ombra analizzate) utilizzando la soglia decisionale operativa standard ($0.30$).

---

### 1. Modello Ufficiale: Addestramento con Ground Truth Sintetica Neurosimbolica

Il modello ufficiale `AttentionPerZoneModel` viene addestrato su GT Sintetica Neurosimbolica (ostacoli fisici reali + affordance del codice della strada). Di seguito i risultati ottenuti nelle due modalita di test:

#### A. Valutazione su Ground Truth Sintetica Neurosimbolica (Anticipazione del Rischio)
Valuta la capacita del modello di anticipare sia gli ostacoli reali sia i varchi plausibili definiti dalla mappa HD:

| Raggio | Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **20m** | Auto | 2.354 | 585 | 14 | 80.1% | 99.4% | **88.7%** |
| | Camion / Bus | 1.159 | 402 | 0 | 74.2% | 100.0% | **85.2%** |
| | VRU (Pedoni / Ciclisti) | 1.458 | 711 | 0 | 67.2% | 100.0% | **80.4%** |
| | Barriere | 1.041 | 580 | 0 | 64.2% | 100.0% | **78.2%** |
| | **Media Globale (20m)** | **6.012** | **2.278** | **14** | **72.5%** | **99.8%** | **84.0%** |
| **25m** | Auto | 3.495 | 877 | 23 | 79.9% | 99.3% | **88.6%** |
| | Camion / Bus | 1.751 | 617 | 0 | 73.9% | 100.0% | **85.0%** |
| | VRU (Pedoni / Ciclisti) | 2.237 | 1.053 | 0 | 68.0% | 100.0% | **80.9%** |
| | Barriere | 2.007 | 899 | 0 | 69.1% | 100.0% | **81.7%** |
| | **Media Globale (25m)** | **9.490** | **3.446** | **23** | **73.4%** | **99.8%** | **84.5%** |

#### B. Valutazione su Ground Truth Reale nuScenes (Rilevamento Ostacoli Fisici)
Valuta l'accuratezza stretta rispetto ai soli ostacoli fisici reali visibili e annotati nei box 3D:

| Raggio | Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **20m** | Auto | 1.362 | 1.577 | 0 | 46.3% | 100.0% | **63.3%** |
| | VRU (Pedoni / Ciclisti) | 687 | 1.482 | 0 | 31.7% | 100.0% | **48.1%** |
| | Barriere | 509 | 1.112 | 0 | 31.4% | 100.0% | **47.8%** |
| | Camion / Bus | 427 | 1.134 | 0 | 27.4% | 100.0% | **43.0%** |
| | **Media Globale (20m)** | **2.985** | **5.305** | **0** | **36.0%** | **100.0%** | **52.9%** |
| **25m** | Auto | 1.894 | 2.478 | 0 | 43.3% | 100.0% | **60.5%** |
| | VRU (Pedoni / Ciclisti) | 957 | 2.333 | 0 | 29.1% | 100.0% | **45.1%** |
| | Barriere | 665 | 2.241 | 0 | 22.9% | 100.0% | **37.2%** |
| | Camion / Bus | 588 | 1.780 | 0 | 24.8% | 100.0% | **39.8%** |
| | **Media Globale (25m)** | **4.104** | **8.832** | **0** | **31.7%** | **100.0%** | **48.2%** |

---

### 2. Modello Baseline: Addestramento con Ground Truth Reale nuScenes (Ablation Study)

Il modello baseline viene addestrato utilizzando unicamente i box 3D fisicamente annotati da nuScenes (senza regole semantiche di affordance):

#### A. Valutazione su Ground Truth Reale nuScenes (a 25m)
| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.878 | 384 | 16 | 83.0% | 99.2% | **90.4%** |
| Camion / Bus | 586 | 205 | 2 | 74.1% | 99.7% | **85.0%** |
| VRU (Pedoni / Ciclisti) | 925 | 244 | 32 | 79.1% | 96.7% | **87.0%** |
| Barriere | 663 | 104 | 2 | 86.4% | 99.7% | **92.6%** |
| **Media Globale (25m)** | **4.052** | **937** | **52** | **81.2%** | **98.7%** | **89.1%** |

#### B. Valutazione su Ground Truth Sintetica Neurosimbolica (a 25m)
| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.959 | 303 | 1.559 | 86.6% | 55.7% | **67.8%** |
| Camion / Bus | 604 | 187 | 1.147 | 76.4% | 34.5% | **47.5%** |
| VRU (Pedoni / Ciclisti) | 978 | 191 | 1.259 | 83.7% | 43.7% | **57.4%** |
| Barriere | 677 | 90 | 1.330 | 88.3% | 33.7% | **48.8%** |
| **Media Globale (25m)** | **4.218** | **771** | **5.295** | **84.5%** | **44.3%** | **58.2%** |

---

### 3. Tabella Comparativa di Ablation Study (a 25 Metri)

| Addestramento Effettuato con | Valutazione Test su | Precision (%) | Recall (%) | F1-Score (%) | Falsi Negativi (Pericoli Persi) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ground Truth Reale nuScenes** *(Baseline)* | GT Reale nuScenes | 81.2% | 98.7% | 89.1% | 52 |
| **Ground Truth Reale nuScenes** *(Baseline)* | GT Sintetica Neurosimbolica | 84.5% | 44.3% | 58.2% | **5.295** *(Crollo sicurezza)* |
| **Ground Truth Sintetica Neurosimbolica** *(Ufficiale)* | GT Reale nuScenes | 31.7% | **100.0%** | 48.2% | **0** *(Zero ostacoli reali persi)* |
| **Ground Truth Sintetica Neurosimbolica** *(Ufficiale)* | GT Sintetica Neurosimbolica | 73.4% | **99.8%** | **84.5%** | **23** *(Massima anticipazione)* |

* **Evidenze Scientifiche**:
  * L'addestramento su **Ground Truth Reale** produce un modello miope che sopprime le allerte sulle zone d'ombra apparentemente vuote, mancando il **55.7%** dei pericoli potenziali (5.295 punti ciechi pericolosi ignorati).
  * L'addestramento su **Ground Truth Sintetica Neurosimbolica** abilita la guida difensiva (**Recall al 99.8%** sui varchi a rischio) azzerando i falsi negativi sugli ostacoli fisici reali (**Recall al 100.0%**).
