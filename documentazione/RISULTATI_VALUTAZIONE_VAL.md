# Report Ufficiale di Valutazione nuScenes - Split VAL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: VAL] - 2026-09-12 21:02:28

> **Split analizzato**: `VAL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Analitica (Bayes)** | GT Reale nuScenes |  25.6% |  41.6% | ** 31.7%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Geometrica 3D) |  60.4% |  45.2% | ** 51.7%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Semantica Naïve) |  61.6% |  51.2% | ** 55.9%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Ibrida Spazio-Sem) |  64.3% |  48.4% | ** 55.2%** |

---


### Modello: Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 250 | 302 | 262 |  45.3% |  48.8% |  47.0% |
| Camion/Bus | 14 | 234 | 76 |   5.6% |  15.6% |   8.3% |
| VRU (Pedoni/Bici) | 102 | 397 | 133 |  20.4% |  43.4% |  27.8% |
| Barriera | 0 | 0 | 13 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **366** | **933** | **484** | ** 28.2%** | ** 43.1%** | ** 34.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 354 | 465 | 364 |  43.2% |  49.3% |  46.1% |
| Camion/Bus | 19 | 355 | 105 |   5.1% |  15.3% |   7.6% |
| VRU (Pedoni/Bici) | 107 | 575 | 190 |  15.7% |  36.0% |  21.9% |
| Barriera | 0 | 0 | 14 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **480** | **1395** | **673** | ** 25.6%** | ** 41.6%** | ** 31.7%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 424 | 128 | 286 |  76.8% |  59.7% |  67.2% |
| Camion/Bus | 80 | 168 | 94 |  32.3% |  46.0% |  37.9% |
| VRU (Pedoni/Bici) | 264 | 235 | 335 |  52.9% |  44.1% |  48.1% |
| Barriera | 0 | 0 | 113 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **768** | **531** | **828** | ** 59.1%** | ** 48.1%** | ** 53.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 640 | 179 | 391 |  78.1% |  62.1% |  69.2% |
| Camion/Bus | 125 | 249 | 134 |  33.4% |  48.3% |  39.5% |
| VRU (Pedoni/Bici) | 368 | 314 | 590 |  54.0% |  38.4% |  44.9% |
| Barriera | 0 | 0 | 258 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **1133** | **742** | **1373** | ** 60.4%** | ** 45.2%** | ** 51.7%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 450 | 102 | 286 |  81.5% |  61.1% |  69.9% |
| Camion/Bus | 106 | 142 | 139 |  42.7% |  43.3% |  43.0% |
| VRU (Pedoni/Bici) | 238 | 261 | 172 |  47.7% |  58.0% |  52.4% |
| Barriera | 0 | 0 | 109 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **794** | **505** | **706** | ** 61.1%** | ** 52.9%** | ** 56.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 683 | 136 | 396 |  83.4% |  63.3% |  72.0% |
| Camion/Bus | 172 | 202 | 210 |  46.0% |  45.0% |  45.5% |
| VRU (Pedoni/Bici) | 300 | 382 | 247 |  44.0% |  54.8% |  48.8% |
| Barriera | 0 | 0 | 247 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **1155** | **720** | **1100** | ** 61.6%** | ** 51.2%** | ** 55.9%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 422 | 130 | 210 |  76.4% |  66.8% |  71.3% |
| Camion/Bus | 84 | 164 | 82 |  33.9% |  50.6% |  40.6% |
| VRU (Pedoni/Bici) | 303 | 196 | 428 |  60.7% |  41.5% |  49.3% |
| Barriera | 0 | 0 | 75 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **809** | **490** | **795** | ** 62.3%** | ** 50.4%** | ** 55.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 644 | 175 | 297 |  78.6% |  68.4% |  73.2% |
| Camion/Bus | 133 | 241 | 132 |  35.6% |  50.2% |  41.6% |
| VRU (Pedoni/Bici) | 428 | 254 | 745 |  62.8% |  36.5% |  46.1% |
| Barriera | 0 | 0 | 112 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **1205** | **670** | **1286** | ** 64.3%** | ** 48.4%** | ** 55.2%** |
