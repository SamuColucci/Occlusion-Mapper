# Report Ufficiale di Valutazione nuScenes - Split VAL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: VAL] - 2026-09-22 19:37:15

> **Split analizzato**: `VAL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   1.8% |  76.0% | **  3.5%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  46.6% |  77.0% | ** 58.1%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  30.7% |  82.0% | ** 44.7%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  50.8% |  97.0% | ** 66.6%** |

---


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 22 | 264 | 2 |   7.7% |  91.7% |  14.2% |
| Camion/Bus | 0 | 138 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 593 | 0 |   0.3% | 100.0% |   0.7% |
| Barriera | 0 | 274 | 2 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **24** | **1269** | **4** | **  1.9%** | ** 85.7%** | **  3.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 68 | 695 | 18 |   8.9% |  79.1% |  16.0% |
| Camion/Bus | 0 | 371 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 1988 | 0 |   0.2% | 100.0% |   0.4% |
| Barriera | 4 | 1065 | 6 |   0.4% |  40.0% |   0.7% |
| **MEDIA GLOBALE** | **76** | **4119** | **24** | **  1.8%** | ** 76.0%** | **  3.5%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 133 | 153 | 4 |  46.5% |  97.1% |  62.9% |
| Camion/Bus | 14 | 124 | 2 |  10.1% |  87.5% |  18.2% |
| VRU (Pedoni/Bici) | 289 | 306 | 4 |  48.6% |  98.6% |  65.1% |
| Barriera | 156 | 118 | 207 |  56.9% |  43.0% |  49.0% |
| **MEDIA GLOBALE** | **592** | **701** | **217** | ** 45.8%** | ** 73.2%** | ** 56.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 355 | 408 | 29 |  46.5% |  92.4% |  61.9% |
| Camion/Bus | 16 | 355 | 2 |   4.3% |  88.9% |   8.2% |
| VRU (Pedoni/Bici) | 909 | 1083 | 5 |  45.6% |  99.5% |  62.6% |
| Barriera | 675 | 394 | 549 |  63.1% |  55.1% |  58.9% |
| **MEDIA GLOBALE** | **1955** | **2240** | **585** | ** 46.6%** | ** 77.0%** | ** 58.1%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 148 | 138 | 7 |  51.7% |  95.5% |  67.1% |
| Camion/Bus | 54 | 84 | 13 |  39.1% |  80.6% |  52.7% |
| VRU (Pedoni/Bici) | 217 | 378 | 1 |  36.5% |  99.5% |  53.4% |
| Barriera | 44 | 230 | 39 |  16.1% |  53.0% |  24.6% |
| **MEDIA GLOBALE** | **463** | **830** | **60** | ** 35.8%** | ** 88.5%** | ** 51.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 414 | 349 | 45 |  54.3% |  90.2% |  67.8% |
| Camion/Bus | 103 | 268 | 37 |  27.8% |  73.6% |  40.3% |
| VRU (Pedoni/Bici) | 552 | 1440 | 1 |  27.7% |  99.8% |  43.4% |
| Barriera | 219 | 850 | 200 |  20.5% |  52.3% |  29.4% |
| **MEDIA GLOBALE** | **1288** | **2907** | **283** | ** 30.7%** | ** 82.0%** | ** 44.7%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 168 | 118 | 4 |  58.7% |  97.7% |  73.4% |
| Camion/Bus | 16 | 122 | 1 |  11.6% |  94.1% |  20.6% |
| VRU (Pedoni/Bici) | 446 | 149 | 3 |  75.0% |  99.3% |  85.4% |
| Barriera | 55 | 219 | 19 |  20.1% |  74.3% |  31.6% |
| **MEDIA GLOBALE** | **685** | **608** | **27** | ** 53.0%** | ** 96.2%** | ** 68.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 462 | 301 | 30 |  60.6% |  93.9% |  73.6% |
| Camion/Bus | 28 | 343 | 1 |   7.5% |  96.6% |  14.0% |
| VRU (Pedoni/Bici) | 1433 | 559 | 4 |  71.9% |  99.7% |  83.6% |
| Barriera | 207 | 862 | 32 |  19.4% |  86.6% |  31.7% |
| **MEDIA GLOBALE** | **2130** | **2065** | **67** | ** 50.8%** | ** 97.0%** | ** 66.6%** |
