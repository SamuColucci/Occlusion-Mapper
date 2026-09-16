# Report Ufficiale di Valutazione nuScenes - Split VAL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: VAL] - 2026-09-16 16:24:24

> **Split analizzato**: `VAL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Geometric)** | GT Reale nuScenes |   1.7% |  82.0% | **  3.3%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Geometrica 3D) |  50.3% |  97.0% | ** 66.3%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Semantica Naïve) |  30.1% |  93.8% | ** 45.6%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Ibrida Spazio-Sem) |  43.2% |  96.3% | ** 59.7%** |
| **Attention + GT Sintetica (Semantic)** | GT Reale nuScenes |   2.0% |  80.0% | **  3.8%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Geometrica 3D) |  46.5% |  74.3% | ** 57.2%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Semantica Naïve) |  37.0% |  95.7% | ** 53.4%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Ibrida Spazio-Sem) |  45.0% |  83.1% | ** 58.4%** |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes |   3.1% |  74.0% | **  6.0%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Geometrica 3D) |  46.3% |  43.0% | ** 44.6%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Semantica Naïve) |  32.8% |  49.3% | ** 39.4%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Ibrida Spazio-Sem) |  46.2% |  49.6% | ** 47.9%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Reale nuScenes |   1.8% |  82.0% | **  3.5%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Geometrica 3D) |  45.3% |  81.6% | ** 58.3%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Semantica Naïve) |  29.7% |  86.6% | ** 44.3%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Ibrida Spazio-Sem) |  44.2% |  91.9% | ** 59.7%** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   1.8% |  78.0% | **  3.6%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  46.5% |  78.3% | ** 58.3%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  30.6% |  83.4% | ** 44.8%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  49.6% |  96.7% | ** 65.6%** |
| **Baseline Analitica (Bayes)** | GT Reale nuScenes |   4.4% |  65.0% | **  8.3%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Geometrica 3D) |  46.8% |  27.0% | ** 34.2%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Semantica Naïve) |  52.7% |  49.1% | ** 50.9%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Ibrida Spazio-Sem) |  60.1% |  40.1% | ** 48.1%** |

---


### Modello: Modello Attention su GT Sintetica Geometrica (Spatially-Constrained)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 21 | 264 | 3 |   7.4% |  87.5% |  13.6% |
| Camion/Bus | 0 | 148 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 592 | 0 |   0.3% | 100.0% |   0.7% |
| Barriera | 2 | 541 | 0 |   0.4% | 100.0% |   0.7% |
| **MEDIA GLOBALE** | **25** | **1545** | **3** | **  1.6%** | ** 89.3%** | **  3.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 68 | 709 | 18 |   8.8% |  79.1% |  15.8% |
| Camion/Bus | 0 | 364 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 1956 | 0 |   0.2% | 100.0% |   0.4% |
| Barriera | 10 | 1786 | 0 |   0.6% | 100.0% |   1.1% |
| **MEDIA GLOBALE** | **82** | **4815** | **18** | **  1.7%** | ** 82.0%** | **  3.3%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 130 | 155 | 7 |  45.6% |  94.9% |  61.6% |
| Camion/Bus | 14 | 134 | 2 |   9.5% |  87.5% |  17.1% |
| VRU (Pedoni/Bici) | 289 | 305 | 4 |  48.7% |  98.6% |  65.2% |
| Barriera | 354 | 189 | 9 |  65.2% |  97.5% |  78.1% |
| **MEDIA GLOBALE** | **787** | **783** | **22** | ** 50.1%** | ** 97.3%** | ** 66.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 357 | 420 | 27 |  45.9% |  93.0% |  61.5% |
| Camion/Bus | 16 | 348 | 2 |   4.4% |  88.9% |   8.4% |
| VRU (Pedoni/Bici) | 897 | 1063 | 17 |  45.8% |  98.1% |  62.4% |
| Barriera | 1195 | 601 | 29 |  66.5% |  97.6% |  79.1% |
| **MEDIA GLOBALE** | **2465** | **2432** | **75** | ** 50.3%** | ** 97.0%** | ** 66.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 145 | 140 | 10 |  50.9% |  93.5% |  65.9% |
| Camion/Bus | 59 | 89 | 8 |  39.9% |  88.1% |  54.9% |
| VRU (Pedoni/Bici) | 208 | 386 | 10 |  35.0% |  95.4% |  51.2% |
| Barriera | 82 | 461 | 1 |  15.1% |  98.8% |  26.2% |
| **MEDIA GLOBALE** | **494** | **1076** | **29** | ** 31.5%** | ** 94.5%** | ** 47.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 422 | 355 | 37 |  54.3% |  91.9% |  68.3% |
| Camion/Bus | 104 | 260 | 36 |  28.6% |  74.3% |  41.3% |
| VRU (Pedoni/Bici) | 532 | 1428 | 21 |  27.1% |  96.2% |  42.3% |
| Barriera | 416 | 1380 | 3 |  23.2% |  99.3% |  37.6% |
| **MEDIA GLOBALE** | **1474** | **3423** | **97** | ** 30.1%** | ** 93.8%** | ** 45.6%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 164 | 121 | 8 |  57.5% |  95.3% |  71.8% |
| Camion/Bus | 16 | 132 | 1 |  10.8% |  94.1% |  19.4% |
| VRU (Pedoni/Bici) | 437 | 157 | 12 |  73.6% |  97.3% |  83.8% |
| Barriera | 71 | 472 | 3 |  13.1% |  95.9% |  23.0% |
| **MEDIA GLOBALE** | **688** | **882** | **24** | ** 43.8%** | ** 96.6%** | ** 60.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 461 | 316 | 31 |  59.3% |  93.7% |  72.7% |
| Camion/Bus | 28 | 336 | 1 |   7.7% |  96.6% |  14.2% |
| VRU (Pedoni/Bici) | 1394 | 566 | 43 |  71.1% |  97.0% |  82.1% |
| Barriera | 233 | 1563 | 6 |  13.0% |  97.5% |  22.9% |
| **MEDIA GLOBALE** | **2116** | **2781** | **81** | ** 43.2%** | ** 96.3%** | ** 59.7%** |


### Modello: Modello Attention su GT Sintetica Semantica (Semantic-Affordance / Naïve)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 21 | 271 | 3 |   7.2% |  87.5% |  13.3% |
| Camion/Bus | 0 | 171 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 523 | 0 |   0.4% | 100.0% |   0.8% |
| Barriera | 1 | 315 | 1 |   0.3% |  50.0% |   0.6% |
| **MEDIA GLOBALE** | **24** | **1280** | **4** | **  1.8%** | ** 85.7%** | **  3.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 68 | 700 | 18 |   8.9% |  79.1% |  15.9% |
| Camion/Bus | 0 | 437 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 1742 | 0 |   0.2% | 100.0% |   0.5% |
| Barriera | 8 | 1100 | 2 |   0.7% |  80.0% |   1.4% |
| **MEDIA GLOBALE** | **80** | **3979** | **20** | **  2.0%** | ** 80.0%** | **  3.8%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 131 | 161 | 6 |  44.9% |  95.6% |  61.1% |
| Camion/Bus | 14 | 157 | 2 |   8.2% |  87.5% |  15.0% |
| VRU (Pedoni/Bici) | 243 | 282 | 50 |  46.3% |  82.9% |  59.4% |
| Barriera | 205 | 111 | 158 |  64.9% |  56.5% |  60.4% |
| **MEDIA GLOBALE** | **593** | **711** | **216** | ** 45.5%** | ** 73.3%** | ** 56.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 351 | 417 | 33 |  45.7% |  91.4% |  60.9% |
| Camion/Bus | 16 | 421 | 2 |   3.7% |  88.9% |   7.0% |
| VRU (Pedoni/Bici) | 772 | 974 | 142 |  44.2% |  84.5% |  58.0% |
| Barriera | 747 | 361 | 477 |  67.4% |  61.0% |  64.1% |
| **MEDIA GLOBALE** | **1886** | **2173** | **654** | ** 46.5%** | ** 74.3%** | ** 57.2%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 149 | 143 | 6 |  51.0% |  96.1% |  66.7% |
| Camion/Bus | 62 | 109 | 5 |  36.3% |  92.5% |  52.1% |
| VRU (Pedoni/Bici) | 208 | 317 | 10 |  39.6% |  95.4% |  56.0% |
| Barriera | 76 | 240 | 7 |  24.1% |  91.6% |  38.1% |
| **MEDIA GLOBALE** | **495** | **809** | **28** | ** 38.0%** | ** 94.6%** | ** 54.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 429 | 339 | 30 |  55.9% |  93.5% |  69.9% |
| Camion/Bus | 131 | 306 | 9 |  30.0% |  93.6% |  45.4% |
| VRU (Pedoni/Bici) | 538 | 1208 | 15 |  30.8% |  97.3% |  46.8% |
| Barriera | 405 | 703 | 14 |  36.6% |  96.7% |  53.0% |
| **MEDIA GLOBALE** | **1503** | **2556** | **68** | ** 37.0%** | ** 95.7%** | ** 53.4%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 164 | 128 | 8 |  56.2% |  95.3% |  70.7% |
| Camion/Bus | 16 | 155 | 1 |   9.4% |  94.1% |  17.0% |
| VRU (Pedoni/Bici) | 369 | 156 | 80 |  70.3% |  82.2% |  75.8% |
| Barriera | 41 | 275 | 33 |  13.0% |  55.4% |  21.0% |
| **MEDIA GLOBALE** | **590** | **714** | **122** | ** 45.2%** | ** 82.9%** | ** 58.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 445 | 323 | 47 |  57.9% |  90.4% |  70.6% |
| Camion/Bus | 28 | 409 | 1 |   6.4% |  96.6% |  12.0% |
| VRU (Pedoni/Bici) | 1201 | 545 | 236 |  68.8% |  83.6% |  75.5% |
| Barriera | 152 | 956 | 87 |  13.7% |  63.6% |  22.6% |
| **MEDIA GLOBALE** | **1826** | **2233** | **371** | ** 45.0%** | ** 83.1%** | ** 58.4%** |


### Modello: Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 20 | 212 | 4 |   8.6% |  83.3% |  15.6% |
| Camion/Bus | 0 | 130 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 1 | 307 | 1 |   0.3% |  50.0% |   0.6% |
| Barriera | 2 | 192 | 0 |   1.0% | 100.0% |   2.0% |
| **MEDIA GLOBALE** | **23** | **841** | **5** | **  2.7%** | ** 82.1%** | **  5.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 65 | 569 | 21 |  10.3% |  75.6% |  18.1% |
| Camion/Bus | 0 | 304 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 932 | 2 |   0.2% |  50.0% |   0.4% |
| Barriera | 7 | 479 | 3 |   1.4% |  70.0% |   2.8% |
| **MEDIA GLOBALE** | **74** | **2284** | **26** | **  3.1%** | ** 74.0%** | **  6.0%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 101 | 131 | 36 |  43.5% |  73.7% |  54.7% |
| Camion/Bus | 10 | 120 | 6 |   7.7% |  62.5% |  13.7% |
| VRU (Pedoni/Bici) | 160 | 148 | 133 |  51.9% |  54.6% |  53.2% |
| Barriera | 142 | 52 | 221 |  73.2% |  39.1% |  51.0% |
| **MEDIA GLOBALE** | **413** | **451** | **396** | ** 47.8%** | ** 51.1%** | ** 49.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 265 | 369 | 119 |  41.8% |  69.0% |  52.1% |
| Camion/Bus | 11 | 293 | 7 |   3.6% |  61.1% |   6.8% |
| VRU (Pedoni/Bici) | 437 | 497 | 477 |  46.8% |  47.8% |  47.3% |
| Barriera | 379 | 107 | 845 |  78.0% |  31.0% |  44.3% |
| **MEDIA GLOBALE** | **1092** | **1266** | **1448** | ** 46.3%** | ** 43.0%** | ** 44.6%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 111 | 121 | 44 |  47.8% |  71.6% |  57.4% |
| Camion/Bus | 33 | 97 | 34 |  25.4% |  49.3% |  33.5% |
| VRU (Pedoni/Bici) | 108 | 200 | 110 |  35.1% |  49.5% |  41.1% |
| Barriera | 41 | 153 | 42 |  21.1% |  49.4% |  29.6% |
| **MEDIA GLOBALE** | **293** | **571** | **230** | ** 33.9%** | ** 56.0%** | ** 42.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 322 | 312 | 137 |  50.8% |  70.2% |  58.9% |
| Camion/Bus | 58 | 246 | 82 |  19.1% |  41.4% |  26.1% |
| VRU (Pedoni/Bici) | 242 | 692 | 311 |  25.9% |  43.8% |  32.5% |
| Barriera | 152 | 334 | 267 |  31.3% |  36.3% |  33.6% |
| **MEDIA GLOBALE** | **774** | **1584** | **797** | ** 32.8%** | ** 49.3%** | ** 39.4%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 128 | 104 | 44 |  55.2% |  74.4% |  63.4% |
| Camion/Bus | 12 | 118 | 5 |   9.2% |  70.6% |  16.3% |
| VRU (Pedoni/Bici) | 233 | 75 | 216 |  75.6% |  51.9% |  61.6% |
| Barriera | 24 | 170 | 50 |  12.4% |  32.4% |  17.9% |
| **MEDIA GLOBALE** | **397** | **467** | **315** | ** 45.9%** | ** 55.8%** | ** 50.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 345 | 289 | 147 |  54.4% |  70.1% |  61.3% |
| Camion/Bus | 18 | 286 | 11 |   5.9% |  62.1% |  10.8% |
| VRU (Pedoni/Bici) | 663 | 271 | 774 |  71.0% |  46.1% |  55.9% |
| Barriera | 64 | 422 | 175 |  13.2% |  26.8% |  17.7% |
| **MEDIA GLOBALE** | **1090** | **1268** | **1107** | ** 46.2%** | ** 49.6%** | ** 47.9%** |


### Modello: Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 22 | 281 | 2 |   7.3% |  91.7% |  13.5% |
| Camion/Bus | 0 | 184 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 592 | 0 |   0.3% | 100.0% |   0.7% |
| Barriera | 2 | 407 | 0 |   0.5% | 100.0% |   1.0% |
| **MEDIA GLOBALE** | **26** | **1464** | **2** | **  1.7%** | ** 92.9%** | **  3.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 70 | 770 | 16 |   8.3% |  81.4% |  15.1% |
| Camion/Bus | 0 | 493 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 1931 | 1 |   0.2% |  75.0% |   0.3% |
| Barriera | 9 | 1296 | 1 |   0.7% |  90.0% |   1.4% |
| **MEDIA GLOBALE** | **82** | **4490** | **18** | **  1.8%** | ** 82.0%** | **  3.5%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 124 | 179 | 13 |  40.9% |  90.5% |  56.4% |
| Camion/Bus | 13 | 171 | 3 |   7.1% |  81.2% |  13.0% |
| VRU (Pedoni/Bici) | 278 | 316 | 15 |  46.8% |  94.9% |  62.7% |
| Barriera | 252 | 157 | 111 |  61.6% |  69.4% |  65.3% |
| **MEDIA GLOBALE** | **667** | **823** | **142** | ** 44.8%** | ** 82.4%** | ** 58.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 341 | 499 | 43 |  40.6% |  88.8% |  55.7% |
| Camion/Bus | 15 | 478 | 3 |   3.0% |  83.3% |   5.9% |
| VRU (Pedoni/Bici) | 873 | 1061 | 41 |  45.1% |  95.5% |  61.3% |
| Barriera | 844 | 461 | 380 |  64.7% |  69.0% |  66.7% |
| **MEDIA GLOBALE** | **2073** | **2499** | **467** | ** 45.3%** | ** 81.6%** | ** 58.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 142 | 161 | 13 |  46.9% |  91.6% |  62.0% |
| Camion/Bus | 58 | 126 | 9 |  31.5% |  86.6% |  46.2% |
| VRU (Pedoni/Bici) | 213 | 381 | 5 |  35.9% |  97.7% |  52.5% |
| Barriera | 65 | 344 | 18 |  15.9% |  78.3% |  26.4% |
| **MEDIA GLOBALE** | **478** | **1012** | **45** | ** 32.1%** | ** 91.4%** | ** 47.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 409 | 431 | 50 |  48.7% |  89.1% |  63.0% |
| Camion/Bus | 117 | 376 | 23 |  23.7% |  83.6% |  37.0% |
| VRU (Pedoni/Bici) | 537 | 1397 | 16 |  27.8% |  97.1% |  43.2% |
| Barriera | 297 | 1008 | 122 |  22.8% |  70.9% |  34.5% |
| **MEDIA GLOBALE** | **1360** | **3212** | **211** | ** 29.7%** | ** 86.6%** | ** 44.3%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 158 | 145 | 14 |  52.1% |  91.9% |  66.5% |
| Camion/Bus | 17 | 167 | 0 |   9.2% | 100.0% |  16.9% |
| VRU (Pedoni/Bici) | 429 | 165 | 20 |  72.2% |  95.5% |  82.3% |
| Barriera | 62 | 347 | 12 |  15.2% |  83.8% |  25.7% |
| **MEDIA GLOBALE** | **666** | **824** | **46** | ** 44.7%** | ** 93.5%** | ** 60.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 440 | 400 | 52 |  52.4% |  89.4% |  66.1% |
| Camion/Bus | 29 | 464 | 0 |   5.9% | 100.0% |  11.1% |
| VRU (Pedoni/Bici) | 1373 | 561 | 64 |  71.0% |  95.5% |  81.5% |
| Barriera | 178 | 1127 | 61 |  13.6% |  74.5% |  23.1% |
| **MEDIA GLOBALE** | **2020** | **2552** | **177** | ** 44.2%** | ** 91.9%** | ** 59.7%** |


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 22 | 274 | 2 |   7.4% |  91.7% |  13.8% |
| Camion/Bus | 0 | 152 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 611 | 0 |   0.3% | 100.0% |   0.7% |
| Barriera | 1 | 319 | 1 |   0.3% |  50.0% |   0.6% |
| **MEDIA GLOBALE** | **25** | **1356** | **3** | **  1.8%** | ** 89.3%** | **  3.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 69 | 715 | 17 |   8.8% |  80.2% |  15.9% |
| Camion/Bus | 0 | 366 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 2006 | 0 |   0.2% | 100.0% |   0.4% |
| Barriera | 5 | 1117 | 5 |   0.4% |  50.0% |   0.9% |
| **MEDIA GLOBALE** | **78** | **4204** | **22** | **  1.8%** | ** 78.0%** | **  3.6%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 133 | 163 | 4 |  44.9% |  97.1% |  61.4% |
| Camion/Bus | 14 | 138 | 2 |   9.2% |  87.5% |  16.7% |
| VRU (Pedoni/Bici) | 291 | 322 | 2 |  47.5% |  99.3% |  64.2% |
| Barriera | 192 | 128 | 171 |  60.0% |  52.9% |  56.2% |
| **MEDIA GLOBALE** | **630** | **751** | **179** | ** 45.6%** | ** 77.9%** | ** 57.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 358 | 426 | 26 |  45.7% |  93.2% |  61.3% |
| Camion/Bus | 16 | 350 | 2 |   4.4% |  88.9% |   8.3% |
| VRU (Pedoni/Bici) | 909 | 1101 | 5 |  45.2% |  99.5% |  62.2% |
| Barriera | 707 | 415 | 517 |  63.0% |  57.8% |  60.3% |
| **MEDIA GLOBALE** | **1990** | **2292** | **550** | ** 46.5%** | ** 78.3%** | ** 58.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 150 | 146 | 5 |  50.7% |  96.8% |  66.5% |
| Camion/Bus | 57 | 95 | 10 |  37.5% |  85.1% |  52.1% |
| VRU (Pedoni/Bici) | 217 | 396 | 1 |  35.4% |  99.5% |  52.2% |
| Barriera | 50 | 270 | 33 |  15.6% |  60.2% |  24.8% |
| **MEDIA GLOBALE** | **474** | **907** | **49** | ** 34.3%** | ** 90.6%** | ** 49.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 423 | 361 | 36 |  54.0% |  92.2% |  68.1% |
| Camion/Bus | 101 | 265 | 39 |  27.6% |  72.1% |  39.9% |
| VRU (Pedoni/Bici) | 552 | 1458 | 1 |  27.5% |  99.8% |  43.1% |
| Barriera | 234 | 888 | 185 |  20.9% |  55.8% |  30.4% |
| **MEDIA GLOBALE** | **1310** | **2972** | **261** | ** 30.6%** | ** 83.4%** | ** 44.8%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 168 | 128 | 4 |  56.8% |  97.7% |  71.8% |
| Camion/Bus | 17 | 135 | 0 |  11.2% | 100.0% |  20.1% |
| VRU (Pedoni/Bici) | 447 | 166 | 2 |  72.9% |  99.6% |  84.2% |
| Barriera | 56 | 264 | 18 |  17.5% |  75.7% |  28.4% |
| **MEDIA GLOBALE** | **688** | **693** | **24** | ** 49.8%** | ** 96.6%** | ** 65.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 462 | 322 | 30 |  58.9% |  93.9% |  72.4% |
| Camion/Bus | 29 | 337 | 0 |   7.9% | 100.0% |  14.7% |
| VRU (Pedoni/Bici) | 1432 | 578 | 5 |  71.2% |  99.7% |  83.1% |
| Barriera | 202 | 920 | 37 |  18.0% |  84.5% |  29.7% |
| **MEDIA GLOBALE** | **2125** | **2157** | **72** | ** 49.6%** | ** 96.7%** | ** 65.6%** |


### Modello: Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 19 | 145 | 5 |  11.6% |  79.2% |  20.2% |
| Camion/Bus | 0 | 45 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 255 | 0 |   0.8% | 100.0% |   1.5% |
| Barriera | 0 | 0 | 2 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **21** | **445** | **7** | **  4.5%** | ** 75.0%** | **  8.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 61 | 506 | 25 |  10.8% |  70.9% |  18.7% |
| Camion/Bus | 0 | 193 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 701 | 0 |   0.6% | 100.0% |   1.1% |
| Barriera | 0 | 0 | 10 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **65** | **1400** | **35** | **  4.4%** | ** 65.0%** | **  8.3%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 111 | 53 | 26 |  67.7% |  81.0% |  73.8% |
| Camion/Bus | 10 | 35 | 6 |  22.2% |  62.5% |  32.8% |
| VRU (Pedoni/Bici) | 148 | 109 | 145 |  57.6% |  50.5% |  53.8% |
| Barriera | 0 | 0 | 363 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **269** | **197** | **540** | ** 57.7%** | ** 33.3%** | ** 42.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 309 | 258 | 75 |  54.5% |  80.5% |  65.0% |
| Camion/Bus | 11 | 182 | 7 |   5.7% |  61.1% |  10.4% |
| VRU (Pedoni/Bici) | 365 | 340 | 549 |  51.8% |  39.9% |  45.1% |
| Barriera | 0 | 0 | 1224 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **685** | **780** | **1855** | ** 46.8%** | ** 27.0%** | ** 34.2%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 121 | 43 | 34 |  73.8% |  78.1% |  75.9% |
| Camion/Bus | 30 | 15 | 37 |  66.7% |  44.8% |  53.6% |
| VRU (Pedoni/Bici) | 147 | 110 | 71 |  57.2% |  67.4% |  61.9% |
| Barriera | 0 | 0 | 83 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **298** | **168** | **225** | ** 63.9%** | ** 57.0%** | ** 60.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 372 | 195 | 87 |  65.6% |  81.0% |  72.5% |
| Camion/Bus | 71 | 122 | 69 |  36.8% |  50.7% |  42.6% |
| VRU (Pedoni/Bici) | 329 | 376 | 224 |  46.7% |  59.5% |  52.3% |
| Barriera | 0 | 0 | 419 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **772** | **693** | **799** | ** 52.7%** | ** 49.1%** | ** 50.9%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 122 | 42 | 50 |  74.4% |  70.9% |  72.6% |
| Camion/Bus | 8 | 37 | 9 |  17.8% |  47.1% |  25.8% |
| VRU (Pedoni/Bici) | 198 | 59 | 251 |  77.0% |  44.1% |  56.1% |
| Barriera | 0 | 0 | 74 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **328** | **138** | **384** | ** 70.4%** | ** 46.1%** | ** 55.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 355 | 212 | 137 |  62.6% |  72.2% |  67.0% |
| Camion/Bus | 10 | 183 | 19 |   5.2% |  34.5% |   9.0% |
| VRU (Pedoni/Bici) | 515 | 190 | 922 |  73.0% |  35.8% |  48.1% |
| Barriera | 0 | 0 | 239 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **880** | **585** | **1317** | ** 60.1%** | ** 40.1%** | ** 48.1%** |
