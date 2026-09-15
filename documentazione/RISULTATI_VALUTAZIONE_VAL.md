# Report Ufficiale di Valutazione nuScenes - Split VAL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: VAL] - 2026-09-15 22:48:26

> **Split analizzato**: `VAL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Geometric)** | GT Reale nuScenes |   1.7% |  93.5% | **  3.3%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Geometrica 3D) |  50.0% |  99.7% | ** 66.6%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Semantica Naïve) |  29.4% |  97.3% | ** 45.2%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Ibrida Spazio-Sem) |  41.0% |  98.8% | ** 58.0%** |
| **Attention + GT Sintetica (Semantic)** | GT Reale nuScenes |   2.0% |  92.6% | **  3.9%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Geometrica 3D) |  46.4% |  77.4% | ** 58.0%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Semantica Naïve) |  35.7% |  98.8% | ** 52.4%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Ibrida Spazio-Sem) |  42.1% |  85.1% | ** 56.4%** |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes |   4.4% |  93.5% | **  8.5%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Geometrica 3D) |  53.0% |  40.2% | ** 45.7%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Semantica Naïve) |  44.0% |  55.4% | ** 49.1%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Ibrida Spazio-Sem) |  32.0% |  29.4% | ** 30.7%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Reale nuScenes |   2.3% |  94.4% | **  4.5%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Geometrica 3D) |  54.4% |  80.6% | ** 65.0%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Semantica Naïve) |  33.9% |  83.3% | ** 48.2%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Ibrida Spazio-Sem) |  46.0% |  82.5% | ** 59.1%** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   1.7% |  87.0% | **  3.3%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  45.3% |  83.3% | ** 58.7%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  28.6% |  87.2% | ** 43.1%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  44.6% |  99.1% | ** 61.5%** |
| **Baseline Analitica (Bayes)** | GT Reale nuScenes |   4.7% |  76.9% | **  8.8%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Geometrica 3D) |  49.6% |  29.3% | ** 36.8%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Semantica Naïve) |  57.5% |  56.4% | ** 57.0%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Ibrida Spazio-Sem) |  58.9% |  42.1% | ** 49.1%** |

---


### Modello: Modello Attention su GT Sintetica Geometrica (Spatially-Constrained)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 37 | 403 | 0 |   8.4% | 100.0% |  15.5% |
| Camion/Bus | 0 | 321 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 621 | 0 |   0.5% | 100.0% |   1.0% |
| Barriera | 4 | 581 | 0 |   0.7% | 100.0% |   1.4% |
| **MEDIA GLOBALE** | **44** | **1926** | **0** | **  2.2%** | **100.0%** | **  4.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 86 | 1121 | 7 |   7.1% |  92.5% |  13.2% |
| Camion/Bus | 0 | 843 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 5 | 2017 | 0 |   0.2% | 100.0% |   0.5% |
| Barriera | 10 | 1903 | 0 |   0.5% | 100.0% |   1.0% |
| **MEDIA GLOBALE** | **101** | **5884** | **7** | **  1.7%** | ** 93.5%** | **  3.3%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 186 | 254 | 0 |  42.3% | 100.0% |  59.4% |
| Camion/Bus | 30 | 291 | 0 |   9.3% | 100.0% |  17.1% |
| VRU (Pedoni/Bici) | 374 | 250 | 0 |  59.9% | 100.0% |  74.9% |
| Barriera | 445 | 140 | 1 |  76.1% |  99.8% |  86.3% |
| **MEDIA GLOBALE** | **1035** | **935** | **1** | ** 52.5%** | ** 99.9%** | ** 68.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 495 | 712 | 8 |  41.0% |  98.4% |  57.9% |
| Camion/Bus | 37 | 806 | 0 |   4.4% | 100.0% |   8.4% |
| VRU (Pedoni/Bici) | 1052 | 970 | 1 |  52.0% |  99.9% |  68.4% |
| Barriera | 1411 | 502 | 1 |  73.8% |  99.9% |  84.9% |
| **MEDIA GLOBALE** | **2995** | **2990** | **10** | ** 50.0%** | ** 99.7%** | ** 66.6%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 217 | 223 | 6 |  49.3% |  97.3% |  65.5% |
| Camion/Bus | 89 | 232 | 7 |  27.7% |  92.7% |  42.7% |
| VRU (Pedoni/Bici) | 182 | 442 | 0 |  29.2% | 100.0% |  45.2% |
| Barriera | 114 | 471 | 0 |  19.5% | 100.0% |  32.6% |
| **MEDIA GLOBALE** | **602** | **1368** | **13** | ** 30.6%** | ** 97.9%** | ** 46.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 602 | 605 | 23 |  49.9% |  96.3% |  65.7% |
| Camion/Bus | 190 | 653 | 26 |  22.5% |  88.0% |  35.9% |
| VRU (Pedoni/Bici) | 512 | 1510 | 0 |  25.3% | 100.0% |  40.4% |
| Barriera | 458 | 1455 | 0 |  23.9% | 100.0% |  38.6% |
| **MEDIA GLOBALE** | **1762** | **4223** | **49** | ** 29.4%** | ** 97.3%** | ** 45.2%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 251 | 189 | 2 |  57.0% |  99.2% |  72.4% |
| Camion/Bus | 33 | 288 | 0 |  10.3% | 100.0% |  18.6% |
| VRU (Pedoni/Bici) | 460 | 164 | 0 |  73.7% | 100.0% |  84.9% |
| Barriera | 88 | 497 | 1 |  15.0% |  98.9% |  26.1% |
| **MEDIA GLOBALE** | **832** | **1138** | **3** | ** 42.2%** | ** 99.6%** | ** 59.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 633 | 574 | 27 |  52.4% |  95.9% |  67.8% |
| Camion/Bus | 41 | 802 | 0 |   4.9% | 100.0% |   9.3% |
| VRU (Pedoni/Bici) | 1503 | 519 | 1 |  74.3% |  99.9% |  85.3% |
| Barriera | 279 | 1634 | 1 |  14.6% |  99.6% |  25.4% |
| **MEDIA GLOBALE** | **2456** | **3529** | **29** | ** 41.0%** | ** 98.8%** | ** 58.0%** |


### Modello: Modello Attention su GT Sintetica Semantica (Semantic-Affordance / Naïve)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 36 | 410 | 1 |   8.1% |  97.3% |  14.9% |
| Camion/Bus | 0 | 340 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 528 | 0 |   0.6% | 100.0% |   1.1% |
| Barriera | 4 | 391 | 0 |   1.0% | 100.0% |   2.0% |
| **MEDIA GLOBALE** | **43** | **1669** | **1** | **  2.5%** | ** 97.7%** | **  4.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 87 | 1159 | 6 |   7.0% |  93.5% |  13.0% |
| Camion/Bus | 0 | 956 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 4 | 1700 | 1 |   0.2% |  80.0% |   0.5% |
| Barriera | 9 | 1105 | 1 |   0.8% |  90.0% |   1.6% |
| **MEDIA GLOBALE** | **100** | **4920** | **8** | **  2.0%** | ** 92.6%** | **  3.9%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 185 | 261 | 1 |  41.5% |  99.5% |  58.5% |
| Camion/Bus | 30 | 310 | 0 |   8.8% | 100.0% |  16.2% |
| VRU (Pedoni/Bici) | 322 | 209 | 52 |  60.6% |  86.1% |  71.2% |
| Barriera | 296 | 99 | 150 |  74.9% |  66.4% |  70.4% |
| **MEDIA GLOBALE** | **833** | **879** | **203** | ** 48.7%** | ** 80.4%** | ** 60.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 496 | 750 | 7 |  39.8% |  98.6% |  56.7% |
| Camion/Bus | 37 | 919 | 0 |   3.9% | 100.0% |   7.5% |
| VRU (Pedoni/Bici) | 903 | 801 | 150 |  53.0% |  85.8% |  65.5% |
| Barriera | 891 | 223 | 521 |  80.0% |  63.1% |  70.5% |
| **MEDIA GLOBALE** | **2327** | **2693** | **678** | ** 46.4%** | ** 77.4%** | ** 58.0%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 222 | 224 | 1 |  49.8% |  99.6% |  66.4% |
| Camion/Bus | 96 | 244 | 0 |  28.2% | 100.0% |  44.0% |
| VRU (Pedoni/Bici) | 180 | 351 | 2 |  33.9% |  98.9% |  50.5% |
| Barriera | 108 | 287 | 6 |  27.3% |  94.7% |  42.4% |
| **MEDIA GLOBALE** | **606** | **1106** | **9** | ** 35.4%** | ** 98.5%** | ** 52.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 619 | 627 | 6 |  49.7% |  99.0% |  66.2% |
| Camion/Bus | 216 | 740 | 0 |  22.6% | 100.0% |  36.9% |
| VRU (Pedoni/Bici) | 507 | 1197 | 5 |  29.8% |  99.0% |  45.8% |
| Barriera | 448 | 666 | 10 |  40.2% |  97.8% |  57.0% |
| **MEDIA GLOBALE** | **1790** | **3230** | **21** | ** 35.7%** | ** 98.8%** | ** 52.4%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 245 | 201 | 8 |  54.9% |  96.8% |  70.1% |
| Camion/Bus | 31 | 309 | 2 |   9.1% |  93.9% |  16.6% |
| VRU (Pedoni/Bici) | 394 | 137 | 66 |  74.2% |  85.7% |  79.5% |
| Barriera | 65 | 330 | 24 |  16.5% |  73.0% |  26.9% |
| **MEDIA GLOBALE** | **735** | **977** | **100** | ** 42.9%** | ** 88.0%** | ** 57.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 634 | 612 | 26 |  50.9% |  96.1% |  66.5% |
| Camion/Bus | 39 | 917 | 2 |   4.1% |  95.1% |   7.8% |
| VRU (Pedoni/Bici) | 1255 | 449 | 249 |  73.7% |  83.4% |  78.2% |
| Barriera | 187 | 927 | 93 |  16.8% |  66.8% |  26.8% |
| **MEDIA GLOBALE** | **2115** | **2905** | **370** | ** 42.1%** | ** 85.1%** | ** 56.4%** |


### Modello: Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 37 | 314 | 0 |  10.5% | 100.0% |  19.1% |
| Camion/Bus | 0 | 159 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 2 | 60 | 1 |   3.2% |  66.7% |   6.2% |
| Barriera | 4 | 270 | 0 |   1.5% | 100.0% |   2.9% |
| **MEDIA GLOBALE** | **43** | **803** | **1** | **  5.1%** | ** 97.7%** | **  9.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 88 | 831 | 5 |   9.6% |  94.6% |  17.4% |
| Camion/Bus | 0 | 426 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 145 | 2 |   2.0% |  60.0% |   3.9% |
| Barriera | 10 | 775 | 0 |   1.3% | 100.0% |   2.5% |
| **MEDIA GLOBALE** | **101** | **2177** | **7** | **  4.4%** | ** 93.5%** | **  8.5%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 164 | 187 | 22 |  46.7% |  88.2% |  61.1% |
| Camion/Bus | 19 | 140 | 11 |  11.9% |  63.3% |  20.1% |
| VRU (Pedoni/Bici) | 37 | 25 | 337 |  59.7% |   9.9% |  17.0% |
| Barriera | 222 | 52 | 224 |  81.0% |  49.8% |  61.7% |
| **MEDIA GLOBALE** | **442** | **404** | **594** | ** 52.2%** | ** 42.7%** | ** 47.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 424 | 495 | 79 |  46.1% |  84.3% |  59.6% |
| Camion/Bus | 23 | 403 | 14 |   5.4% |  62.2% |   9.9% |
| VRU (Pedoni/Bici) | 84 | 64 | 969 |  56.8% |   8.0% |  14.0% |
| Barriera | 677 | 108 | 735 |  86.2% |  47.9% |  61.6% |
| **MEDIA GLOBALE** | **1208** | **1070** | **1797** | ** 53.0%** | ** 40.2%** | ** 45.7%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 204 | 147 | 19 |  58.1% |  91.5% |  71.1% |
| Camion/Bus | 45 | 114 | 51 |  28.3% |  46.9% |  35.3% |
| VRU (Pedoni/Bici) | 20 | 42 | 162 |  32.3% |  11.0% |  16.4% |
| Barriera | 69 | 205 | 45 |  25.2% |  60.5% |  35.6% |
| **MEDIA GLOBALE** | **338** | **508** | **277** | ** 40.0%** | ** 55.0%** | ** 46.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 550 | 369 | 75 |  59.8% |  88.0% |  71.2% |
| Camion/Bus | 102 | 324 | 114 |  23.9% |  47.2% |  31.8% |
| VRU (Pedoni/Bici) | 45 | 103 | 467 |  30.4% |   8.8% |  13.6% |
| Barriera | 306 | 479 | 152 |  39.0% |  66.8% |  49.2% |
| **MEDIA GLOBALE** | **1003** | **1275** | **808** | ** 44.0%** | ** 55.4%** | ** 49.1%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 188 | 163 | 65 |  53.6% |  74.3% |  62.3% |
| Camion/Bus | 17 | 142 | 16 |  10.7% |  51.5% |  17.7% |
| VRU (Pedoni/Bici) | 44 | 18 | 416 |  71.0% |   9.6% |  16.9% |
| Barriera | 41 | 233 | 48 |  15.0% |  46.1% |  22.6% |
| **MEDIA GLOBALE** | **290** | **556** | **545** | ** 34.3%** | ** 34.7%** | ** 34.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 483 | 436 | 177 |  52.6% |  73.2% |  61.2% |
| Camion/Bus | 24 | 402 | 17 |   5.6% |  58.5% |  10.3% |
| VRU (Pedoni/Bici) | 108 | 40 | 1396 |  73.0% |   7.2% |  13.1% |
| Barriera | 115 | 670 | 165 |  14.6% |  41.1% |  21.6% |
| **MEDIA GLOBALE** | **730** | **1548** | **1755** | ** 32.0%** | ** 29.4%** | ** 30.7%** |


### Modello: Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 37 | 323 | 0 |  10.3% | 100.0% |  18.6% |
| Camion/Bus | 0 | 127 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 571 | 0 |   0.5% | 100.0% |   1.0% |
| Barriera | 4 | 458 | 0 |   0.9% | 100.0% |   1.7% |
| **MEDIA GLOBALE** | **44** | **1479** | **0** | **  2.9%** | **100.0%** | **  5.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 87 | 948 | 6 |   8.4% |  93.5% |  15.4% |
| Camion/Bus | 0 | 354 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 5 | 1739 | 0 |   0.3% | 100.0% |   0.6% |
| Barriera | 10 | 1308 | 0 |   0.8% | 100.0% |   1.5% |
| **MEDIA GLOBALE** | **102** | **4349** | **6** | **  2.3%** | ** 94.4%** | **  4.5%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 163 | 197 | 23 |  45.3% |  87.6% |  59.7% |
| Camion/Bus | 17 | 110 | 13 |  13.4% |  56.7% |  21.7% |
| VRU (Pedoni/Bici) | 348 | 226 | 26 |  60.6% |  93.0% |  73.4% |
| Barriera | 361 | 101 | 85 |  78.1% |  80.9% |  79.5% |
| **MEDIA GLOBALE** | **889** | **634** | **147** | ** 58.4%** | ** 85.8%** | ** 69.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 459 | 576 | 44 |  44.3% |  91.3% |  59.7% |
| Camion/Bus | 21 | 333 | 16 |   5.9% |  56.8% |  10.7% |
| VRU (Pedoni/Bici) | 924 | 820 | 129 |  53.0% |  87.7% |  66.1% |
| Barriera | 1019 | 299 | 393 |  77.3% |  72.2% |  74.7% |
| **MEDIA GLOBALE** | **2423** | **2028** | **582** | ** 54.4%** | ** 80.6%** | ** 65.0%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 203 | 157 | 20 |  56.4% |  91.0% |  69.6% |
| Camion/Bus | 44 | 83 | 52 |  34.6% |  45.8% |  39.5% |
| VRU (Pedoni/Bici) | 172 | 402 | 10 |  30.0% |  94.5% |  45.5% |
| Barriera | 98 | 364 | 16 |  21.2% |  86.0% |  34.0% |
| **MEDIA GLOBALE** | **517** | **1006** | **98** | ** 33.9%** | ** 84.1%** | ** 48.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 582 | 453 | 43 |  56.2% |  93.1% |  70.1% |
| Camion/Bus | 111 | 243 | 105 |  31.4% |  51.4% |  38.9% |
| VRU (Pedoni/Bici) | 430 | 1314 | 82 |  24.7% |  84.0% |  38.1% |
| Barriera | 386 | 932 | 72 |  29.3% |  84.3% |  43.5% |
| **MEDIA GLOBALE** | **1509** | **2942** | **302** | ** 33.9%** | ** 83.3%** | ** 48.2%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 195 | 165 | 58 |  54.2% |  77.1% |  63.6% |
| Camion/Bus | 10 | 117 | 23 |   7.9% |  30.3% |  12.5% |
| VRU (Pedoni/Bici) | 425 | 149 | 35 |  74.0% |  92.4% |  82.2% |
| Barriera | 65 | 397 | 24 |  14.1% |  73.0% |  23.6% |
| **MEDIA GLOBALE** | **695** | **828** | **140** | ** 45.6%** | ** 83.2%** | ** 58.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 539 | 496 | 121 |  52.1% |  81.7% |  63.6% |
| Camion/Bus | 14 | 340 | 27 |   4.0% |  34.1% |   7.1% |
| VRU (Pedoni/Bici) | 1316 | 428 | 188 |  75.5% |  87.5% |  81.0% |
| Barriera | 180 | 1138 | 100 |  13.7% |  64.3% |  22.5% |
| **MEDIA GLOBALE** | **2049** | **2402** | **436** | ** 46.0%** | ** 82.5%** | ** 59.1%** |


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 37 | 415 | 0 |   8.2% | 100.0% |  15.1% |
| Camion/Bus | 0 | 318 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 621 | 0 |   0.5% | 100.0% |   1.0% |
| Barriera | 1 | 397 | 3 |   0.3% |  25.0% |   0.5% |
| **MEDIA GLOBALE** | **41** | **1751** | **3** | **  2.3%** | ** 93.2%** | **  4.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 86 | 1191 | 7 |   6.7% |  92.5% |  12.6% |
| Camion/Bus | 0 | 880 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 5 | 2020 | 0 |   0.2% | 100.0% |   0.5% |
| Barriera | 3 | 1337 | 7 |   0.2% |  30.0% |   0.4% |
| **MEDIA GLOBALE** | **94** | **5428** | **14** | **  1.7%** | ** 87.0%** | **  3.3%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 186 | 266 | 0 |  41.2% | 100.0% |  58.3% |
| Camion/Bus | 29 | 289 | 1 |   9.1% |  96.7% |  16.7% |
| VRU (Pedoni/Bici) | 374 | 250 | 0 |  59.9% | 100.0% |  74.9% |
| Barriera | 276 | 122 | 170 |  69.3% |  61.9% |  65.4% |
| **MEDIA GLOBALE** | **865** | **927** | **171** | ** 48.3%** | ** 83.5%** | ** 61.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 496 | 781 | 7 |  38.8% |  98.6% |  55.7% |
| Camion/Bus | 36 | 844 | 1 |   4.1% |  97.3% |   7.9% |
| VRU (Pedoni/Bici) | 1053 | 972 | 0 |  52.0% | 100.0% |  68.4% |
| Barriera | 917 | 423 | 495 |  68.4% |  64.9% |  66.6% |
| **MEDIA GLOBALE** | **2502** | **3020** | **503** | ** 45.3%** | ** 83.3%** | ** 58.7%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 221 | 231 | 2 |  48.9% |  99.1% |  65.5% |
| Camion/Bus | 88 | 230 | 8 |  27.7% |  91.7% |  42.5% |
| VRU (Pedoni/Bici) | 182 | 442 | 0 |  29.2% | 100.0% |  45.2% |
| Barriera | 60 | 338 | 54 |  15.1% |  52.6% |  23.4% |
| **MEDIA GLOBALE** | **551** | **1241** | **64** | ** 30.7%** | ** 89.6%** | ** 45.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 613 | 664 | 12 |  48.0% |  98.1% |  64.5% |
| Camion/Bus | 193 | 687 | 23 |  21.9% |  89.4% |  35.2% |
| VRU (Pedoni/Bici) | 512 | 1513 | 0 |  25.3% | 100.0% |  40.4% |
| Barriera | 261 | 1079 | 197 |  19.5% |  57.0% |  29.0% |
| **MEDIA GLOBALE** | **1579** | **3943** | **232** | ** 28.6%** | ** 87.2%** | ** 43.1%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 251 | 201 | 2 |  55.5% |  99.2% |  71.2% |
| Camion/Bus | 32 | 286 | 1 |  10.1% |  97.0% |  18.2% |
| VRU (Pedoni/Bici) | 460 | 164 | 0 |  73.7% | 100.0% |  84.9% |
| Barriera | 82 | 316 | 7 |  20.6% |  92.1% |  33.7% |
| **MEDIA GLOBALE** | **825** | **967** | **10** | ** 46.0%** | ** 98.8%** | ** 62.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 649 | 628 | 11 |  50.8% |  98.3% |  67.0% |
| Camion/Bus | 40 | 840 | 1 |   4.5% |  97.6% |   8.7% |
| VRU (Pedoni/Bici) | 1504 | 521 | 0 |  74.3% | 100.0% |  85.2% |
| Barriera | 269 | 1071 | 11 |  20.1% |  96.1% |  33.2% |
| **MEDIA GLOBALE** | **2462** | **3060** | **23** | ** 44.6%** | ** 99.1%** | ** 61.5%** |


### Modello: Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 32 | 220 | 5 |  12.7% |  86.5% |  22.1% |
| Camion/Bus | 0 | 72 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 3 | 267 | 0 |   1.1% | 100.0% |   2.2% |
| Barriera | 0 | 0 | 4 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **35** | **559** | **9** | **  5.9%** | ** 79.5%** | ** 11.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 78 | 708 | 15 |   9.9% |  83.9% |  17.7% |
| Camion/Bus | 0 | 287 | 0 |   0.0% |   0.0% |   0.0% |
| VRU (Pedoni/Bici) | 5 | 700 | 0 |   0.7% | 100.0% |   1.4% |
| Barriera | 0 | 0 | 10 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **83** | **1695** | **25** | **  4.7%** | ** 76.9%** | **  8.8%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 87 | 21 |  65.5% |  88.7% |  75.3% |
| Camion/Bus | 20 | 52 | 10 |  27.8% |  66.7% |  39.2% |
| VRU (Pedoni/Bici) | 173 | 97 | 201 |  64.1% |  46.3% |  53.7% |
| Barriera | 0 | 0 | 446 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **358** | **236** | **678** | ** 60.3%** | ** 34.6%** | ** 43.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 450 | 336 | 53 |  57.3% |  89.5% |  69.8% |
| Camion/Bus | 26 | 261 | 11 |   9.1% |  70.3% |  16.0% |
| VRU (Pedoni/Bici) | 405 | 300 | 648 |  57.4% |  38.5% |  46.1% |
| Barriera | 0 | 0 | 1412 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **881** | **897** | **2124** | ** 49.6%** | ** 29.3%** | ** 36.8%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 196 | 56 | 27 |  77.8% |  87.9% |  82.5% |
| Camion/Bus | 45 | 27 | 51 |  62.5% |  46.9% |  53.6% |
| VRU (Pedoni/Bici) | 139 | 131 | 43 |  51.5% |  76.4% |  61.5% |
| Barriera | 0 | 0 | 114 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **380** | **214** | **235** | ** 64.0%** | ** 61.8%** | ** 62.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 562 | 224 | 63 |  71.5% |  89.9% |  79.7% |
| Camion/Bus | 120 | 167 | 96 |  41.8% |  55.6% |  47.7% |
| VRU (Pedoni/Bici) | 340 | 365 | 172 |  48.2% |  66.4% |  55.9% |
| Barriera | 0 | 0 | 458 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **1022** | **756** | **789** | ** 57.5%** | ** 56.4%** | ** 57.0%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 182 | 70 | 71 |  72.2% |  71.9% |  72.1% |
| Camion/Bus | 11 | 61 | 22 |  15.3% |  33.3% |  21.0% |
| VRU (Pedoni/Bici) | 208 | 62 | 252 |  77.0% |  45.2% |  57.0% |
| Barriera | 0 | 0 | 89 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **401** | **193** | **434** | ** 67.5%** | ** 48.0%** | ** 56.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 505 | 281 | 155 |  64.2% |  76.5% |  69.8% |
| Camion/Bus | 14 | 273 | 27 |   4.9% |  34.1% |   8.5% |
| VRU (Pedoni/Bici) | 528 | 177 | 976 |  74.9% |  35.1% |  47.8% |
| Barriera | 0 | 0 | 280 |   0.0% |   0.0% |   0.0% |
| **MEDIA GLOBALE** | **1047** | **731** | **1438** | ** 58.9%** | ** 42.1%** | ** 49.1%** |
