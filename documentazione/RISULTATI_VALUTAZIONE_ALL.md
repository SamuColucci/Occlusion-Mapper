# Report Ufficiale di Valutazione nuScenes - Split ALL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: ALL] - 2026-09-16 16:13:16

> **Split analizzato**: `ALL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Geometric)** | GT Reale nuScenes |   2.1% |  79.3% | **  4.0%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Geometrica 3D) |  54.7% |  96.7% | ** 69.9%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Semantica Naïve) |  33.9% |  91.4% | ** 49.5%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Ibrida Spazio-Sem) |  47.7% |  93.5% | ** 63.2%** |
| **Attention + GT Sintetica (Semantic)** | GT Reale nuScenes |   2.3% |  69.3% | **  4.4%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Geometrica 3D) |  50.6% |  71.8% | ** 59.4%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Semantica Naïve) |  42.9% |  92.9% | ** 58.7%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Ibrida Spazio-Sem) |  49.8% |  78.4% | ** 60.9%** |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes |   3.2% |  53.1% | **  6.1%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Geometrica 3D) |  50.0% |  37.8% | ** 43.0%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Semantica Naïve) |  34.2% |  39.4% | ** 36.7%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Ibrida Spazio-Sem) |  53.0% |  44.4% | ** 48.3%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Reale nuScenes |   2.2% |  77.8% | **  4.2%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Geometrica 3D) |  48.9% |  80.9% | ** 61.0%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Semantica Naïve) |  32.8% |  82.6% | ** 46.9%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Ibrida Spazio-Sem) |  48.3% |  88.7% | ** 62.6%** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   2.1% |  72.1% | **  4.2%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  49.8% |  77.3% | ** 60.5%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  33.9% |  80.4% | ** 47.7%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  55.4% |  95.5% | ** 70.1%** |
| **Baseline Analitica (Bayes)** | GT Reale nuScenes |   5.0% |  71.0% | **  9.4%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Geometrica 3D) |  48.7% |  31.9% | ** 38.5%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Semantica Naïve) |  54.6% |  54.4% | ** 54.5%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Ibrida Spazio-Sem) |  62.0% |  45.0% | ** 52.1%** |

---


### Modello: Modello Attention su GT Sintetica Geometrica (Spatially-Constrained)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 113 | 942 | 11 |  10.7% |  91.1% |  19.2% |
| Camion/Bus | 5 | 730 | 2 |   0.7% |  71.4% |   1.3% |
| VRU (Pedoni/Bici) | 59 | 2301 | 0 |   2.5% | 100.0% |   4.9% |
| Barriera | 30 | 2076 | 12 |   1.4% |  71.4% |   2.8% |
| **MEDIA GLOBALE** | **207** | **6049** | **25** | **  3.3%** | ** 89.2%** | **  6.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 184 | 2238 | 44 |   7.6% |  80.7% |  13.9% |
| Camion/Bus | 6 | 1296 | 7 |   0.5% |  46.2% |   0.9% |
| VRU (Pedoni/Bici) | 103 | 7346 | 8 |   1.4% |  92.8% |   2.7% |
| Barriera | 82 | 6843 | 39 |   1.2% |  67.8% |   2.3% |
| **MEDIA GLOBALE** | **375** | **17723** | **98** | **  2.1%** | ** 79.3%** | **  4.0%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 667 | 388 | 37 |  63.2% |  94.7% |  75.8% |
| Camion/Bus | 185 | 550 | 11 |  25.2% |  94.4% |  39.7% |
| VRU (Pedoni/Bici) | 1297 | 1063 | 49 |  55.0% |  96.4% |  70.0% |
| Barriera | 1466 | 640 | 67 |  69.6% |  95.6% |  80.6% |
| **MEDIA GLOBALE** | **3615** | **2641** | **164** | ** 57.8%** | ** 95.7%** | ** 72.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1315 | 1107 | 84 |  54.3% |  94.0% |  68.8% |
| Camion/Bus | 207 | 1095 | 16 |  15.9% |  92.8% |  27.1% |
| VRU (Pedoni/Bici) | 3648 | 3801 | 85 |  49.0% |  97.7% |  65.2% |
| Barriera | 4728 | 2197 | 154 |  68.3% |  96.8% |  80.1% |
| **MEDIA GLOBALE** | **9898** | **8200** | **339** | ** 54.7%** | ** 96.7%** | ** 69.9%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 668 | 387 | 42 |  63.3% |  94.1% |  75.7% |
| Camion/Bus | 347 | 388 | 50 |  47.2% |  87.4% |  61.3% |
| VRU (Pedoni/Bici) | 915 | 1445 | 66 |  38.8% |  93.3% |  54.8% |
| Barriera | 352 | 1754 | 18 |  16.7% |  95.1% |  28.4% |
| **MEDIA GLOBALE** | **2282** | **3974** | **176** | ** 36.5%** | ** 92.8%** | ** 52.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1451 | 971 | 127 |  59.9% |  92.0% |  72.5% |
| Camion/Bus | 555 | 747 | 167 |  42.6% |  76.9% |  54.8% |
| VRU (Pedoni/Bici) | 2275 | 5174 | 221 |  30.5% |  91.1% |  45.8% |
| Barriera | 1857 | 5068 | 62 |  26.8% |  96.8% |  42.0% |
| **MEDIA GLOBALE** | **6138** | **11960** | **577** | ** 33.9%** | ** 91.4%** | ** 49.5%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 771 | 284 | 63 |  73.1% |  92.4% |  81.6% |
| Camion/Bus | 118 | 617 | 10 |  16.1% |  92.2% |  27.3% |
| VRU (Pedoni/Bici) | 1759 | 601 | 95 |  74.5% |  94.9% |  83.5% |
| Barriera | 417 | 1689 | 19 |  19.8% |  95.6% |  32.8% |
| **MEDIA GLOBALE** | **3065** | **3191** | **187** | ** 49.0%** | ** 94.2%** | ** 64.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1598 | 824 | 136 |  66.0% |  92.2% |  76.9% |
| Camion/Bus | 145 | 1157 | 20 |  11.1% |  87.9% |  19.8% |
| VRU (Pedoni/Bici) | 5616 | 1833 | 376 |  75.4% |  93.7% |  83.6% |
| Barriera | 1272 | 5653 | 64 |  18.4% |  95.2% |  30.8% |
| **MEDIA GLOBALE** | **8631** | **9467** | **596** | ** 47.7%** | ** 93.5%** | ** 63.2%** |


### Modello: Modello Attention su GT Sintetica Semantica (Semantic-Affordance / Naïve)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 114 | 961 | 10 |  10.6% |  91.9% |  19.0% |
| Camion/Bus | 5 | 776 | 2 |   0.6% |  71.4% |   1.3% |
| VRU (Pedoni/Bici) | 38 | 1893 | 21 |   2.0% |  64.4% |   3.8% |
| Barriera | 24 | 1065 | 18 |   2.2% |  57.1% |   4.2% |
| **MEDIA GLOBALE** | **181** | **4695** | **51** | **  3.7%** | ** 78.0%** | **  7.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 188 | 2224 | 40 |   7.8% |  82.5% |  14.2% |
| Camion/Bus | 6 | 1528 | 7 |   0.4% |  46.2% |   0.8% |
| VRU (Pedoni/Bici) | 82 | 6440 | 29 |   1.3% |  73.9% |   2.5% |
| Barriera | 52 | 4005 | 69 |   1.3% |  43.0% |   2.5% |
| **MEDIA GLOBALE** | **328** | **14197** | **145** | **  2.3%** | ** 69.3%** | **  4.4%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 669 | 406 | 35 |  62.2% |  95.0% |  75.2% |
| Camion/Bus | 184 | 597 | 12 |  23.6% |  93.9% |  37.7% |
| VRU (Pedoni/Bici) | 1018 | 913 | 328 |  52.7% |  75.6% |  62.1% |
| Barriera | 756 | 333 | 777 |  69.4% |  49.3% |  57.7% |
| **MEDIA GLOBALE** | **2627** | **2249** | **1152** | ** 53.9%** | ** 69.5%** | ** 60.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1303 | 1109 | 96 |  54.0% |  93.1% |  68.4% |
| Camion/Bus | 202 | 1332 | 21 |  13.2% |  90.6% |  23.0% |
| VRU (Pedoni/Bici) | 2995 | 3527 | 738 |  45.9% |  80.2% |  58.4% |
| Barriera | 2851 | 1206 | 2031 |  70.3% |  58.4% |  63.8% |
| **MEDIA GLOBALE** | **7351** | **7174** | **2886** | ** 50.6%** | ** 71.8%** | ** 59.4%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 679 | 396 | 31 |  63.2% |  95.6% |  76.1% |
| Camion/Bus | 363 | 418 | 34 |  46.5% |  91.4% |  61.6% |
| VRU (Pedoni/Bici) | 830 | 1101 | 151 |  43.0% |  84.6% |  57.0% |
| Barriera | 337 | 752 | 33 |  30.9% |  91.1% |  46.2% |
| **MEDIA GLOBALE** | **2209** | **2667** | **249** | ** 45.3%** | ** 89.9%** | ** 60.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1484 | 928 | 94 |  61.5% |  94.0% |  74.4% |
| Camion/Bus | 666 | 868 | 56 |  43.4% |  92.2% |  59.0% |
| VRU (Pedoni/Bici) | 2297 | 4225 | 199 |  35.2% |  92.0% |  50.9% |
| Barriera | 1790 | 2267 | 129 |  44.1% |  93.3% |  59.9% |
| **MEDIA GLOBALE** | **6237** | **8288** | **478** | ** 42.9%** | ** 92.9%** | ** 58.7%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 774 | 301 | 60 |  72.0% |  92.8% |  81.1% |
| Camion/Bus | 116 | 665 | 12 |  14.9% |  90.6% |  25.5% |
| VRU (Pedoni/Bici) | 1421 | 510 | 433 |  73.6% |  76.6% |  75.1% |
| Barriera | 185 | 904 | 251 |  17.0% |  42.4% |  24.3% |
| **MEDIA GLOBALE** | **2496** | **2380** | **756** | ** 51.2%** | ** 76.8%** | ** 61.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1568 | 844 | 166 |  65.0% |  90.4% |  75.6% |
| Camion/Bus | 144 | 1390 | 21 |   9.4% |  87.3% |  17.0% |
| VRU (Pedoni/Bici) | 4777 | 1745 | 1215 |  73.2% |  79.7% |  76.3% |
| Barriera | 747 | 3310 | 589 |  18.4% |  55.9% |  27.7% |
| **MEDIA GLOBALE** | **7236** | **7289** | **1991** | ** 49.8%** | ** 78.4%** | ** 60.9%** |


### Modello: Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 70 | 684 | 54 |   9.3% |  56.5% |  15.9% |
| Camion/Bus | 5 | 442 | 2 |   1.1% |  71.4% |   2.2% |
| VRU (Pedoni/Bici) | 49 | 1206 | 10 |   3.9% |  83.1% |   7.5% |
| Barriera | 21 | 577 | 21 |   3.5% |  50.0% |   6.6% |
| **MEDIA GLOBALE** | **145** | **2909** | **87** | **  4.7%** | ** 62.5%** | **  8.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 131 | 1628 | 97 |   7.4% |  57.5% |  13.2% |
| Camion/Bus | 6 | 871 | 7 |   0.7% |  46.2% |   1.3% |
| VRU (Pedoni/Bici) | 77 | 3451 | 34 |   2.2% |  69.4% |   4.2% |
| Barriera | 37 | 1534 | 84 |   2.4% |  30.6% |   4.4% |
| **MEDIA GLOBALE** | **251** | **7484** | **222** | **  3.2%** | ** 53.1%** | **  6.1%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 405 | 349 | 299 |  53.7% |  57.5% |  55.6% |
| Camion/Bus | 90 | 357 | 106 |  20.1% |  45.9% |  28.0% |
| VRU (Pedoni/Bici) | 776 | 479 | 570 |  61.8% |  57.7% |  59.7% |
| Barriera | 439 | 159 | 1094 |  73.4% |  28.6% |  41.2% |
| **MEDIA GLOBALE** | **1710** | **1344** | **2069** | ** 56.0%** | ** 45.3%** | ** 50.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 794 | 965 | 605 |  45.1% |  56.8% |  50.3% |
| Camion/Bus | 98 | 779 | 125 |  11.2% |  43.9% |  17.8% |
| VRU (Pedoni/Bici) | 1776 | 1752 | 1957 |  50.3% |  47.6% |  48.9% |
| Barriera | 1200 | 371 | 3682 |  76.4% |  24.6% |  37.2% |
| **MEDIA GLOBALE** | **3868** | **3867** | **6369** | ** 50.0%** | ** 37.8%** | ** 43.0%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 399 | 355 | 311 |  52.9% |  56.2% |  54.5% |
| Camion/Bus | 144 | 303 | 253 |  32.2% |  36.3% |  34.1% |
| VRU (Pedoni/Bici) | 419 | 836 | 562 |  33.4% |  42.7% |  37.5% |
| Barriera | 126 | 472 | 244 |  21.1% |  34.1% |  26.0% |
| **MEDIA GLOBALE** | **1088** | **1966** | **1370** | ** 35.6%** | ** 44.3%** | ** 39.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 892 | 867 | 686 |  50.7% |  56.5% |  53.5% |
| Camion/Bus | 277 | 600 | 445 |  31.6% |  38.4% |  34.6% |
| VRU (Pedoni/Bici) | 989 | 2539 | 1507 |  28.0% |  39.6% |  32.8% |
| Barriera | 490 | 1081 | 1429 |  31.2% |  25.5% |  28.1% |
| **MEDIA GLOBALE** | **2648** | **5087** | **4067** | ** 34.2%** | ** 39.4%** | ** 36.7%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 503 | 251 | 331 |  66.7% |  60.3% |  63.4% |
| Camion/Bus | 86 | 361 | 42 |  19.2% |  67.2% |  29.9% |
| VRU (Pedoni/Bici) | 975 | 280 | 879 |  77.7% |  52.6% |  62.7% |
| Barriera | 112 | 486 | 324 |  18.7% |  25.7% |  21.7% |
| **MEDIA GLOBALE** | **1676** | **1378** | **1576** | ** 54.9%** | ** 51.5%** | ** 53.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1015 | 744 | 719 |  57.7% |  58.5% |  58.1% |
| Camion/Bus | 103 | 774 | 62 |  11.7% |  62.4% |  19.8% |
| VRU (Pedoni/Bici) | 2683 | 845 | 3309 |  76.0% |  44.8% |  56.4% |
| Barriera | 297 | 1274 | 1039 |  18.9% |  22.2% |  20.4% |
| **MEDIA GLOBALE** | **4098** | **3637** | **5129** | ** 53.0%** | ** 44.4%** | ** 48.3%** |


### Modello: Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 117 | 1110 | 7 |   9.5% |  94.4% |  17.3% |
| Camion/Bus | 5 | 823 | 2 |   0.6% |  71.4% |   1.2% |
| VRU (Pedoni/Bici) | 52 | 2128 | 7 |   2.4% |  88.1% |   4.6% |
| Barriera | 27 | 1549 | 15 |   1.7% |  64.3% |   3.3% |
| **MEDIA GLOBALE** | **201** | **5610** | **31** | **  3.5%** | ** 86.6%** | **  6.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 194 | 2608 | 34 |   6.9% |  85.1% |  12.8% |
| Camion/Bus | 6 | 1681 | 7 |   0.4% |  46.2% |   0.7% |
| VRU (Pedoni/Bici) | 93 | 7209 | 18 |   1.3% |  83.8% |   2.5% |
| Barriera | 75 | 5066 | 46 |   1.5% |  62.0% |   2.9% |
| **MEDIA GLOBALE** | **368** | **16564** | **105** | **  2.2%** | ** 77.8%** | **  4.2%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 654 | 573 | 50 |  53.3% |  92.9% |  67.7% |
| Camion/Bus | 179 | 649 | 17 |  21.6% |  91.3% |  35.0% |
| VRU (Pedoni/Bici) | 1208 | 972 | 138 |  55.4% |  89.7% |  68.5% |
| Barriera | 1028 | 548 | 505 |  65.2% |  67.1% |  66.1% |
| **MEDIA GLOBALE** | **3069** | **2742** | **710** | ** 52.8%** | ** 81.2%** | ** 64.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1267 | 1535 | 132 |  45.2% |  90.6% |  60.3% |
| Camion/Bus | 199 | 1488 | 24 |  11.8% |  89.2% |  20.8% |
| VRU (Pedoni/Bici) | 3454 | 3848 | 279 |  47.3% |  92.5% |  62.6% |
| Barriera | 3360 | 1781 | 1522 |  65.4% |  68.8% |  67.0% |
| **MEDIA GLOBALE** | **8280** | **8652** | **1957** | ** 48.9%** | ** 80.9%** | ** 61.0%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 661 | 566 | 49 |  53.9% |  93.1% |  68.2% |
| Camion/Bus | 334 | 494 | 63 |  40.3% |  84.1% |  54.5% |
| VRU (Pedoni/Bici) | 819 | 1361 | 162 |  37.6% |  83.5% |  51.8% |
| Barriera | 242 | 1334 | 128 |  15.4% |  65.4% |  24.9% |
| **MEDIA GLOBALE** | **2056** | **3755** | **402** | ** 35.4%** | ** 83.6%** | ** 49.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1432 | 1370 | 146 |  51.1% |  90.7% |  65.4% |
| Camion/Bus | 579 | 1108 | 143 |  34.3% |  80.2% |  48.1% |
| VRU (Pedoni/Bici) | 2224 | 5078 | 272 |  30.5% |  89.1% |  45.4% |
| Barriera | 1312 | 3829 | 607 |  25.5% |  68.4% |  37.2% |
| **MEDIA GLOBALE** | **5547** | **11385** | **1168** | ** 32.8%** | ** 82.6%** | ** 46.9%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 780 | 447 | 54 |  63.6% |  93.5% |  75.7% |
| Camion/Bus | 118 | 710 | 10 |  14.3% |  92.2% |  24.7% |
| VRU (Pedoni/Bici) | 1634 | 546 | 220 |  75.0% |  88.1% |  81.0% |
| Barriera | 354 | 1222 | 82 |  22.5% |  81.2% |  35.2% |
| **MEDIA GLOBALE** | **2886** | **2925** | **366** | ** 49.7%** | ** 88.7%** | ** 63.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1582 | 1220 | 152 |  56.5% |  91.2% |  69.8% |
| Camion/Bus | 146 | 1541 | 19 |   8.7% |  88.5% |  15.8% |
| VRU (Pedoni/Bici) | 5487 | 1815 | 505 |  75.1% |  91.6% |  82.5% |
| Barriera | 969 | 4172 | 367 |  18.8% |  72.5% |  29.9% |
| **MEDIA GLOBALE** | **8184** | **8748** | **1043** | ** 48.3%** | ** 88.7%** | ** 62.6%** |


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 116 | 975 | 8 |  10.6% |  93.5% |  19.1% |
| Camion/Bus | 5 | 707 | 2 |   0.7% |  71.4% |   1.4% |
| VRU (Pedoni/Bici) | 54 | 2320 | 5 |   2.3% |  91.5% |   4.4% |
| Barriera | 22 | 1144 | 20 |   1.9% |  52.4% |   3.6% |
| **MEDIA GLOBALE** | **197** | **5146** | **35** | **  3.7%** | ** 84.9%** | **  7.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 188 | 2309 | 40 |   7.5% |  82.5% |  13.8% |
| Camion/Bus | 6 | 1395 | 7 |   0.4% |  46.2% |   0.8% |
| VRU (Pedoni/Bici) | 102 | 7666 | 9 |   1.3% |  91.9% |   2.6% |
| Barriera | 45 | 4186 | 76 |   1.1% |  37.2% |   2.1% |
| **MEDIA GLOBALE** | **341** | **15556** | **132** | **  2.1%** | ** 72.1%** | **  4.2%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 669 | 422 | 35 |  61.3% |  95.0% |  74.5% |
| Camion/Bus | 164 | 548 | 32 |  23.0% |  83.7% |  36.1% |
| VRU (Pedoni/Bici) | 1306 | 1068 | 40 |  55.0% |  97.0% |  70.2% |
| Barriera | 767 | 399 | 766 |  65.8% |  50.0% |  56.8% |
| **MEDIA GLOBALE** | **2906** | **2437** | **873** | ** 54.4%** | ** 76.9%** | ** 63.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1315 | 1182 | 84 |  52.7% |  94.0% |  67.5% |
| Camion/Bus | 184 | 1217 | 39 |  13.1% |  82.5% |  22.7% |
| VRU (Pedoni/Bici) | 3664 | 4104 | 69 |  47.2% |  98.2% |  63.7% |
| Barriera | 2746 | 1485 | 2136 |  64.9% |  56.2% |  60.3% |
| **MEDIA GLOBALE** | **7909** | **7988** | **2328** | ** 49.8%** | ** 77.3%** | ** 60.5%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 668 | 423 | 42 |  61.2% |  94.1% |  74.2% |
| Camion/Bus | 317 | 395 | 80 |  44.5% |  79.8% |  57.2% |
| VRU (Pedoni/Bici) | 880 | 1494 | 101 |  37.1% |  89.7% |  52.5% |
| Barriera | 181 | 985 | 189 |  15.5% |  48.9% |  23.6% |
| **MEDIA GLOBALE** | **2046** | **3297** | **412** | ** 38.3%** | ** 83.2%** | ** 52.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1449 | 1048 | 129 |  58.0% |  91.8% |  71.1% |
| Camion/Bus | 555 | 846 | 167 |  39.6% |  76.9% |  52.3% |
| VRU (Pedoni/Bici) | 2372 | 5396 | 124 |  30.5% |  95.0% |  46.2% |
| Barriera | 1020 | 3211 | 899 |  24.1% |  53.2% |  33.2% |
| **MEDIA GLOBALE** | **5396** | **10501** | **1319** | ** 33.9%** | ** 80.4%** | ** 47.7%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 797 | 294 | 37 |  73.1% |  95.6% |  82.8% |
| Camion/Bus | 119 | 593 | 9 |  16.7% |  93.0% |  28.3% |
| VRU (Pedoni/Bici) | 1766 | 608 | 88 |  74.4% |  95.3% |  83.5% |
| Barriera | 375 | 791 | 61 |  32.2% |  86.0% |  46.8% |
| **MEDIA GLOBALE** | **3057** | **2286** | **195** | ** 57.2%** | ** 94.0%** | ** 71.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1643 | 854 | 91 |  65.8% |  94.8% |  77.7% |
| Camion/Bus | 150 | 1251 | 15 |  10.7% |  90.9% |  19.2% |
| VRU (Pedoni/Bici) | 5861 | 1907 | 131 |  75.5% |  97.8% |  85.2% |
| Barriera | 1158 | 3073 | 178 |  27.4% |  86.7% |  41.6% |
| **MEDIA GLOBALE** | **8812** | **7085** | **415** | ** 55.4%** | ** 95.5%** | ** 70.1%** |


### Modello: Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 111 | 714 | 13 |  13.5% |  89.5% |  23.4% |
| Camion/Bus | 4 | 527 | 3 |   0.8% |  57.1% |   1.5% |
| VRU (Pedoni/Bici) | 53 | 1303 | 6 |   3.9% |  89.8% |   7.5% |
| Barriera | 23 | 7 | 19 |  76.7% |  54.8% |  63.9% |
| **MEDIA GLOBALE** | **191** | **2551** | **41** | **  7.0%** | ** 82.3%** | ** 12.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 181 | 1832 | 47 |   9.0% |  79.4% |  16.2% |
| Camion/Bus | 5 | 1128 | 8 |   0.4% |  38.5% |   0.9% |
| VRU (Pedoni/Bici) | 98 | 3364 | 13 |   2.8% |  88.3% |   5.5% |
| Barriera | 52 | 34 | 69 |  60.5% |  43.0% |  50.2% |
| **MEDIA GLOBALE** | **336** | **6358** | **137** | **  5.0%** | ** 71.0%** | **  9.4%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 604 | 221 | 100 |  73.2% |  85.8% |  79.0% |
| Camion/Bus | 162 | 369 | 34 |  30.5% |  82.7% |  44.6% |
| VRU (Pedoni/Bici) | 781 | 575 | 565 |  57.6% |  58.0% |  57.8% |
| Barriera | 24 | 6 | 1509 |  80.0% |   1.6% |   3.1% |
| **MEDIA GLOBALE** | **1571** | **1171** | **2208** | ** 57.3%** | ** 41.6%** | ** 48.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1189 | 824 | 210 |  59.1% |  85.0% |  69.7% |
| Camion/Bus | 176 | 957 | 47 |  15.5% |  78.9% |  26.0% |
| VRU (Pedoni/Bici) | 1834 | 1628 | 1899 |  53.0% |  49.1% |  51.0% |
| Barriera | 62 | 24 | 4820 |  72.1% |   1.3% |   2.5% |
| **MEDIA GLOBALE** | **3261** | **3433** | **6976** | ** 48.7%** | ** 31.9%** | ** 38.5%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 596 | 229 | 114 |  72.2% |  83.9% |  77.7% |
| Camion/Bus | 274 | 257 | 123 |  51.6% |  69.0% |  59.1% |
| VRU (Pedoni/Bici) | 756 | 600 | 225 |  55.8% |  77.1% |  64.7% |
| Barriera | 23 | 7 | 347 |  76.7% |   6.2% |  11.5% |
| **MEDIA GLOBALE** | **1649** | **1093** | **809** | ** 60.1%** | ** 67.1%** | ** 63.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1332 | 681 | 246 |  66.2% |  84.4% |  74.2% |
| Camion/Bus | 496 | 637 | 226 |  43.8% |  68.7% |  53.5% |
| VRU (Pedoni/Bici) | 1772 | 1690 | 724 |  51.2% |  71.0% |  59.5% |
| Barriera | 53 | 33 | 1866 |  61.6% |   2.8% |   5.3% |
| **MEDIA GLOBALE** | **3653** | **3041** | **3062** | ** 54.6%** | ** 54.4%** | ** 54.5%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 631 | 194 | 203 |  76.5% |  75.7% |  76.1% |
| Camion/Bus | 94 | 437 | 34 |  17.7% |  73.4% |  28.5% |
| VRU (Pedoni/Bici) | 1061 | 295 | 793 |  78.2% |  57.2% |  66.1% |
| Barriera | 23 | 7 | 413 |  76.7% |   5.3% |   9.9% |
| **MEDIA GLOBALE** | **1809** | **933** | **1443** | ** 66.0%** | ** 55.6%** | ** 60.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1313 | 700 | 421 |  65.2% |  75.7% |  70.1% |
| Camion/Bus | 110 | 1023 | 55 |   9.7% |  66.7% |  16.9% |
| VRU (Pedoni/Bici) | 2676 | 786 | 3316 |  77.3% |  44.7% |  56.6% |
| Barriera | 52 | 34 | 1284 |  60.5% |   3.9% |   7.3% |
| **MEDIA GLOBALE** | **4151** | **2543** | **5076** | ** 62.0%** | ** 45.0%** | ** 52.1%** |
