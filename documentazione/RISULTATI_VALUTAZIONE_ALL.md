# Report Ufficiale di Valutazione nuScenes - Split ALL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: ALL] - 2026-09-15 22:46:44

> **Split analizzato**: `ALL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Attention + GT Sintetica (Geometric)** | GT Reale nuScenes |   1.9% |  85.7% | **  3.7%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Geometrica 3D) |  49.8% |  99.0% | ** 66.3%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Semantica Naïve) |  30.2% |  95.8% | ** 45.9%** |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica (Ibrida Spazio-Sem) |  41.9% |  97.4% | ** 58.6%** |
| **Attention + GT Sintetica (Semantic)** | GT Reale nuScenes |   2.3% |  80.4% | **  4.4%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Geometrica 3D) |  47.0% |  74.3% | ** 57.6%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Semantica Naïve) |  38.4% |  96.9% | ** 55.0%** |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica (Ibrida Spazio-Sem) |  42.9% |  79.3% | ** 55.7%** |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes |   4.8% |  83.4% | **  9.1%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Geometrica 3D) |  55.0% |  42.2% | ** 47.7%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Semantica Naïve) |  44.7% |  54.8% | ** 49.2%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Ibrida Spazio-Sem) |  32.8% |  29.5% | ** 31.1%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Reale nuScenes |   2.6% |  87.1% | **  5.0%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Geometrica 3D) |  53.1% |  80.0% | ** 63.8%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Semantica Naïve) |  32.8% |  78.9% | ** 46.3%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Ibrida Spazio-Sem) |  47.8% |  84.2% | ** 60.9%** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   2.0% |  80.4% | **  3.9%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  44.3% |  79.9% | ** 57.0%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  28.9% |  83.2% | ** 42.8%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  46.3% |  97.7% | ** 62.8%** |
| **Baseline Analitica (Bayes)** | GT Reale nuScenes |   5.0% |  77.7% | **  9.4%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Geometrica 3D) |  48.1% |  33.3% | ** 39.3%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Semantica Naïve) |  53.9% |  59.5% | ** 56.6%** |
| **Baseline Analitica (Bayes)** | GT Sintetica (Ibrida Spazio-Sem) |  57.6% |  46.6% | ** 51.5%** |

---


### Modello: Modello Attention su GT Sintetica Geometrica (Spatially-Constrained)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 1317 | 6 |  11.1% |  96.5% |  20.0% |
| Camion/Bus | 5 | 1250 | 2 |   0.4% |  71.4% |   0.8% |
| VRU (Pedoni/Bici) | 60 | 2322 | 4 |   2.5% |  93.8% |   4.9% |
| Barriera | 39 | 2113 | 12 |   1.8% |  76.5% |   3.5% |
| **MEDIA GLOBALE** | **269** | **7002** | **24** | **  3.7%** | ** 91.8%** | **  7.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 235 | 3729 | 22 |   5.9% |  91.4% |  11.1% |
| Camion/Bus | 7 | 2869 | 7 |   0.2% |  50.0% |   0.5% |
| VRU (Pedoni/Bici) | 93 | 7789 | 6 |   1.2% |  93.9% |   2.3% |
| Barriera | 84 | 7139 | 35 |   1.2% |  70.6% |   2.3% |
| **MEDIA GLOBALE** | **419** | **21526** | **70** | **  1.9%** | ** 85.7%** | **  3.7%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 813 | 669 | 6 |  54.9% |  99.3% |  70.7% |
| Camion/Bus | 226 | 1029 | 2 |  18.0% |  99.1% |  30.5% |
| VRU (Pedoni/Bici) | 1389 | 993 | 18 |  58.3% |  98.7% |  73.3% |
| Barriera | 1563 | 589 | 32 |  72.6% |  98.0% |  83.4% |
| **MEDIA GLOBALE** | **3991** | **3280** | **58** | ** 54.9%** | ** 98.6%** | ** 70.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1636 | 2328 | 23 |  41.3% |  98.6% |  58.2% |
| Camion/Bus | 257 | 2619 | 7 |   8.9% |  97.3% |  16.4% |
| VRU (Pedoni/Bici) | 3889 | 3993 | 21 |  49.3% |  99.5% |  66.0% |
| Barriera | 5147 | 2076 | 58 |  71.3% |  98.9% |  82.8% |
| **MEDIA GLOBALE** | **10929** | **11016** | **109** | ** 49.8%** | ** 99.0%** | ** 66.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 822 | 660 | 17 |  55.5% |  98.0% |  70.8% |
| Camion/Bus | 441 | 814 | 16 |  35.1% |  96.5% |  51.5% |
| VRU (Pedoni/Bici) | 723 | 1659 | 71 |  30.4% |  91.1% |  45.5% |
| Barriera | 442 | 1710 | 21 |  20.5% |  95.5% |  33.8% |
| **MEDIA GLOBALE** | **2428** | **4843** | **125** | ** 33.4%** | ** 95.1%** | ** 49.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1843 | 2121 | 63 |  46.5% |  96.7% |  62.8% |
| Camion/Bus | 781 | 2095 | 79 |  27.2% |  90.8% |  41.8% |
| VRU (Pedoni/Bici) | 2000 | 5882 | 99 |  25.4% |  95.3% |  40.1% |
| Barriera | 2002 | 5221 | 47 |  27.7% |  97.7% |  43.2% |
| **MEDIA GLOBALE** | **6626** | **15319** | **288** | ** 30.2%** | ** 95.8%** | ** 45.9%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 967 | 515 | 31 |  65.2% |  96.9% |  78.0% |
| Camion/Bus | 188 | 1067 | 2 |  15.0% |  98.9% |  26.0% |
| VRU (Pedoni/Bici) | 1676 | 706 | 60 |  70.4% |  96.5% |  81.4% |
| Barriera | 347 | 1805 | 15 |  16.1% |  95.9% |  27.6% |
| **MEDIA GLOBALE** | **3178** | **4093** | **108** | ** 43.7%** | ** 96.7%** | ** 60.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1994 | 1970 | 110 |  50.3% |  94.8% |  65.7% |
| Camion/Bus | 215 | 2661 | 7 |   7.5% |  96.8% |  13.9% |
| VRU (Pedoni/Bici) | 5806 | 2076 | 88 |  73.7% |  98.5% |  84.3% |
| Barriera | 1185 | 6038 | 38 |  16.4% |  96.9% |  28.1% |
| **MEDIA GLOBALE** | **9200** | **12745** | **243** | ** 41.9%** | ** 97.4%** | ** 58.6%** |


### Modello: Modello Attention su GT Sintetica Semantica (Semantic-Affordance / Naïve)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 1330 | 6 |  11.0% |  96.5% |  19.8% |
| Camion/Bus | 5 | 1267 | 2 |   0.4% |  71.4% |   0.8% |
| VRU (Pedoni/Bici) | 56 | 1842 | 8 |   3.0% |  87.5% |   5.7% |
| Barriera | 34 | 1255 | 17 |   2.6% |  66.7% |   5.1% |
| **MEDIA GLOBALE** | **260** | **5694** | **33** | **  4.4%** | ** 88.7%** | **  8.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 237 | 3703 | 20 |   6.0% |  92.2% |  11.3% |
| Camion/Bus | 7 | 3113 | 7 |   0.2% |  50.0% |   0.4% |
| VRU (Pedoni/Bici) | 86 | 6233 | 13 |   1.4% |  86.9% |   2.7% |
| Barriera | 63 | 3990 | 56 |   1.6% |  52.9% |   3.0% |
| **MEDIA GLOBALE** | **393** | **17039** | **96** | **  2.3%** | ** 80.4%** | **  4.4%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 809 | 686 | 10 |  54.1% |  98.8% |  69.9% |
| Camion/Bus | 225 | 1047 | 3 |  17.7% |  98.7% |  30.0% |
| VRU (Pedoni/Bici) | 1091 | 807 | 316 |  57.5% |  77.5% |  66.0% |
| Barriera | 959 | 330 | 636 |  74.4% |  60.1% |  66.5% |
| **MEDIA GLOBALE** | **3084** | **2870** | **965** | ** 51.8%** | ** 76.2%** | ** 61.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1634 | 2306 | 25 |  41.5% |  98.5% |  58.4% |
| Camion/Bus | 256 | 2864 | 8 |   8.2% |  97.0% |  15.1% |
| VRU (Pedoni/Bici) | 3116 | 3203 | 794 |  49.3% |  79.7% |  60.9% |
| Barriera | 3194 | 859 | 2011 |  78.8% |  61.4% |  69.0% |
| **MEDIA GLOBALE** | **8200** | **9232** | **2838** | ** 47.0%** | ** 74.3%** | ** 57.6%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 833 | 662 | 6 |  55.7% |  99.3% |  71.4% |
| Camion/Bus | 455 | 817 | 2 |  35.8% |  99.6% |  52.6% |
| VRU (Pedoni/Bici) | 699 | 1199 | 95 |  36.8% |  88.0% |  51.9% |
| Barriera | 435 | 854 | 28 |  33.7% |  94.0% |  49.7% |
| **MEDIA GLOBALE** | **2422** | **3532** | **131** | ** 40.7%** | ** 94.9%** | ** 56.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1886 | 2054 | 20 |  47.9% |  99.0% |  64.5% |
| Camion/Bus | 853 | 2267 | 7 |  27.3% |  99.2% |  42.9% |
| VRU (Pedoni/Bici) | 1994 | 4325 | 105 |  31.6% |  95.0% |  47.4% |
| Barriera | 1966 | 2087 | 83 |  48.5% |  95.9% |  64.4% |
| **MEDIA GLOBALE** | **6699** | **10733** | **215** | ** 38.4%** | ** 96.9%** | ** 55.0%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 953 | 542 | 45 |  63.7% |  95.5% |  76.5% |
| Camion/Bus | 181 | 1091 | 9 |  14.2% |  95.3% |  24.8% |
| VRU (Pedoni/Bici) | 1349 | 549 | 387 |  71.1% |  77.7% |  74.2% |
| Barriera | 231 | 1058 | 131 |  17.9% |  63.8% |  28.0% |
| **MEDIA GLOBALE** | **2714** | **3240** | **572** | ** 45.6%** | ** 82.6%** | ** 58.7%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 2002 | 1938 | 102 |  50.8% |  95.2% |  66.2% |
| Camion/Bus | 208 | 2912 | 14 |   6.7% |  93.7% |  12.4% |
| VRU (Pedoni/Bici) | 4502 | 1817 | 1392 |  71.2% |  76.4% |  73.7% |
| Barriera | 774 | 3279 | 449 |  19.1% |  63.3% |  29.3% |
| **MEDIA GLOBALE** | **7486** | **9946** | **1957** | ** 42.9%** | ** 79.3%** | ** 55.7%** |


### Modello: Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 762 | 6 |  17.8% |  96.5% |  30.1% |
| Camion/Bus | 5 | 691 | 2 |   0.7% |  71.4% |   1.4% |
| VRU (Pedoni/Bici) | 49 | 296 | 15 |  14.2% |  76.6% |  24.0% |
| Barriera | 39 | 1169 | 12 |   3.2% |  76.5% |   6.2% |
| **MEDIA GLOBALE** | **258** | **2918** | **35** | **  8.1%** | ** 88.1%** | ** 14.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 239 | 2122 | 18 |  10.1% |  93.0% |  18.3% |
| Camion/Bus | 7 | 1634 | 7 |   0.4% |  50.0% |   0.8% |
| VRU (Pedoni/Bici) | 78 | 768 | 21 |   9.2% |  78.8% |  16.5% |
| Barriera | 84 | 3541 | 35 |   2.3% |  70.6% |   4.5% |
| **MEDIA GLOBALE** | **408** | **8065** | **81** | **  4.8%** | ** 83.4%** | **  9.1%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 590 | 337 | 229 |  63.6% |  72.0% |  67.6% |
| Camion/Bus | 119 | 577 | 109 |  17.1% |  52.2% |  25.8% |
| VRU (Pedoni/Bici) | 174 | 171 | 1233 |  50.4% |  12.4% |  19.9% |
| Barriera | 934 | 274 | 661 |  77.3% |  58.6% |  66.6% |
| **MEDIA GLOBALE** | **1817** | **1359** | **2232** | ** 57.2%** | ** 44.9%** | ** 50.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1240 | 1121 | 419 |  52.5% |  74.7% |  61.7% |
| Camion/Bus | 133 | 1508 | 131 |   8.1% |  50.4% |  14.0% |
| VRU (Pedoni/Bici) | 376 | 470 | 3534 |  44.4% |   9.6% |  15.8% |
| Barriera | 2908 | 717 | 2297 |  80.2% |  55.9% |  65.9% |
| **MEDIA GLOBALE** | **4657** | **3816** | **6381** | ** 55.0%** | ** 42.2%** | ** 47.7%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 640 | 287 | 199 |  69.0% |  76.3% |  72.5% |
| Camion/Bus | 227 | 469 | 230 |  32.6% |  49.7% |  39.4% |
| VRU (Pedoni/Bici) | 197 | 148 | 597 |  57.1% |  24.8% |  34.6% |
| Barriera | 286 | 922 | 177 |  23.7% |  61.8% |  34.2% |
| **MEDIA GLOBALE** | **1350** | **1826** | **1203** | ** 42.5%** | ** 52.9%** | ** 47.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1497 | 864 | 409 |  63.4% |  78.5% |  70.2% |
| Camion/Bus | 386 | 1255 | 474 |  23.5% |  44.9% |  30.9% |
| VRU (Pedoni/Bici) | 457 | 389 | 1642 |  54.0% |  21.8% |  31.0% |
| Barriera | 1447 | 2178 | 602 |  39.9% |  70.6% |  51.0% |
| **MEDIA GLOBALE** | **3787** | **4686** | **3127** | ** 44.7%** | ** 54.8%** | ** 49.2%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 627 | 300 | 371 |  67.6% |  62.8% |  65.1% |
| Camion/Bus | 104 | 592 | 86 |  14.9% |  54.7% |  23.5% |
| VRU (Pedoni/Bici) | 238 | 107 | 1498 |  69.0% |  13.7% |  22.9% |
| Barriera | 233 | 975 | 129 |  19.3% |  64.4% |  29.7% |
| **MEDIA GLOBALE** | **1202** | **1974** | **2084** | ** 37.8%** | ** 36.6%** | ** 37.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1373 | 988 | 731 |  58.2% |  65.3% |  61.5% |
| Camion/Bus | 120 | 1521 | 102 |   7.3% |  54.1% |  12.9% |
| VRU (Pedoni/Bici) | 597 | 249 | 5297 |  70.6% |  10.1% |  17.7% |
| Barriera | 692 | 2933 | 531 |  19.1% |  56.6% |  28.5% |
| **MEDIA GLOBALE** | **2782** | **5691** | **6661** | ** 32.8%** | ** 29.5%** | ** 31.1%** |


### Modello: Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 912 | 6 |  15.3% |  96.5% |  26.4% |
| Camion/Bus | 5 | 470 | 2 |   1.1% |  71.4% |   2.1% |
| VRU (Pedoni/Bici) | 64 | 2279 | 0 |   2.7% | 100.0% |   5.3% |
| Barriera | 39 | 1735 | 12 |   2.2% |  76.5% |   4.3% |
| **MEDIA GLOBALE** | **273** | **5396** | **20** | **  4.8%** | ** 93.2%** | **  9.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 238 | 2757 | 19 |   7.9% |  92.6% |  14.6% |
| Camion/Bus | 7 | 1055 | 7 |   0.7% |  50.0% |   1.3% |
| VRU (Pedoni/Bici) | 97 | 7198 | 2 |   1.3% |  98.0% |   2.6% |
| Barriera | 84 | 5205 | 35 |   1.6% |  70.6% |   3.1% |
| **MEDIA GLOBALE** | **426** | **16215** | **63** | **  2.6%** | ** 87.1%** | **  5.0%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 642 | 435 | 177 |  59.6% |  78.4% |  67.7% |
| Camion/Bus | 104 | 371 | 124 |  21.9% |  45.6% |  29.6% |
| VRU (Pedoni/Bici) | 1346 | 997 | 61 |  57.4% |  95.7% |  71.8% |
| Barriera | 1276 | 498 | 319 |  71.9% |  80.0% |  75.7% |
| **MEDIA GLOBALE** | **3368** | **2301** | **681** | ** 59.4%** | ** 83.2%** | ** 69.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1379 | 1616 | 280 |  46.0% |  83.1% |  59.3% |
| Camion/Bus | 115 | 947 | 149 |  10.8% |  43.6% |  17.3% |
| VRU (Pedoni/Bici) | 3582 | 3713 | 328 |  49.1% |  91.6% |  63.9% |
| Barriera | 3756 | 1533 | 1449 |  71.0% |  72.2% |  71.6% |
| **MEDIA GLOBALE** | **8832** | **7809** | **2206** | ** 53.1%** | ** 80.0%** | ** 63.8%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 673 | 404 | 166 |  62.5% |  80.2% |  70.3% |
| Camion/Bus | 208 | 267 | 249 |  43.8% |  45.5% |  44.6% |
| VRU (Pedoni/Bici) | 776 | 1567 | 18 |  33.1% |  97.7% |  49.5% |
| Barriera | 366 | 1408 | 97 |  20.6% |  79.0% |  32.7% |
| **MEDIA GLOBALE** | **2023** | **3646** | **530** | ** 35.7%** | ** 79.2%** | ** 49.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1623 | 1372 | 283 |  54.2% |  85.2% |  66.2% |
| Camion/Bus | 366 | 696 | 494 |  34.5% |  42.6% |  38.1% |
| VRU (Pedoni/Bici) | 1895 | 5400 | 204 |  26.0% |  90.3% |  40.3% |
| Barriera | 1572 | 3717 | 477 |  29.7% |  76.7% |  42.8% |
| **MEDIA GLOBALE** | **5456** | **11185** | **1458** | ** 32.8%** | ** 78.9%** | ** 46.3%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 717 | 360 | 281 |  66.6% |  71.8% |  69.1% |
| Camion/Bus | 86 | 389 | 104 |  18.1% |  45.3% |  25.9% |
| VRU (Pedoni/Bici) | 1661 | 682 | 75 |  70.9% |  95.7% |  81.4% |
| Barriera | 278 | 1496 | 84 |  15.7% |  76.8% |  26.0% |
| **MEDIA GLOBALE** | **2742** | **2927** | **544** | ** 48.4%** | ** 83.4%** | ** 61.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1579 | 1416 | 525 |  52.7% |  75.0% |  61.9% |
| Camion/Bus | 98 | 964 | 124 |   9.2% |  44.1% |  15.3% |
| VRU (Pedoni/Bici) | 5421 | 1874 | 473 |  74.3% |  92.0% |  82.2% |
| Barriera | 851 | 4438 | 372 |  16.1% |  69.6% |  26.1% |
| **MEDIA GLOBALE** | **7949** | **8692** | **1494** | ** 47.8%** | ** 84.2%** | ** 60.9%** |


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 165 | 1408 | 6 |  10.5% |  96.5% |  18.9% |
| Camion/Bus | 5 | 1171 | 2 |   0.4% |  71.4% |   0.8% |
| VRU (Pedoni/Bici) | 61 | 2312 | 3 |   2.6% |  95.3% |   5.0% |
| Barriera | 29 | 1204 | 22 |   2.4% |  56.9% |   4.5% |
| **MEDIA GLOBALE** | **260** | **6095** | **33** | **  4.1%** | ** 88.7%** | **  7.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 235 | 4063 | 22 |   5.5% |  91.4% |  10.3% |
| Camion/Bus | 7 | 2893 | 7 |   0.2% |  50.0% |   0.5% |
| VRU (Pedoni/Bici) | 93 | 7768 | 6 |   1.2% |  93.9% |   2.3% |
| Barriera | 58 | 4806 | 61 |   1.2% |  48.7% |   2.3% |
| **MEDIA GLOBALE** | **393** | **19530** | **96** | **  2.0%** | ** 80.4%** | **  3.9%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 813 | 760 | 6 |  51.7% |  99.3% |  68.0% |
| Camion/Bus | 223 | 953 | 5 |  19.0% |  97.8% |  31.8% |
| VRU (Pedoni/Bici) | 1387 | 986 | 20 |  58.4% |  98.6% |  73.4% |
| Barriera | 791 | 442 | 804 |  64.2% |  49.6% |  55.9% |
| **MEDIA GLOBALE** | **3214** | **3141** | **835** | ** 50.6%** | ** 79.4%** | ** 61.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1637 | 2661 | 22 |  38.1% |  98.7% |  55.0% |
| Camion/Bus | 252 | 2648 | 12 |   8.7% |  95.5% |  15.9% |
| VRU (Pedoni/Bici) | 3855 | 4006 | 55 |  49.0% |  98.6% |  65.5% |
| Barriera | 3079 | 1785 | 2126 |  63.3% |  59.2% |  61.2% |
| **MEDIA GLOBALE** | **8823** | **11100** | **2215** | ** 44.3%** | ** 79.9%** | ** 57.0%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 829 | 744 | 10 |  52.7% |  98.8% |  68.7% |
| Camion/Bus | 392 | 784 | 65 |  33.3% |  85.8% |  48.0% |
| VRU (Pedoni/Bici) | 716 | 1657 | 78 |  30.2% |  90.2% |  45.2% |
| Barriera | 196 | 1037 | 267 |  15.9% |  42.3% |  23.1% |
| **MEDIA GLOBALE** | **2133** | **4222** | **420** | ** 33.6%** | ** 83.5%** | ** 47.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1875 | 2423 | 31 |  43.6% |  98.4% |  60.4% |
| Camion/Bus | 755 | 2145 | 105 |  26.0% |  87.8% |  40.2% |
| VRU (Pedoni/Bici) | 2006 | 5855 | 93 |  25.5% |  95.6% |  40.3% |
| Barriera | 1113 | 3751 | 936 |  22.9% |  54.3% |  32.2% |
| **MEDIA GLOBALE** | **5749** | **14174** | **1165** | ** 28.9%** | ** 83.2%** | ** 42.8%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 990 | 583 | 8 |  62.9% |  99.2% |  77.0% |
| Camion/Bus | 187 | 989 | 3 |  15.9% |  98.4% |  27.4% |
| VRU (Pedoni/Bici) | 1683 | 690 | 53 |  70.9% |  96.9% |  81.9% |
| Barriera | 322 | 911 | 40 |  26.1% |  89.0% |  40.4% |
| **MEDIA GLOBALE** | **3182** | **3173** | **104** | ** 50.1%** | ** 96.8%** | ** 66.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 2078 | 2220 | 26 |  48.3% |  98.8% |  64.9% |
| Camion/Bus | 214 | 2686 | 8 |   7.4% |  96.4% |  13.7% |
| VRU (Pedoni/Bici) | 5796 | 2065 | 98 |  73.7% |  98.3% |  84.3% |
| Barriera | 1138 | 3726 | 85 |  23.4% |  93.0% |  37.4% |
| **MEDIA GLOBALE** | **9226** | **10697** | **217** | ** 46.3%** | ** 97.7%** | ** 62.8%** |


### Modello: Baseline Analitica: Probabilità Condizionata Bayesiana (No Rete Neurale)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 160 | 841 | 11 |  16.0% |  93.6% |  27.3% |
| Camion/Bus | 4 | 670 | 3 |   0.6% |  57.1% |   1.2% |
| VRU (Pedoni/Bici) | 58 | 1270 | 6 |   4.4% |  90.6% |   8.3% |
| Barriera | 28 | 7 | 23 |  80.0% |  54.9% |  65.1% |
| **MEDIA GLOBALE** | **250** | **2788** | **43** | **  8.2%** | ** 85.3%** | ** 15.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 228 | 2363 | 29 |   8.8% |  88.7% |  16.0% |
| Camion/Bus | 6 | 1487 | 8 |   0.4% |  42.9% |   0.8% |
| VRU (Pedoni/Bici) | 91 | 3371 | 8 |   2.6% |  91.9% |   5.1% |
| Barriera | 55 | 31 | 64 |  64.0% |  46.2% |  53.7% |
| **MEDIA GLOBALE** | **380** | **7252** | **109** | **  5.0%** | ** 77.7%** | **  9.4%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 764 | 237 | 55 |  76.3% |  93.3% |  84.0% |
| Camion/Bus | 208 | 466 | 20 |  30.9% |  91.2% |  46.1% |
| VRU (Pedoni/Bici) | 764 | 564 | 643 |  57.5% |  54.3% |  55.9% |
| Barriera | 30 | 5 | 1565 |  85.7% |   1.9% |   3.7% |
| **MEDIA GLOBALE** | **1766** | **1272** | **2283** | ** 58.1%** | ** 43.6%** | ** 49.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1529 | 1062 | 130 |  59.0% |  92.2% |  72.0% |
| Camion/Bus | 234 | 1259 | 30 |  15.7% |  88.6% |  26.6% |
| VRU (Pedoni/Bici) | 1842 | 1620 | 2068 |  53.2% |  47.1% |  50.0% |
| Barriera | 67 | 19 | 5138 |  77.9% |   1.3% |   2.5% |
| **MEDIA GLOBALE** | **3672** | **3960** | **7366** | ** 48.1%** | ** 33.3%** | ** 39.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 779 | 222 | 60 |  77.8% |  92.8% |  84.7% |
| Camion/Bus | 352 | 322 | 105 |  52.2% |  77.0% |  62.2% |
| VRU (Pedoni/Bici) | 689 | 639 | 105 |  51.9% |  86.8% |  64.9% |
| Barriera | 28 | 7 | 435 |  80.0% |   6.0% |  11.2% |
| **MEDIA GLOBALE** | **1848** | **1190** | **705** | ** 60.8%** | ** 72.4%** | ** 66.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1753 | 838 | 153 |  67.7% |  92.0% |  78.0% |
| Camion/Bus | 654 | 839 | 206 |  43.8% |  76.0% |  55.6% |
| VRU (Pedoni/Bici) | 1655 | 1807 | 444 |  47.8% |  78.8% |  59.5% |
| Barriera | 55 | 31 | 1994 |  64.0% |   2.7% |   5.2% |
| **MEDIA GLOBALE** | **4117** | **3515** | **2797** | ** 53.9%** | ** 59.5%** | ** 56.6%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 802 | 199 | 196 |  80.1% |  80.4% |  80.2% |
| Camion/Bus | 131 | 543 | 59 |  19.4% |  68.9% |  30.3% |
| VRU (Pedoni/Bici) | 976 | 352 | 760 |  73.5% |  56.2% |  63.7% |
| Barriera | 28 | 7 | 334 |  80.0% |   7.7% |  14.1% |
| **MEDIA GLOBALE** | **1937** | **1101** | **1349** | ** 63.8%** | ** 58.9%** | ** 61.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1682 | 909 | 422 |  64.9% |  79.9% |  71.7% |
| Camion/Bus | 144 | 1349 | 78 |   9.6% |  64.9% |  16.8% |
| VRU (Pedoni/Bici) | 2514 | 948 | 3380 |  72.6% |  42.7% |  53.7% |
| Barriera | 57 | 29 | 1166 |  66.3% |   4.7% |   8.7% |
| **MEDIA GLOBALE** | **4397** | **3235** | **5046** | ** 57.6%** | ** 46.6%** | ** 51.5%** |
