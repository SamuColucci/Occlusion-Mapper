# Report Ufficiale di Valutazione nuScenes - Split VAL

## 🏆 Risultati Valutazione Ufficiale nuScenes [Split: VAL] - 2026-09-28 18:36:23

> **Split analizzato**: `VAL` (Ufficiale nuScenes)

### Tabella Comparativa di Sintesi a 25 Metri (Triplo Confronto)

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes |   6.1% |  27.1% | ** 10.0%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Geometrica 3D) |  55.9% |   7.0% | ** 12.5%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Semantica Naïve) |  43.4% |   8.8% | ** 14.7%** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica (Ibrida Spazio-Sem) |  60.6% |   8.8% | ** 15.3%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Reale nuScenes |   6.3% |  32.6% | ** 10.5%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Geometrica 3D) |  53.7% |   7.9% | ** 13.8%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Semantica Naïve) |  47.0% |  11.2% | ** 18.2%** |
| **Baseline Attention + GT Reale (Positives Only)** | GT Sintetica (Ibrida Spazio-Sem) |  64.9% |  11.0% | ** 18.9%** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes |   1.6% |  49.1% | **  3.1%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Geometrica 3D) |  63.1% |  54.3% | ** 58.3%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Semantica Naïve) |  39.5% |  55.0% | ** 46.0%** |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica (Ibrida Spazio-Sem) |  84.5% |  83.5% | ** 84.0%** |

---


### Modello: Modello Baseline Attention su GT Reale Completa (18k zone nuScenes)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 155 | 3664 | 321 |   4.1% |  32.6% |   7.2% |
| Camion/Bus | 34 | 1770 | 69 |   1.9% |  33.0% |   3.6% |
| VRU (Pedoni/Bici) | 98 | 2705 | 204 |   3.5% |  32.5% |   6.3% |
| Barriera | 432 | 789 | 468 |  35.4% |  48.0% |  40.7% |
| **MEDIA GLOBALE** | **719** | **8928** | **1062** | **  7.5%** | ** 40.4%** | ** 12.6%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 240 | 6553 | 735 |   3.5% |  24.6% |   6.2% |
| Camion/Bus | 58 | 3015 | 131 |   1.9% |  30.7% |   3.6% |
| VRU (Pedoni/Bici) | 130 | 4452 | 392 |   2.8% |  24.9% |   5.1% |
| Barriera | 572 | 1345 | 1433 |  29.8% |  28.5% |  29.2% |
| **MEDIA GLOBALE** | **1000** | **15365** | **2691** | **  6.1%** | ** 27.1%** | ** 10.0%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1977 | 1842 | 4438 |  51.8% |  30.8% |  38.6% |
| Camion/Bus | 590 | 1214 | 1685 |  32.7% |  25.9% |  28.9% |
| VRU (Pedoni/Bici) | 2254 | 549 | 14726 |  80.4% |  13.3% |  22.8% |
| Barriera | 1130 | 91 | 17972 |  92.5% |   5.9% |  11.1% |
| **MEDIA GLOBALE** | **5951** | **3696** | **38821** | ** 61.7%** | ** 13.3%** | ** 21.9%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3260 | 3533 | 10429 |  48.0% |  23.8% |  31.8% |
| Camion/Bus | 648 | 2425 | 1954 |  21.1% |  24.9% |  22.8% |
| VRU (Pedoni/Bici) | 3459 | 1123 | 45703 |  75.5% |   7.0% |  12.9% |
| Barriera | 1776 | 141 | 62633 |  92.6% |   2.8% |   5.4% |
| **MEDIA GLOBALE** | **9143** | **7222** | **120719** | ** 55.9%** | **  7.0%** | ** 12.5%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1985 | 1834 | 4714 |  52.0% |  29.6% |  37.7% |
| Camion/Bus | 739 | 1065 | 3473 |  41.0% |  17.5% |  24.6% |
| VRU (Pedoni/Bici) | 785 | 2018 | 9335 |  28.0% |   7.8% |  12.1% |
| Barriera | 628 | 593 | 5737 |  51.4% |   9.9% |  16.6% |
| **MEDIA GLOBALE** | **4137** | **5510** | **23259** | ** 42.9%** | ** 15.1%** | ** 22.3%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3566 | 3227 | 12051 |  52.5% |  22.8% |  31.8% |
| Camion/Bus | 1254 | 1819 | 6546 |  40.8% |  16.1% |  23.1% |
| VRU (Pedoni/Bici) | 1281 | 3301 | 27471 |  28.0% |   4.5% |   7.7% |
| Barriera | 994 | 923 | 27218 |  51.9% |   3.5% |   6.6% |
| **MEDIA GLOBALE** | **7095** | **9270** | **73286** | ** 43.4%** | **  8.8%** | ** 14.7%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 2688 | 1131 | 5791 |  70.4% |  31.7% |  43.7% |
| Camion/Bus | 671 | 1133 | 1185 |  37.2% |  36.2% |  36.7% |
| VRU (Pedoni/Bici) | 2425 | 378 | 18891 |  86.5% |  11.4% |  20.1% |
| Barriera | 530 | 691 | 4867 |  43.4% |   9.8% |  16.0% |
| **MEDIA GLOBALE** | **6314** | **3333** | **30734** | ** 65.5%** | ** 17.0%** | ** 27.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4475 | 2318 | 14263 |  65.9% |  23.9% |  35.1% |
| Camion/Bus | 767 | 2306 | 1479 |  25.0% |  34.1% |  28.8% |
| VRU (Pedoni/Bici) | 3935 | 647 | 71128 |  85.9% |   5.2% |   9.9% |
| Barriera | 735 | 1182 | 16447 |  38.3% |   4.3% |   7.7% |
| **MEDIA GLOBALE** | **9912** | **6453** | **103317** | ** 60.6%** | **  8.8%** | ** 15.3%** |


### Modello: Modello Baseline Attention su GT Reale Positives-Only (Sole zone con ostacoli)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 145 | 3588 | 331 |   3.9% |  30.5% |   6.9% |
| Camion/Bus | 49 | 1148 | 54 |   4.1% |  47.6% |   7.5% |
| VRU (Pedoni/Bici) | 98 | 3119 | 204 |   3.0% |  32.5% |   5.6% |
| Barriera | 352 | 489 | 548 |  41.9% |  39.1% |  40.4% |
| **MEDIA GLOBALE** | **644** | **8344** | **1137** | **  7.2%** | ** 36.2%** | ** 12.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 285 | 7901 | 690 |   3.5% |  29.2% |   6.2% |
| Camion/Bus | 77 | 1970 | 112 |   3.8% |  40.7% |   6.9% |
| VRU (Pedoni/Bici) | 187 | 7289 | 335 |   2.5% |  35.8% |   4.7% |
| Barriera | 654 | 871 | 1351 |  42.9% |  32.6% |  37.1% |
| **MEDIA GLOBALE** | **1203** | **18031** | **2488** | **  6.3%** | ** 32.6%** | ** 10.5%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1835 | 1898 | 4580 |  49.2% |  28.6% |  36.2% |
| Camion/Bus | 523 | 674 | 1752 |  43.7% |  23.0% |  30.1% |
| VRU (Pedoni/Bici) | 2226 | 991 | 14754 |  69.2% |  13.1% |  22.0% |
| Barriera | 733 | 108 | 18369 |  87.2% |   3.8% |   7.4% |
| **MEDIA GLOBALE** | **5317** | **3671** | **39455** | ** 59.2%** | ** 11.9%** | ** 19.8%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3673 | 4513 | 10016 |  44.9% |  26.8% |  33.6% |
| Camion/Bus | 593 | 1454 | 2009 |  29.0% |  22.8% |  25.5% |
| VRU (Pedoni/Bici) | 4711 | 2765 | 44451 |  63.0% |   9.6% |  16.6% |
| Barriera | 1344 | 181 | 63065 |  88.1% |   2.1% |   4.1% |
| **MEDIA GLOBALE** | **10321** | **8913** | **119541** | ** 53.7%** | **  7.9%** | ** 13.8%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1960 | 1773 | 4739 |  52.5% |  29.3% |  37.6% |
| Camion/Bus | 603 | 594 | 3609 |  50.4% |  14.3% |  22.3% |
| VRU (Pedoni/Bici) | 1249 | 1968 | 8871 |  38.8% |  12.3% |  18.7% |
| Barriera | 446 | 395 | 5919 |  53.0% |   7.0% |  12.4% |
| **MEDIA GLOBALE** | **4258** | **4730** | **23138** | ** 47.4%** | ** 15.5%** | ** 23.4%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4249 | 3937 | 11368 |  51.9% |  27.2% |  35.7% |
| Camion/Bus | 996 | 1051 | 6804 |  48.7% |  12.8% |  20.2% |
| VRU (Pedoni/Bici) | 2930 | 4546 | 25822 |  39.2% |  10.2% |  16.2% |
| Barriera | 866 | 659 | 27346 |  56.8% |   3.1% |   5.8% |
| **MEDIA GLOBALE** | **9041** | **10193** | **71340** | ** 47.0%** | ** 11.2%** | ** 18.2%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 2442 | 1291 | 6037 |  65.4% |  28.8% |  40.0% |
| Camion/Bus | 541 | 656 | 1315 |  45.2% |  29.1% |  35.4% |
| VRU (Pedoni/Bici) | 2584 | 633 | 18732 |  80.3% |  12.1% |  21.1% |
| Barriera | 449 | 392 | 4948 |  53.4% |   8.3% |  14.4% |
| **MEDIA GLOBALE** | **6016** | **2972** | **31032** | ** 66.9%** | ** 16.2%** | ** 26.1%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4847 | 3339 | 13891 |  59.2% |  25.9% |  36.0% |
| Camion/Bus | 622 | 1425 | 1624 |  30.4% |  27.7% |  29.0% |
| VRU (Pedoni/Bici) | 6217 | 1259 | 68846 |  83.2% |   8.3% |  15.1% |
| Barriera | 800 | 725 | 16382 |  52.5% |   4.7% |   8.6% |
| **MEDIA GLOBALE** | **12486** | **6748** | **100743** | ** 64.9%** | ** 11.0%** | ** 18.9%** |


### Modello: Modello Attention su GT Sintetica Ibrida Spazio-Semantica (Spatio-Semantic Affordance)

#### 1. GT Reale nuScenes (3D Reali)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 362 | 9163 | 114 |   3.8% |  76.1% |   7.2% |
| Camion/Bus | 66 | 2577 | 37 |   2.5% |  64.1% |   4.8% |
| VRU (Pedoni/Bici) | 182 | 17246 | 120 |   1.0% |  60.3% |   2.1% |
| Barriera | 410 | 5099 | 490 |   7.4% |  45.6% |  12.8% |
| **MEDIA GLOBALE** | **1020** | **34085** | **761** | **  2.9%** | ** 57.3%** | **  5.5%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 646 | 20470 | 329 |   3.1% |  66.3% |   5.8% |
| Camion/Bus | 83 | 3669 | 106 |   2.2% |  43.9% |   4.2% |
| VRU (Pedoni/Bici) | 367 | 68304 | 155 |   0.5% |  70.3% |   1.1% |
| Barriera | 717 | 17554 | 1288 |   3.9% |  35.8% |   7.1% |
| **MEDIA GLOBALE** | **1813** | **109997** | **1878** | **  1.6%** | ** 49.1%** | **  3.1%** |

#### 2. GT Sintetica Geometrica (Fitting 3D)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 5932 | 3593 | 483 |  62.3% |  92.5% |  74.4% |
| Camion/Bus | 1172 | 1471 | 1103 |  44.3% |  51.5% |  47.7% |
| VRU (Pedoni/Bici) | 12404 | 5024 | 4576 |  71.2% |  73.1% |  72.1% |
| Barriera | 4864 | 645 | 14238 |  88.3% |  25.5% |  39.5% |
| **MEDIA GLOBALE** | **24372** | **10733** | **20400** | ** 69.4%** | ** 54.4%** | ** 61.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 12638 | 8478 | 1051 |  59.9% |  92.3% |  72.6% |
| Camion/Bus | 1287 | 2465 | 1315 |  34.3% |  49.5% |  40.5% |
| VRU (Pedoni/Bici) | 40197 | 28474 | 8965 |  58.5% |  81.8% |  68.2% |
| Barriera | 16378 | 1893 | 48031 |  89.6% |  25.4% |  39.6% |
| **MEDIA GLOBALE** | **70500** | **41310** | **59362** | ** 63.1%** | ** 54.3%** | ** 58.3%** |

#### 3. GT Sintetica Semantica (Regole Naïve)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 5855 | 3670 | 844 |  61.5% |  87.4% |  72.2% |
| Camion/Bus | 1241 | 1402 | 2971 |  47.0% |  29.5% |  36.2% |
| VRU (Pedoni/Bici) | 6997 | 10431 | 3123 |  40.1% |  69.1% |  50.8% |
| Barriera | 1602 | 3907 | 4763 |  29.1% |  25.2% |  27.0% |
| **MEDIA GLOBALE** | **15695** | **19410** | **11701** | ** 44.7%** | ** 57.3%** | ** 50.2%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 13021 | 8095 | 2596 |  61.7% |  83.4% |  70.9% |
| Camion/Bus | 1758 | 1994 | 6042 |  46.9% |  22.5% |  30.4% |
| VRU (Pedoni/Bici) | 22284 | 46387 | 6468 |  32.5% |  77.5% |  45.7% |
| Barriera | 7143 | 11128 | 21069 |  39.1% |  25.3% |  30.7% |
| **MEDIA GLOBALE** | **44206** | **67604** | **36175** | ** 39.5%** | ** 55.0%** | ** 46.0%** |

#### 4. GT Sintetica Ibrida (Spazio-Semantica)

**Raggio 20m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 7807 | 1718 | 672 |  82.0% |  92.1% |  86.7% |
| Camion/Bus | 1299 | 1344 | 557 |  49.1% |  70.0% |  57.7% |
| VRU (Pedoni/Bici) | 15874 | 1554 | 5442 |  91.1% |  74.5% |  81.9% |
| Barriera | 3879 | 1630 | 1518 |  70.4% |  71.9% |  71.1% |
| **MEDIA GLOBALE** | **28859** | **6246** | **8189** | ** 82.2%** | ** 77.9%** | ** 80.0%** |

**Raggio 25m**

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 17182 | 3934 | 1556 |  81.4% |  91.7% |  86.2% |
| Camion/Bus | 1498 | 2254 | 748 |  39.9% |  66.7% |  49.9% |
| VRU (Pedoni/Bici) | 63210 | 5461 | 11853 |  92.0% |  84.2% |  88.0% |
| Barriera | 12645 | 5626 | 4537 |  69.2% |  73.6% |  71.3% |
| **MEDIA GLOBALE** | **94535** | **17275** | **18694** | ** 84.5%** | ** 83.5%** | ** 84.0%** |
