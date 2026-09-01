# Tabella Ufficiale dei Risultati Certificati (Tesi di Laurea)

Questo documento raccoglie tutti i risultati sperimentali definitivi calcolati sull'intero dataset **nuScenes (404 fotogrammi, oltre 18.000 zone d'ombra)**:
* **Architettura Principale**: `AttentionPerZoneModel` (Backbone Residuo con Squeeze-and-Excitation Channel Attention + Modulazione FiLM con Mappa HD + LayerNorm).
* **Loss**: `AsymmetricLoss` ($\gamma_+ = 1.0, \gamma_- = 4.0, \text{clip} = 0.05$).

---

## 1. Modello Ufficiale (Addestrato con GT Sintetica Neurosimbolica): Valutazione su GT Sintetica
> **Scopo**: Valuta la capacita predittiva sia sugli ostacoli reali annotati sia sui **varchi plausibili del codice stradale** (affordance pedonale su marciapiedi/strisce e sagome veicolari su asfalto).

### A. Raggio Operativo Standard: 20 Metri
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 2.354 | 585 | 14 | 80.1% | 99.4% | **88.7%** |
| Camion / Bus | 1.159 | 402 | 0 | 74.2% | 100.0% | **85.2%** |
| VRU (Pedoni / Ciclisti) | 1.458 | 711 | 0 | 67.2% | 100.0% | **80.4%** |
| Barriere / Muri | 1.041 | 580 | 0 | 64.2% | 100.0% | **78.2%** |
| **MEDIA GLOBALE (Macro)** | **6.012** | **2.278** | **14** | **72.5%** | **99.8%** | **84.0%** |

### B. Raggio Esteso: 25 Metri
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3.495 | 877 | 23 | 79.9% | 99.3% | **88.6%** |
| Camion / Bus | 1.751 | 617 | 0 | 73.9% | 100.0% | **85.0%** |
| VRU (Pedoni / Ciclisti) | 2.237 | 1.053 | 0 | 68.0% | 100.0% | **80.9%** |
| Barriere / Muri | 2.007 | 899 | 0 | 69.1% | 100.0% | **81.7%** |
| **MEDIA GLOBALE (Macro)** | **9.490** | **3.446** | **23** | **73.4%** | **99.8%** | **84.5%** |

---

## 2. Modello Ufficiale (Addestrato con GT Sintetica Neurosimbolica): Valutazione su GT Reale 3D nuScenes
> **Scopo**: Valuta l'accuratezza stretta sugli ostacoli fisici reali visibili nelle telecamere e annotati dagli operatori umani.

### A. Raggio Operativo Standard: 20 Metri
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.362 | 1.577 | 0 | 46.3% | 100.0% | **63.3%** |
| VRU (Pedoni / Ciclisti) | 687 | 1.482 | 0 | 31.7% | 100.0% | **48.1%** |
| Barriere / Muri | 509 | 1.112 | 0 | 31.4% | 100.0% | **47.8%** |
| Camion / Bus | 427 | 1.134 | 0 | 27.4% | 100.0% | **43.0%** |
| **MEDIA GLOBALE (Macro)** | **2.985** | **5.305** | **0** | **36.0%** | **100.0%** | **52.9%** |

### B. Raggio Esteso: 25 Metri
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.894 | 2.478 | 0 | 43.3% | 100.0% | **60.5%** |
| VRU (Pedoni / Ciclisti) | 957 | 2.333 | 0 | 29.1% | 100.0% | **45.1%** |
| Barriere / Muri | 665 | 2.241 | 0 | 22.9% | 100.0% | **37.2%** |
| Camion / Bus | 588 | 1.780 | 0 | 24.8% | 100.0% | **39.8%** |
| **MEDIA GLOBALE (Macro)** | **4.104** | **8.832** | **0** | **31.7%** | **100.0%** | **48.2%** |

---

## 3. Confronto Architetturale: Baseline CNN vs Modello Finale ad Attenzione (a 25m)

| Metrica Chiave | Modello Base (CNN Semplice + Late Concat) | Modello Avanzato (SE-Attention + FiLM Modulation) | Delta di Miglioramento |
| :--- | :---: | :---: | :--- |
| Precision (GT Sintetica) | 30.0% | **73.4%** | +43.4 punti percentuali |
| Recall (GT Sintetica) | 75.9% | **99.8%** | +23.9 punti percentuali |
| F1-Score (GT Sintetica) | 42.7% | **84.5%** | F1 Raddoppiato (+41.8%) |
| IoU (Intersezione su Unione) | 27.9% | **73.4%** | Quasi Triplicato (+45.5%) |
| F1-Score (GT Reale) | 25.4% | **48.2%** | Quasi Raddoppiato (+22.8%) |
| Latenza di Inferenza | 0.8 ms / ombra | **1.2 ms / ombra** | Pienamente Real-Time (> 100 FPS) |

---

## 4. Studio di Ablazione sulla Supervisione: Addestramento con GT Reale vs Addestramento con GT Neurosimbolica (a 25m)

### A. Modello Baseline Addestrato con Sola GT Reale nuScenes

#### 1. Valutato su GT Reale nuScenes (25m):
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.878 | 384 | 16 | 83.0% | 99.2% | **90.4%** |
| Camion / Bus | 586 | 205 | 2 | 74.1% | 99.7% | **85.0%** |
| VRU (Pedoni / Ciclisti) | 925 | 244 | 32 | 79.1% | 96.7% | **87.0%** |
| Barriere / Muri | 663 | 104 | 2 | 86.4% | 99.7% | **92.6%** |
| **MEDIA GLOBALE (Macro)** | **4.052** | **937** | **52** | **81.2%** | **98.7%** | **89.1%** |

#### 2. Valutato su GT Sintetica Neurosimbolica (25m):
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 1.959 | 303 | 1.559 | 86.6% | 55.7% | **67.8%** |
| Camion / Bus | 604 | 187 | 1.147 | 76.4% | 34.5% | **47.5%** |
| VRU (Pedoni / Ciclisti) | 978 | 191 | 1.259 | 83.7% | 43.7% | **57.4%** |
| Barriere / Muri | 677 | 90 | 1.330 | 88.3% | 33.7% | **48.8%** |
| **MEDIA GLOBALE (Macro)** | **4.218** | **771** | **5.295** | **84.5%** | **44.3%** | **58.2%** |

---

### B. Tabella Riassuntiva di Confronto (Macro a 25m)

| Addestramento Effettuato con | Valutazione Test su | Precision (%) | Recall (%) | F1-Score (%) | Falsi Negativi (Pericoli Persi) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ground Truth Reale nuScenes** *(Baseline)* | GT Reale nuScenes | 81.2% | 98.7% | 89.1% | 52 |
| **Ground Truth Reale nuScenes** *(Baseline)* | GT Sintetica Neurosimbolica | 84.5% | 44.3% | 58.2% | **5.295** *(Crollo sicurezza)* |
| **Ground Truth Sintetica Neurosimbolica** *(Ufficiale)* | GT Reale nuScenes | 31.7% | **100.0%** | 48.2% | **0** *(Zero ostacoli reali persi)* |
| **Ground Truth Sintetica Neurosimbolica** *(Ufficiale)* | GT Sintetica Neurosimbolica | 73.4% | **99.8%** | **84.5%** | **23** *(Massima protezione)* |

* **Evidenze Scientifiche**:
  * L'addestramento su **Ground Truth Reale** costringe la rete a sopprimere le predizioni sui punti ciechi privi di ostacoli visibili, causando la perdita del **55.7%** dei varchi di pericolo potenziale (5.295 pericoli persi su marciapiedi e incroci).
  * L'addestramento su **Ground Truth Sintetica Neurosimbolica** conferisce alla rete la corretta anticipazione del rischio (**Recall 99.8%**) preservando la totale sicurezza rispetto agli ostacoli fisici reali (**Recall 100.0%**, 0 ostacoli persi).