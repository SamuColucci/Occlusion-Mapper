# Tabella Ufficiale dei Risultati e Guida Metodologica (Tesi di Laurea)

> **Dataset**: nuScenes `v1.0-trainval` · **Valutazione**: split ufficiale `val` (150 scene, 6.019 fotogrammi mai visti in addestramento) · **Raggio operativo**: 25 m · **Ultimo aggiornamento**: 2026-09-29
>
> Le tabelle complete, anche a 20 m, sono generate automaticamente in [`RISULTATI_VALUTAZIONE_VAL.md`](RISULTATI_VALUTAZIONE_VAL.md) da `valutazione/evaluate_final_official.py`. Le figure corrispondenti sono in [`immagini_tesi/`](immagini_tesi/).

Questo documento raccoglie:
1. **Guida Metodologica**: modello `AttentionPerZoneModel`, addestramento, funzione di costo e vincolo neuro-simbolico, come implementati nel codice attuale.
2. **Protocollo di Valutazione**: split, zone analizzate e calibrazione delle soglie.
3. **Matrice dei Risultati**: ogni modello (per GT di addestramento) valutato su ogni GT.
4. **Tabelle Dettagliate per Categoria** a 25 m, per ogni modello e ogni GT.
5. **Analisi Critica**: quale modello è migliore, per quale obiettivo, e con quali limiti.

---

# 🧠 PARTE 1: Come Funziona il Modello, il Training e la Loss

## 1. Il Modello Neurale: `AttentionPerZoneModel`
L'architettura è una rete di **Deep Learning Multimodale** che riceve in ingresso due flussi di informazioni per ogni zona occlusa:
* **Flusso Visivo 2D (11 canali BEV $64 \times 64$)**: griglia LiDAR a terra, sagoma del cono d'ombra (dal Raycaster), maschere semantiche della Mappa HD (asfalto carrabile, marciapiedi, strisce pedonali) e maschere spaziali degli ostacoli visibili.
* **Flusso Scalare Topologico (9 numeri)**:
  $$\text{scalars} = [\text{area}, \text{distanza}, \text{larghezza\_obb}, \text{lunghezza\_obb}, \text{asfalto\_f}, \text{marciapiede\_f}, \text{strisce\_f}, \text{banchina\_f}, \text{terreno\_f}]$$

### 🧩 I 5 Passaggi Chiave dell'Architettura
1. **Channel Attention Iniziale (Squeeze-and-Excitation)**: ricalibra l'importanza degli 11 canali di input con un peso tra 0 e 1 per canale.
2. **Spina Dorsale Convoluzionale Residua (4 Stadi)**: estrae gerarchicamente le feature ($64 \to 32 \to 16 \to 8 \to 4$), ridotte a **128 feature visive**. Le connessioni residue (*shortcut $1 \times 1$*) evitano la dispersione del gradiente.
3. **Modulazione Semantica FiLM (Feature-wise Linear Modulation)**: i 9 scalari, tramite una MLP ($9 \to 64 \to 128$), generano moltiplicatori $\gamma$ e traslazioni $\beta$ che modulano le feature visive:
   $$\text{Vis\_Modulato} = \text{Vis\_Feat} \cdot (1.0 + \tanh(\gamma)) + \beta$$
4. **Testa Decisionale con Dropout (0.25)**: fonde visione e geometria ($128 + 128 = 256 \to 128 \to 64 \to 6$) ed emette 6 logit per le classi `Auto, Camion/Bus, Pedone, Moto, Bici, Barriera`. In valutazione Pedone, Moto e Bici sono raggruppati nella macro-classe **VRU** (massimo delle tre probabilità).
5. **Vincolo Neuro-Simbolico Integrato (Maschera di Compatibilità)**: `build_compatibility_mask()` produce una maschera $M \in \{0, 1\}^6$ e il `forward()` somma $-10000$ ai logit delle classi escluse:
   - **Occludente umano, bici/moto o sagoma stretta** (larghezza $< 0.95\,\text{m}$) $\implies$ ammessi solo Pedone e Bici.
   - **Occludente basso** (altezza $< 1.50\,\text{m}$, non pesante e non manufatto) $\implies$ Camion/Bus escluso: il LiDAR montato a 1.84 m vedrebbe sopra l'occludente.
   - **Zona lontana dalla strada** ($\text{asfalto\_f} < 5\%$ e $\text{banchina\_f} < 15\%$) $\implies$ Auto, Camion/Bus e Moto esclusi.

---

## 2. Il Processo di Addestramento (Train)
* **Dati**: 700 scene di train di `v1.0-trainval` (28.130 fotogrammi, **505.984 zone d'ombra**), salvati su disco in 29 blocchi da 1.000 fotogrammi (patch in float16, ~42 GB per modalità).
* **Ordine**: a ogni epoca i blocchi vengono attraversati tutti in ordine rimescolato e, all'interno di ciascun blocco, i campioni sono mescolati (*shuffle*).
* **Batch**: 64 campioni.
* **Durata**: **5 epoche** su `v1.0-trainval` (20 su `v1.0-mini`).
* **Ottimizzatore**: `AdamW` (learning rate $10^{-3}$, weight decay $10^{-4}$), gradient clipping a norma 1.0.
* **Scheduler**: `CosineAnnealingLR` fino a $10^{-5}$.
* **Modalità di supervisione**:
  - `hybrid`: GT sintetica ibrida spazio-semantica (geometria 3D + affordance HD-Map).
  - `real`: GT reale nuScenes (ostacoli annotati effettivamente presenti nella zona d'ombra).
  - `positives_only`: GT reale, ma solo sulle zone che contengono almeno un ostacolo (**~21%** delle zone di train).

---

## 3. La Funzione di Costo: `Asymmetric Loss (ASL)`
Nelle scene urbane la grande maggioranza delle zone d'ombra è vuota: una BCE simmetrica porterebbe la rete a non segnalare mai pericoli. L'**Asymmetric Loss** (Ridnik et al., ICCV 2021) separa il trattamento di positivi e negativi:

$$\mathcal{L} = - \left[ y \cdot w_{\text{pos}} \cdot (1-p)^{\gamma_{\text{pos}}} \cdot \log(p) \;+\; (1-y) \cdot p_m^{\gamma_{\text{neg}}} \cdot \log(1 - p_m) \right], \quad p_m = \max(p - m, 0)$$

* **Parametri usati in addestramento**: $\gamma_{\text{pos}} = 0.5$, $\gamma_{\text{neg}} = 4.0$, margin shift $m = 0.05$.
* **Pesi dei positivi** $w_{\text{pos}}$ (Auto, Camion/Bus, Pedone, Moto, Bici, Barriera):
  - `hybrid`: $[1.2, 2.2, 2.0, 1.8, 1.8, 3.5]$
  - `real` e `positives_only`: $[1.5, 2.5, 2.5, 2.0, 2.0, 2.5]$
* **Effetto collaterale sulla calibrazione**: attenuando fortemente i negativi facili ($\gamma_{\text{neg}} = 4$) e pesando i positivi, l'ASL spinge le probabilità verso l'alto. Le soglie di decisione vanno quindi **calibrate per modello** (Parte 2).

---

# 📐 PARTE 2: Protocollo di Valutazione

* **Split**: `val` ufficiale nuScenes, 150 scene e **6.019 fotogrammi** mai visti in addestramento.
* **Raggio**: metriche calcolate entro 20 m e 25 m dal veicolo ego (qui si riportano quelle a 25 m).
* **Metriche**: TP, FP, FN, Precision, Recall ed F1 per macro-classe (Auto, Camion/Bus, VRU, Barriera) e in media globale (micro-media sui conteggi).
* **Vincoli d'area in valutazione**: Auto predetta solo in zone $\geq 3.5\,\text{m}^2$, Camion/Bus solo in zone $\geq 8.0\,\text{m}^2$.
* **Target di valutazione**: ogni modello viene confrontato con 4 GT: Reale nuScenes, Sintetica Geometrica, Sintetica Semantica e Sintetica Ibrida.

### Calibrazione delle soglie
Le soglie fisse tarate su `v1.0-mini` (0.25 – 0.28) risultavano inferiori a quasi tutte le probabilità dei modelli addestrati su `v1.0-trainval`: le predizioni coincidevano con la maschera di compatibilità e i tre modelli ottenevano metriche identiche. Le soglie sono state quindi calibrate con `valutazione/calibra_soglie.py`: per ogni modello e macro-classe si sceglie la soglia che massimizza l'F1 rispetto alla **GT di addestramento del modello**, su 6 blocchi della cache di **train** (90.131 zone). Lo split `val` resta così inedito.

| Modello | Auto | Camion/Bus | VRU | Barriera |
| :--- | :---: | :---: | :---: | :---: |
| `hybrid` | 0.63 | 0.64 | 0.69 | 0.72 |
| `real` | 0.60 | 0.58 | 0.58 | 0.62 |
| `positives_only` | 0.78 | 0.66 | 0.74 | 0.82 |

**Capacità di separazione indipendente dalla soglia** (AUC per classe sulla cache di train, rispetto alla GT di addestramento; dettagli in `diagnostica/auc_ap_risultati.txt`): `hybrid` 0.90 – 0.95, `real` 0.83 – 0.92, `positives_only` 0.50 – 0.82.

---

# 📊 PARTE 3: Matrice dei Risultati a 25 Metri (Split VAL)

Righe: GT su cui il modello è stato **addestrato**. Colonne: GT su cui è **valutato**. Ogni cella riporta **F1** (Precision / Recall). In grassetto la diagonale, cioè la valutazione sulla stessa GT dell'addestramento.

| Modello (GT di addestramento) | GT Reale nuScenes | GT Sint. Geometrica | GT Sint. Semantica | GT Sint. Ibrida |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid** (GT Sintetica Ibrida) | 3.1% (1.6 / 49.1) | 58.3% (63.1 / 54.3) | 46.0% (39.5 / 55.0) | **84.0% (84.5 / 83.5)** |
| **Real** (GT Reale completa) | **10.0% (6.1 / 27.1)** | 12.5% (55.9 / 7.0) | 14.7% (43.4 / 8.8) | 15.3% (60.6 / 8.8) |
| **Positives-Only** (GT Reale, solo zone con ostacoli) | **10.5% (6.3 / 32.6)** | 13.8% (53.7 / 7.9) | 18.2% (47.0 / 11.2) | 18.9% (64.9 / 11.0) |

**Non rivalutati su `v1.0-trainval`**: i modelli addestrati su GT Sintetica Geometrica e Semantica (non addestrati) e la Baseline Bayesiana (probabilità precalcolate non generate).

---

# 📋 PARTE 4: Tabelle Dettagliate per Categoria (25 Metri)

## Modello: Attention + GT Sintetica Ibrida (`hybrid`)

#### Valutazione su GT Sintetica Ibrida

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 17182 | 3934 | 1556 |  81.4% |  91.7% |  86.2% |
| Camion/Bus | 1498 | 2254 | 748 |  39.9% |  66.7% |  49.9% |
| VRU (Pedoni/Bici) | 63210 | 5461 | 11853 |  92.0% |  84.2% |  88.0% |
| Barriera | 12645 | 5626 | 4537 |  69.2% |  73.6% |  71.3% |
| **MEDIA GLOBALE** | **94535** | **17275** | **18694** | ** 84.5%** | ** 83.5%** | ** 84.0%** |

#### Valutazione su GT Reale nuScenes

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 646 | 20470 | 329 |   3.1% |  66.3% |   5.8% |
| Camion/Bus | 83 | 3669 | 106 |   2.2% |  43.9% |   4.2% |
| VRU (Pedoni/Bici) | 367 | 68304 | 155 |   0.5% |  70.3% |   1.1% |
| Barriera | 717 | 17554 | 1288 |   3.9% |  35.8% |   7.1% |
| **MEDIA GLOBALE** | **1813** | **109997** | **1878** | **  1.6%** | ** 49.1%** | **  3.1%** |

#### Valutazione su GT Sintetica Geometrica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 12638 | 8478 | 1051 |  59.9% |  92.3% |  72.6% |
| Camion/Bus | 1287 | 2465 | 1315 |  34.3% |  49.5% |  40.5% |
| VRU (Pedoni/Bici) | 40197 | 28474 | 8965 |  58.5% |  81.8% |  68.2% |
| Barriera | 16378 | 1893 | 48031 |  89.6% |  25.4% |  39.6% |
| **MEDIA GLOBALE** | **70500** | **41310** | **59362** | ** 63.1%** | ** 54.3%** | ** 58.3%** |

#### Valutazione su GT Sintetica Semantica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 13021 | 8095 | 2596 |  61.7% |  83.4% |  70.9% |
| Camion/Bus | 1758 | 1994 | 6042 |  46.9% |  22.5% |  30.4% |
| VRU (Pedoni/Bici) | 22284 | 46387 | 6468 |  32.5% |  77.5% |  45.7% |
| Barriera | 7143 | 11128 | 21069 |  39.1% |  25.3% |  30.7% |
| **MEDIA GLOBALE** | **44206** | **67604** | **36175** | ** 39.5%** | ** 55.0%** | ** 46.0%** |

---

## Modello: Attention + GT Reale completa (`real`)

#### Valutazione su GT Reale nuScenes

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 240 | 6553 | 735 |   3.5% |  24.6% |   6.2% |
| Camion/Bus | 58 | 3015 | 131 |   1.9% |  30.7% |   3.6% |
| VRU (Pedoni/Bici) | 130 | 4452 | 392 |   2.8% |  24.9% |   5.1% |
| Barriera | 572 | 1345 | 1433 |  29.8% |  28.5% |  29.2% |
| **MEDIA GLOBALE** | **1000** | **15365** | **2691** | **  6.1%** | ** 27.1%** | ** 10.0%** |

#### Valutazione su GT Sintetica Ibrida

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4475 | 2318 | 14263 |  65.9% |  23.9% |  35.1% |
| Camion/Bus | 767 | 2306 | 1479 |  25.0% |  34.1% |  28.8% |
| VRU (Pedoni/Bici) | 3935 | 647 | 71128 |  85.9% |   5.2% |   9.9% |
| Barriera | 735 | 1182 | 16447 |  38.3% |   4.3% |   7.7% |
| **MEDIA GLOBALE** | **9912** | **6453** | **103317** | ** 60.6%** | **  8.8%** | ** 15.3%** |

#### Valutazione su GT Sintetica Geometrica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3260 | 3533 | 10429 |  48.0% |  23.8% |  31.8% |
| Camion/Bus | 648 | 2425 | 1954 |  21.1% |  24.9% |  22.8% |
| VRU (Pedoni/Bici) | 3459 | 1123 | 45703 |  75.5% |   7.0% |  12.9% |
| Barriera | 1776 | 141 | 62633 |  92.6% |   2.8% |   5.4% |
| **MEDIA GLOBALE** | **9143** | **7222** | **120719** | ** 55.9%** | **  7.0%** | ** 12.5%** |

#### Valutazione su GT Sintetica Semantica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3566 | 3227 | 12051 |  52.5% |  22.8% |  31.8% |
| Camion/Bus | 1254 | 1819 | 6546 |  40.8% |  16.1% |  23.1% |
| VRU (Pedoni/Bici) | 1281 | 3301 | 27471 |  28.0% |   4.5% |   7.7% |
| Barriera | 994 | 923 | 27218 |  51.9% |   3.5% |   6.6% |
| **MEDIA GLOBALE** | **7095** | **9270** | **73286** | ** 43.4%** | **  8.8%** | ** 14.7%** |

---

## Modello: Attention + GT Reale, solo zone con ostacoli (`positives_only`)

#### Valutazione su GT Reale nuScenes

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 285 | 7901 | 690 |   3.5% |  29.2% |   6.2% |
| Camion/Bus | 77 | 1970 | 112 |   3.8% |  40.7% |   6.9% |
| VRU (Pedoni/Bici) | 187 | 7289 | 335 |   2.5% |  35.8% |   4.7% |
| Barriera | 654 | 871 | 1351 |  42.9% |  32.6% |  37.1% |
| **MEDIA GLOBALE** | **1203** | **18031** | **2488** | **  6.3%** | ** 32.6%** | ** 10.5%** |

#### Valutazione su GT Sintetica Ibrida

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4847 | 3339 | 13891 |  59.2% |  25.9% |  36.0% |
| Camion/Bus | 622 | 1425 | 1624 |  30.4% |  27.7% |  29.0% |
| VRU (Pedoni/Bici) | 6217 | 1259 | 68846 |  83.2% |   8.3% |  15.1% |
| Barriera | 800 | 725 | 16382 |  52.5% |   4.7% |   8.6% |
| **MEDIA GLOBALE** | **12486** | **6748** | **100743** | ** 64.9%** | ** 11.0%** | ** 18.9%** |

#### Valutazione su GT Sintetica Geometrica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 3673 | 4513 | 10016 |  44.9% |  26.8% |  33.6% |
| Camion/Bus | 593 | 1454 | 2009 |  29.0% |  22.8% |  25.5% |
| VRU (Pedoni/Bici) | 4711 | 2765 | 44451 |  63.0% |   9.6% |  16.6% |
| Barriera | 1344 | 181 | 63065 |  88.1% |   2.1% |   4.1% |
| **MEDIA GLOBALE** | **10321** | **8913** | **119541** | ** 53.7%** | **  7.9%** | ** 13.8%** |

#### Valutazione su GT Sintetica Semantica

| Categoria Semantica | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Auto | 4249 | 3937 | 11368 |  51.9% |  27.2% |  35.7% |
| Camion/Bus | 996 | 1051 | 6804 |  48.7% |  12.8% |  20.2% |
| VRU (Pedoni/Bici) | 2930 | 4546 | 25822 |  39.2% |  10.2% |  16.2% |
| Barriera | 866 | 659 | 27346 |  56.8% |   3.1% |   5.8% |
| **MEDIA GLOBALE** | **9041** | **10193** | **71340** | ** 47.0%** | ** 11.2%** | ** 18.2%** |

---

# 🎯 PARTE 5: Qual è il Modello Migliore e Perché? (Risposta per la Tesi)

La risposta dipende dall'obiettivo, perché i modelli non imparano lo stesso compito:
- le **GT sintetiche** segnano le zone dove un ostacolo **può fisicamente e legalmente trovarsi** (anticipazione del rischio);
- la **GT reale** segna solo le zone dove nel log nuScenes un ostacolo **c'era davvero** (rilevamento).

### 🥇 Per l'anticipazione del rischio: `hybrid`
1. **F1 84.0%** sulla GT Ibrida con Precision (84.5%) e Recall (83.5%) bilanciate, su 6.019 fotogrammi inediti.
2. **VRU**: F1 88.0% (Precision 92.0%, Recall 84.2%). **Auto**: F1 86.2% (Recall 91.7%).
3. Le classi più difficili restano **Camion/Bus** (F1 49.9%) e **Barriere** (F1 71.3%), comunque molto migliorate rispetto a `v1.0-mini` (14.0% e 31.7%).
4. È anche il modello migliore sulle altre due GT sintetiche (58.3% Geometrica, 46.0% Semantica), molto sopra `real` e `positives_only` (12.5% – 18.2%).

**Cautela interpretativa**: la GT Ibrida è generata da regole che usano la stessa geometria delle ombre e la stessa Mappa HD date in ingresso alla rete. L'84% misura quanto bene la rete **riproduce e generalizza queste regole** su scene nuove, non quanti ostacoli reali intercetta.

### Per il rilevamento degli ostacoli reali: nessun modello è soddisfacente
* `real` e `positives_only` raggiungono **F1 10.0% – 10.5%** sulla GT Reale, contro il 3.1% di `hybrid`.
* `hybrid` intercetta di più (Recall 49.1% contro 27.1%) ma con Precision molto bassa (1.6%): segnala come potenzialmente occupate molte zone che nel log erano vuote. È il comportamento atteso da un modello di anticipazione prudenziale.
* Le cause principali sono:
  - **Classi rarissime**: i VRU reali sono presenti solo nel ~3% delle zone compatibili, per cui anche un buon ordinamento produce Precision bassa (per `real`, AUC 0.83 – 0.92 ma Average Precision VRU 0.02 – 0.24).
  - **Informazione insufficiente in un singolo fotogramma**: dalla forma dell'ombra e dalla mappa si ricava se un ostacolo *può* esserci, non se c'è. È l'*observation bias* di nuScenes: un'area occlusa sul marciapiede è pericolosa a priori anche se in quel decimo di secondo non vi è transitato nessuno.
  - **`positives_only`** non vede mai zone vuote in addestramento e non può imparare a rispondere "nessun ostacolo"; è il modello con la separazione più debole (AUC 0.50 – 0.82).

### Confronto con `v1.0-mini`
Su `mini_val` (81 fotogrammi, soglie fisse 0.25 – 0.28) `hybrid` otteneva F1 66.6% sulla GT Ibrida con Recall 97%. Il confronto non è omogeneo: oltre al dataset sono cambiate le soglie (ora calibrate per modello), e i Recall molto alti su mini erano in parte dovuti a soglie basse che segnalavano quasi tutte le zone compatibili. I risultati su 6.019 fotogrammi sono statisticamente molto più affidabili.

### Possibili miglioramenti
1. Riportare sulla GT Reale curve Precision–Recall e Average Precision, confrontate con un classificatore casuale alla stessa prevalenza.
2. Più epoche per `hybrid` e `real` (la loss di `real` stava ancora scendendo dopo 5 epoche) e pesi di classe ritarati su trainval.
3. Aggiungere a `positives_only` una quota di zone vuote.
4. Introdurre il **contesto temporale** (ostacoli visti nei fotogrammi precedenti ed entrati nell'ombra), la leva principale per migliorare il rilevamento degli ostacoli reali.
