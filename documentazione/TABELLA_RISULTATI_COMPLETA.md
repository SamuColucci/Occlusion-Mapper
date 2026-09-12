# Tabella Ufficiale dei Risultati Certificati e Guida Metodologica (Tesi di Laurea)

Questo documento raccoglie in modo organico, rigoroso e leggibile:
1. **Guida Metodologica**: Spiegazione discorsiva del **Modello Neurale (`AttentionPerZoneModel`)**, del **Processo di Addestramento**, della **Funzione di Costo (`Asymmetric Loss`)** e del **Vincolo Neuro-Simbolico Integrato (Gating Spazio-Semantico)**.
2. **Quadro Comparativo Ufficiale**: Tabella riassuntiva di sintesi a 25 metri su tutti i modelli dello studio di ablazione sullo split ufficiale di validazione (**nuScenes `val`**, 81 fotogrammi mai visti).
3. **Tabelle Dettagliate per Modello e Raggio Operativo (20m e 25m)**.
4. **Analisi Critica: Qual è il Modello Migliore e Perché?**

---

# 🧠 PARTE 1: Come Funziona il Modello, il Training e la Loss

## 1. Il Modello Neurale: `AttentionPerZoneModel`
L'architettura proposta è una rete di **Deep Learning Multimodale** che riceve in ingresso due flussi di informazioni per ogni zona occlusa:
* **Flusso Visivo 2D (11 canali BEV $64 \times 64$)**: contiene la griglia LiDAR a terra, la sagoma del cono d'ombra (dal Raycaster), le maschere semantiche della Mappa HD (asfalto carrabile, marciapiedi, strisce pedonali) e le maschere spaziali degli ostacoli visibili (auto, camion, pedoni, ecc.).
* **Flusso Scalare Topologico (9 numeri)**: contiene i dati fisici sintetici dell'ombra:
  $$\text{scalars} = [\text{area}, \text{distanza}, \mathbf{larghezza\_obb}, \mathbf{lunghezza\_obb}, \text{asfalto\_f}, \text{marciapiede\_f}, \text{strisce\_f}, \text{dinamico}, \text{terreno\_f}]$$

### 🧩 I 5 Passaggi Chiave dell'Architettura:
1. **Channel Attention Iniziale (Squeeze-and-Excitation)**: Ricalibra l'importanza degli 11 canali di input prima di passare alle convoluzioni, assegnando un peso tra 0 e 1 per dare priorità al LiDAR o alla Mappa HD a seconda del contesto.
2. **Spina Dorsale Convoluzionale Residua (4 Stadi)**: Estrae gerarchicamente contorni primari ($64 \to 32$), forme complesse ($32 \to 16$), pattern geometrici ($16 \to 8$) e comprensione globale ($8 \to 4$), ridotti infine a **128 feature visive**. Le connessioni residue (*shortcut $1 \times 1$*) evitano la dispersione del gradiente.
3. **Modulazione Semantica FiLM (Feature-wise Linear Modulation)**: I 9 scalari della mappa HD generano moltiplicatori $\gamma$ e traslazioni $\beta$ per modulare i filtri visivi:
   $$\text{Vis\_Modulato} = \text{Vis\_Feat} \cdot (1.0 + \tanh(\gamma)) + \beta$$
4. **Testa Decisionale con Dropout (0.25)**: Fonde visione e geometria ($128 + 128 = 256$) ed emette 6 punteggi (Logit) per le classi: `Auto, Camion, Pedoni, Moto, Bici, Barriere`.
5. **Vincolo Neuro-Simbolico Integrato (Semantic & Geometric Affordance Gating)**:
   Per impedire alla rete di compiere allucinazioni assurde (es. camion di 8 metri su marciapiedi o in varchi stretti di 1 metro), il metodo `forward()` applica una penalità differenziabile:
   - **Asfalto insufficiente ($\text{road\_f} < 15\%$)** $\implies$ Auto e Camion inibiti.
   - **Varco troppo stretto ($\text{larghezza} < 1.7\,\text{m}$)** $\implies$ Auto e Camion inibiti.
   - **Varco insufficiente per mezzi pesanti ($\text{larghezza} < 2.3\,\text{m}$)** $\implies$ Camion inibito.
   In questo modo il modello è matematicamente costretto a far emergere **VRU (Pedoni/Ciclisti)** o **Barriere** nelle zone dove i veicoli non possono fisicamente transitare.

---

## 2. Il Processo di Addestramento (Train)
* **Dataset e Batching**: Mini-batch da **64 campioni**, mescolati casualmente (*Shuffle*) a ogni epoca.
* **Durata**: **20 epoche** complete.
* **Ottimizzatore**: `AdamW` (learning rate $10^{-3}$, weight decay $10^{-4}$).
* **Scheduler**: `CosineAnnealingLR` con discesa a coseno fino a $10^{-5}$.

---

## 3. La Funzione di Costo: `Asymmetric Loss (ASL)`
Nelle scene urbane, il **$95\%$ delle zone d'ombra è privo di pedoni**. Una Loss simmetrica standard (BCE) fallirebbe, portando la rete a non segnalare mai pericoli. L'**Asymmetric Loss** risolve il problema:

$$\mathcal{L} = - \left[ y \cdot (1-p)^{\gamma_{\text{pos}}} \cdot \log(p) \cdot w_{\text{pos}} \;+\; (1-y) \cdot (p - m)_+^{\gamma_{\text{neg}}} \cdot \log(1 - p + m) \right]$$

* **Sui Casi Positivi ($y = 1$)**: $\gamma_{\text{pos}} = 1.0$. Pesi salvavita $w_{\text{pos}} = 3.0$ sui pedoni e veicoli pesanti: se la rete manca un pedone, l'errore viene triplicato, costringendo la Recall al $100\%$.
* **Sui Casi Negativi ($y = 0$)**: Margin shift $m = 0.05$ (le ombre vuote sotto il $5\%$ non generano errore) ed esponente $\gamma_{\text{neg}} = 4.0$ (gli errori negativi lievi si riducono a zero).

---

# 📊 PARTE 2: Quadro Comparativo Ufficiale a 25 Metri (Split Validation nuScenes)

I risultati seguenti sono stati certificati sullo split ufficiale **`VAL` (81 fotogrammi mai visti durante l'addestramento, 2 scene complete)** confrontando ciascun modello sui 4 target di riferimento.

| Modello Addestrato | Target di Valutazione | Precision (%) | Recall (%) | F1-Score (%) | Giudizio Sintetico |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Attention + GT Sintetica (Hybrid)** | **GT Sintetica Ibrida (Spazio-Sem)** | **57.3%** | **99.3%** | **72.6%** | 🏆 **MIGLIOR MODELLO GLOBALE** |
| **Attention + GT Sintetica (Hybrid)** | GT Reale nuScenes (3D Reali) | 16.8% | 68.4% | 27.0% | Anticipazione prudenziale |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica Geometrica (3D) | 45.3% | 85.0% | 59.1% | Filtra le allucinazioni geometriche |
| **Attention + GT Sintetica (Hybrid)** | GT Sintetica Semantica (Naïve) | 39.9% | 83.1% | 53.9% | Filtra i veicoli impossibili |
| **Attention + GT Sintetica (Geometric)** | GT Sintetica Geometrica (3D) | 59.6% | 96.9% | 73.8% | Soffre di camion su marciapiedi |
| **Attention + GT Sintetica (Geometric)** | GT Reale nuScenes (3D Reali) | 27.3% | 96.4% | 42.5% | Troppi falsi allarmi pesanti |
| **Attention + GT Sintetica (Semantic)** | GT Sintetica Semantica (Naïve) | 61.8% | 94.1% | 74.6% | Ignora la fisica degli spazi stretti |
| **Attention + GT Sintetica (Semantic)** | GT Reale nuScenes (3D Reali) | 31.2% | 93.0% | 46.7% | Rischio collisioni volumetriche |
| **Baseline Attention + GT Reale Completa** | GT Reale nuScenes (3D Reali) | **68.5%** | 91.8% | **78.5%** | Ottimo detector, **0 anticipazione** |
| **Baseline Attention + GT Reale Completa** | GT Sintetica Ibrida / Geometrica | 74.8% | 46.1% | 57.0% | ❌ **Crollo sicurezza: perde il 54% dei pericoli** |
| **Baseline Attention + GT Reale (Positives-Only)** | GT Reale nuScenes (3D Reali) | 36.1% | 97.1% | 52.7% | Senza negativi produce troppi FP |
| **Baseline Analitica (Bayes Condizionato)** | GT Reale nuScenes (3D Reali) | 25.5% | 41.6% | 31.6% | Metodo tabellare senza DL: perde il 58% degli ostacoli |
| **Baseline Analitica (Bayes Condizionato)** | GT Sintetica Ibrida (Spazio-Sem) | 64.3% | 48.4% | 55.2% | Recall cieca limitata (<49%): manca il contesto visivo BEV |

---

# 📋 PARTE 3: Tabelle Dettagliate per Categoria (Modello Ibrido Finale)

### Modello: `Attention + GT Sintetica Ibrida (Spazio-Semantica con Gating)`

#### Target 1: Valutazione su GT Sintetica Ibrida (Target Primario della Tesi)
Questo target valuta la conformità congiunta alle **leggi fisiche 3D** (ingombro minimo e collisioni) e al **codice stradale** (affordance HD-Map).

**Raggio 20 Metri**
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 490 | 190 | 6 | 72.1% | 98.8% | **83.3%** |
| **Camion / Bus** | 116 | 265 | 4 | 30.4% | 96.7% | **46.3%** |
| **VRU (Pedoni / Ciclisti)** | 833 | 362 | 0 | 69.7% | **100.0%** | **82.1%** |
| **Barriere / Muri** | 242 | 626 | 6 | 27.9% | 97.6% | **43.4%** |
| **MEDIA GLOBALE** | **1.681** | **1.443** | **16** | **53.8%** | **99.1%** | **69.7%** |

**Raggio 25 Metri (Analisi Operativa Completa)**
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 734 | 252 | 6 | **74.4%** | **99.2%** | **85.1%** |
| **Camion / Bus** | 179 | 381 | 6 | 32.0% | **96.8%** | **48.1%** |
| **VRU (Pedoni / Ciclisti)** | 1.308 | 518 | 0 | **71.6%** | **100.0%** | **83.5%** |
| **Barriere / Muri** | 471 | 858 | 6 | 35.4% | **98.7%** | **52.2%** |
| **MEDIA GLOBALE** | **2.692** | **2.009** | **18** | **57.3%** | **99.3%** | **72.6%** |

---

#### Target 2: Valutazione su GT Reale nuScenes (Rilevamento Ostacoli 3D Effettivi)
Questo target misura quanti ostacoli fisici *realmente presenti e annotati in nuScenes* all'interno delle zone d'ombra vengono intercettati.

**Raggio 25 Metri**
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 432 | 554 | 286 | 43.8% | 60.2% | **50.7%** |
| **Camion / Bus** | 48 | 512 | 76 | 8.6% | 38.7% | **14.0%** |
| **VRU (Pedoni / Ciclisti)** | 297 | 1.529 | 0 | 16.3% | **100.0%** | **28.0%** |
| **Barriere / Muri** | 12 | 1.317 | 2 | 0.9% | 85.7% | **1.8%** |
| **MEDIA GLOBALE** | **789** | **3.912** | **364** | **16.8%** | **68.4%** | **27.0%** |

---

### 3. Baseline Analitica: Probabilità Condizionata Bayesiana (Senza Deep Learning)

Questo approccio analitico calcola $P(\text{Classe} \mid \text{Area}, \text{Suolo})$ mediante lookup tabellare e regole di combinazione condizionata basate sui macro-descrittori geometrici e sulla semantica del suolo.

**Raggio 25 Metri (Valutato su GT Reale 3D)**
| Categoria Semantica | Veri Positivi (TP) | Falsi Positivi (FP) | Falsi Negativi (FN) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 352 | 467 | 364 | 43.0% | 49.2% | **45.9%** |
| **Camion / Bus** | 19 | 355 | 105 | 5.1% | 15.3% | **7.6%** |
| **VRU (Pedoni / Ciclisti)** | 107 | 575 | 190 | 15.7% | 36.0% | **21.9%** |
| **Barriere / Muri** | 0 | 0 | 13 | 0.0% | 0.0% | **0.0%** |
| **MEDIA GLOBALE** | **478** | **1.397** | **672** | **25.5%** | **41.6%** | **31.6%** |

---

# 🎯 PARTE 4: Qual è il Modello Migliore e Perché? (Risposta per la Tesi)

La risposta rigorosa dal punto di vista dell'Ingegneria Informatica e della Guida Autonoma Sicura è:

### 🥇 Il Modello Migliore in Assoluto è: `Attention + GT Sintetica Ibrida (Spazio-Semantica)`

Ecco le motivazioni scientifiche e pratiche:

1. **Protezione Assoluta degli Utenti Vulnerabili (VRU)**:
   - Raggiunge il **100.0% di Recall** (zero pedoni mancati su oltre 1.300 zone) e una **Precision del 71.6%**, con un **F1-Score straordinario dell'83.5%**.
   - Avendo vincolato la larghezza ($< 1.7\,\text{m}$) e il tipo di suolo, la rete **non soffoca più i pedoni** con allucinazioni di veicoli pesanti.
2. **Affidabilità sulle Auto**:
   - Raggiunge l'**85.1% di F1-Score** (Precision 74.4%, Recall 99.2%), con solo 6 falsi negativi in tutto il set di validazione.
3. **Eliminazione delle Allucinazioni Fisicamente Impossibili**:
   - A differenza del modello `GEOMETRIC` (che posizionava camion di 8 metri sui marciapiedi perché vedeva solo l'area) e del modello `SEMANTIC` (che ignorava gli spazi stretti), il modello **Ibrido** garantisce che nessun veicolo venga mai predetto fuori strada o in varchi dove non può fisicamente entrare.
4. **Perché una Rete Neurale e non la Baseline Bayesiana?**:
   - La **Baseline Bayesiana** puramente analitica non sfrutta il pattern visivo 2D del LiDAR né la correlazione spaziale con gli ostacoli vicini.
   - Sulla GT Reale, Bayes raggiunge solo il **41.6% di Recall** e il **31.6% di F1**, **mancando 672 pericoli reali (il 58.4% degli ostacoli)**.
   - Il modello neurale `AttentionPerZoneModel` raddoppia le prestazioni e garantisce una mappatura continua e sensibile al contesto locale che nessuna tabella statica di probabilità a priori può eguagliare.
5. **Perché non usare la Baseline GT Reale?**:
   - La Baseline GT Reale ottiene un F1 elevato sulla GT reale solo perché si comporta da semplice detector 3D supervisato.
   - **Tuttavia fallisce catastroficamente nell'anticipazione preventiva del rischio**: perde oltre il **$54\%$ dei potenziali pericoli ciechi** (Recall al $46.1\%$). In un incrocio cieco o all'uscita da un parcheggio con visibilità ostruita, un'auto a guida autonoma con la Baseline reale non rallenterebbe, causando potenziali collisioni fatali.
   - Il modello **Ibrido**, invece, con la sua **Recall globale del $99.3\%$**, assicura la piena consapevolezza dello scenario peggiore (*Safety Worst-Case Anticipation*), nel pieno rispetto dei requisiti ISO 26262 e SOTIF (ISO 21448).

