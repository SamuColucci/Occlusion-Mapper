# Guida all'Esecuzione degli Agenti Decisionali a Runtime

Questa guida illustra come lanciare, testare e comprendere il funzionamento dei due agenti decisionali sviluppati per la stima delle probabilità nelle zone d'occlusione.

Gli agenti implementano due filosofie di calcolo separate ed indipendenti, ciascuna declinata in due varianti (**Diretto** e **Near-Miss**), per un totale di **4 modalità di stima del rischio**:
*   **Approccio A (Agente Bayesiano Dinamico)**: Modello classico probabilistico-statistico basato su tempo di cecità e dati storici frequentisti.
    *   *Diretto*: Prior ricavate da sovrapposizioni esatte.
    *   *Near-Miss*: Prior ricavate da vicinanza entro 2.0 metri.
*   **Approccio B (Agente Neurale Deep Learning)**: Modello basato su una rete neurale convoluzionale UNet che apprende le relazioni spaziali della strada.
    *   *Diretto*: Griglia di campionamento esatta dell'ombra.
    *   *Near-Miss*: Griglia allargata con buffer di 2.0 metri.

---

## Dashboard Interattiva
```powershell
python run_project.py
```
Selezionando l'opzione **`6. Esegui Pipeline Completa (RUN ALL)`** lo script eseguirà in sequenza automatica l'intero ciclo di calcoli geometrici, addestramento ed esportazione offline, generando simultaneamente tutti e 4 i dataset probabilistici nelle rispettive cartelle.

---

## 🧠 APPROCCIO A: Agente Bayesiano Dinamico (Statistico)

### Come viene calcolata la stima?
L'agente a runtime calcola la probabilità finale moltiplicando tre fattori chiave:
1.  **L'Hit Rate statico $P(H \mid S)$ (dal CSV)**: La frequenza storica con cui un oggetto reale è stato trovato geometricamente all'interno (o in prossimità) delle ombre proiettate da una determinata barriera.
2.  **Il Risk Score temporale $R(t) = 1 - e^{-\lambda \cdot t}$**: Il decadimento probabilistico continuo. Più a lungo una zona rimane cieca ($t$), più la probabilità di presenza cresce asintoticamente verso il 100%.
3.  **Filtri Geometrici Anti-Falso Positivo**: Prima di conteggiare un hit, lo script esclude le auto/pedoni reali che distano meno di $3\text{ metri}$ dall'ostacolo proiettante o che non si trovano fisicamente *dietro* di esso rispetto al LiDAR.

---

### 📐 Formulazione Matematica della Probabilità di Bayes nel Progetto

Il calcolo della probabilità che un ostacolo invisibile (target nascosto $H$) sia presente all'interno di una zona d'ombra (occlusione) proiettata da un ostacolo visibile (sorgente $S$) al secondo $t$ di cecità continuativa, è formulato unendo il **Teorema di Bayes** classico alla **Teoria dell'Affidabilità Temporale**.

#### 1. Il Teorema di Bayes Statistico (A Priori)
La probabilità a priori condizionata che indica la presenza di una specifica classe di ostacolo nascosto $H$, data una determinata sorgente proiettante $S$, è descritta dalla formula classica di Bayes:

$$P(H \mid S) = \frac{P(S \mid H) \cdot P(H)}{P(S)}$$

Nel nostro caso specifico, i singoli termini sono definiti come segue:
*   **$P(H \mid S)$ [Probabilità a Posteriori Condizionata]**: È la probabilità che, data una zona d'ombra proiettata da una sorgente visibile di classe $S$ (es. *Bus*), vi sia un ostacolo nascosto di classe $H$ (es. *Pedone*) al suo interno.
*   **$P(S \mid H)$ [Verosimiglianza o Likelihood]**: È la probabilità che un ostacolo di classe $H$ (nascosto) si trovi posizionato proprio dietro a una sorgente visibile di classe $S$, venendone coperto geometricamente rispetto al LiDAR.
*   **$P(H)$ [Prior dell'Ostacolo Nascosto]**: È la probabilità incondizionata che un ostacolo di tipo $H$ esista nell'ambiente (la frequenza complessiva di Pedoni, Auto o Camion rilevati nell'intero dataset).
*   **$P(S)$ [Evidenza o Evidence]**: È la probabilità incondizionata che una sorgente visibile proietti una zona d'occlusione rilevante all'interno del campo visivo.

#### 2. Stima Frequentista a Priori
Per calcolare concretamente $P(H \mid S)$ eliminando l'errore di discretizzazione spaziale, l'estrattore [bayesian_prior_extractor.py](file:///c:/Users/samue/Desktop/Tirocinio/bayesian_prior_extractor.py) analizza l'intero dataset storico in modo frequentista contando le intersezioni fisiche tra le bounding box reali tracciate dal dataset e i poligoni d'ombra LiDAR (oppure i coni allargati nel caso del near-miss):

$$P(H \mid S) = \frac{N(H \cap S)}{N(S)}$$

#### 3. Estensione Dinamico-Temporale a Runtime
La sola probabilità statica $P(H \mid S)$ non descrive lo stato del traffico a runtime. L'agente unisce quindi il fattore temporale cumulativo esponenziale $R(t)$, ottenendo la **Probabilità Bayesiana Dinamica Finalizzata**:

$$P(H \mid S, t) = R(t) \cdot P(H \mid S) = \left( 1 - e^{-\lambda \cdot t} \right) \cdot P(H \mid S)$$

Definizione delle variabili temporali:
*   **$t$ (Persistenza Temporale)**: Durata continuativa in secondi per cui la sorgente $S$ ha proiettato l'occlusione senza interruzioni ($t = \text{Frame Consecutivi} \times dt$, con $dt = 0.5\text{s}$).
*   **$\lambda$ (Costante di Rischio)**: Coefficiente di crescita fissato a $0.15\text{ s}^{-1}$. Determina la rapidità con cui il fattore di cecità temporale satura verso il $100\%$.

#### 4. Probabilità Combinata Finale (Unione Probabilistica)
Per determinare la probabilità totale che un ostacolo sia presente nell'ombra o nelle sue immediate vicinanze, viene applicato l'OR probabilistico (unione stocastica) tra la probabilità diretta $P_{\text{dir}}$ (Hit) e la probabilità vicina $P_{\text{near}}$ (Near-Miss):

$$P_{\text{finale}} = P_{\text{dir}} + P_{\text{near}} - \left( P_{\text{dir}} \cdot P_{\text{near}} \right)$$

Questo valore combinato finale viene:
1. Calcolato dinamicamente e mostrato nella colonna **`Fin`** del visualizzatore comparativo affiancato.
2. Salvato fisicamente nei file JSON probabilistici della cartella `_near` sotto la voce **`combined_probabilities`**, per permettere analisi statistiche offline immediate.

---

## 🧠 APPROCCIO B: Agente Neurale (Deep Learning / UNet)

Questo approccio si basa sull'apprendimento automatico spaziale (Image-to-Image translation) tramite una rete neurale convoluzionale Encoder-Decoder **UNet multi-classe** definita in `unet_model.py`.

### 1. Input e Output della UNet
*   **Input (4 canali $200 \times 200$)**: Poligoni d'ombra geometrici, punti LiDAR, bounding box degli ostacoli caster e il rischio temporale.
*   **Output (3 canali $200 \times 200$)**: Heatmap probabilistiche pixel-per-pixel corrispondenti a tre categorie distinte:
    *   **Canale 0**: Probabilità di presenza di un'**Auto** nascosta.
    *   **Canale 1**: Probabilità di presenza di un **Pedone** nascosto.
    *   **Canale 2**: Probabilità di presenza di un **Camion** nascosto.
*   **Buffer Near-Miss**: Quando l'agente neurale lavora in modalità Near-Miss, l'estrazione locale del massimo di probabilità dal canale della griglia viene effettuata applicando una dilatazione geometrica spaziale di 2.0 metri al poligono d'ombra originale prima del campionamento della bounding box.

---

## 🗺️ MODULAZIONE SEMANTICA DELLA SUPERFICIE (Road, Sidewalk, Terrain)

Per migliorare le stime di rischio a runtime ed evitare falsi allarmi e frenate non necessarie (ad es. per ombre cadenti su prati, marciapiedi o cespugli), gli agenti estraggono le superfici HD di NuScenes a monte durante la fase batch offline e scrivono i risultati direttamente nei file JSON probabilistici:
*   **Strada (`drivable_area`)**
*   **Marciapiede (`sidewalk`)**
*   **Prato (`terrain`)**

Per ciascun cono d'ombra $P_{occ}$, calcoliamo le frazioni di area occupate su ciascuna superficie ($\alpha_{road}$, $\alpha_{side}$, $\alpha_{terr}$) ed una frazione residua indefinita ($\alpha_{other} = 1.0 - \alpha_{road} - \alpha_{side} - \alpha_{terr}$). Questa frazione è salvata nei campi `road_fraction`, `sidewalk_fraction`, `terrain_fraction` del JSON.

### 1. Moltiplicatori di Modulazione delle Probabilità

A seconda di dove si trova l'occlusione, le probabilità stimate per le **quattro** classi vengono modulate usando le frazioni come pesi:

$$P'_C = P_C \cdot \left( \alpha_{road} \cdot M_{C,road} + \alpha_{side} \cdot M_{C,side} + \alpha_{terr} \cdot M_{C,terr} + \alpha_{other} \cdot M_{C,other} \right)$$

I coefficienti di modulazione semantica $M$ sono definiti come segue:

| Classe | Strada (`road`) | Marciapiede (`side`) | Prato (`terr`) | Altro (`other`) | Note e Fonti di Letteratura Scientifica di Riferimento |
|---|---|---|---|---|---|
| **Auto** | $1.0$ | $0.05$ | $0.01$ | $0.1$ | **L. Yin & H. Zhang (2021)**: Veicoli transitano quasi esclusivamente in carreggiata (esposizione su marciapiedi ~5% per manovre/sosta, e ~1% su prati sterrati). |
| **Pedone** | $0.25$ | $1.0$ | $0.15$ | $0.4$ | **L. Yin & H. Zhang (2021)**: L'esposizione pedonale media sulla carreggiata stradale è stimata a ~25% rispetto ai marciapiedi urbani. |
| **Camion** | $1.0$ | $0.01$ | $0.00$ | $0.05$ | Transito pesantemente confinato a carreggiate per vincoli fisici e di consolidamento. |
| **Animale/Altro** | $0.03$ | $0.05$ | $1.0$ | $0.2$ | **J. O. Abraham & S. A. Mumma (2021)**: Studio dell'esposizione spaziale e del comportamento della fauna selvatica, che staziona prevalentemente nei corridoi naturali (terrain). |

### 2. Modulazione del Rischio Connesso

Il rischio finale di collisione associato all'occlusione (mostrato a schermo ed usato per colorarla in verde, giallo o rosso neon) viene ridimensionato in base alla pericolosità logica della superficie e salvato nel campo `risk_score` del JSON:

$$\text{Rischio Modulato} = \text{Rischio base} \cdot \left( \alpha_{road} \cdot 1.0 + \alpha_{side} \cdot 0.4 + \alpha_{terr} \cdot 0.15 + \alpha_{other} \cdot 0.3 \right)$$

*   **Strada**: Pericolo massimo ($100\%$ del rischio originario).
*   **Marciapiede**: Pericolo moderato ($40\%$ del rischio originario, dovuto a potenziale discesa dei pedoni).
*   **Prato**: Pericolo minimo ($15\%$ del rischio originario, l'ego vehicle non transita sull'erba).

---

## 📈 CONFRONTO COMPARATIVO DI RUNTIME (Confronto Completo)

Eseguendo lo script di confronto:
```powershell
python visualizzatori/verify_runtime_comparison.py
```
*   **Caricamento Simultaneo**: Vengono caricati contemporaneamente tutti e 4 i dataset (Diretto e Near-Miss per entrambi gli approcci).
*   **Visualizzazione HUD**: Nel grafico Matplotlib, ciascuna classe mostra affiancati i risultati del caso **Diretto** e del caso **Near-Miss** per ogni ombra scansionata, indicando in tempo reale le percentuali per ciascuna superficie (`Strada: XX%`, `Marc.: YY%`, `Prato: ZZ%`).
*   **Log a Console**: La console stampa una tabella dettagliata con i Delta ($P_{Bayes} - P_{UNet}$) per entrambe le modalità contemporaneamente, facilitando il confronto immediato dei comportamenti.
