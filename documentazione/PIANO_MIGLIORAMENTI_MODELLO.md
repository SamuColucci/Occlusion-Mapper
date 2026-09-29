# Piano dei Miglioramenti del Modello (dopo la valutazione su `v1.0-trainval`)

> Questo piano parte dai risultati misurati su `v1.0-trainval` (split `val`, 6.019 fotogrammi, 25 m, soglie calibrate), riportati in [`TABELLA_RISULTATI_COMPLETA.md`](TABELLA_RISULTATI_COMPLETA.md). Le proposte architetturali a lungo termine sono raccolte in [`MIGLIORAMENTI_DA_FARE.md`](MIGLIORAMENTI_DA_FARE.md).
>
> Gli interventi sono ordinati dal meno al più costoso. Per ognuno sono indicati il problema osservato, la modifica, il costo stimato sulla macchina attuale (CPU 16 core, Quadro P4000, dataset su SSD esterno in USB 2.0) e come verificarne l'effetto.

---

## 0. Principi da Mantenere

Le modifiche proposte **non cambiano l'impostazione del lavoro**, ma la rafforzano:

1. **Pipeline per zona**: Raycasting LiDAR → zone d'ombra → patch BEV a 11 canali + 9 descrittori scalari → `AttentionPerZoneModel` (Squeeze-and-Excitation + FiLM).
2. **Vincolo neuro-simbolico**: la maschera di compatibilità fisico-semantica resta parte del modello e agisce sui logit, sia in addestramento sia in inferenza.
3. **Asymmetric Loss** come funzione di costo per lo sbilanciamento di classe.
4. **GT Sintetica Ibrida** come target principale di anticipazione del rischio, con la GT Reale come riferimento di rilevamento.
5. **Calibrazione delle soglie per modello** su dati di addestramento, **mai sullo split `val`**, che resta inedito per la valutazione finale.
6. **Nessun post-processing correttivo** sulle predizioni: ciò che si valuta è l'uscita della rete.

---

## 1. Quadro di Partenza

| Modello | F1 su GT Ibrida | F1 su GT Reale | Osservazione principale |
| :--- | :---: | :---: | :--- |
| `hybrid` | **84.0%** | 3.1% | Ottimo sulla GT Ibrida; classi deboli Camion/Bus (49.9%) e Barriere (71.3%) |
| `real` | 15.3% | **10.0%** | Loss quasi ferma (0.416 → 0.411 in 5 epoche); VRU reali nel ~3% delle zone |
| `positives_only` | 18.9% | **10.5%** | Non vede mai zone vuote in addestramento; AUC 0.50 – 0.82 |

Problemi di metodo emersi:
- le soglie fisse tarate su `v1.0-mini` non erano adatte ai modelli trainval (risolto con la calibrazione);
- la calibrazione usa blocchi della stessa cache di **train** su cui il modello è stato addestrato, quindi le soglie possono essere leggermente ottimistiche;
- la valutazione applica due vincoli d'area **fuori dalla rete** (Auto solo se area ≥ 3.5 m², Camion/Bus solo se area ≥ 8.0 m²), in contrasto con il principio 6.

---

## 2. Interventi a Costo Nullo (senza riaddestrare)

### 2.1 Metriche indipendenti dalla soglia sullo split `val`
- **Problema**: un singolo F1 dipende dalla soglia scelta e, sulla GT Reale con classi rarissime, non basta a dire se un modello è migliore di un altro.
- **Modifica**: calcolare per ogni modello, GT e classe la **curva Precision–Recall**, l'**Average Precision (AP)** e l'**AUC**, e confrontare l'AP con quella di un classificatore casuale (pari alla prevalenza della classe). La cache `valutazione/cache_valutazione_val.json` contiene già, per ogni zona, le probabilità `probs_4` e la GT `gt_target_4`: non serve rieseguire la valutazione.
- **Costo**: pochi minuti.
- **Verifica**: tabella AP per modello × GT × classe e figure delle curve PR in `documentazione/immagini_tesi/`.

### 2.2 Sensibilità alle soglie
- **Modifica**: riportare per `hybrid` e `real` come variano Precision, Recall ed F1 al variare della soglia (dalla stessa cache), per mostrare che il punto scelto dalla calibrazione non è un caso fortunato.
- **Costo**: pochi minuti.

---

## 3. Interventi sul Protocollo (riaddestramento non necessario, o minimo)

### 3.1 Set di calibrazione separato dall'addestramento
- **Problema**: le soglie sono tarate su zone che il modello ha già visto in addestramento.
- **Modifica**: riservare circa il **10% delle scene di train** come set di calibrazione (e di *early stopping*, § 4.1), escludendole dall'addestramento. La calibrazione resta per modello e per classe, sempre fuori da `val`.
- **Costo**: una rigenerazione dell'elenco delle scene e un riaddestramento (da combinare con § 4).
- **Verifica**: le soglie calibrate dovrebbero cambiare di poco; se cambiano molto, le attuali erano ottimistiche.

### 3.2 Vincoli d'area dentro la maschera neuro-simbolica
- **Problema**: i vincoli `area ≥ 3.5 m²` (Auto) e `area ≥ 8.0 m²` (Camion/Bus) sono applicati solo in valutazione, dopo la rete. La rete durante l'addestramento non li conosce, e sono tra le cause possibili dei 748 Camion/Bus mancati da `hybrid` sulla GT Ibrida.
- **Modifica**: spostare questi vincoli in `build_compatibility_mask()`, così che valgano identici in addestramento, calibrazione e valutazione, come gli altri vincoli fisici. In alternativa, se la GT sintetica non li rispetta, rimuoverli.
- **Costo**: modifica di poche righe; richiede di rigenerare la cache e riaddestrare.
- **Verifica**: F1 e Recall di Camion/Bus sulla GT Ibrida.

### 3.3 Calibrazione delle probabilità (opzionale)
- **Modifica**: applicare sul set di calibrazione un *temperature scaling* per classe, così che le probabilità siano interpretabili come frequenze (ad esempio 0.3 = ostacolo presente in circa il 30% dei casi). Le soglie per classe restano, ma diventano più stabili tra un addestramento e l'altro.
- **Costo**: basso.

---

## 4. Interventi sull'Addestramento (riusano le cache esistenti)

### 4.1 Più epoche con *early stopping*
- **Problema**: 5 epoche su trainval. La loss di `real` scende ancora pochissimo, ma lo scheduler a coseno arriva a $10^{-5}$ già alla quinta epoca: la rete non ha modo di continuare a imparare.
- **Modifica**: 15 – 20 epoche, conservando il checkpoint con la migliore Average Precision sul set di calibrazione (§ 3.1).
- **Costo**: circa 9 – 12 minuti per epoca con cache su USB 2.0 (misurati), cioè 2.5 – 4 ore per modello (molto meno con l'SSD su USB 3).
- **Verifica**: curva della loss e dell'AP per epoca; F1 su `val` con soglie ricalibrate.

### 4.2 Pesi delle classi ricalcolati su trainval
- **Problema**: i pesi dei positivi della ASL sono stati scelti su `v1.0-mini`. Su trainval le prevalenze sono diverse (GT Ibrida: Auto 41%, Camion/Bus 7%, VRU 53%, Barriere 16%; GT Reale: Auto 18%, Camion/Bus 5%, VRU 3%, Barriere 4%).
- **Modifica**: ricavare i pesi dalla prevalenza reale di ogni classe nella cache, ad esempio $w_c = \sqrt{(1 - p_c) / p_c}$, limitati a un intervallo ragionevole (1 – 5).
- **Costo**: calcolo immediato; riaddestramento.
- **Verifica**: F1 per classe, in particolare Camion/Bus e Barriere per `hybrid`, VRU per `real`.

### 4.3 $\gamma_{\text{neg}}$ per classe
- **Modifica**: già proposto in `MIGLIORAMENTI_DA_FARE.md` (§ 2.A). Ridurre $\gamma_{\text{neg}}$ sulle classi rare della GT Reale (VRU, Camion/Bus), mantenendolo a 4 sulle classi frequenti. Va valutato insieme al § 4.2, cambiando un parametro alla volta.
- **Costo**: riaddestramento.

### 4.4 `positives_only` con una quota di zone vuote
- **Problema**: il modello non vede mai zone vuote in addestramento (usa solo il ~21% delle zone), quindi non può imparare a rispondere "nessun ostacolo".
- **Modifica**: mantenere tutte le zone con ostacoli e aggiungere un campione casuale di zone vuote a rapporto fisso (ad esempio 1:1 e 1:3), confrontandolo con `real` (rapporto naturale ~1:4). L'esperimento diventa uno **studio sul bilanciamento dei negativi**.
- **Costo**: modifica del filtro in `carica_blocco()`; riaddestramento breve (la cache è la stessa di `real`).
- **Verifica**: AUC e AP sulla GT Reale.

---

## 5. Interventi sul Modello e sui Dati

### 5.1 Canale di memoria temporale (12° canale BEV)
- **Problema**: da un solo fotogramma la rete vede solo la forma dell'ombra e la mappa, quindi sa dove un ostacolo **può** esserci, non se c'è. È la causa principale dei risultati bassi sulla GT Reale.
- **Modifica**: il canale di memoria descritto in `MIGLIORAMENTI_DA_FARE.md` (§ 1.B): ostacoli visti nei fotogrammi precedenti, riportati nelle coordinate del fotogramma corrente con la posa dell'ego, con decadimento nel tempo. Resta un **ingresso** della rete: nessun filtro a valle, coerente con il principio 6.
- **Costo**: alto. Estrazione sequenziale dei fotogrammi, nuova cache (~42 GB per modalità), riaddestramento a 12 canali.
- **Verifica**: F1 e AP sulla GT Reale, soprattutto per VRU e Barriere statiche.

### 5.2 Loss pesata per distanza
- **Modifica**: già proposto in `MIGLIORAMENTI_DA_FARE.md` (§ 2.B). Aumentare il peso degli errori sulle zone vicine all'ego, $w(d) = 1 + \alpha / \max(d, 2)$.
- **Costo**: basso; riaddestramento.
- **Verifica**: metriche a 20 m rispetto a 25 m.

---

## 6. Esperimenti da Completare per la Tesi

- **Modelli `geometric` e `semantic` su trainval**: senza questi l'ablazione non mostra che la GT Ibrida sia migliore delle due GT sintetiche separate. Circa 2.5 ore ciascuno (cache + addestramento) più la valutazione.
- **Baseline Bayesiana**: generare le probabilità in `extracted_occlusions_probabilities/` e valutarla con gli altri modelli.
- **Valutazione anche su train** (o su un sottoinsieme di scene di train) per il confronto TRAIN / VAL nella figura di recap: su tutto il train servono circa 7.5 ore.
- **Ripetizioni con seed diversi** (almeno 2 – 3 per `hybrid`) per riportare media e deviazione standard.

---

## 7. Cosa Non Fare

- **Tarare le soglie sullo split `val`**: gonfierebbe i risultati. Lo script `valutazione/tune_thresholds_and_nms.py` lavora su `val` e va usato solo come analisi esplorativa, non per i numeri della tesi.
- **Correggere le predizioni a valle della rete** con regole manuali: i vincoli vanno nella maschera neuro-simbolica, che agisce anche in addestramento.
- **Cambiare più parametri insieme**: ogni modifica dei §§ 3 – 5 va valutata da sola rispetto all'attuale `hybrid`, per poterne attribuire l'effetto.
- **Confrontare modelli su GT diverse come se fossero lo stesso compito**: il confronto va sempre letto per target di valutazione.

---

## 8. Ordine Consigliato

| # | Intervento | § | Riaddestramento | Costo stimato | Effetto atteso |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | AP, curve PR e sensibilità alle soglie su `val` | 2.1 – 2.2 | No | Minuti | Valutazione più solida, soprattutto su GT Reale |
| 2 | Set di calibrazione separato + *early stopping* | 3.1, 4.1 | Sì | 3 – 4 h per modello | Soglie non ottimistiche; `real` e `hybrid` più addestrati |
| 3 | Vincoli d'area nella maschera | 3.2 | Sì (nuova cache) | ~3.5 h per modello | Camion/Bus di `hybrid` |
| 4 | Pesi di classe da prevalenza trainval | 4.2 | Sì | ~1 – 2 h per modello | Classi rare |
| 5 | `positives_only` con zone vuote | 4.4 | Sì (breve) | ~1 h per rapporto | Ablazione più significativa |
| 6 | `geometric`, `semantic`, Bayes su trainval | 6 | Sì | ~8 h complessive | Ablazione completa per la tesi |
| 7 | Canale di memoria temporale | 5.1 | Sì (nuovi dati) | Giorni | Il miglioramento maggiore atteso su GT Reale |

> **Nota**: nel documento `MIGLIORAMENTI_DA_FARE.md` la descrizione dell'architettura (backbone ResNet-18) e alcuni valori di riferimento (ad esempio Recall 96.4% su GT Reale) risalgono a versioni precedenti e non corrispondono al modello e ai risultati attuali.
