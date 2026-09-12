# 🚀 Roadmap Miglioramenti Futuri & Architetture SOTA

Questo documento raccoglie in modo strutturato e scientifico tutte le proposte di miglioramento architetturale, metodologico e di loss function discusse per il sistema di **Anticipazione del Rischio di Occlusione Neurosimbolico (BEV)**.

Una copia identica è conservata anche in [`documentazione/MIGLIORAMENTI_DA_FARE.md`](file:///c:/Users/samue/Desktop/Tirocinio/documentazione/MIGLIORAMENTI_DA_FARE.md).

---

## 1. Architetture Neurale SOTA (Spatial & Temporal)

Attualmente l'architettura ufficiale impiega:
- **ResNet-18 Backbone** troncata (prime 3 fasi conv) per estrarre feature multiscala;
- **Channel Attention (SE-Block / Squeeze-and-Excitation)** per ricalibrare l'importanza dei canali di mappa (LiDAR, Coni d'ombra, HD-Map, Dinamica veicolo);
- **Spatial Multi-Scale ROI-Pooling / Per-Zone Pooling**: estrazione di feature fisse per ciascuna zona occlusa convogliate all'MLP predittivo.

### A. Spatial Attention & Coordinate Attention (CBAM)
- **Idea**: Oltre alla sola *Channel Attention* (che decide *QUALI* feature pesare tra i canali di input), integrare un blocco di **Spatial Attention** (come in CBAM - *Convolutional Block Attention Module*, Woo et al., ECCV 2018) oppure **Coordinate Attention** (Hou et al., CVPR 2021).
- **Vantaggio Specifico**:
  - La Spatial Attention calcola una maschera 2D di peso: impara a concentrarsi esplicitamente sui **bordi di separazione geometrici** tra cono d'ombra e carreggiata stradale, dove l'incertezza visiva è massima.
  - La Coordinate Attention fattorizza l'attenzione lungo l'asse X e Y, preservando la precisa localizzazione spaziale degli ostacoli senza degradare con il global pooling.
- **Overhead**: Praticamente nullo (< 2 ms per frame), mantiene il processing pienamente real-time (> 30 FPS).

### B. Modellazione Temporale Sequenziale (ConvLSTM / GRU / BEV-Former Temporal Fusion)
- **Idea**: Attualmente l'inferenza è **single-frame** (analizza il frame $t$ staticamente). Nel mondo reale, le occlusioni sono fenomeni dinamici.
- **Come Funzionerebbe**:
  - Un modulo **ConvLSTM** o **Gated Recurrent Unit (ConvGRU)** riceve la sequenza di mappe BEV $[t-4, t-3, t-2, t-1, t]$.
  - Se un pedone o un ciclista era visibile al frame $t-2$ e scompare dietro un camion al frame $t$, la memoria ricorrente dell'architettura mantiene viva l'informazione di persistenza (*Object Permanence*).
  - La probabilità di rischio nella zona occlusa sale quasi al 100% perché la rete "si ricorda" che un agente è appena entrato nell'ombra.
- **Costo**: Richiede l'estensione del dataset loader a sequenze temporali contigue (nuScenes sample annotations + sweeps intermedi a 20Hz).

### C. Graph Neural Networks (GNN / Scene Graphs Topologici)
- **Idea**: Rappresentare la scena non solo come raster BEV, ma come un **Grafo di Relazioni Topologiche**:
  - **Nodi**: Ego-Vehicle, Veicoli occlusori (camion, bus, auto parcheggiate), Zone d'ombra geometriche, Aree stradali (marciapiede, attraversamento pedonale, corsia).
  - **Archi**: Relazioni spaziali e causali (*"il camion X occlude la zona Y"*, *"la zona Y interseca l'attraversamento pedonale Z"*).
- **Vantaggio**: Integrazione esplicita del ragionamento causale/relazionale tra chi blocca la vista e dove si trova il pericolo potenziale.

---

## 2. Funzione di Loss Attuale (ASL) e Possibili Evoluzioni

> [!NOTE]
> Nel nostro sistema **non usiamo né BCE né Focal Loss standard**: utilizziamo già la **Asymmetric Loss (ASL, Ridnik et al., ICCV 2021)** con parametri calibrati ($\gamma_- = 4.0$, $\gamma_+ = 1.0$, $\text{clip} = 0.05$) e pesi differenziati per classe (`pos_weights`). L'ASL è già superiore a BCE e Focal Loss per via del soft-thresholding sui negativi facili.

Tuttavia, all'interno della famiglia ASL, ci sono margini di ulteriore perfezionamento:

### A. ASL con Parametri Adattivi per Classe ($\gamma_{-, c}$)
- **Idea**: Attualmente usiamo $\gamma_- = 4.0$ identico per tutte le categorie.
- **Miglioramento**: Per classi a bassissima frequenza e critiche per la vita (VRU - Pedoni e Ciclisti), abbassare $\gamma_{-, \text{VRU}} = 2.0$ o $1.5$ per evitare che la rete sopprima le predizioni borderline sui pedoni, mantenendo invece $\gamma_{-, \text{Car}} = 4.0$ o $5.0$ sulle auto dove l'abbondanza di dati permette una pulizia più aggressiva dei falsi allarmi.

### B. Distance-Weighted ASL (Safety-Critical Weighting)
- **Razionale**: Nel contesto automotive (ISO 26262 / SOTIF), sbagliare a predire un ostacolo a **5 metri** dall'auto ha un impatto catastrofico (urto imminente), mentre a **24 metri** il veicolo ha ancora secondi per frenare o ripianificare la traiettoria.
- **Implementazione**: Moltiplicare la loss di ciascuna zona per un peso iperbolico inversamente proporzionale alla distanza radiale:
  $$w_{\text{safety}}(d) = 1.0 + \frac{\alpha}{\max(d, 2.0)}$$
  In questo modo la rete è penalizzata molto più duramente se fallisce una previsione a corto raggio.

---

## 3. Fusione Multi-Sensoriale Avanzata (Radar & Mappe Vettoriali)

1. **Radar Doppler BEV Channel**:
   - I segnali Radar mmWave (nativi in nuScenes su 5 sensori radar) penetrano parzialmente la pioggia e rimbalzano sotto il pianale dei veicoli.
   - Forniscono velocità radiale istantanea ($v_r$), consentendo alla rete di rilevare se qualcosa si sta muovendo *all'interno* del cono d'ombra prima ancora che sia visibile al LiDAR.
2. **Distance-Transform Map Channels**:
   - Oltre alla maschera binaria di marciapiedi e corsie, passare alla rete la distanza euclidea continua dai bordi carreggiata (Distance Field Map), facilitando la localizzazione spaziale.

---

## 4. Tabella di Priorità e Fattibilità

| Miglioramento | Complessità | Impatto Prestazioni | Compatibilità Real-time | Priorità Tesi |
| :--- | :---: | :---: | :---: | :---: |
| **Spatial Attention (CBAM)** | Bassa | Medio-Alto (+2-3% F1 sui bordi) | Eccellente (<2ms) | ⭐⭐⭐ Consigliato per Tesi |
| **Distance-Weighted ASL** | Minima | Medio (+sicurezza a corto raggio) | Immediato (0ms runtime) | ⭐⭐⭐ Consigliato per Tesi |
| **ConvLSTM Temporale** | Media | Altissimo (persistenza oggetti) | Buono (~15-20ms) | ⭐⭐ Sezione "Sviluppi Futuri" |
| **Radar Doppler Channels** | Media | Alto (robustezza dinamica) | Ottimo (<5ms) | ⭐⭐ Sezione "Sviluppi Futuri" |
| **Scene Graph GNN** | Alta | Teorico/Accademico | Medio (~30-50ms) | ⭐ Sezione "Sviluppi Futuri" |

---

## 5. Come Valorizzare Questi Punti nella Discussione della Tesi

1. **Evidenziare che l'attuale architettura Neurosimbolica è un trade-off ottimale**:
   - Recall **96.4%** su GT Reale e **96.9%** su GT Geometrica con soli **5.8 ms** di latenza ad inferenza.
   - Qualsiasi veicolo autonomo reale (es. standard ISO 26262 e ASIL-D) privilegia un'architettura snella, deterministica e a bassa latenza rispetto a modelli transformer pesanti che richiedono GPU da data-center.
2. **Dimostrare padronanza della Loss Function ASL**:
   - Spiegare alla commissione perché la Asymmetric Loss è stata scelta al posto di BCE e Focal Loss classica: il dynamic margin clipping ($\text{clip} = 0.05$) elimina il gradiente dei background facili, evitando che la miriade di zone d'ombra vuote soffochi il gradiente dei veri ostacoli.
