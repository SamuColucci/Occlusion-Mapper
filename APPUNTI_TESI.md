# Appunti, Fonti e Note Metodologiche per la Tesi di Laurea

Questo documento raccoglie in modo strutturato le spiegazioni fisiche, geometriche e le fonti scientifiche degli algoritmi implementati nel software di calcolo delle occlusioni 3D (Voxel Grid). È pensato per essere usato direttamente come traccia o bozza per la scrittura dei capitoli della tesi.

---

## 1. Architettura Generale della Pipeline (Voxel-Based Raycasting)
Il software ha il compito di generare la griglia di visibilità 3D (conosciuto/sconosciuto/occluso) proiettando i dati LiDAR grezzi e i Bounding Box 3D degli ostacoli in una griglia voxel discreta.

```
                  [ NuScenes Dataset (Data Input) ]
                                 │
                                 ▼
                     [ SOTARayCaster (__init__) ]
                                 │
       ┌─────────────────────────┴────────────────────────┐
       ▼                                                  ▼
[ 1. Griglia di Conoscenza ]                    [ 2. Coni d'Ombra Generati ]
  (generate_known_zone)                           (generate_box_shadows)
  • Tracciamento raggi LiDAR                      • Proiezione geometrica ostacoli
  • Mappatura voxel visibili/vuoti                • Calcolo ombra locale (np.min)
       │                                                  │
       └─────────────────────────┬────────────────────────┘
                                 ▼
                   [ 3. Visibilità Finale ]
                   (extract_occlusion_zones)
                   • Sottrazione Conoscenza 2D/3D
                   • Chiusura Morfologica 5x5 SciPy
                                 │
                                 ▼
                 [ 4. Esportazione Poligonale JSON ]
                     (save_occlusions_to_json)
                     • Scanning Radiale (Shell Mapping)
                     • Sanificazione Shapely .buffer(0)
```
---

## 2. Configurazione e Dipendenze Globali
* **Librerie utili**: `numpy` (matrici) [Harris 2020], `scipy.ndimage` (morfologia) [SciPy 2001], `shapely` (poligoni) [GEOS/Gillies 2007], `nuscenes` (dataset SDK) [Caesar 2020].
* **Parametri griglia**: `GRID_RANGE = 40.0m`, `VOXEL_SIZE = 0.4m` (standard benchmark **Occ3D** [Tian 2023]), `GRID_DIM = 200` (risoluzione orizzontale), `Z_DIM = 24` (risoluzione verticale).

---

## 3. Classe `PseudoBox`
Classe helper per trattare uniformemente ostacoli dinamici (da dataset) ed ostacoli manmade statici (da clustering LiDAR).

* **`__init__(...)`**:
  * Riceve i confini 3D allineati agli assi ($min$ e $max$ per $X, Y, Z$) e associa `name` e `token`.
  * Calcola il **centro geometrico 3D (centroide)** del box prendendo il punto medio: $(min + max) / 2.0$.
* **`corners()`**:
  * Calcola gli 8 vertici 3D della scatola (AABB) restituendo una matrice $(3, 8)$ conforme all'SDK NuScenes.

---

## 4. Classe `SOTARayCaster`
Classe principale che implementa la pipeline voxel di visibilità ed ombra.

### A. Costruttore `__init__(...)`
* **Caricamento LiDAR**: Legge il file binario `.bin` in una matrice a 5 colonne ($X, Y, Z, Intensity, Ring\_Index$) ed estrae le coordinate 3D metriche in `self.pts`.
* **Allocazione Matrici NumPy 3D** (dimensione $200 \times 200 \times 24$):
  * `self.grid`: spazio conosciuto/attraversato dai raggi ($1 = \text{visto}$).
  * `self.internal_shadows`: cuneo orizzontale per il contenimento dell'ombra interna.
  * `self.box_shadows`: coni d'ombra proiettati dai Bounding Box degli ostacoli ($1 = \text{ombra}$).
* **Ottimizzazioni**: Predisposizione delle variabili di cache lazy `_all_boxes = None` e `_lidar_occ = None`.

### B. Meccanismo di Cache Lazy
* **`_get_all_boxes()`**: Interroga l'SDK NuScenes `get_sample_data` una sola volta per frame per estrarre i Bounding Box 3D, memorizzandoli per le chiamate successive dello stesso frame.
* **`_get_lidar_occ()`**: Esegue la discretizzazione della nuvola LiDAR tridimensionale sulla griglia voxel ($200 \times 200 \times 24$). La conversione da coordinate metriche reali $(x, y, z)$ a indici voxel discreti $(i_x, i_y, i_z)$ avviene tramite le relazioni:
  $$i_x = \lfloor x/\text{VOXEL\_SIZE} \rfloor + \text{GRID\_DIM}/2$$
  $$i_y = \lfloor y/\text{VOXEL\_SIZE} \rfloor + \text{GRID\_DIM}/2$$
  $$i_z = \lfloor (z + z_{off})/\text{VOXEL\_SIZE} \rfloor$$
  *dove $z_{off} = (Z\_DIM / 4) \times \text{VOXEL\_SIZE} = 2.4\text{m}$ per consentire la mappatura spaziale della strada posta sotto la quota del sensore LiDAR dell'auto.*
* **Nota di Scope Temporale (Tesi)**: La cache è **locale alla singola istanza** di `SOTARayCaster` (che rappresenta un singolo frame). Ogni volta che si passa al frame successivo, viene istanziato un nuovo oggetto `SOTARayCaster` con cache vuota (`None`), garantendo che i dati LiDAR di un fotogramma non inquinino o si sovrappongano a quelli del fotogramma successivo.

### C. Rilevamento Ostacoli Statici (`detect_static_manmade_boxes`)
Algoritmo euristico per identificare muri ed edifici (manmade) privi di annotazioni 3D:
1. **Filtrazione LiDAR e Esclusione Dinamica**:
   * Limiti di quota $Z \in [-0.5\text{m}, 4.0\text{m}]$ (esclusione di asfalto e alberi).
   * **Mascheramento dei Veicoli (`inside_mask`)**: Si calcola l'AABB 2D dei veicoli dinamici tramite `corners()` e si eliminano i punti LiDAR che ricadono al loro interno (con tolleranza di $0.5$ metri) per evitare la duplicazione spuria di ostacoli statici.
2. **Discretizzazione BEV 2D a Risoluzione $0.8\text{m}$**:
   * I punti statici rimasti vengono proiettati su una griglia piana. Questa discretizzazione funge da **filtro passa-basso spaziale**, fondendo punti LiDAR sparsi appartenenti allo stesso muro in celle adiacenti accese.
3. **Clustering a Componenti Connesse (`scipy.ndimage.label`)**:
   * Algoritmo di etichettatura che raggruppa le celle adiacenti accese. Ogni gruppo continuo riceve un ID numerico univoco (`labeled_grid`), che viene poi riassociato ai punti 3D originali per raggrupparli in singoli oggetti.
4. **Segmentazione dei Muri e Ottimizzazione ad Estremi (`extremes`)**:
   * I cluster con estensione lineare superiore a $5.0\text{m}$ vengono suddivisi in sotto-segmenti di lunghezza massima $5.0\text{m}$ (evita macro-box artificiali).
   * **Logica degli Estremi**: Vengono istanziate PseudoBox solo in corrispondenza del **primo** e dell'**ultimo** segmento di ciascun muro continuo. Poiché i segmenti centrali sono geometricamente auto-occlusi dalla prospettiva del sensore ego, proiettare solo le estremità produce la medesima ombra complessiva ottimizzando i tempi di CPU.

### D. Spazio Conosciuto (`generate_known_zone`)
Algoritmo di *Space Carving* per marcare i voxel visibili/liberi attraversati dai raggi LiDAR:
1. **Depth Buffer Sferico ($1200 \times 400$)**:
   * Conversione delle coordinate cartesiane LiDAR $(x,y,z)$ in coordinate sferiche $(r, az, el)$.
   * Mappatura discreta in indici $(u, v)$ per generare un buffer sferico a 360° che contiene la distanza massima visiva registrata.
2. **Gap Filling (Interpolazione)**:
   * Applicazione di un filtro di massimo `3x5` (`scipy.ndimage.maximum_filter`) per chiudere i buchi verticali neri causati dalla discretizzazione angolare dei canali fisici del LiDAR.
3. **Space Carving Vettorializzato**:
   * Generazione della mesh geometrica dei voxel tramite `meshgrid`.
   * Conversione di ciascun centro-voxel in coordinate sferiche.
   * **known_mask**: Un voxel è marcato come libero/visibile (`self.grid = 1`) se la sua distanza dall'ego-vehicle è inferiore alla distanza massima registrata nel buffer sferico in quella specifica direzione:
     $$\text{distancia\_voxel} < \text{depth\_buffer}[u_{voxel}, v_{voxel}]$$

### E. Simulazione Coni d'Ombra (`generate_box_shadows`)
Algoritmo che calcola i coni d'ombra tridimensionali proiettati dai Bounding Box degli ostacoli:
1. **Filtro di Occupazione**: Si verifica che il volume 3D di ciascun ostacolo contenga almeno un punto LiDAR visibile tramite la cache `lidar_occ`.
2. **Proiezione Angolare del Cono**: Si calcolano gli angoli azimutali estremi (`az_min`, `az_max`) formati dai vertici del box.
3. **Split Limit Dinamico per Singolo Raggio**:
   * Per ciascun raggio angolare $u$ attivo, si estraggono i riflessi LiDAR retrostanti.
   * Se vi sono riflessi stradali o muri, il cono d'ombra viene forzatamente troncato al **primo punto visibile incontrato** (`np.min(hits_behind)`). In caso contrario, l'ombra si estende fino a un massimo di $30.0$ metri per coprire le zone d'angolo vuote.
4. **Surgical Wall Clipping (Taglio Muro)**: Si verifica se il cono intercetta pareti verticali statiche sottomappa (inclinazione verticale elevata) e si taglia l'ombra davanti al muro.
5. **Spatial Ownership (Risoluzione Sovrapposizioni)**: Qualora due coni d'ombra si sovrappongano sullo stesso voxel BEV, esso viene assegnato univocamente all'ostacolo con il centroide geometrico più vicino (criterio di *Distanza Euclidea Minima dal Centro*), popolando la mappa `cone_ownership`.

### F. Cunei d'Ombra Interni (`find_object_occlusion_wedges`)
Algoritmo che identifica lo spazio d'ombra racchiuso longitudinalmente tra due ostacoli consecutivi in colonna (es. tra auto anteriore e auto posteriore/muro):
1. **Maschera di Rispetto LiDAR (`safety_mask`)**:
   * Si applica una dilatazione binaria $3 \times 3$ (`scipy.ndimage.binary_dilation`) su tutti i punti occupati.
   * Rende lo spazio LiDAR inaccessibile all'ombra interna (magenta) per prevenire sovrapposizioni su asfalto o elementi visibili posteriori.
2. **Contrazione del Cuneo**:
   * Si restringe il cuneo angolare di $0.03\text{ rad}$ ($\approx 1.7^\circ$) per lato per evitare sbavature geometriche sui fianchi dell'ostacolo sorgente.
3. **Campionamento a 3 Raggi e Interpolazione**:
   * Si proiettano tre raggi esplorativi (Sinistro, Centro, Destro) dal centro del veicolo in primo piano.
   * Si calcolano le tre distanze di collisione con l'ostacolo retrostante (`d_L`, `d_C`, `d_R`).
   * La distanza limite d'ombra `d_limit` in ciascun voxel viene calcolata per **interpolazione bilineare** tra i raggi adiacenti, tracciando una sagoma d'ombra fedele al profilo dell'ostacolo posteriore.
4. **Filtrazione Rumore Spaziale**:
   * Si etichettano le isole d'ombra tramite `scipy.ndimage.label`.
   * Si scartano i micro-cluster spuri aventi area inferiore a 20 pixel BEV.

### G. Estrazione Zone di Occlusione Finali (`extract_occlusion_zones`)
Pipeline di fusione e sanificazione per la generazione della griglia voxel 3D definitiva:
1. **Sottrazione Ombre Interne dallo Spazio Noto**:
   * Si escludono i voxel appartenenti ai cunei d'ombra magenta dalla mappa dello spazio visibile:
     $$\text{final\_grid} = \text{grid} \setminus \text{internal\_shadows}$$
2. **Proiezione BEV piana della Visibilità**:
   * Si riduce l'asse verticale (Z) calcolando la visibilità BEV 2D: $\text{bev\_visibility} = \max_Z(\text{final\_grid})$. Se un raggio LiDAR ha attraversato anche solo una quota di una coordinata piana, l'intera colonna è considerata visibile.
   * Si azzera l'ombra dei box ovunque la colonna BEV sia marcata come visibile (nessun voxel visibile può essere occluso).
3. **Chiusura Morfologica BEV 2D (5x5)**:
   * Si proietta l'ombra 3D su piano piano 2D e si applica `scipy.ndimage.binary_closing` con un elemento strutturante quadrato di $5 \times 5$ voxel (pari a $2.0\text{m} \times 2.0\text{m}$).
   * **Utilità scientifica**: Salda le spaccature radiali e i vuoti d'aria (*Sweep Line Artifacts*) causati dalla scansione LiDAR a cerchi concentrici discreti sul terreno, generando un blocco occluso compatto e continuo.
4. **Ricostruzione della Griglia Voxel 3D**:
   * La mappa piana corretta viene riproiettata verticalmente sulla griglia dei box per generare `self.occluded_final` in 3D.

### H. Esportazione Poligoni JSON (`save_occlusions_to_json`)
Converte la griglia voxel 3D in poligoni vettoriali metrici esportabili per il sistema di pianificazione:
1. **Raggruppamento per Ownership**:
   * La mappa finale BEV viene scansionata pixel per pixel. Ogni pixel rosso viene associato al token dell'ostacolo che lo ha generato tramite la mappa `cone_ownership` calcolata nel passo E.
2. **Soglia Dinamica per Categoria**:
   * Pedoni/biciclette: $\geq 3$ pixel. Veicoli: $\geq 5$ pixel. Ostacoli statici: $\geq 2$ pixel. Questo adatta la sensibilità alla dimensione fisica del caster.
3. **Shell Mapping (Scanning Radiale)**:
   * I pixel d'ombra vengono convertiti in coordinate polari $(r, \theta)$.
   * Si campionano 12 raggi equidistanti nell'intervallo angolare. Per ciascun raggio si estraggono le distanze minima ($r_{near}$, bordo interno) e massima ($r_{far}$, bordo esterno).
   * L'unione dei due archi (esterno + interno rovesciato) genera un poligono chiuso a "guscio" (*shell*).
4. **Sanificazione Geometrica (Shapely)**:
   * Il poligono grezzo viene costruito come `ShapelyPolygon`.
   * `.buffer(0)` sana eventuali auto-intersezioni. `.simplify(0.2)` riduce i vertici con tolleranza di $0.2\text{m}$ (metà dimensione voxel).
5. **Serializzazione JSON**: Ciascun poligono viene esportato con: nome, token, distanza dal sensore, bounding box metrica, vertici del poligono, e area ($\text{m}^2$).

### I. Utilità Operative
* **`plot_bev()`** *(debug)*: Visualizzazione diagnostica matplotlib a 4 quadranti BEV (spazio noto, noto pulito, coni grezzi, risultato finale). Usato solo in modalità `--mode single` durante lo sviluppo; non coinvolto nella generazione del dataset.
* **`batch_process_dataset()`** *(produzione)*: Esegue l'intera pipeline (Fasi 1–5) su un intervallo di frame NuScenes, salvando un file JSON per ciascun sample. Supporta la riprendibilità automatica (salta i file già generati).

---

## 5. Classe OcclusionValidator (`validate_json_occlusion.py`)
Visualizzatore interattivo matplotlib per ispezionare e validare visivamente i poligoni di occlusione esportati dalla pipeline.

### Funzionamento
1. **Caricamento**: Scansiona il filesystem alla ricerca dei file JSON generati (`extracted_occlusions/*.json`). Per ciascun file, recupera il `lidar_token` dal JSON e carica la nube di punti LiDAR binaria corrispondente dal dataset NuScenes.
2. **Rendering BEV**: Sovrappone i poligoni di occlusione (letti dal JSON) alla nube di punti LiDAR proiettata in Bird's Eye View. Ogni poligono è colorato per categoria (ciano = veicoli, verde = pedoni, rosa = muri statici) e riempito con trasparenza.
3. **Navigazione Interattiva da Tastiera**:
   * **Frecce SU/GIU**: Cambio frame (carica il file JSON successivo/precedente e la relativa nube LiDAR).
   * **Frecce DESTRA/SINISTRA**: Scorrimento tra le singole occlusioni del frame corrente.
4. **Metadati Visualizzati**: Il titolo della finestra mostra in tempo reale: nome file, indice frame, tipo di ostacolo, token, distanza dal sensore e area in metri quadri.

---

## 6. Moduli di Analisi Probabilistica e Temporale

Per estrarre valore informativo dai dati geometrici discretizzati e supportare il processo decisionale di un pianificatore di traiettorie (Motion Planner), sono stati introdotti due moduli statistico-probabilistici. 

> [!NOTE]
> **Filosofia di Progetto (White-Box vs Black-Box)**:
> A differenza dei moderni approcci basati su reti neurali profonde (Deep Learning, come UNet o GNN) che operano come "scatole nere" non verificabili formalmente, la pipeline qui proposta è **interamente deterministica e trasparente (White-Box)**. 
> Essa si basa esclusivamente sulla geometria computazionale, sulla statistica frequentista classica (Teorema di Bayes) e su distribuzioni stocastiche note (esponenziale cumulativa). Questo garantisce spiegabilità matematica al 100% per ogni stima di rischio calcolata, un requisito fondamentale per la certificazione di sicurezza nei veicoli autonomi.

I moduli sono strutturati in due approcci indipendenti coordinati da un agente decisore a runtime:

### A. Analisi Bayesiana delle Categorie (`bayesian_prior_extractor.py`)

#### Formolazione Matematica e Concetto di Hit Rate
Sia $C_{hidden}$ la classe dell'agente realmente presente ma nascosto all'interno di una zona d'ombra (es. *Pedestrian*, *Car*) e sia $C_{source}$ la classe dell'ostacolo visibile che genera quella specifica ombra (es. *Truck*, *Bus*). Vogliamo calcolare la probabilità condizionata di presenza $P(C_{hidden} \mid C_{source}, d)$, dove $d$ rappresenta la fascia di distanza dall'ego-vehicle.

Nel nostro impianto statistico, questo valore coincide con il concetto di **Hit Rate (Tasso di Impatto)**, calcolato per via empirico-frequentista (frequenza relativa degli eventi favorevoli sui casi totali possibili) analizzando l'intero dataset storico NuScenes:

$$\text{Hit Rate} = P(C_{hidden} \mid C_{source}) \approx \frac{N(C_{hidden} \cap C_{source})}{N(C_{source})}$$

Dove:
*   **$N(C_{source})$ (Casi Possibili)**: È il numero totale di coni d'ombra (occlusioni) generati da ostacoli di tipo $C_{source}$ in tutto il dataset.
*   **$N(C_{hidden} \cap C_{source})$ (Casi Favorevoli / Hit Reali)**: È il numero di volte in cui un oggetto reale di tipo $C_{hidden}$ (es. un'auto reale) si trovava geometricamente all'interno dell'ombra proiettata da $C_{source}$ (sotto i vincoli dei filtri anti-falso positivo).

#### Componenti del Teorema di Bayes
Esprimendo questa probabilità condizionata secondo il formalismo classico del **Teorema di Bayes**, possiamo scomporla nelle seguenti componenti:

$$P(C_{hidden} \mid C_{source}) = \frac{P(C_{source} \mid C_{hidden}) \cdot P(C_{hidden})}{P(C_{source})}$$

Ciascuna componente rappresenta un fattore fisico e probabilistico ben definito:
1.  **$P(C_{hidden} \mid C_{source})$ (Probabilità a Posteriori / Posterior)**: È la stima finale che l'agente calcola per quantificare il rischio di collisione.
2.  **$P(C_{source} \mid C_{hidden})$ (Verosimiglianza / Likelihood)**: È la probabilità geometrica che, data la presenza di un ostacolo $C_{hidden}$ nell'ambiente, esso finisca per essere coperto proprio da un ostacolo di categoria $C_{source}$. È direttamente proporzionale alla grandezza e alla forma dell'ostacolo proiettante (es. un camion ha una verosimiglianza di occlusione molto più alta rispetto a una moto).
3.  **$P(C_{hidden})$ (Probabilità a Priori / Prior)**: È la probabilità di presenza generica di quell'oggetto nel territorio di guida (corrispondente alla densità di traffico del dataset per quella determinata categoria).
4.  **$P(C_{source})$ (Evidenza / Evidence)**: È la probabilità di incontrare un ostacolo proiettante $C_{source}$ lungo il percorso, agendo come fattore di normalizzazione.

#### Filtri Anti-Falso Positivo
1.  **Filtro ID**: Si scarta l'intersezione dell'oggetto con se stesso:
    $$\text{ID}_{hidden} \neq \text{ID}_{source}$$
2.  **Filtro Distanza Radiale (Direzionalità)**: L'oggetto nascosto deve essere posizionato dietro l'ostacolo rispetto al punto di vista del sensore ego:
    $$d(ego, center_{hidden}) \ge d(ego, center_{source}) - 1.0\text{m}$$
3.  **Filtro Prossimità Relativa**: Se due veicoli sono accodati ed estremamente vicini (es. entro 3 metri), l'intersezione non costituisce una reale situazione di occlusione invisibile imprevedibile:
    $$\|center_{hidden} - center_{source}\|_2 \ge 3.0\text{m}$$
4.  **Fasce Metriche**: Le probabilità vengono calcolate in modo differenziato per tre fasce di distanza $d$:
    *   $d \in [0, 10)\text{m}$ (Fascia a corto raggio, alta criticità)
    *   $d \in [10, 20)\text{m}$ (Fascia a medio raggio)
    *   $d \in [20, 40)\text{m}$ (Fascia a lungo raggio)

---

### B. Analisi Temporale di Persistenza (`temporal_persistence_tracker.py`)
Traccia in sequenza temporale keyframe per keyframe, con un passo temporale di campionamento $dt = 0.5\text{s}$ ($2\text{ Hz}$), la persistenza temporale delle ombre.

#### Formolazione Matematica
Più a lungo una zona di spazio stradale rimane occlusa alla vista dei sensori (stato di cecità), più aumenta l'incertezza e con essa la probabilità che un agente dinamico non rilevato sia penetrato all'interno di quell'ombra.
Modelliamo questo fenomeno stocastico tramite una **funzione di distribuzione cumulativa esponenziale** (CDF di una variabile aleatoria esponenziale che descrive il tempo di arrivo/intrusione di un oggetto nell'ombra):

$$P(\text{presenza} \mid t_{pers}) = 1 - e^{-\lambda \cdot t_{pers}}$$

Dove:
*   $t_{pers}$ rappresenta il tempo cumulativo continuo di persistenza dell'occlusione in secondi:
    $$t_{pers} = N_{frames} \cdot dt$$
*   $\lambda$ è il tasso di crescita del rischio stocastico (hazard rate), impostato empiricamente a $\lambda = 0.15$. Un valore di $0.15$ implica che dopo $3.5\text{ secondi}$ di cecità continua ($7$ frame), il punteggio di rischio probabilistico supera la soglia critica del $65\%$:
    $$P(\text{presenza} \mid t_{pers} = 3.5\text{s}) = 1 - e^{-0.15 \cdot 7} \approx 0.6501$$

Questo risk score normalizzato $R \in [0, 1]$ viene iniettato nei JSON per consentire a un pianificatore di percorso di penalizzare i percorsi adiacenti o interni alle zone cieche più persistenti.

---

### C. Strumento di Validazione degli Hit (`verify_bayesian_hits.py`)
Interfaccia interattiva che isola visivamente i singoli casi di hit bayesiano (in rosso il poligono d'occlusione geometrico, in verde il footprint 2D reale dell'oggetto occultato) sovrapponendoli ai punti LiDAR per una validazione qualitativa ed estrazione di screenshot.

---

### D. Agente Decisionale a Runtime (`bayesian_occlusion_agent.py`)
Il modulo di integrazione finale che simula il comportamento dell'agente a bordo del veicolo autonomo. A runtime, per ogni frame e per ogni ombra rilevata, calcola la stima probabilistica semantica degli oggetti nascosti combinando i due approcci precedenti (frequentista e temporale).

#### Formolazione Matematica Definitiva
Sia $O_i$ la zona d'ombra corrente e $W$ la larghezza geometrica del cono estratta via **Oriented Bounding Box (OBB)**. La probabilità stimata di trovare un oggetto nascosto della classe $C_{hidden}$ è:

$$P(C_{hidden} \mid \vec{\alpha}, W) = \min\!\left( P_{\text{base}}(C_{hidden}, \vec{\alpha}) \cdot K_{\text{width}}(W, C_{hidden}),\; 1.0 \right)$$

Dove:
*   $P_{\text{base}}(C_{hidden}, \vec{\alpha})$: **Probabilità di base semantica** ponderata sulle frazioni di area della superficie:
    $$P_{\text{base}} = \alpha_{road} \cdot B_{C,road} + \alpha_{side} \cdot B_{C,side} + \alpha_{terr} \cdot B_{C,terr} + \alpha_{other} \cdot B_{C,other}$$
    I valori $B_C$ sono interpretabili direttamente (es. $B_{Pedone,side}=0.35$, $B_{Auto,road}=0.30$) e rappresentano la probabilità a priori di trovare la classe su quella superficie pura. La prior empirica CSV $P(C_{hidden} \mid C_{source})$, estratta frequentisticamente dal dataset, è salvata separatamente come campo `empirical_prior` nel JSON per la tesi ma non entra nella formula per evitare effetto depressivo su sorgenti statiche (dove è ~1-3%).

*   $K_{\text{width}}(W, C_{hidden})$: **Coefficiente di larghezza OBB** — vincolo fisico. Si applica solo su zona stradale ($\alpha_{road} \geq 0.1$):

| Larghezza OBB $W$ | Auto | Camion | Pedone | Bicicletta | Altro |
|:---|:---:|:---:|:---:|:---:|:---:|
| $W < 1.0\text{m}$ | $0.00$ | $0.00$ | $1.00$ | $0.50$ | $0.20$ |
| $1.0 \leq W < 1.8\text{m}$ | $0.05$ | $0.00$ | $1.00$ | $1.00$ | $0.30$ |
| $1.8 \leq W < 2.5\text{m}$ | $0.50$ | $0.20$ | $1.00$ | $1.00$ | $0.60$ |
| $2.5 \leq W < 4.0\text{m}$ | $1.00$ | $0.50$ | $0.80$ | $1.00$ | $0.80$ |
| $W \geq 4.0\text{m}$ | $1.00$ | $1.00$ | $0.60$ | $1.00$ | $1.00$ |

Il **risk_score** (CDF esponenziale $R(t_{pers}) = 1 - e^{-\lambda t}$) non entra nella formula delle probabilità di classe: rimane come indicatore di urgenza del cono per il pianificatore di traiettoria.

#### Memoria Temporale Causale (solo frame passati)
Se un oggetto della classe $C_k$ era **confermato visibile** (visibility\_token $\in \{3, 4\}$, copertura $>40\%$) nei frame precedenti della stessa scena ed il suo footprint BEV interseca geometricamente la zona d'ombra corrente, la sua stima viene alzata a:

$$P(C_k) \leftarrow \max\!\left( P(C_k),\; 0.92 \right)$$

Il valore $0.92$ (e non $1.0$) lascia un margine dell'$8\%$ di incertezza posizionale: l'oggetto potrebbe essersi spostato al limite dell'ombra nell'intervallo inter-frame. Solo il frame passato è considerato (causalità): i frame futuri non sono accessibili all'agente a bordo del veicolo in tempo reale.

---

### E. Approccio Neurale basato su Deep Learning (`unet_model.py`, `dataset_generator.py`, `train_neural_agent.py`, `neural_occlusion_agent.py`)
In alternativa e a titolo di confronto rispetto al modello Bayesiano classico, è stata implementata una pipeline di stima basata su **Reti Neurali Convoluzionali Profonde (Deep Learning)** con un'architettura **UNet in coordinate BEV (Bird's Eye View)**.

#### 1. Architettura UNet BEV (`unet_model.py`)
La rete implementata è una **UNet multi-classe** strutturata come segue:
*   **Input (4 canali $200 \times 200$)**:
    *   Canale 0: Poligoni delle ombre geometriche rasterizzati (0/1).
    *   Canale 1: Nube LiDAR discretizzata proiettata a terra (0/1).
    *   Canale 2: Bounding Box degli ostacoli noti (caster) rasterizzati (0/1).
    *   Canale 3: Mappa del rischio temporale $R(t_{pers})$ rasterizzato.
*   **Encoder (Fase di Contrazione)**: 3 blocchi di doppia convoluzione ($3 \times 3$) seguiti da Batch Normalization, attivazione ReLU e Max-Pooling ($2 \times 2$) che dimezza la risoluzione spaziale aumentando i canali di feature ($32 \rightarrow 64 \rightarrow 128$). Questa fase estrae le relazioni semantiche di alto livello sull'ambiente circostante (es. curvature, incroci).
*   **Bottleneck**: Punto a minima risoluzione spaziale ($25 \times 25$) e massimo numero di feature ($256$).
*   **Decoder (Fase di Espansione)**: 3 blocchi di up-sampling (convoluzione trasposta) che raddoppiano la risoluzione orizzontale.
*   **Skip Connections**: Connettono direttamente le feature ad alta risoluzione geometrica dell'Encoder con i corrispondenti blocchi del Decoder, preservando la forma esatta delle zone d'ombra.
*   **Output (3 canali $200 \times 200$ con attivazione Sigmoide)**:
    *   Canale 0: Stima probabilistica della presenza di un'**Auto** nascosta.
    *   Canale 1: Stima probabilistica della presenza di un **Pedone** nascosto.
    *   Canale 2: Stima probabilistica della presenza di un **Camion** nascosto.

#### 2. Dataset con Caching in RAM (`dataset_generator.py`)
Per ovviare al massiccio collo di bottiglia di I/O indotto dalla lettura dinamica dei file binari LiDAR e dalla proiezione delle bounding box 3D durante ogni epoca, è stato implementato un sistema di **caching in memoria RAM**. 
All'avvio, l'intero dataset di 404 frame (NuScenes Mini) viene pre-elaborato e caricato in RAM (occupazione totale: $\approx 450\text{MB}$). Questo consente alla CPU di prelevare istantaneamente le coppie input-target, abbattendo il tempo di una singola epoca di training da $202\text{ secondi}$ a **meno di $2\text{ secondi}$** (un'accelerazione di oltre 100 volte).

#### 3. Loss Function e Ottimizzazione (`train_neural_agent.py`)
L'addestramento pixel-to-pixel viene minimizzato tramite la **Binary Cross-Entropy Loss (BCE Loss)** calcolata indipendentemente per ciascun canale semantico di output rispetto alla Ground Truth:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

L'ottimizzazione dei pesi del modello viene effettuata tramite l'algoritmo **Adam Optimizer** (learning rate: $10^{-3}$, batch size: $8$, epoche: $5$). Il modello addestrato viene esportato in `best_model.pth`.

---

### F. Confronto Comparativo delle Metodologie (`verify_runtime_comparison.py`)
Il modulo comparativo permette di affiancare in modo sincronizzato le risposte fornite dall'agente Bayesiano (sinistra) e dall'agente neurale (destra), stampando a terminale la tabella delle differenze (Delta percentuali) per ciascun frame ed occlusione:

$$\text{Delta} = P_{\text{Bayes}}(C_{\text{hidden}} \text{ presente}) - P_{\text{UNet}}(C_{\text{hidden}} \text{ presente})$$

#### Matrice di Confronto Qualitativo per la Tesi

| Metodologia | Approccio Bayesiano Classico | Approccio Neurale (UNet) |
| :--- | :--- | :--- |
| **Natura del Modello** | Deterministico / White-Box (100% spiegabile) | Stocastico / Black-Box (Scatola nera probabilistica) |
| **Input Dati** | Poligoni vettoriali discreti e metadati temporali | Matrice raster BEV 4 canali continua |
| **Output** | Stima numerica discreta per ciascuna ombra | Heatmap densa di probabilità pixel-per-pixel |
| **Risorse Hardware** | Bassissime (Esecuzione istantanea su CPU) | Medio-alte (Richiede GPU per training ed inferenza) |
| **Sensibilità al Contesto**| Assente (Ignora la semantica stradale circostante) | Altissima (Riconosce incroci, marciapiedi e corsie) |
| **Addestramento** | Non richiesto (Statistiche ricavate a priori) | Obbligatorio (Richiede migliaia di campioni reali) |

#### Analisi dei Risultati e Discussione Critica (Tesi)
Dalle misurazioni a console nel modulo comparativo, si osserva che nei primi frame di cecità il Delta è sistematicamente negativo (compreso tra $-5\%$ e $-9\%$). 
Questo fenomeno si spiega metodologicamente a causa dello **sbilanciamento delle classi (Class Imbalance)** del dataset NuScenes Mini (dove meno del $3\%$ delle aree d'ombra contiene un ostacolo reale). La rete UNet, addestrata per poche epoche su dati così sbilanciati, tende a formulare una predizione di base "conservativa" distribuita uniformemente intorno a valori bassi ($4\%-9\%$) per minimizzare la loss. 
Al contrario, il modello Bayesiano ridimensiona la probabilità a ridosso dello $0\%$ per via del limitato tempo di cecità accumulato ($R(t) \approx 0.0$). 
Questo dimostra il trade-off teorico: il modello Bayesiano offre risposte nette ma rigide; il modello neurale offre stime sensibili alla topologia stradale ma richiede dataset completi (es. NuScenes Train Set da 28.000 frame) e funzioni di loss pesate (Focal Loss) per evitare il livellamento conservativo delle predizioni.

---

## 7. Riferimenti Bibliografici e Fonti

### Dataset
* **NuScenes**: H. Caesar et al., *"nuScenes: A Multimodal Dataset for Autonomous Driving"*, CVPR 2020. — Dataset multimodale con annotazioni 3D Bounding Box e sensore LiDAR a 32 canali (Velodyne HDL-32E). Utilizzato come sorgente dati per l'intera pipeline.
* **Occ3D**: Y. Tian et al., *"Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving"*, NeurIPS 2023. — Benchmark di riferimento per la predizione di occupancy 3D. Il nome della pipeline (`occ3d_occlusion_explorer`) richiama questo lavoro.

### Deep Learning e Reti Neurali
* **UNet**: O. Ronneberger, P. Fischer, T. Brox, *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI 2015. — Architettura Encoder-Decoder con skip connections originale, qui adattata per la segmentazione semantica delle griglie BEV.
* **Adam Optimizer**: D.P. Kingma, J. Ba, *"Adam: A Method for Stochastic Optimization"*, ICLR 2015. — Algoritmo di ottimizzazione stocastica di primo ordine a gradiente disceso con adattamento dei momenti, utilizzato per il training della UNet.
* **PyTorch**: A. Paszke et al., *"PyTorch: An Imperative Style, High-Performance Deep Learning Library"*, NeurIPS 2019. — Framework di machine learning open source utilizzato per implementare l'architettura `UNetBEV` e il Dataloader.

### Algoritmi e Tecniche Consolidate
* **Space Carving (Voxel Carving)**: K.N. Kutulakos, S.M. Seitz, *"A Theory of Shape by Space Carving"*, International Journal of Computer Vision (IJCV), 2000. — Metodo di ricostruzione 3D che deduce lo spazio libero tracciando raggi dalla sorgente ai punti di impatto. Applicato nella Fase 1 (`generate_known_zone`) per marcare i voxel attraversati come "visibili".
* **Ray Casting**: A. Appel, *"Some Techniques for Shading Machine Renderings of Solids"*, AFIPS Spring Joint Conference, 1968. — Tecnica fondamentale della computer graphics per la proiezione di raggi da un punto di vista. Utilizzata nelle Fasi 1 e 2 per la simulazione sferica dei raggi LiDAR.
* **Depth Buffer (Z-Buffer)**: E. Catmull, *"A Subdivision Algorithm for Computer Display of Curved Surfaces"*, PhD Thesis, University of Utah, 1974. — Struttura dati per la determinazione della visibilità. Il `depth_buffer` sferico nella Fase 1 ne è un'estensione adattata alle coordinate polari.
* **Operazioni Morfologiche (Chiusura, Dilatazione)**: J. Serra, *"Image Analysis and Mathematical Morphology"*, Academic Press, 1982. — La chiusura morfologica $5 \times 5$ della Fase 4 (`binary_closing`) e la dilatazione di sicurezza $3 \times 3$ della Fase 3 (`binary_dilation`) derivano da questa teoria.
* **Componenti Connesse (Connected Components Labeling)**: A. Rosenfeld, J.L. Pfaltz, *"Sequential Operations in Digital Picture Processing"*, Journal of the ACM, 1966. — L'algoritmo di etichettatura usato nella Fase C (`scipy.ndimage.label`) per il clustering dei muri statici si basa su questo metodo.

### Sicurezza Stradale ed Esposizione al Rischio (Studi Recenti 2021-2022)
* **Pedestrian & Vehicle Exposure**: L. Yin, H. Zhang, *"Building Walkable and Safe Neighborhoods: Assessing the Built Environment Characteristics for Pedestrian Safety in Buffalo, NY"*, 2021. — Studio sulla relazione spaziale tra marciapiedi ed esposizione del rischio. [Documento Locale](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioPedonaleL.Yin,H.Zhang.md).
* **HD Maps & Perception Safety**: Y. Li et al., *"HD Map-Based Occlusion Reasoning for Autonomous Driving Perception"*, IEEE / arXiv, 2022. — Analizza come l'integrazione delle mappe HD e dei priori semantici legati alle superfici consenta di ridurre i falsi positivi. [Documento Locale](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioMappeHDY.Li.md).

### Librerie Software
* **SciPy `ndimage`**: Modulo di elaborazione immagini N-dimensionali (filtri, labeling, morfologia). Fonte: `scipy.ndimage` — SciPy Reference Guide.
* **Shapely**: Libreria Python per geometria computazionale planare (costruzione/validazione/semplificazione poligoni). Basata su GEOS (JTS Topology Suite). Utilizzata nella Fase 5 per `.buffer(0)` (sanificazione auto-intersezioni) e `.simplify()` (riduzione vertici).
* **NumPy `meshgrid`**: Generazione di griglie N-dimensionali per la vettorializzazione del ray-tracing. Fonte: NumPy Reference.
* **NuScenes DevKit**: SDK Python ufficiale per l'accesso ai dati NuScenes (`nuscenes-devkit`). Utilizzato per `get_sample_data()`, `corners()`, e il caricamento dei file `.bin` LiDAR.

---

## 8. Modulazione Semantica Spaziale delle Superfici (Mappe HD e Studi del Rischio Recenti)

Per ridurre i falsi allarmi e rendere la stima delle probabilità di presenza nelle zone d'occlusione realistica e scientificamente fondata, la pipeline integra la mappa HD di NuScenes. I coni d'ombra LiDAR vengono intersecati con i layer semantici estratti: Strada (`drivable_area`), Marciapiede (`walkway`) e Prato (`terrain`, calcolato geometricamente come area residua).

La probabilità semantica $P_C$ per ciascuna classe viene modulata in base alle frazioni di area occupate sulle superfici ($\alpha_{road}, \alpha_{side}, \alpha_{terr}$) ed una frazione residua indefinita ($\alpha_{other}$):

$$P'_C = P_C \cdot \left( \alpha_{road} \cdot M_{C,road} + \alpha_{side} \cdot M_{C,side} + \alpha_{terr} \cdot M_{C,terr} + \alpha_{other} \cdot M_{C,other} \right)$$

I moltiplicatori semantici $M$ sono stati ricavati ed allineati a recenti studi di sicurezza stradale ed esposizione spaziale del rischio ([Yin & Zhang 2021](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioPedonaleL.Yin,H.Zhang.md), [Li et al. 2022](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioMappeHDY.Li.md)):

| Classe | Strada (`road`) | Marciapiede (`side`) | Prato (`terr`) | Altro (`other`) | Note e Fonti di Letteratura Scientifica di Riferimento |
|---|---|---|---|---|---|
| **Auto** | $1.0$ | $0.05$ | $0.01$ | $0.1$ | **[L. Yin & H. Zhang (2021)](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioPedonaleL.Yin,H.Zhang.md)**: Veicoli transitano quasi esclusivamente in carreggiata. L'esposizione su marciapiedi (infrazioni/sosta abusiva) o zone verdi è bassa (stimata al ~5% e ~1% rispettivamente). |
| **Pedone** | $0.25$ | $1.0$ | $0.15$ | $0.4$ | **[L. Yin & H. Zhang (2021)](file:///c:/Users/samue/Desktop/Tirocinio/FontiMD/RischioPedonaleL.Yin,H.Zhang.md)**: L'esposizione pedonale media sulla carreggiata stradale è stimata a ~25% rispetto al transito sicuro sulle aree pedonali dedicate. |
| **Camion** | $1.0$ | $0.01$ | $0.00$ | $0.05$ | Transito pesantemente confinato a carreggiate per vincoli fisici e di consolidamento. |
| **Bicicletta** | $0.80$ | $0.60$ | $0.10$ | $0.30$ | Transito misto su strada e aree pedonali/marciapiedi; presenza molto limitata su erba/terrain non consolidato. |
| **Altro** | $0.20$ | $0.30$ | $0.50$ | $0.40$ | Ostacoli generici (moto, bus, cassonetti, oggetti mobili). Raggruppa tutte le restanti priorità empiriche estratte dal dataset. |

### Modulazione del Rischio Connesso
Il rischio stocastico finale di collisione associato all'occlusione viene scalato in base alla pericolosità logica della superficie per la guida autonoma:

$$\text{Rischio Modulato} = \text{Rischio base} \cdot \left( \alpha_{road} \cdot 1.0 + \alpha_{side} \cdot 0.4 + \alpha_{terr} \cdot 0.15 + \alpha_{other} \cdot 0.3 \right)$$

*   **Strada**: Pericolo massimo ($100\%$ del rischio, area di transito dell'ego vehicle).
*   **Marciapiede**: Pericolo moderato ($40\%$, dovuto a potenziale invasione di carreggiata da parte di pedoni).
*   **Prato**: Pericolo minimo ($15\%$, l'ego vehicle non transita sull'erba).
