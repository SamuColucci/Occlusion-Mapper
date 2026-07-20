# Guida Dettagliata per la Riscrittura da Zero di `occ3d_occlusion_explorer.py` (con Architettura Adapter)

Questa guida è stata strutturata per supportarti nella stesura manuale di **`occ3d_occlusion_explorer.py`** e del modulo di astrazione **`dataset_adapters.py`** per la tua tesi di laurea. L'obiettivo è farti comprendere a fondo l'interazione tra i dati fisici (LiDAR, HD Map) e l'astrazione ad oggetti che permette di slegare la pipeline di calcolo dal dataset specifico (NuScenes).

---

## ⚙️ Il Pattern Adapter: Principi, Funzionamento e Requisiti

Il **Pattern Adapter** (definito dalla *Gang of Four* come uno dei design pattern strutturali fondamentali) ha l'obiettivo di risolvere l'incompatibilità tra due interfacce diverse. Nel contesto di un sistema di guida autonoma, questo pattern è essenziale per garantire la portabilità dell'algoritmo di stima.

### 🧠 Su cosa si basa?
Il pattern si basa su tre concetti chiave dell'ingegneria del software:
1.  **Disaccoppiamento (Loose Coupling)**: Il codice che calcola l'occlusione e stima le probabilità non deve sapere "chi" ha fornito i dati. Se l'algoritmo conoscesse le tabelle interne del database NuScenes, sarebbe impossibile riutilizzarlo su un altro dataset senza riscriverlo da capo.
2.  **Astrazione dei Dati Sensoriali**: I vari sensori e mappe dei veicoli autonomi utilizzano standard differenti. L'Adapter funge da "traduttore universale", nascondendo la complessità dei formati di input originali.
3.  **Unificazione dei Sistemi di Riferimento (Ego BEV)**: La matematica del Ray-Tracing e delle intersezioni semantiche necessita di un sistema di coordinate locale riferito all'auto (Ego frame). L'Adapter si assume l'esclusiva responsabilità di convertire tutte le coordinate spaziali (spesso memorizzate in coordinate globali UTM o GPS) in coordinate cartesiane locali centrate sul sensore LiDAR.

### 📋 Di cosa ha bisogno per funzionare?
Per poter implementare un Adapter concreto (come `NuScenesAdapter`), lo strato di astrazione necessita di accedere a tre sorgenti di informazioni fondamentali dal dataset sorgente:
*   **Dati Sensoriali Grezzi**: La nuvola di punti LiDAR 3D (espresse come distanze metriche relative al sensore).
*   **Dati di Stato del Veicolo (Ego Pose)**: La posizione assoluta ed il quaternione di rotazione del veicolo nel mondo reale in quell'istante temporale (necessari per allineare la mappa globale al veicolo).
*   **Metadati degli Ostacoli (Annotazioni)**: I Bounding Box 3D degli oggetti circostanti con la loro classificazione semantica (es. vettura, pedone) e i loro ID persistenti (per tracciare la memoria temporale tra frame successivi).
*   ** HD Map (Informazioni Semantiche)**: I vettori o poligoni della strada e dei marciapiedi che circondano il veicolo.

### 🏛️ Architettura delle Classi e Flusso di Esecuzione (SOLID DIP)

Il collegamento tra le classi segue il **Principio di Inversione delle Dipendenze (DIP)**: i moduli ad alto livello (Ray-Caster, Agente Bayesiano) non dipendono dai moduli a basso livello (SDK NuScenes), ma entrambi dipendono da un'astrazione comune.

```
       +----------------------------+
       |   BaseDatasetAdapter       |  <--- INTERFACCIA (Astratta)
       +----------------------------+
                     ^
                     |
       +----------------------------+
       |     NuScenesAdapter        |  <--- IMPLEMENTAZIONE (Concreta)
       +----------------------------+
                     |
                     |  restituisce
                     v
       +----------------------------+
       |   Formato Standard (Diz)   |  <--- CONTRATTO DATI (LocalBox)
       +----------------------------+
                     |
                     |  passato a
                     v
       +----------------------------+
       |  SOTARayCaster / Agent     |  <--- CLIENT PRINCIPALE (Algoritmi)
       +----------------------------+
```

1.  **L'Interfaccia (`BaseDatasetAdapter`)**: Dichiara le firme dei metodi. Garantisce che chiunque la estenda (es. un futuro `CarlaAdapter` o `KittiAdapter`) esponga esattamente le stesse funzioni.
2.  **L'Implementazione (`NuScenesAdapter`)**: È l'unica classe che importa l'SDK NuScenes. Esegue le trasformazioni geometriche e restituisce i dati convertiti in un dizionario Python con tipi primitivi e oggetti `LocalBox`.
3.  **L'Oggetto Standard (`LocalBox`)**: Sostituisce l'oggetto `Box` di NuScenes. In questo modo il client principale può fare `box.corners()` o `box.center` senza curarsi se il box proviene da NuScenes o Waymo.
4.  **Il Flusso d'Esecuzione (Main Core)**:
    *   Nel punto di ingresso (`__main__` o `batch_process_dataset`), istanziamo l'adapter concreto:
        `adapter = NuScenesAdapter(dataroot="./nuscenes")`
    *   Nel ciclo dei frame, estraiamo i dati:
        `frame_data = adapter.get_sample_data(idx)`
    *   Passiamo i dati standard al motore geometrico:
        `caster = SOTARayCaster(frame_data)`
        `caster.generate_known_zone()`
    
    In questo modo, il codice algoritmico (`SOTARayCaster`) rimane **immutato ed agnostico** rispetto alla sorgente dei dati.

---

## 🏗️ 1. L'Architettura Adapter per il Disaccoppiamento dei Dataset

Per evitare che il Ray-Caster e l'Agente Bayesiano dipendano direttamente dalle API proprietarie di NuScenes, utilizziamo questo strato intermedio per convertire qualsiasi sorgente dati in un **formato locale standardizzato (Ego BEV)**.

```
+------------------+      +--------------------+      +---------------------------+
|  NuScenes / SDK  | ---> |  NuScenesAdapter   | ---> | Formato Locale Standard   |
+------------------+      +--------------------+      | (LocalBox, LiDAR, Mappe)  |
                                                      +---------------------------+
                                                                    |
                                                                    v
                                                      +---------------------------+
                                                      |   SOTARayCaster (Core)    |
                                                      +---------------------------+
```

### 📁 Struttura del file `dataset_adapters.py`

Crea un file chiamato `dataset_adapters.py` che fungerà da gestore dei dati. All'interno implementerai tre elementi:

#### A. La Classe `LocalBox`
Questa classe rappresenta un Bounding Box 3D nel sistema di coordinate locale dell'auto (Ego frame). Deve esporre un'interfaccia standard per simulare un oggetto `Box` reale:
*   `__init__(self, center, wlh, corners_3d, name, token, visibility)`: memorizza il centroide $(X,Y,Z)$, dimensioni (width, length, height), gli 8 vertici 3D calcolati, la categoria, l'instance token persistente e la visibilità.
*   `corners(self)`: ritorna la matrice `(3, 8)` dei vertici.

#### B. La Classe Base Astratta `BaseDatasetAdapter`
Definisce il contratto comune per tutti i futuri dataset (es. KITTI, Waymo, Carla):
*   `get_num_samples(self) -> int`
*   `get_scene_indices(self) -> dict` (ritorna scene_token -> indici campioni ordinati cronologicamente).
*   `get_sample_data(self, idx: int) -> dict` (restituisce il dizionario standardizzato locale).

#### C. L'Adapter Concreto `NuScenesAdapter`
Contiene tutta la logica specifica per interrogare NuScenes:
1.  **Caricamento punti LiDAR**: Legge il file binario `.bin` in locale.
2.  **Trasformazione dei Box**: Recupera i box 3D in coordinate sensore (`boxes_ego` restituiti da `nusc.get_sample_data` sono già convertiti nel sistema di riferimento locale del LiDAR). Estrae `instance_token` e `visibility_token` dal database delle annotazioni NuScenes.
3.  **Rototraslazione delle Mappe HD**: Per estrarre le curve stradali in coordinate locali dell'ego-vehicle:
    *   Ottiene la posizione globale dell'ego vehicle $(t_x, t_y, t_z)$ e il quaternione di orientamento $q$.
    *   Costruisce la matrice di rotazione inversa: $R_{\text{inv}} = (q^{-1})_{\text{matrix}}$.
    *   Per ogni punto globale della mappa $\mathbf{P}_{\text{global}}$, calcola il corrispondente punto locale $\mathbf{P}_{\text{local}}$:
        $$\mathbf{P}_{\text{local}} = (\mathbf{P}_{\text{global}} - \mathbf{T}) \cdot R_{\text{inv}}^T$$

---

## 📌 2. Parametri Geometrici della Griglia Voxel (BEV)

Il Ray-Caster discretizza il mondo piana in una griglia 3D:
*   `GRID_RANGE = 40.0`: raggio di azione orizzontale in metri.
*   `VOXEL_SIZE = 0.4`: risoluzione spaziale (ogni voxel è $40\text{ cm} \times 40\text{ cm} \times 40\text{ cm}$).
*   `GRID_DIM = 200`: numero di voxel per lato ($2 \times 40 / 0.4 = 200$).
*   `Z_DIM = 24`: risoluzione verticale. Copre un'altezza reale di $9.6\text{ metri}$ (da $-2.4\text{ m}$ a $+7.2\text{ m}$ rispetto al sensore).

---

## 🚀 3. Sviluppo del Ray-Caster: `SOTARayCaster`

La classe `SOTARayCaster` non importa più `NuScenes`. Il suo costruttore accetta un dizionario standardizzato `frame_data` fornito dall'Adapter.

### Fase 3.1: Inizializzazione (`__init__`)
Alloca le tre matrici voxel 3D `(200, 200, 24)` inizializzate a zero:
*   `self.grid`: mappa dello spazio libero visibile (conosciuto).
*   `self.internal_shadows`: buffer per l'ombra propria degli ostacoli.
*   `self.box_shadows`: griglia dei coni d'ombra proiettati.

---

### Fase 3.2: Creazione del Depth Buffer Sferico (`generate_known_zone`)
Per implementare lo **Space Carving** in modo super-efficiente e vettorializzato, convertiamo i punti LiDAR cartesiani in coordinate sferiche angolari:

1.  **Conversione Sferica**: Per ogni punto LiDAR locale $(x, y, z)$, calcola:
    *   Distanza: $r = \sqrt{x^2 + y^2 + z^2}$
    *   Azimut: $\theta = \text{arctan2}(y, x)$ (angolo sul piano $X-Y$)
    *   Elevazione: $\phi = \text{arcsin}\left(\frac{z}{r}\right)$ (angolo verticale)
2.  **Mappatura del Depth Buffer**: Crea una griglia angolare sferica di risoluzione $1200 \times 400$ pixel. Calcola gli indici interi discreti di proiezione:
    $$u = \lfloor \frac{\theta + \pi}{2\pi} \times 1199 \rfloor, \quad v = \lfloor \frac{\phi + \pi/2}{\pi} \times 399 \rfloor$$
    Salva il raggio massimo $r$ per ciascuna cella $(u, v)$ per definire il limite di profondità visibile.
3.  **Filtro Antirumore**: Applica un filtro di massimo 2D (`scipy.ndimage.maximum_filter` di dimensione `(3, 5)`) sul buffer sferico per "chiudere" i vuoti artificiali tra i canali di scansione laser del LiDAR.
4.  **Space Carving**: Per ogni cella $(X,Y,Z)$ del nostro spazio voxel, calcola la sua posizione sferica proiettata $(u, v)$ nel depth buffer sferico ed ottieni la distanza visiva massima $d_{\text{limit}}$:
    *   Se la distanza euclidea del voxel dall'origine è minore di $d_{\text{limit}}$, imposta il voxel come noto: `self.grid[voxel] = 1`.

---

### Fase 3.3: Proiezione Geometrica delle Ombre (`generate_box_shadows`)
Proietta i coni d'ombra dietro ad ogni ostacolo (veicoli dinamici passati dall'adapter e muri statici rilevati dai voxel occupati):

1.  **Ombra Angolare BEV**: Estrai i 4 vertici 2D BEV del box dell'oggetto. Calcola gli angoli azimutali dal sensore. L'intervallo angolare dell'ombra sarà $[\theta_{\text{min}}, \theta_{\text{max}}]$ e inizierà a partire da $r_{\text{min}}$ (distanza minima dei vertici).
2.  **Arresto Dinamico (Ray Termination)**: Per ciascun angolo del cono d'ombra, cerca nella mappa sferica il primo punto LiDAR visibile a una distanza maggiore del box. L'ombra si ferma in corrispondenza di questo punto (evita che l'ombra attraversi pareti o ostacoli visibili).
3.  **Surgical Wall Clipping**: Se un raggio d'ombra colpisce una parete ad altezza elevata (rilevata da punti LiDAR verticali statici), interrompi geometricamente la proiezione dell'ombra davanti ad essa.
4.  **Ownership Spaziale**: In caso di sovrapposizioni, assegna ciascun pixel d'ombra all'ostacolo (`box.token`) geometricamente più vicino.

---

### Fase 3.4: Estrazione dei Poligoni a Guscio via Shell-Mapping Radiale

Una volta calcolata la griglia 3D delle occlusioni, proiettala in 2D BEV (`np.max(..., axis=2)`) ed applica una **chiusura morfologica** con kernel `(5, 5)` ($2.0\text{ m} \times 2.0\text{ m}$) per saldare micro-fessure radiali.

Per estrarre il poligono BEV pulito dell'ombra di ciascun ostacolo, implementa l'algoritmo di **Shell Mapping (Scansione Radiale)**:

1.  Recupera l'insieme dei pixel 2D dell'ombra appartenenti all'ostacolo dal registro di ownership.
2.  Converti i pixel in coordinate polari $(r, \theta)$ centrate sull'auto.
3.  Suddividi l'arco d'ombra $[\theta_{\text{min}}, \theta_{\text{max}}]$ in $N = 12$ raggi angolari equidistanti.
4.  Per ogni raggio angolare $\theta_k$:
    *   Filtra i pixel vicini ad esso (tolleranza $\pm 0.05\text{ rad}$).
    *   Trova il pixel più vicino (bordo interno $r_{\text{near}}$) e il più lontano (bordo esterno $r_{\text{far}}$).
    *   Calcola le coordinate cartesiane corrispondenti:
        $$\mathbf{P}_{\text{near}} = (r_{\text{near}} \cos\theta_k, \, r_{\text{near}} \sin\theta_k), \quad \mathbf{P}_{\text{far}} = (r_{\text{far}} \cos\theta_k, \, r_{\text{far}} \sin\theta_k)$$
5.  Crea il poligono unendo la catena dei punti esterni con la catena dei punti interni invertita:
    $$\text{Vertici Poligono} = [\mathbf{P}_{\text{far}, 0}, \ldots, \mathbf{P}_{\text{far}, 11}, \mathbf{P}_{\text{near}, 11}, \ldots, \mathbf{P}_{\text{near}, 0}]$$
6.  Sana eventuali auto-intersezioni dovute alla discretizzazione del reticolo tramite Shapely: `poly = ShapelyPolygon(vertices).buffer(0)`.
7.  Salva nel JSON del frame: area $\text{m}^2$, coordinate, distanza dall'ego e token dell'ostacolo che la proietta.

---

## 📚 4. Riferimenti per la Tesi di Laurea

*   **Adapter Pattern (Ingegneria del Software)**: Consulta il testo *Design Patterns* della "Gang of Four" per spiegare come questo pattern disaccoppia la logica algoritmica dal formato dei dati.
*   **Space Carving**: *OctoMap: An efficient probabilistic 3D mapping framework* (Hornung et al.). Spiega la teoria di tracciamento dello spazio libero LiDAR.
*   **Chiusura Morfologica**: *Digital Image Processing* (Gonzalez & Woods). Spiega come la dilatazione seguita dall'erosione permetta di congiungere segmenti d'ombra radiali vicini.
