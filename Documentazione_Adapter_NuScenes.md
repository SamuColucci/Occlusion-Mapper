# Documentazione: Pattern Adapter e NuScenesDatasetAdapter

Questo documento spiega nel dettaglio le scelte progettuali e implementative fatte per collegare il dataset NuScenes all'infrastruttura di analisi delle occlusioni (Ray-Caster e Agente Bayesiano).

## 1. Perché usare il Pattern Adapter?

In ingegneria del software, il **Pattern Adapter** viene utilizzato per far comunicare due interfacce incompatibili. Nel nostro caso, il Ray-Caster ha bisogno di dati in un formato specifico (nuvole di punti locali, bounding box con ID persistenti, poligoni semantici), ma ogni dataset (NuScenes, Waymo, KITTI) espone e memorizza i dati in modi completamente diversi.

Abbiamo quindi creato una classe astratta `AdapterDataset` che definisce i metodi standard che il nostro sistema si aspetta:
- `get_num_samples()`
- `get_scene_indices()`
- `get_sample_data(idx)`

La classe `NusceneDatasetAdapter` eredita da questa classe astratta e **traduce** i dati grezzi del database NuScenes nel formato standard richiesto dal nostro sistema. In questo modo:
- Il Ray-Caster non deve sapere nulla di come funziona NuScenes.
- Se un domani volessimo usare il dataset Waymo, basterebbe creare un `WaymoDatasetAdapter` senza toccare una riga di codice dell'Agente Bayesiano o del Ray-Caster. L'infrastruttura centrale rimane "pulita" e disaccoppiata dai dataset.

---

## 2. Implementazione di NusceneDatasetAdapter

### 2.1 Inizializzazione (`__init__`)
All'istanziazione, l'Adapter salva il `dataroot` (tramite `super().__init__(dataroot)`), carica in memoria l'oggetto `NuScenes` (che rappresenta il database relazionale) ed estrae la lista di tutti i fotogrammi (`self.all_samples = self.nusc.sample`).

### 2.2 Raggruppamento per Scene (`get_scene_indices`)
NuScenes è diviso in scene (es. 20 secondi consecutivi di guida). Questo metodo raggruppa tutti gli indici dei fotogrammi appartenenti alla stessa scena e li ordina cronologicamente.
1. Utilizziamo `enumerate(self.all_samples)` per ottenere l'indice numerico di ogni campione.
2. Usiamo `defaultdict(list)` per creare un dizionario `scenes` dove la chiave è il token della scena e il valore è una lista di tuple `(indice, timestamp)`.
3. Infine, ordiniamo cronologicamente la lista in base al timestamp usando `.sort(key=lambda x: x[1])` ed estraiamo solo gli indici per avere una lista pulita di indici da passare in ordine al nostro sistema.

### 2.3 Estrazione dei Dati (`get_sample_data`)
Questo è il cuore dell'Adapter: è il "fattorino" che raccoglie i dati, li normalizza nello stesso sistema di riferimento locale e li pacchetta.

#### A. Nuvola di Punti LiDAR
NuScenes usa una catena di puntatori. Dal `sample` otteniamo l'ID del sensore `LIDAR_TOP`. Tramite l'ID interroghiamo il database per avere il percorso del file `.bin`. I punti LiDAR sono salvati nativamente nel sistema **Ego-Frame** (il sistema di riferimento centrato sull'auto), quindi ci basta leggerli con `numpy` ed estrarre le coordinate X, Y, Z ignorando l'intensità.

#### B. Bounding Box e Agente Bayesiano
Il metodo `get_sample_data(lidar_token)` di NuScenes ci restituisce le scatole (Bounding Box) già ruotate in coordinate Ego. Tuttavia, il box nudo contiene solo la geometria. 
Per il nostro **Agente Bayesiano**, è fondamentale la **memoria temporale** (sapere che un'auto al frame 1 è la stessa del frame 2). Per questo iteriamo sui box e interroghiamo la `sample_annotation` per estrarre:
- L'`instance_token`: un ID persistente che identifica l'oggetto nel tempo.
- La visibilità: quanto è occluso l'oggetto.
Convertiamo poi il tutto in una nostra classe standard `LocalBox`.

#### C. La Mappa HD e la Rototraslazione
Questa è la parte matematicamente più complessa.
- **Il problema:** I punti LiDAR e i Box sono nel sistema locale dell'auto (dove l'auto è al centro (0,0) e l'asse X punta sempre avanti). La Mappa HD, invece, usa coordinate geografiche assolute globali. Se le sovrapponessimo così com'è, le ombre proiettate cadrebbero nei punti sbagliati.
- **La soluzione:**
  1. Estraiamo la posa dell'auto nel mondo reale in quel momento (`ego_pose`).
  2. Creiamo un **Quaternione** per gestire in modo robusto la rotazione e calcoliamo la matrice di **rotazione inversa** (`R_inv`).
  3. Risaliamo dalla scena al `log` per capire in quale località ci troviamo (es. `singapore-onenorth`) e carichiamo la mappa vettoriale corrispondente.
  4. Definiamo un'area di interesse (`box_coords`) di 100x100 metri attorno all'auto, per non dover elaborare l'intera mappa della città.
  5. Suddividiamo la mappa nei 4 layer semantici principali utili all'Agente Bayesiano: strada (`drivable_area`), marciapiede (`walkway`), parcheggio (`carpark_area`), strisce (`ped_crossing`).
  6. **Rototraslazione**: Per ogni poligono estratto, prendiamo le coordinate (X,Y) globali, **trasliamo** (sottraendo la posizione dell'auto) e poi **ruotiamo** (moltiplicando per la matrice `R_inv.T`) in modo da allineare la mappa globale all'orientamento frontale del LiDAR.

Il metodo restituisce infine un singolo dizionario unificato, pronto per essere consumato dall'algoritmo di Ray-Casting.
