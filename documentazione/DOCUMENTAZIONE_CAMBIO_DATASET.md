# Guida all'Integrazione di Nuovi Dataset (Pattern Adapter)

Questo documento descrive come estendere il software di analisi delle occlusioni 3D per supportare nuovi dataset di guida autonoma (es. Waymo Open Dataset, KITTI, Argoverse).

Grazie all'architettura basata sul **Pattern Adapter**, l'aggiunta di un nuovo dataset **non richiede alcuna modifica** al cuore del programma (Ray-Caster, Agente Bayesiano, Chiusura Morfologica). Il motore geometrico e probabilistico lavora esclusivamente su un formato dati standardizzato e non sa nulla delle API specifiche dei dataset.

---

## 1. Il Ruolo dell'AdapterDataset

Nel file `AdapterDataset.py` è definita l'interfaccia astratta che tutti i dataset devono rispettare. Un "Adapter" funge da traduttore: legge i dati nel formato proprietario del dataset (es. i file `.bin` e le API di NuScenes) e li converte in coordinate locali e dizionari Python standard comprensibili al nostro software.

Per integrare un nuovo dataset (es. Waymo), è sufficiente creare una nuova classe (es. `WaymoDatasetAdapter`) che eredita da `AdapterDataset` e implementa tre metodi fondamentali.

---

## 2. I Passaggi per Aggiungere un Nuovo Dataset

### Step 1: Creare la Classe e il Costruttore
Crea un nuovo file Python (es. `WaymoDatasetAdapter.py`) e implementa il costruttore `__init__`. Qui dovrai inizializzare l'SDK del nuovo dataset e conservare le liste interne dei frame/campioni.

```python
from AdapterDataset import AdapterDataset, LocalBox

class WaymoDatasetAdapter(AdapterDataset):
    def __init__(self, dataroot: str):
        super().__init__(dataroot)
        # Inizializza l'API del nuovo dataset qui
        # self.waymo_api = ...
```

### Step 2: Implementare `get_num_samples()`
Ritorna semplicemente il numero totale di frame validi (campioni) presenti nel dataset. Questo serve per i cicli e le barre di avanzamento.

```python
    def get_num_samples(self) -> int:
        return len(self.all_frames)
```

### Step 3: Implementare `get_scene_indices()`
Questo metodo deve restituire un dizionario che raggruppa gli indici numerici dei frame in base alla scena (sequenza di guida) a cui appartengono, ordinati cronologicamente.

- **Chiave**: Identificativo univoco della scena (stringa).
- **Valore**: Lista degli indici numerici dei frame, in ordine di tempo.

```python
    def get_scene_indices(self) -> dict:
        # Ritorna: { "scena_1": [0, 1, 2, 3], "scena_2": [4, 5, 6] }
        pass
```

### Step 4: Implementare `get_sample_data(idx)`
Questo è il motore di conversione principale. Riceve l'indice numerico di un fotogramma e deve restituire un dizionario contenente nuvola di punti, oggetti ostacolo (bounding box) e poligoni semantici, tutti espressi in **coordinate locali Ego-centriche** (auto al centro `[0,0]`, asse X in avanti).

Il dizionario restituito **deve avere esattamente questa struttura**:

```python
    def get_sample_data(self, idx: int) -> dict:
        # 1. Carica i punti LiDAR nel sistema locale (array nx3)
        # 2. Estrai gli ostacoli convertendoli in oggetti LocalBox
        # 3. Interroga la mappa HD rototraslando i poligoni in coordinate locali
        
        return {
            "lidar_token": "token_univoco_sensore",
            "sample_token": "token_univoco_frame",
            "scene_token": "token_della_scena",
            "points": pts_local,          # np.array di shape (N, 3)
            "boxes": local_boxes,         # Lista di oggetti LocalBox
            "semantic_map": semantic_map  # Dizionario di poligoni (strada, marciapiedi, ecc.)
        }
```

---

## 3. Classi e Formati Standard Richiesti

### L'oggetto `LocalBox`
Ogni ostacolo fisico deve essere inserito nella lista `boxes` istanziando la classe standard `LocalBox`. 
**Parametro Fondamentale:** Il campo `token` deve essere l'**ID persistente** dell'oggetto fisico. Lo stesso veicolo in due frame consecutivi deve avere lo stesso token affinché l'Agente Bayesiano possa sfruttare la memoria temporale.

```python
from AdapterDataset import LocalBox

lbox = LocalBox(
    center=centro_3d,          # np.array [x, y, z] in coord. locali
    wlh=dimensioni,            # np.array [width, length, height]
    corners_3d=vertici,        # np.array (3, 8) dei vertici
    name="vehicle.car",        # Nome categoria standard
    token="ID_persistente",    # ID univoco dell'oggetto tracciato nel tempo
    visibility="4"             # Livello visibilità / occlusione
)
```

### Il Dizionario `semantic_map`
I poligoni della mappa (che nei dataset sono solitamente in coordinate globali metriche o GPS) devono essere **sempre** traslati e ruotati per farli combaciare con l'orientamento locale del LiDAR. I layer richiesti sono:

```python
semantic_map = {
    'drivable_area': [array_poligono_1, array_poligono_2],
    'walkway': [array_poligono_3],
    'carpark_area': [],
    'ped_crossing': []
}
```
*Ogni elemento della lista è un numpy array 2D con i vertici (X,Y) del poligono espressi in metri dal centro del veicolo.*

---

## 4. Vantaggi di questa Architettura
Con questa struttura, il giorno in cui si deciderà di testare il software su Waymo o Argoverse, sarà sufficiente scrivere un file Python da circa 150-200 righe. Il programmatore non dovrà preoccuparsi di come funziona il calcolo delle ombre o l'aggiornamento delle probabilità Bayesiane. 
Se il nuovo `Adapter` supererà i test, il sistema funzionerà istantaneamente.

---

## 5. Switch Dinamico tramite Factory Pattern

Per rendere il punto di ingresso del programma (`EstrazioneZoneOccluse.py`) completamente universale e agnostico rispetto ai dataset installati, non effettuiamo l'importazione diretta delle classi specifiche (es. `from nuscenesDatasetAdapter import ...`).

Utilizziamo invece il **Factory Pattern**: una funzione dedicata che istanzia l'Adapter corretto a runtime in base a una stringa di configurazione. Gli import vengono fatti *localmente* all'interno dell'if: questo garantisce che se su un server non è installata l'SDK di Waymo, il programma non crasherà finché non chiederemo esplicitamente di usare Waymo.

```python
def create_adapter(dataset_name: str, dataroot: str):
    """
    Factory che restituisce l'implementazione concreta di AdapterDataset
    richiesta dalla configurazione dell'utente.
    """
    if dataset_name == "nuscenes":
        from nuscenesDatasetAdapter import nuscenesDatasetAdapter
        return nuscenesDatasetAdapter(dataroot)
        
    elif dataset_name == "waymo":
        # from waymoDatasetAdapter import waymoDatasetAdapter
        # return waymoDatasetAdapter(dataroot)
        pass
        
    else:
        raise ValueError(f"Il dataset {dataset_name} non è attualmente supportato.")
```

Con questo approccio, il file `main` si limiterà a chiamare `create_adapter("waymo", "./data")` e lavorerà sempre in totale astrazione.
