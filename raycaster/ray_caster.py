# Script con input il dataset NuScenes permette di ottenere in output le zone di occlusioni creati dagli oggetti
# in prossimità del veicolo ego, ovvero il veicolo su cui è presente il lidar. 
# Al fine di analizzarle per il miglioramento del sistema di guida autonoma.

# Libreria che permette di gestire la griglia sferica (voxel) e BEV (Bird's Eye View), 
# ovvero vista dall'alto della mappa 3D in modo da mostrarla in maniera bidimensionale
# essendo tutte gestite con le matrici NumPy.
import numpy as np

## Libreria usata per raggruppare dinamicamente i pixel delle ombre associandoli all'ID (token) dell'ostacolo corrispondente.
from collections import defaultdict

# Libreria utile per l'elaborazione di immagini e la segmentazione di oggetti
from scipy import ndimage

# Libreria utile per la geometria computazionale e la creazione di poligoni
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# Impostazioni iniziali della griglia
# GRID_RANGE: range di estensione della griglia (metri)
# VOXEL_SIZE: dimensione dei voxel (metri)
# GRID_DIM: dimensione della griglia in pixel (bidimensionale)
# Z_DIM: numero di voxel lungo l'asse verticale Z (l'altezza della griglia 3D)
# CASTER_CATEGORIES: categorie di oggetti da considerare

GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE)
Z_DIM = 24
CASTER_CATEGORIES = ["car", "truck", "bus", "trailer", "construction", "human", "bicycle", "motorcycle"]

# Classe per contenere i bordi di tutti gli oggetti sprovvisti dal dataset
# Dato che gli oggetti statici ne sono sprovvisti il codice genera automaticamente i bordi
class BoundingBox:
    # Metodo costruttore
    # min_x, max_x, min_y, max_y, min_z, max_z: coordinate minime e massime del bounding box
    # name: nome del bounding box
    # token: token del bounding box
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z, name, token):
        self.name = name
        self.token = token
        # Calcolo il centro del bounding box, al fine di calcolare la distanza con il lidar
        self.center = np.array([
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0
        ])
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

    # Metodo usato per ottenere i vertici del bounding box, simulando il comportamento del dataset
    # x_vals, y_vals, z_vals: coordinate dei vertici del bounding box
    # np.array: array contenente i vertici del bounding box
    # Restituisce gli 8 vertici tridimensionali del box
    def corners(self):
        # Faccia Superiore (tetto della scatola: usa max_z)
        # 1. Angolo in alto a destra:   max_x, max_y
        # 2. Angolo in basso a destra:  max_x, min_y
        # 3. Angolo in basso a sinistra: min_x, min_y
        # 4. Angolo in alto a sinistra:  min_x, max_y
        
        # Faccia Inferiore (base della scatola: usa min_z)
        # 5. Angolo in alto a destra:   max_x, max_y
        # 6. Angolo in basso a destra:  max_x, min_y
        # 7. Angolo in basso a sinistra: min_x, min_y
        # 8. Angolo in alto a sinistra:  min_x, max_y
        x_vals = [self.max_x, self.max_x, self.min_x, self.min_x, self.max_x, self.max_x, self.min_x, self.min_x]
        y_vals = [self.max_y, self.min_y, self.min_y, self.max_y, self.max_y, self.min_y, self.min_y, self.max_y]
        z_vals = [self.max_z, self.max_z, self.max_z, self.max_z, self.min_z, self.min_z, self.min_z, self.min_z]
        return np.array([x_vals, y_vals, z_vals])



# Classe per la gestione della logica di estrazione delle zone occluse a partire dai frame estartti dal dataset
# Calcola inizialmente la zona nota dal lidar e poi esegue una simulazione di raggi (ray casting) per stimare le zone non visibili.
# In seguito esegue una differenza fra la zona nota con quella delle occlusioni stimate per ottenere solo la zona occlusa, con un filtro per gestire eventuali zone occluse interne a zone note 
# causate da ostacoli uno dietro l'altro di altezza diversa
# Occlusione_Finale = Occlusione_Stimata AND (NOT Zona_Nota), con aggiunta dei rispettivi filtri per ogni tipo di zona occlusa
class RayCaster:
    # Metodo costruttore
    # frame_data: dizionario contenente i dati standardizzati del frame corrente
    # verbose: booleano per l'output su console
    def __init__(self, frame_data, verbose=True):
        # Salvo il flag per abilitare/disabilitare i messaggi in console
        self.verbose = verbose
        # Salvo il token del lidar
        self.lidar_token = frame_data["lidar_token"]
        # Salvo i punti del lidar
        self.pts = frame_data["points"]
        # Salvo la mappa semantica HD (strada, marciapiedi, parcheggi, strisce)
        self.semantic_map = frame_data.get("semantic_map", {})
        # Creo la griglia 3D per la zona nota dal lidar
        self.grid = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Creo la griglia 3D per le occlusioni interne (shadows interposte da più ostacoli)
        self.internal_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Creo la griglia 3D per i coni d'ombra simulati dai bounding box proiettati da tutti gli ostacoli
        self.box_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Creo la griglia 3D per le occlusioni finali
        self.occluded_final = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Salvo i bounding box del frame
        self._all_boxes = frame_data["boxes"]
        # Inizializzo a None la cache per la mappa 3D dei voxel occupati dai punti LiDAR
        self._lidar_occ = None




    # Metodo usato per ritornare i bounding box del frame degli oggetti dinamici ovvero già provvisti di bounding box
    # Restituisce i bounding box del frame, es. veicoli, pedoni
    def _get_all_boxes(self):
        return self._all_boxes
    
    # Metodo usato per ritornare i voxel occupati dai punti LiDAR
    # Restituisce i voxel occupati dai punti LiDAR, ovvero i cubi occupati dai punti LiDAR (fisicamente)
    def _get_lidar_occ(self):
        # Se la cache è vuota, calcola i voxel occupati dai punti LiDAR
        if self._lidar_occ is None:
            # Inizializzo una griglia 3D vuota per la mappa di occupazione fisica
            lidar_occ = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
            # Offset per centrare la griglia
            z_off = (Z_DIM // 4) * VOXEL_SIZE
            # Trasformazione delle coordinate dei punti da metriche (continue) a indici voxel (discreti)
            ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
            iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
            iz_p = ((self.pts[:, 2] + z_off) / VOXEL_SIZE).astype(int)
            # Maschera booleana per i voxel all'interno della griglia
            m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM) & (iz_p >= 0) & (iz_p < Z_DIM)
            # Accende a 1 i voxel in cui è caduto almeno un punto LiDAR (materia solida)
            lidar_occ[ix_p[m], iy_p[m], iz_p[m]] = 1
            # Aggiorno la cache
            self._lidar_occ = lidar_occ
        return self._lidar_occ


    # Metodo usato per calcolare e creae i bounding box degli oggetti statici (ovvero quelli non provvisti di bounding box)
    # Restituisce i bounding box degli oggetti statici, es. alberi, muretti, panchine...
    def detect_static_manmade_boxes(self, dynamic_vehicles):

        if self.verbose:
            print("Clustering geometrico e segmentazione degli ostacoli statici (manmade)...")
        # Crea una copia dei punti del LiDAR
        pts_static = self.pts.copy()

        # Maschera per filtrare i punti in base all'altezza (z),
        # manteniamo solo i punti che si trovano tra 0.5 e 4 metri di altezza da terra
        m_z = (pts_static[:, 2] > -0.5) & (pts_static[:, 2] < 4.0)
        # Applica la maschera ai punti statici
        pts_static = pts_static[m_z]

        # Distanza dei punti dal veicolo
        dist_to_ego = np.linalg.norm(pts_static[:, :2], axis=1)
       
        # Scarta i punti del veicolo e quelli troppo vicini (3 metri)
        pts_static = pts_static[dist_to_ego > 3.0]

        # Se il veicolo ha il bounding box, scarta i punti all'interno del veicolo
        inside_mask = np.zeros(len(pts_static), dtype=bool)
        
        # Per ogni veicolo dinamico
        for box in dynamic_vehicles:
            # Calcola gli angoli del bounding box
            corners = box.corners()
            # Trova i punti minimi e massimi del bounding box
            min_c = np.min(corners, axis=1)
            max_c = np.max(corners, axis=1)
            # Crea una maschera per i punti all'interno del bounding box (con un piccolo buffer)
            inside = (pts_static[:, 0] >= min_c[0] - 0.5) & (pts_static[:, 0] <= max_c[0] + 0.5) &                     (pts_static[:, 1] >= min_c[1] - 0.5) & (pts_static[:, 1] <= max_c[1] + 0.5)
            # Aggiorna la maschera totale (OR logico)
            inside_mask |= inside
        
        # Applica la maschera ai punti statici (scarta i punti all'interno dei veicoli dinamici)
        pts_static = pts_static[~inside_mask]
        
        # Risoluzione del clustering (dimensione dei voxel 2D in metri), permettendo al muro di essere considerato un unico ostacolo,
        # anche se composto da più punti anche se separati da pochi cm di distanza
        CLUST_RES = 0.8
        # Dimensione della griglia di clustering (2D)
        CLUST_DIM = int((GRID_RANGE * 2) / CLUST_RES)
        # Crea una griglia 2D vuota per il clustering, su cui proiettare i punti del lidar statici
        grid_bev = np.zeros((CLUST_DIM, CLUST_DIM), dtype=np.uint8)
        # Converte le coordinate da metri a indici della griglia
        # Aggiunge CLUST_DIM//2 per traslare il sensore (che è a 0,0 metri) nel centro esatto della matrice Numpy, evitando indici negativi
        ix_c = ((pts_static[:, 0] / CLUST_RES) + CLUST_DIM//2).astype(int)
        iy_c = ((pts_static[:, 1] / CLUST_RES) + CLUST_DIM//2).astype(int)

        # Maschera per i punti all'interno della griglia di clustering
        m_grid = (ix_c >= 0) & (ix_c < CLUST_DIM) & (iy_c >= 0) & (iy_c < CLUST_DIM)
        # Accende a 1 i voxel della griglia in cui sono caduti i punti statici
        grid_bev[ix_c[m_grid], iy_c[m_grid]] = 1
        # Etichetta i cluster (oggetti separati) al fine di separarli tra loro
        labeled_grid, num_clusters = ndimage.label(grid_bev)
        # Etichetta ogni punto con l'ID del suo cluster
        pt_labels = np.zeros(len(pts_static), dtype=int)
        # Assegna a ciascun punto l'ID del cluster a cui appartiene
        pt_labels[m_grid] = labeled_grid[ix_c[m_grid], iy_c[m_grid]]


        # Lista per contenere i bounding box degli oggetti statici
        static_boxes = []
        # Contatore per i bounding box
        box_counter = 0
        # Dimensione massima di un bounding box
        MAX_BOX_SIZE = 5.0
        # Scorre tutti i cluster
        for c_id in range(1, num_clusters + 1):
            # Punti appartenenti al cluster corrente
            cluster_pts = pts_static[pt_labels == c_id]
            # Se il cluster ha meno di 15 punti, viene ignorato
            if len(cluster_pts) < 15:
                continue
            # Coordinate minime e massime del cluster, al fine di trovare le dimensioni del box
            min_x, max_x = np.min(cluster_pts[:, 0]), np.max(cluster_pts[:, 0])
            min_y, max_y = np.min(cluster_pts[:, 1]), np.max(cluster_pts[:, 1])
            min_z, max_z = np.min(cluster_pts[:, 2]), np.max(cluster_pts[:, 2])

            # Dimensioni del cluster
            dx = max_x - min_x
            dy = max_y - min_y

            # Se il cluster è più lungo di MAX_BOX_SIZE (5 metri) e più lungo che largo, viene diviso in più bounding box
            if dx > MAX_BOX_SIZE and dx >= dy:
                # Numero di segmenti in cui dividere il cluster
                num_segments = int(np.ceil(dx / MAX_BOX_SIZE))
                # Coordinate degli spigoli dei segmenti
                x_edges = np.linspace(min_x, max_x, num_segments + 1)
                # Lista temporanea per i bounding box dei segmenti
                temp_boxes = []


                # Scorre i segmenti
                for seg_i in range(num_segments):
                    # Coordinate minime e massime del segmento
                    s_min_x, s_max_x = x_edges[seg_i], x_edges[seg_i+1]
                    # Maschera per i punti all'interno del segmento
                    m_seg = (cluster_pts[:, 0] >= s_min_x) & (cluster_pts[:, 0] <= s_max_x)
                    # Punti appartenenti al segmento
                    seg_pts = cluster_pts[m_seg]
                    # Se il segmento ha meno di 8 punti, viene ignorato
                    if len(seg_pts) < 8: continue
                    # Coordinate minime e massime del segmento
                    s_min_y, s_max_y = np.min(seg_pts[:, 1]), np.max(seg_pts[:, 1])
                    # Coordinate minime e massime del segmento
                    s_min_z, s_max_z = np.min(seg_pts[:, 2]), np.max(seg_pts[:, 2])
                    # Aggiunge il bounding box del segmento alla lista
                    temp_boxes.append((s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z))

                # Se ci sono segmenti validi
                if temp_boxes:
                    # Prende il primo e l'ultimo segmento ( gli estremi del cluster ) -> Per evitare di dividere in box piu piccoli cose come i muri
                    extremes = [temp_boxes[0]]
                    # Se ci sono più segmenti, aggiunge l'ultimo
                    if len(temp_boxes) > 1:
                        extremes.append(temp_boxes[-1])
                    # Scorre i segmenti estremi

                    for s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z in extremes:
                        # Assegna un token univoco al bounding box
                        token = f"static_manmade_{box_counter}"
                        box_counter += 1
                        # Aggiunge il bounding box alla lista
                        static_boxes.append(BoundingBox(s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z, "static.manmade", token))
            
            # Se il cluster è più largo di MAX_BOX_SIZE (5 metri) e più largo che alto, viene diviso in più bounding box
            elif dy > MAX_BOX_SIZE and dy > dx:
                # Numero di segmenti in cui dividere il cluster
                num_segments = int(np.ceil(dy / MAX_BOX_SIZE))
                y_edges = np.linspace(min_y, max_y, num_segments + 1)
                temp_boxes = []

                # Scorre i segmenti
                for seg_i in range(num_segments):
                    # Coordinate minime e massime del segmento
                    s_min_y, s_max_y = y_edges[seg_i], y_edges[seg_i+1]
                    # Maschera per i punti all'interno del segmento
                    m_seg = (cluster_pts[:, 1] >= s_min_y) & (cluster_pts[:, 1] <= s_max_y)
                    # Punti appartenenti al segmento
                    seg_pts = cluster_pts[m_seg]
                    # Se il segmento ha meno di 8 punti, viene ignorato
                    if len(seg_pts) < 8: continue
                    # Coordinate minime e massime del segmento
                    s_min_x, s_max_x = np.min(seg_pts[:, 0]), np.max(seg_pts[:, 0])
                    # Coordinate minime e massime del segmento
                    s_min_z, s_max_z = np.min(seg_pts[:, 2]), np.max(seg_pts[:, 2])
                    # Aggiunge il bounding box del segmento alla lista
                    temp_boxes.append((s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z))

                # Se ci sono segmenti validi
                if temp_boxes:
                    # Mantiene solo il primo e l'ultimo segmento del muro (gli estremi).
                    # Questo perché, geometricamente, i confini della zona occlusa sono determinati 
                    # solo dagli spigoli esterni dell'ostacolo. I segmenti centrali genererebbero 
                    # ombre ridondanti (completamente contenute all'interno di quelle degli estremi),
                    # causando solo uno spreco inutile di potenza di calcolo.
                    extremes = [temp_boxes[0]]
                    # Se ci sono più segmenti, aggiunge l'ultimo
                    if len(temp_boxes) > 1:
                        extremes.append(temp_boxes[-1])

                    # Scorre i segmenti estremi
                    for s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z in extremes:
                        # Assegna un token univoco al bounding box
                        token = f"static_manmade_{box_counter}"
                        box_counter += 1
                        # Aggiunge il bounding box alla lista
                        static_boxes.append(BoundingBox(s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z, "static.manmade", token))
            
            # Se il cluster non è abbastanza lungo o largo, viene trattato come un singolo bounding box
            else:
                # Assegna un token univoco al bounding box
                token = f"static_manmade_{box_counter}"
                box_counter += 1
                # Aggiunge il bounding box alla lista
                static_boxes.append(BoundingBox(min_x, max_x, min_y, max_y, min_z, max_z, "static.manmade", token))
        
        # Stampa il numero di pseudo-box statici rilevati
        if self.verbose:
            print(f"Rilevati {len(static_boxes)} pseudo-box statici (manmade).")

        # Restituisce la lista dei bounding box statici
        return static_boxes



    # Metodo per calcolare lo spazio noto visto dal sensore attraverso l'utilizzo della tecnica del Raycasting
    # Sfrutta una mappa 3D suddivisa in cubetti (Voxels) in cui ogni cubetto può essere: 
    # Spazio noto: valore 1
    # Spazio non conosciuto: valore 0
    def generate_known_zone(self):
        if self.verbose:
            print("Analisi dei blocchi attraversati (Ray-Tracing simulation)...")
        
        # Definisce la risoluzione della "sfera panoramica" del sensore: 1200 spicchi orizzontali e 400 verticali
        AZ_RES, EL_RES = 1200, 400 
        # Inizializza la mappa di profondità (depth_buffer) con -1.0, indicando che inziale mente nessun raggio ha colpito nulla
        depth_buffer = np.full((AZ_RES, EL_RES), -1.0)
        # Calcola la distanza di ogni singolo punto dal sensore
        r = np.linalg.norm(self.pts, axis=1)
        # Calcola l'azimuth (angolo orizzontale) per ciascun punto, convertendo le coordinate cartesiane in coordinate polari
        az = np.arctan2(self.pts[:, 1], self.pts[:, 0])
        # Calcola l'elevazione (angolo verticale) per ciascun punto, convertendo le coordinate cartesiane in coordinate polari
        el = np.arcsin(np.clip(self.pts[:, 2] / (r + 1e-6), -1, 1))
        # Converte l'angolo orizzontale continuo in un indice intero (pixel u) per la matrice del buffer
        u = ((az + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(int)
        # Converte l'angolo verticale continuo in un indice intero (pixel v) per la matrice del buffer
        v = ((el + np.pi/2) / np.pi * (EL_RES - 1)).astype(int)

        # Maschera per considerare solo i punti LiDAR entro il nostro raggio di interesse (es. 40 metri)
        mask_range = r < GRID_RANGE
        # Applica la maschera agli indici e alle distanze
        u_masked = u[mask_range]
        v_masked = v[mask_range]
        r_masked = r[mask_range]
        # Aggiorna il buffer di profondità con le distanze dei punti
        np.maximum.at(depth_buffer, (u_masked, v_masked), r_masked)

        if self.verbose:
            print("Chiusura buchi tra i raggi (Interpolazione angolare)...")
        # Allarga i valori massimi ai pixel adiacenti (finestra 3x5) per tappare i "buchi" vuoti lasciati dai raggi laser troppo sottili
        depth_buffer = ndimage.maximum_filter(depth_buffer, size=(3, 5))


        self.depth_buffer_ref = depth_buffer
        # Crea le coordinate cartesiane per la griglia di voxels
        x = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z = (np.arange(Z_DIM) - Z_DIM//4) * VOXEL_SIZE 
        # Crea una griglia di coordinate cartesiane 3D
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        # Calcola la distanza di ogni voxel dal sensore
        dist = np.sqrt(xv**2 + yv**2 + zv**2)


        # Maschera per i voxel che rientrano nel raggio di interesse
        valid_mask = dist <= GRID_RANGE
        # Calcola l'angolo orizzontale e verticale sotto cui il sensore "vede" ogni singolo voxel 3D
        az_v = np.arctan2(yv, xv)
        el_v = np.arcsin(np.clip(zv / (dist + 1e-6), -1.0, 1.0))


        # Converte questi angoli nei "pixel" sferici (u,v) usati prima
        u_v = ((az_v + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(np.int32)
        v_v = ((el_v + np.pi/2) / np.pi * (EL_RES - 1)).astype(np.int32)


        # Limita gli indici ai bordi validi della griglia
        u_v = np.clip(u_v, 0, AZ_RES - 1)
        v_v = np.clip(v_v, 0, EL_RES - 1)
        # Recupera il limite di profondità per ogni voxel
        d_limit = depth_buffer[u_v, v_v]
        # Un voxel viene considerato "Spazio Noto" (illuminato dal laser) SE E SOLO SE:
        # 1. Si trova entro i limiti geometrici della mappa (valid_mask)
        # 2. In quella direzione c'era effettivamente un muro/ostacolo (d_limit > 0)
        # 3. Il voxel si trova FISICAMENTE PRIMA di quell'ostacolo (dist < d_limit)
        known_mask = valid_mask & (d_limit > 0) & (dist < d_limit)
        # Accende a 1 i voxel della griglia generale
        self.grid[known_mask] = 1
        return self.grid

    # Metodo per simualare le ombre proiettate dagli oggetti visti dal lidar
    # Sfruttiamo il concetto di proiezione geometrica 2D
    def generate_box_shadows(self):
        if self.verbose:
            print("Simulazione coni d'ombra in corso...")

        # Inizializza la mappa 3D delle ombre (tutta a zeri)
        # e la griglia per ricordare a chi appartiene ogni ombra (vettorizzata)
        self.box_shadows = np.zeros_like(self.grid)
        self.cone_ownership = {}
        self.cone_ownership_grid = np.full((GRID_DIM, GRID_DIM), None, dtype=object)
        
        # Raccoglie tutti i veicoli dinamici (scartando categorie inutili, come i coni stradali)
        vehicles = [b for b in self._get_all_boxes() if any(c in b.name.lower() for c in CASTER_CATEGORIES)]
        # Raccoglie gli oggetti statici
        self.static_boxes = self.detect_static_manmade_boxes(vehicles)

        # Fonde gli oggetti dinamici con quelli statici per avere un unica lista di oggetti
        all_casters = self.static_boxes + vehicles

        # Calcoliamo le coordinate della mappa 2D
        x_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z_off = (Z_DIM // 4) * VOXEL_SIZE

        # Crea una griglia 2D di coordinate x, y
        xv_2d, yv_2d = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Calcola la distanza (dv) e l'angolo (av) di OGNI PIXEL 2D dal sensore
        dv_2d = np.sqrt(xv_2d**2 + yv_2d**2)
        av_2d = np.arctan2(yv_2d, xv_2d)

        # Per ogni cella, calcola la distanza dal punto più vicino al sensore
        min_dist_to_center = np.full((GRID_DIM, GRID_DIM), np.inf)
        # Recupera la griglia di occupazione e il buffer di profondità
        lidar_occ = self._get_lidar_occ()
        depth_buffer = self.depth_buffer_ref
        AZ_RES = depth_buffer.shape[0]

        # Mappa gli angoli della griglia 2D negli indici della sfera LiDAR (u) usati in precedenza
        u_v_2d = ((av_2d + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(np.int32)

        # Limita gli indici ai bordi validi della griglia
        u_v_2d = np.clip(u_v_2d, 0, AZ_RES - 1)

        # Proietta ogni oggetto (caster) sulla griglia 2D per calcolarne l'ombra
        for box in all_casters:
            # Estrae le coordinate 3D degli 8 vertici del veicolo
            corners_3d = box.corners()
            # Trova il rettangolo minimo (min_c, max_c) che racchiude il veicolo
            min_c, max_c = np.min(corners_3d, axis=1), np.max(corners_3d, axis=1)
            
            # Calcola le coordinate 2D di interesse in metri
            r_r = np.clip([int((min_c[0]+GRID_RANGE)/VOXEL_SIZE), int((max_c[0]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            c_r = np.clip([int((min_c[1]+GRID_RANGE)/VOXEL_SIZE), int((max_c[1]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            z_r = np.clip([int((min_c[2]+z_off)/VOXEL_SIZE), int((max_c[2]+z_off)/VOXEL_SIZE)], 0, Z_DIM-1)

            # Se dentro il Bounding Box dell'auto non ci è caduto neanche un punto LiDAR,
            # significa che l'auto non è fisicamente visibile (magari è dietro un muro)
            # Quindi non può proiettare ombra
            if not np.any(lidar_occ[r_r[0]:r_r[1]+1, c_r[0]:c_r[1]+1, z_r[0]:z_r[1]+1]):
                continue

            # Proietta i vertici del veicolo sulla griglia 2D (ignora la quota z)
            corners = box.corners()[:2, :].T
            # Calcola la distanza minima e massima dal sensore ai vertici del veicolo
            r_min = np.min(np.linalg.norm(corners, axis=1))
            r_max = np.max(np.linalg.norm(corners, axis=1))
            az_c = np.arctan2(corners[:, 1], corners[:, 0])

            # Controllo per evitare che un'auto presente esattamente dietro il sensore proietti un ombra a 360°
            # In tal caso, l'ampiezza dell'angolo sarebbe di 2pi (360°) e si verificherebbe un errore di wrapping
            # Il wrapping si verifica quando l'angolo cambia bruscamente da +180° a -180° (o viceversa)
            if np.max(az_c) - np.min(az_c) > np.pi: 
                az_c[az_c < 0] += 2 * np.pi
            
            # Estrae l'intervallo di angoli (az_min, az_max) che racchiude il veicolo
            az_min, az_max = np.min(az_c), np.max(az_c)
            av_shifted = av_2d.copy()
            if az_max > np.pi:
                av_shifted[av_shifted < az_min] += 2 * np.pi

            # # Identifica tutti i raggi LiDAR (u) che sbattono esattamente contro quest'auto
            active_u = np.unique(u_v_2d[(dv_2d > r_min) & (av_shifted >= az_min) & (av_shifted <= az_max)])
            # Inizialmente supponiamo che l'ombra sia infinita (lunga 30 metri dietro l'auto)
            ray_limits = np.full(AZ_RES, r_min + 30.0)

            # Interroga la sfera LiDAR per vedere se i raggi hanno colpito qualcosa dietro l'auto (oltre r_max, per non mozzare bus e furgoni)
            for u in active_u:
                hits_behind = depth_buffer[u, :][depth_buffer[u, :] > (r_max + 1.0)]
                if len(hits_behind) > 0:
                    ray_limits[u] = np.min(hits_behind)
            # Se c'è un muro, l'ombra dell'auto viene tagliata e finisce contro il muro
            limit_grid = ray_limits[u_v_2d]

            # CONDIZIONE DELL'OMBRA: un pixel sul terreno è in ombra SE:
            # 1. Si trova DIETRO l'auto (dv_2d > r_min) ma PRIMA del muro successivo (dv_2d <= limit_grid)
            # 2. Si trova ESATTAMENTE NELLO SPICCHIO ANGOLARE dell'auto (tra az_min e az_max)
            cand_mask = (dv_2d > r_min) & (dv_2d <= limit_grid)
            cand_mask &= (av_shifted >= az_min) & (av_shifted <= az_max)

            # Se non c'è nessuna cella in ombra, passa all'auto successiva
            if not np.any(cand_mask):
                continue
            # Aumenta il margine di sicurezza
            d_wall_limit = np.full(AZ_RES, np.inf)
            active_u = np.unique(u_v_2d[cand_mask])

            # Cerca i muri che bloccano l'ombra
            for u in active_u:
                # Prende tutti i punti dietro l'auto (dal centro in poi)
                d_hits_wall = depth_buffer[u, 200:] 
                # Filtra quelli più lontani del bordo posteriore dell'auto (r_max + 1.0)
                wall_hits = d_hits_wall[d_hits_wall > (r_max + 1.0)]
                # Se trova almeno 2 muri, taglia l'ombra al primo (più vicino)
                if len(wall_hits) >= 2:
                    d_wall_limit[u] = np.min(wall_hits)

            # Ricalcola la maschera con il limite di profondità corretto
            limit_grid = d_wall_limit[u_v_2d]
            cand_mask &= (dv_2d <= limit_grid)

            # Se non ci sono celle valide, passa all'auto successiva
            if not np.any(cand_mask):
                continue
            # Assegna l'ombra all'auto
            self.box_shadows[cand_mask, :] = 1
            # Trova le celle che sono ombre valide

            # Salva nella griglia NumPy a chi appartiene l'ombra (Vettorizzato, zero cicli for)
            dist_to_center = np.sqrt((xv_2d - box.center[0])**2 + (yv_2d - box.center[1])**2)
            better_mask = cand_mask & (dist_to_center < min_dist_to_center)
            min_dist_to_center[better_mask] = dist_to_center[better_mask]
            self.cone_ownership_grid[better_mask] = box.token

        # Restituisce la griglia 3D con tutte le ombre di tutte le auto e palazzi
        return self.box_shadows


    # Metodo per simulare le zone occluse interne che altrimenti sarebbero indicate come note per il funzionamento alla base del raycasting
    # Essendo zone occluse circoscritte da zone note che necessitano anch'esse di un'analisi
    # Se abbiamo un'auto a destra, e un palazzo ancora più a destra, i laser del LiDAR (essendo sul tetto)
    # spesso passano sopra l'auto e colpiscono il palazzo. Il Ray-Tracing standard 3D considererebbe
    # lo spazio tra l'auto e il palazzo come "Noto/Libero". Tuttavia, a livello della strada, 
    # quello spazio è totalmente cieco
    # Questo metodo lavora in 2D per forzare la creazione, dato che il problema dovuto agli ostacoli è in prossimità dell'asfalto
    # di un'ombra (Cuneo) tra l'auto e il primo ostacolo retrostante, mettendo in sicurezza la mappa
    def find_object_occlusion_wedges(self):
        if self.verbose:
            print("Pulizia e mascheramento cunei (Safety Buffer)...")

        # Inizializza la mappa 3D delle ombre interne (i cunei) a zero
        self.internal_shadows = np.zeros_like(self.grid)
        # Crea la mappa 2D degli oggetti occupati, utile per identificare dove sono gli ostacoli
        # In cui 1 significa che c'è sicuramento un ostacolo
        occupied_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.uint8)
        ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM)
        occupied_bev[ix_p[m], iy_p[m]] = 1

        # Dilata questi ostacoli di 1 pixel (binary_dilation)
        # Questo crea un "bordo di sicurezza" attorno ai muri veri, per evitare che i nostri 
        # cunei finti vadano a sovrascrivere muri reali
        safety_mask = ndimage.binary_dilation(occupied_bev, structure=np.ones((3, 3)))

        # Pre-calcola la griglia 2D di coordinate, distanze ed angoli (una sola volta per tutte le auto)
        x_c = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y_c = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        xv_2d, yv_2d = np.meshgrid(x_c, y_c, indexing='ij')
        dv_2d = np.sqrt(xv_2d**2 + yv_2d**2)
        av_2d = np.arctan2(yv_2d, xv_2d)

        # Cicla solo sui veicoli grossi (auto, camion, bus). I pedoni o le bici non creano grossi cunei.
        for box in self._get_all_boxes():
            if not any(c in box.name.lower() for c in ["vehicle", "truck", "bus"]): continue
            center = box.center
            dist_A = np.linalg.norm(center[:2])
            corners = box.corners()

            # Calcola l'apertura angolare (spicchio) del veicolo
            az_c = np.arctan2(corners[1], corners[0])
            if np.max(az_c) - np.min(az_c) > np.pi: az_c[az_c < 0] += 2*np.pi
            
            # Definisce 3 angoli critici: Sinistra (az_L), Centro (az_C) e Destra (az_R).
            # I +/- 0.03 radianti restringono leggermente lo spicchio per non sbavare fuori dall'auto
            az_L, az_R = np.min(az_c) + 0.03, np.max(az_c) - 0.03
            az_C = (az_L + az_R) / 2

            # Spara un raggio immaginario dal retro dell'auto dritto verso l'esterno, 
            # per vedere a che distanza c'è un muro (o un'altra auto) a fare da tappo
            # Inizializza la distanza del muro a -1 (che significa "non trovato")
            found_d_back = -1
            
            # Crea 40 punti di controllo lungo una linea dritta
            # Parte da 4 metri dietro l'auto (dist_A + 4) per non colpire l'auto stessa,
            # e arriva fino al limite della mappa (GRID_RANGE)
            for d in np.linspace(dist_A + 4, GRID_RANGE, 40):
                # Calcola le coordinate metriche X e Y del punto di controllo
                # lungo l'angolo centrale (az_C)
                tx, ty = d * np.cos(az_C), d * np.sin(az_C)
                
                # Converte i metri in coordinate pixel (gx, gy) della griglia 2D
                gx, gy = int(tx/VOXEL_SIZE + GRID_DIM//2), int(ty/VOXEL_SIZE + GRID_DIM//2)
                
                # Assicura che i pixel non escano fuori dai bordi della mappa
                if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM:
                    if occupied_bev[gx, gy]:
                        found_d_back = d
                        break


            # Se ha trovato un muro solido, E questo muro dista più di 5 metri dall'auto 
            # (se è più vicino, non c'è abbastanza spazio fisico per nascondere un pericolo grosso)
            if found_d_back > 0 and (found_d_back - dist_A) > 5.0:
                dists_back = []
                
                # Spara 3 raggi laser separati (Sinistro, Centrale, Destro)
                for angle in [az_L, az_C, az_R]:
                    d_edge = GRID_RANGE
                    for d in np.linspace(dist_A + 2, GRID_RANGE, 30):
                        tx, ty = d * np.cos(angle), d * np.sin(angle)
                        gx, gy = int(tx/VOXEL_SIZE + GRID_DIM//2), int(ty/VOXEL_SIZE + GRID_DIM//2)
                        # Se il raggio colpisce un ostacolo, si ferma
                        if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM and occupied_bev[gx, gy]:
                            d_edge = d
                            break
                    # Salva le distanze di impatto dei 3 raggi
                    dists_back.append(d_edge)
                
                # Assegna le 3 distanze trovate a Sinistra, Centro e Destra
                d_L, d_C, d_R = dists_back
                
                # Vettorizzazione NumPy della maschera dei cunei (priva di doppi cicli for)
                av_shifted = av_2d.copy()
                if az_R > np.pi:
                    av_shifted[av_shifted < az_L] += 2 * np.pi

                t = np.where(
                    av_shifted < az_C,
                    (av_shifted - az_L) / (az_C - az_L + 1e-6),
                    (av_shifted - az_C) / (az_R - az_C + 1e-6)
                )
                d_limit = np.where(
                    av_shifted < az_C,
                    d_L * (1 - t) + d_C * t,
                    d_C * (1 - t) + d_R * t
                )

                in_wedge = (az_L <= av_shifted) & (av_shifted <= az_R)
                in_shadow = in_wedge & (dv_2d >= dist_A + 2.0) & (dv_2d < d_limit - 1.0) & (~safety_mask)
                self.internal_shadows[in_shadow, :] = 1

        # Estrae solo il livello del terreno (z=0) dalle ombre interne appena calcolate
        shadow_bev = self.internal_shadows[:, :, 0]
        
        # Etichetta i gruppi di pixel d'ombra uniti (come abbiamo fatto per i cluster dei palazzi)
        labeled, num_feat = ndimage.label(shadow_bev)
        
        if num_feat > 0:
            # Calcola la grandezza (in numero di pixel) di ciascuna macchia d'ombra
            sizes = ndimage.sum(shadow_bev, labeled, range(num_feat + 1))
            
            # Crea una maschera per identificare le macchie più piccole di 20 pixel (le "briciole")
            mask_noise = (sizes < 20)
            
            # Cancella (imposta a 0) tutte le briciole dalla mappa 2D
            shadow_bev[mask_noise[labeled]] = 0
            
            # Alza questi cunei d'ombra per tutta l'altezza della griglia 3D via broadcasting (senza cicli for)
            self.internal_shadows[:] = shadow_bev[:, :, np.newaxis]
                
        # Restituisce il volume 3D finale dei cunei di sicurezza pulito
        return self.internal_shadows



    # Metodo per generare la zona d'ombra finale e permetterne la sua estrazione
    def extract_occlusion_zones(self):
        if self.verbose:
            print("Calcolo visibilità finale con proiezione verticale...")
            
        # Crea la griglia finale tenendo conto delle ombre interne
        self.final_grid = self.grid.copy()
        
        # Forza a 0 (Non Noto) tutti i voxel che cadono nei cunei di sicurezza interni,
        # correggendo così l'errore del laser che scavalcava le auto
        self.final_grid[self.internal_shadows == 1] = 0
        
        # Se c'è anche un solo voxel noto (1) in tutta la colonna verticale Z, 
        # consideriamo quel pixel 2D a terra come "Visibile" (cioè sicuro)
        bev_visibility = np.max(self.final_grid, axis=2)
        
        # Prendiamo le ombre grezze (le potenziali zone rosse calcolate dietro i bounding box)
        temp_occluded = self.box_shadows.copy().astype(bool)

        # Cancella le ombre dove la visibilità è accertata usando broadcasting 3D
        temp_occluded &= (bev_visibility == 0)[:, :, np.newaxis]
        
        # Schiaccia di nuovo le ombre verificate sul pavimento (2D)
        final_red_map = np.max(temp_occluded, axis=2)
        
        # Tappa eventuali minuscoli buchi di 1-2 pixel
        # all'interno delle grosse macchie d'ombra, per rendere la mappa più pulita da vedere.
        closed_red_map = ndimage.binary_closing(final_red_map, structure=np.ones((5, 5)))
        
        # Salviamo la mappa in modo che extract_polygons possa usarla
        self.closed_red_map = closed_red_map
        
        # Restituisce la mappa 2D definitiva delle zone occluse
        return closed_red_map



    # È il metodo principale che viene richiamato dall'esterno per calcolare la mappa
    # Esegue in ordine rigoroso i 4 step della simulazione geometrica del Ray-Caster
    def get_occlusion_mask(self):
        self.generate_known_zone()
        self.generate_box_shadows()
        self.find_object_occlusion_wedges()
        return self.extract_occlusion_zones()



    # Metodo di estrazione finale: converte la mappa a pixel in forme vettoriali (Poligoni)
    # Raggruppa i pixel in ombra in base al veicolo che li ha generati (tramite cone_ownership)
    # e calcola l'area effettiva (in metri quadrati) di ogni zona occlusa
    def extract_polygons(self):
        if self.verbose:
            print("Estrazione poligoni in corso (Geometric-Ownership Mode)...")
            
        vehicles = [b for b in self._get_all_boxes() if any(c in b.name.lower() for c in CASTER_CATEGORIES)]
        all_casters = self.static_boxes + vehicles
        occlusions_output = []
        
        # Prende la mappa rossa 2D schiacciata a terra
        final_red_map = self.closed_red_map.astype(np.uint8)
        
        # Crea un dizionario in cui la chiave è il token del veicolo, e il valore è la lista dei pixel 
        # Trova direttamente le coordinate dei soli pixel d'ombra (evita 40.000 cicli for inutili)
        object_occlusions = defaultdict(list)
        shadow_pixels = np.argwhere(final_red_map > 0)
        for ix, iy in shadow_pixels:
            token = self.cone_ownership_grid[ix, iy]
            if token:
                object_occlusions[token].append((int(ix), int(iy)))
                        
        # Pre-costruisce l'unione dei poligoni della mappa semantica per il calcolo delle frazioni di superficie
        def _build_layer_union(poly_list):
            polys = []
            for coords in poly_list:
                if len(coords) >= 3:
                    try:
                        p = ShapelyPolygon(coords)
                        if not p.is_valid: p = p.buffer(0)
                        if not p.is_empty: polys.append(p)
                    except Exception: pass
            return unary_union(polys) if polys else None

        road_union = _build_layer_union(self.semantic_map.get('drivable_area', []))
        side_union = _build_layer_union(self.semantic_map.get('walkway', []))
        carpark_union = _build_layer_union(self.semantic_map.get('carpark_area', []))
        crosswalk_union = _build_layer_union(self.semantic_map.get('ped_crossing', []))

        # Elaborazione di ogni singolo ostacolo
        for box in all_casters:
            intersection_pixels = object_occlusions.get(box.token)
            
            # Decide la soglia minima di pixel affinché un'ombra sia considerata valida
            is_small = any(c in box.name.lower() for c in ["human", "bicycle", "motorcycle"])
            if "static" in box.name.lower():
                min_px = 2
            else:
                min_px = 3 if is_small else 5
                
            # Se l'ostacolo non ha proiettato ombre, passa al prossimo
            if not intersection_pixels:
                continue
            # Se l'ombra è troppo piccola (sotto la soglia), la scarta come rumore
            if len(intersection_pixels) < min_px:
                continue
                
            # Trasforma gli indici dei pixel (riga, colonna) in coordinate metriche reali (X, Y)
            int_pixels_arr = np.array(intersection_pixels)
            r_idx, c_idx = int_pixels_arr[:, 0], int_pixels_arr[:, 1]
            xv_int = (r_idx - GRID_DIM//2) * VOXEL_SIZE
            yv_int = (c_idx - GRID_DIM//2) * VOXEL_SIZE
            
            # Calcola distanza e angolo di ogni pixel dell'ombra rispetto al sensore
            az_int = np.arctan2(yv_int, xv_int)
            dist_int = np.sqrt(xv_int**2 + yv_int**2)
            if np.max(az_int) - np.min(az_int) > np.pi: 
                az_int[az_int < 0] += 2*np.pi
                
            a_min, a_max = np.min(az_int), np.max(az_int)
            
            # Taglia l'ombra in 12 "fette" angolari con tolleranza dinamica proporzionale al passo
            angular_step = (a_max - a_min) / 12.0
            tolerance = max(angular_step / 2.0, 0.01)
            sample_angs = np.linspace(a_min, a_max, 12)
            inner_pts, outer_pts = [], []
            for ang in sample_angs:
                m_ray = (np.abs(az_int - ang) < tolerance) | (np.abs(az_int + 2*np.pi - ang) < tolerance)
                if np.any(m_ray):
                    r_near = np.min(dist_int[m_ray])
                    r_far = np.max(dist_int[m_ray])
                    inner_pts.append([round(float(r_near * np.cos(ang)), 2), round(float(r_near * np.sin(ang)), 2)])
                    outer_pts.append([round(float(r_far * np.cos(ang)), 2), round(float(r_far * np.sin(ang)), 2)])
                    
            if not inner_pts:
                continue
                
            # Unisce i punti lontani e i punti vicini (al contrario) per chiudere un perimetro poligonale grezzo
            polygon_m = outer_pts + inner_pts[::-1]
            sh_poly = None
            
            # Usa la libreria shapely per eliminare vertici inutili
            try:
                sh_poly = ShapelyPolygon(polygon_m)
                if not sh_poly.is_valid:
                    sh_poly = sh_poly.buffer(0)
                sh_poly_simple = sh_poly.simplify(0.2, preserve_topology=True)
                if not sh_poly_simple.is_valid:
                    sh_poly_simple = sh_poly_simple.buffer(0)
                
                # ponytail: se il poligono si è diviso in più parti disgiunte, prendiamo la componente con area maggiore
                if sh_poly_simple.geom_type == 'MultiPolygon':
                    sh_poly_simple = max(sh_poly_simple.geoms, key=lambda p: p.area)
                    
                polygon_m = [[round(x, 2), round(y, 2)] for x, y in sh_poly_simple.exterior.coords]
                # Rimuove il punto duplicato finale se presente
                if len(polygon_m) > 1 and polygon_m[0] == polygon_m[-1]:
                    polygon_m = polygon_m[:-1]
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Shapely fallito per {box.token}: {e}")

            # Calcolo delle frazioni di superficie del terreno (road, sidewalk, carpark, crosswalk, terrain)
            occ_area = sh_poly.area if (sh_poly is not None and not sh_poly.is_empty) else 0.0
            if occ_area > 0:
                road_frac = round(min(float(sh_poly.intersection(road_union).area / occ_area) if road_union else 0.0, 1.0), 4)
                side_frac = round(min(float(sh_poly.intersection(side_union).area / occ_area) if side_union else 0.0, 1.0), 4)
                carpark_frac = round(min(float(sh_poly.intersection(carpark_union).area / occ_area) if carpark_union else 0.0, 1.0), 4)
                crosswalk_frac = round(min(float(sh_poly.intersection(crosswalk_union).area / occ_area) if crosswalk_union else 0.0, 1.0), 4)
                terrain_frac = round(max(0.0, 1.0 - road_frac - side_frac - carpark_frac - crosswalk_frac), 4)
            else:
                road_frac, side_frac, carpark_frac, crosswalk_frac, terrain_frac = 0.0, 0.0, 0.0, 0.0, 1.0
                
            # Calcola la bounding box che racchiude l'intera ombra
            bbox = [float(np.min(xv_int)), float(np.min(yv_int)),
                    float(np.max(xv_int)), float(np.max(yv_int))]
                    
            # Costruisce un Dizionario per questo specifico ostacolo, contenente tutti i dati pronti per l'uso
            occlusions_output.append({
                "object_name": box.name,
                "object_token": box.token,
                "distance_m": round(float(np.linalg.norm(box.center[:2])), 2),
                "occlusion_bbox_m": bbox,
                "polygon_points_m": polygon_m,
                "area_sqm": round(len(intersection_pixels) * (VOXEL_SIZE**2), 2),
                "road_fraction": road_frac,
                "sidewalk_fraction": side_frac,
                "carpark_fraction": carpark_frac,
                "crosswalk_fraction": crosswalk_frac,
                "terrain_fraction": terrain_frac
            })
            
        return occlusions_output
