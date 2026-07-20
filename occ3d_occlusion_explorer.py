import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import matplotlib
import os
import json
from collections import defaultdict
from scipy import ndimage
from shapely.geometry import Polygon as ShapelyPolygon

# Parametri geometrici di configurazione
GRID_RANGE = 40.0 # Raggio di copertura piana (metri)
VOXEL_SIZE = 0.4 # Risoluzione spaziale (metri)
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE) # Numero di voxel per lato (200)
Z_DIM = 24 # Risoluzione verticale (24 voxel per 9.6 metri totali)

# Classi semantiche abilitate come ostacoli dinamici per la ricerca delle zone occluse(auto, pedoni, ecc.)
CASTER_CATEGORIES = ["vehicle", "truck", "bus", "trailer", "human.pedestrian", "bicycle", "motorcycle"]

class PseudoBox:
    """
    Classe helper creata per replicare l'interfaccia dell'oggetto `Box` di NuScenes per tutti gli oggetti privi di box predefiniti. 
    Consente di inserire barriere, muri ed edifici (manmade) nella pipeline 
    di simulazione e poligonizzazione in modo trasparente.
    """
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z, name, token):
        # Riceve i confini 3D allineati agli assi ($min$ e $max$ per $X, Y, Z$).
        self.name = name
        self.token = token
        # Calcola il centro geometrico 3D (centroide) del box prendendo il punto medio delle coordinate esterne: $(min + max) / 2.0$.
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
        
    def corners(self):
        # Calcola gli 8 vertici 3D del box orientato lungo gli assi (AABB)
        # Segue la convenzione standard di output a 8 colonne (3 x 8)
        x_vals = [self.max_x, self.max_x, self.min_x, self.min_x, self.max_x, self.max_x, self.min_x, self.min_x]
        y_vals = [self.max_y, self.min_y, self.min_y, self.max_y, self.max_y, self.min_y, self.min_y, self.max_y]
        z_vals = [self.max_z, self.max_z, self.max_z, self.max_z, self.min_z, self.min_z, self.min_z, self.min_z]
        return np.array([x_vals, y_vals, z_vals])

class SOTARayCaster:
    # Inizializzazione del Raycaster: gestisce l'input standardizzato (Ego BEV frame) e prepara i buffer per la griglia voxel 3D
    def __init__(self, frame_data, verbose=True):
        # 1. Recupero dei metadati e dei punti LiDAR locali dall'interfaccia standard
        self.verbose = verbose
        self.lidar_token = frame_data["lidar_token"]
        self.pts = frame_data["points"]
        # Ricostruiamo un finto pc_full per compatibilità se ring index o intensity non sono presenti
        self.pc_full = np.column_stack([self.pts, np.zeros((len(self.pts), 2))])
        
        # 2. Allocazione delle griglie voxel 3D (dimensioni: 200 x 200 x 24 voxel)
        self.grid = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        self.internal_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        self.box_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)

        # 3. Inizializzazione dei Bounding Box standardizzati passati dall'adapter
        self._all_boxes = frame_data["boxes"]
        self._lidar_occ = None

    # Ritorna i Bounding Box standardizzati del frame
    def _get_all_boxes(self):
        return self._all_boxes

    # Converte la nuvola di punti LiDAR da coordinate continue metriche a indici voxel discreti (con cache lazy)
    def _get_lidar_occ(self):
        if self._lidar_occ is None:
            # 1. Alloca una griglia 3D vuota per ospitare i punti occupati
            lidar_occ = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
            
            # 2. Definisce l'offset per l'altezza Z (incluso lo spazio sotto l'ego-vehicle fino a -2.4 metri)
            z_off = (Z_DIM // 4) * VOXEL_SIZE
            
            # 3. Discretizza le coordinate X, Y, Z in indici interi della griglia (arrotondati a voxel da 40cm)
            ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
            iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
            iz_p = ((self.pts[:, 2] + z_off) / VOXEL_SIZE).astype(int)
            
            # 4. Applica una maschera per eliminare i punti esterni ai confini fisici della griglia
            m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM) & (iz_p >= 0) & (iz_p < Z_DIM)
            
            # 5. Segna a 1 i voxel in cui cade almeno un punto LiDAR
            lidar_occ[ix_p[m], iy_p[m], iz_p[m]] = 1
            self._lidar_occ = lidar_occ
        return self._lidar_occ

    # Rileva ed estrae pseudo-box 3D dagli elementi statici del LiDAR (es. muri e pareti di edifici)
    # Questa funzione isola la parte immobile dell'ambiente per poterne calcolare l'ombra in raycasting.
    def detect_static_manmade_boxes(self, dynamic_vehicles):
        if self.verbose:
            print("Clustering geometrico e segmentazione degli ostacoli statici (manmade)...")
            
        # 1. FILTRAZIONE GEOMETRICA INIZIALE DELLA NUBE DI PUNTI LiDAR
        pts_static = self.pts.copy()
        
        # Filtro quota verticale (Z): serve a rimuovere i punti del terreno/asfalto (z < -0.5m)
        # ed elementi aerei irrilevanti come foglie degli alberi o pali alti (z > 4.0m).
        m_z = (pts_static[:, 2] > -0.5) & (pts_static[:, 2] < 4.0)
        pts_static = pts_static[m_z]
        
        # Filtro distanza (R): esclude i riflessi LiDAR generati dalla scocca dell'ego-vehicle stesso
        # entro un raggio d'azione di 3 metri dall'origine.
        dist_to_ego = np.linalg.norm(pts_static[:, :2], axis=1)
        pts_static = pts_static[dist_to_ego > 3.0]
        
        # Rimozione dei punti LiDAR che ricadono dentro le Bounding Box dei veicoli dinamici (mobili).
        # Evita "muri fantasma" sovrapposti alle auto in movimento.
        inside_mask = np.zeros(len(pts_static), dtype=bool)
        for box in dynamic_vehicles:
            corners = box.corners()
            min_c = np.min(corners, axis=1)
            max_c = np.max(corners, axis=1)
            # Determina quali punti LiDAR cadono nell'AABB 2D proiettata del veicolo (con offset di tolleranza di 0.5m)
            inside = (pts_static[:, 0] >= min_c[0] - 0.5) & (pts_static[:, 0] <= max_c[0] + 0.5) & \
                     (pts_static[:, 1] >= min_c[1] - 0.5) & (pts_static[:, 1] <= max_c[1] + 0.5)
            inside_mask |= inside
        pts_static = pts_static[~inside_mask]
        
        # 2. PROIEZIONE E VOXELIZZAZIONE BEV 2D A BASSA RISOLUZIONE (0.8m)
        # Funge da filtro spaziale passa-basso: i punti LiDAR vicini (es. lungo lo stesso muro)
        # vengono accesi all'interno delle stesse celle adiacenti della matrice discretizzata.
        CLUST_RES = 0.8
        CLUST_DIM = int((GRID_RANGE * 2) / CLUST_RES)
        grid_bev = np.zeros((CLUST_DIM, CLUST_DIM), dtype=np.uint8)
        
        ix_c = ((pts_static[:, 0] / CLUST_RES) + CLUST_DIM//2).astype(int)
        iy_c = ((pts_static[:, 1] / CLUST_RES) + CLUST_DIM//2).astype(int)
        m_grid = (ix_c >= 0) & (ix_c < CLUST_DIM) & (iy_c >= 0) & (iy_c < CLUST_DIM)
        grid_bev[ix_c[m_grid], iy_c[m_grid]] = 1
        
        # 3. LABELING A COMPONENTI CONNESSE (ndimage.label)
        # Algoritmo di scansione dell'immagine BEV per raggruppare celle adiacenti (connettività a 8 vicini).
        # Raggruppa i pixel accesi in cluster discreti di ostacoli. Restituisce num_clusters gruppi.
        labeled_grid, num_clusters = ndimage.label(grid_bev)
        
        # Mappatura inversa: riassegna a ciascun punto LiDAR originario l'ID numerico del cluster BEV
        pt_labels = np.zeros(len(pts_static), dtype=int)
        pt_labels[m_grid] = labeled_grid[ix_c[m_grid], iy_c[m_grid]]
        
        static_boxes = []
        box_counter = 0
        MAX_BOX_SIZE = 5.0 # Dimensione massima in metri per ogni segmento di muro (evita scatole giganti)
        
        # 4. PARTIZIONAMENTO E SOTTO-SEGMENTAZIONE DELLE PSEUDOBOX STATICHE
        for c_id in range(1, num_clusters + 1):
            cluster_pts = pts_static[pt_labels == c_id]
            if len(cluster_pts) < 15: # Ignora cluster microscopici o spuri (rumore di fondo)
                continue
                
            # Calcola i confini minimi e massimi 3D (AABB) del cluster corrente
            min_x, max_x = np.min(cluster_pts[:, 0]), np.max(cluster_pts[:, 0])
            min_y, max_y = np.min(cluster_pts[:, 1]), np.max(cluster_pts[:, 1])
            min_z, max_z = np.min(cluster_pts[:, 2]), np.max(cluster_pts[:, 2])
            
            dx = max_x - min_x
            dy = max_y - min_y
            
            # Scenario A: Il muro è allungato prevalentemente lungo la direzione X
            if dx > MAX_BOX_SIZE and dx >= dy:
                num_segments = int(np.ceil(dx / MAX_BOX_SIZE))
                x_edges = np.linspace(min_x, max_x, num_segments + 1)
                temp_boxes = []
                # Suddivide il muro in fette lunghe al massimo MAX_BOX_SIZE
                for seg_i in range(num_segments):
                    s_min_x, s_max_x = x_edges[seg_i], x_edges[seg_i+1]
                    m_seg = (cluster_pts[:, 0] >= s_min_x) & (cluster_pts[:, 0] <= s_max_x)
                    seg_pts = cluster_pts[m_seg]
                    if len(seg_pts) < 8: continue
                    
                    s_min_y, s_max_y = np.min(seg_pts[:, 1]), np.max(seg_pts[:, 1])
                    s_min_z, s_max_z = np.min(seg_pts[:, 2]), np.max(seg_pts[:, 2])
                    
                    temp_boxes.append((s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z))
                
                # OTTIMIZZAZIONE ESTREMI (Boundary Boxes Only):
                # Generiamo le PseudoBox solo in corrispondenza del primo e dell'ultimo segmento.
                # La porzione centrale del muro continuo è geometricamente auto-occlusa rispetto al sensore (origine),
                # per cui proiettare solo gli estremi genera lo stesso cono d'ombra totale risparmiando il 50% di raycasting.
                if temp_boxes:
                    extremes = [temp_boxes[0]]
                    if len(temp_boxes) > 1:
                        extremes.append(temp_boxes[-1])
                    
                    for s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z in extremes:
                        token = f"static_manmade_{box_counter}"
                        box_counter += 1
                        static_boxes.append(PseudoBox(s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z, "static.manmade", token))
                        
            # Scenario B: Il muro è allungato prevalentemente lungo la direzione Y
            elif dy > MAX_BOX_SIZE and dy > dx:
                num_segments = int(np.ceil(dy / MAX_BOX_SIZE))
                y_edges = np.linspace(min_y, max_y, num_segments + 1)
                temp_boxes = []
                # Suddivide il muro in fette lungo l'asse Y
                for seg_i in range(num_segments):
                    s_min_y, s_max_y = y_edges[seg_i], y_edges[seg_i+1]
                    m_seg = (cluster_pts[:, 1] >= s_min_y) & (cluster_pts[:, 1] <= s_max_y)
                    seg_pts = cluster_pts[m_seg]
                    if len(seg_pts) < 8: continue
                    
                    s_min_x, s_max_x = np.min(seg_pts[:, 0]), np.max(seg_pts[:, 0])
                    s_min_z, s_max_z = np.min(seg_pts[:, 2]), np.max(seg_pts[:, 2])
                    
                    temp_boxes.append((s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z))
                
                # Ottimizzazione estremi
                if temp_boxes:
                    extremes = [temp_boxes[0]]
                    if len(temp_boxes) > 1:
                        extremes.append(temp_boxes[-1])
                    
                    for s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z in extremes:
                        token = f"static_manmade_{box_counter}"
                        box_counter += 1
                        static_boxes.append(PseudoBox(s_min_x, s_max_x, s_min_y, s_max_y, s_min_z, s_max_z, "static.manmade", token))
            else:
                # Scenario C: Ostacolo di piccole dimensioni (nessuna segmentazione richiesta)
                token = f"static_manmade_{box_counter}"
                box_counter += 1
                static_boxes.append(PseudoBox(min_x, max_x, min_y, max_y, min_z, max_z, "static.manmade", token))
                
        if self.verbose:
            print(f"Rilevati {len(static_boxes)} pseudo-box statici (manmade).")
        return static_boxes

    # Marca come NOTI/VISIBILI tutti i voxel che sono stati attraversati dai raggi LiDAR (Space Carving)
    def generate_known_zone(self):
        if self.verbose:
            print("Analisi dei blocchi attraversati (Ray-Tracing simulation)...")
        
        # 1. COSTRUZIONE DI UN DEPTH BUFFER SFERICO AD ALTA RISOLUZIONE
        # Definiamo la griglia angolare sferica: 1200 pixel azimutali (piani) e 400 elevazionali (verticali)
        AZ_RES, EL_RES = 1200, 400 
        depth_buffer = np.full((AZ_RES, EL_RES), -1.0) # Inizializzato a -1.0 (sconosciuto)

        # Conversione dei punti LiDAR reali 3D da coordinate cartesiane (X, Y, Z) a coordinate sferiche
        r = np.linalg.norm(self.pts, axis=1) # Raggio (distanza euclidea dal sensore)
        az = np.arctan2(self.pts[:, 1], self.pts[:, 0]) # Angolo azimutale sul piano orizzontale
        el = np.arcsin(np.clip(self.pts[:, 2] / (r + 1e-6), -1, 1)) # Angolo di elevazione verticale

        # Discretizzazione in indici di pixel interi per la mappa sferica
        u = ((az + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(int)
        v = ((el + np.pi/2) / np.pi * (EL_RES - 1)).astype(int)

        # Riempimento della mappa sferica: memorizza il raggio massimo registrato per ciascun pixel angolare
        for i in range(len(r)):
            if r[i] < GRID_RANGE:
                if r[i] > depth_buffer[u[i], v[i]]:
                    depth_buffer[u[i], v[i]] = r[i]

        # 1.5 INTERPOLAZIONE ANGOLARE (GAP FILLING): Tappa i buchi neri artificiali tra i fasci laser
        # Applica un filtro di massimo 3x5 per estendere localmente il raggio di visibilità,
        # riducendo i falsi positivi di occlusione tra i cerchi concentrici dei canali LiDAR.
        if self.verbose:
            print("Chiusura buchi tra i raggi (Interpolazione angolare)...")
        depth_buffer = ndimage.maximum_filter(depth_buffer, size=(3, 5))
        
        # Salviamo la mappa sferica calcolata per poterla riutilizzare nella logica dell'ombra dei veicoli
        self.depth_buffer_ref = depth_buffer

        # 2. RAY-TRACING VETTORIALIZZATO SULLA GRIGLIA VOXEL 3D
        # Generiamo le coordinate reali metriche per ciascuno dei 200 x 200 x 24 voxel del mondo discreto
        x = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z = (np.arange(Z_DIM) - Z_DIM//4) * VOXEL_SIZE 

        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij') # Crea la griglia 3D di coordinate cartesiane
        dist = np.sqrt(xv**2 + yv**2 + zv**2) # Calcola la distanza euclidea di ogni voxel dall'origine
        valid_mask = dist <= GRID_RANGE # Limita il calcolo alla portata massima definita (40m)

        # Convertiamo la posizione di ciascun voxel in coordinate sferiche angolari (per associarlo alla mappa sferica)
        az_v = np.arctan2(yv, xv)
        el_v = np.arcsin(np.clip(zv / (dist + 1e-6), -1.0, 1.0))
        u_v = ((az_v + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(np.int32)
        v_v = ((el_v + np.pi/2) / np.pi * (EL_RES - 1)).astype(np.int32)

        # Evita errori di fuori-limite (out-of-bounds) durante l'indicizzazione
        u_v = np.clip(u_v, 0, AZ_RES - 1)
        v_v = np.clip(v_v, 0, EL_RES - 1)

        # Determina la massima distanza visiva registrata dal LiDAR in quella direzione angolare
        d_limit = depth_buffer[u_v, v_v]
        
        # SPACE CARVING: un voxel è VISIBILE (noto, grid=1) se si trova all'interno della portata,
        # la direzione è valida (d_limit > 0) e la sua distanza dal sensore è minore di quella massima registrata
        # (ovvero il raggio laser è passato attraverso di esso prima di colpire il bersaglio).
        known_mask = valid_mask & (d_limit > 0) & (dist < d_limit)
        self.grid[known_mask] = 1

        return self.grid

    # Calcola e voxelizza i coni d'ombra tridimensionali proiettati da tutti gli ostacoli stradali
    def generate_box_shadows(self):
        if self.verbose:
            print("Simulazione coni d'ombra in corso...")
        self.box_shadows = np.zeros_like(self.grid)
        self.cone_ownership = {} # Associazione geometrica: (coordinata_voxel_x, coordinata_voxel_y) -> token_ostacolo
        
        # Filtra i box dinamici estratti dal dataset in base alle categorie caster abilitate
        vehicles = [b for b in self._get_all_boxes() if any(c in b.name.lower() for c in CASTER_CATEGORIES)]
        
        # Rileva ed estrae i box per i muri e le barriere statiche
        self.static_boxes = self.detect_static_manmade_boxes(vehicles)
        
        # Combina tutti gli ostacoli dinamici e statici in un'unica lista di caster
        all_casters = self.static_boxes + vehicles
        
        # Coordinate metriche dei voxel lungo gli assi X, Y, Z
        x_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z_off = (Z_DIM // 4) * VOXEL_SIZE

        # Griglia 2D di coordinate cartesiane, distanze e angoli per il BEV (Bird's Eye View)
        xv_2d, yv_2d = np.meshgrid(x_coords, y_coords, indexing='ij')
        dv_2d = np.sqrt(xv_2d**2 + yv_2d**2)
        av_2d = np.arctan2(yv_2d, xv_2d)

        # Mappa per tracciare la distanza minima dal centroide del box (risolve le sovrapposizioni d'ombra)
        min_dist_to_center = np.full((GRID_DIM, GRID_DIM), np.inf)

        # Recupera la griglia dei punti LiDAR voxelizzati (dalla cache lazy)
        lidar_occ = self._get_lidar_occ()

        # Ottiene la mappa di profondità sferica pre-calcolata per il calcolo dei limiti
        depth_buffer = self.depth_buffer_ref
        AZ_RES = depth_buffer.shape[0]

        # Associa preventivamente ogni coordinata BEV all'indice angolare corrispondente del depth buffer
        u_v_2d = ((av_2d + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(np.int32)
        u_v_2d = np.clip(u_v_2d, 0, AZ_RES - 1)

        # Proietta le ombre per ciascun ostacolo dinamico o statico
        for box in all_casters:
            # 1. FILTRO DI OCCUPAZIONE LiDAR: l'ostacolo deve contenere almeno un punto LiDAR visibile
            corners_3d = box.corners()
            min_c, max_c = np.min(corners_3d, axis=1), np.max(corners_3d, axis=1)
            r_r = np.clip([int((min_c[0]+GRID_RANGE)/VOXEL_SIZE), int((max_c[0]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            c_r = np.clip([int((min_c[1]+GRID_RANGE)/VOXEL_SIZE), int((max_c[1]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            z_r = np.clip([int((min_c[2]+z_off)/VOXEL_SIZE), int((max_c[2]+z_off)/VOXEL_SIZE)], 0, Z_DIM-1)
            
            # Se nessun punto LiDAR ricade nel volume 3D dell'ostacolo, lo ignoriamo (oggetto invisibile)
            if not np.any(lidar_occ[r_r[0]:r_r[1]+1, c_r[0]:c_r[1]+1, z_r[0]:z_r[1]+1]):
                continue

            # 2. CALCOLO DEGLI ANGOLI DI COPERTURA DEL CONO D'OMBRA (2D)
            corners = box.corners()[:2, :].T
            r_min = np.min(np.linalg.norm(corners, axis=1)) # Distanza minima del box (inizio dell'ombra)
            az_c = np.arctan2(corners[:, 1], corners[:, 0])
            # Gestione della discontinuità periodica dell'angolo a 180°
            if np.max(az_c) - np.min(az_c) > np.pi: az_c[az_c < 0] += 2 * np.pi
            az_min, az_max = np.min(az_c), np.max(az_c)
            
            av_shifted = av_2d.copy()
            if az_max > np.pi:
                av_shifted[av_shifted < az_min] += 2 * np.pi

            # 3. LOGICA SPLIT SHADOW LIMIT PER RAGGIO ANGOLARE (DINAMICO CON np.min)
            # Determina per ciascun raggio angolare attivo del cono dove arrestare l'ombra
            active_u = np.unique(u_v_2d[(dv_2d > r_min) & (av_shifted >= az_min) & (av_shifted <= az_max)])
            ray_limits = np.full(AZ_RES, r_min + 30.0)  # Default per zone cieche/vuote: 30 metri dall'auto
            for u in active_u:
                # Estrae i riflessi LiDAR posizionati dietro l'ostacolo lungo questo specifico raggio angolare
                hits_behind = depth_buffer[u, :][depth_buffer[u, :] > (r_min + 3.0)]
                if len(hits_behind) > 0:
                    # Se c'è strada o muro visibile, l'ombra si ferma rigidamente al PRIMO punto visibile incontrato
                    ray_limits[u] = np.min(hits_behind)
            
            # Applica i limiti dinamici di raggio calcolati su tutta la griglia BEV
            limit_grid = ray_limits[u_v_2d]
                
            cand_mask = (dv_2d > r_min) & (dv_2d <= limit_grid)
            cand_mask &= (av_shifted >= az_min) & (av_shifted <= az_max)

            if not np.any(cand_mask):
                continue

            # 4. SURGICAL WALL CLIPPING (Taglio netto al primo muro/edificio reale)
            # Controlla se lungo la direzione c'è un muro statico che blocca fisicamente la proiezione
            d_wall_limit = np.full(AZ_RES, np.inf)
            active_u = np.unique(u_v_2d[cand_mask])
            for u in active_u:
                # Cerca collisioni verticali sottomappa (inclinazioni verticali elevate del sensore)
                d_hits_wall = depth_buffer[u, 200:] 
                wall_hits = d_hits_wall[d_hits_wall > (r_min + 5.0)]
                if len(wall_hits) >= 2:
                    d_wall_limit[u] = np.min(wall_hits) # Taglia l'ombra davanti alla parete del muro
            
            limit_grid = d_wall_limit[u_v_2d]
            cand_mask &= (dv_2d <= limit_grid)

            if not np.any(cand_mask):
                continue

            # 5. ASSEGNAZIONE E PROIEZIONE 3D
            # Applica la maschera di ombra BEV su tutte le 24 fette verticali dell'altezza (Z)
            self.box_shadows[cand_mask, :] = 1
            
            # GESTIONE OWNERSHIP SPAZIALE: Associa ogni cella d'ombra all'ostacolo più vicino geometricamente
            # Calcola la distanza euclidea di ogni cella d'ombra dal centroide di questa scatola
            dist_to_center = np.sqrt((xv_2d - box.center[0])**2 + (yv_2d - box.center[1])**2)
            better_mask = cand_mask & (dist_to_center < min_dist_to_center)
            min_dist_to_center[better_mask] = dist_to_center[better_mask]
            
            # Popola la mappa di ownership per determinare a quale poligono appartiene ciascuna coordinata
            for ix, iy in np.argwhere(better_mask):
                self.cone_ownership[(ix, iy)] = box.token

        return self.box_shadows

    # Calcola le zone occluse racchiuse internamente tra ostacoli successivi in fila (cunei d'ombra magenta)
    def find_object_occlusion_wedges(self):
        if self.verbose:
            print("Pulizia e mascheramento cunei (Safety Buffer)...")
        self.internal_shadows = np.zeros_like(self.grid)
        
        # 1. GENERAZIONE MASCHERA DI RISPETTO LiDAR (DILATION)
        occupied_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.uint8)
        ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM)
        occupied_bev[ix_p[m], iy_p[m]] = 1
        
        # Espande di 1.2 metri (dilatazione 3x3) i punti LiDAR per creare un margine di rispetto.
        # Impedisce che l'ombra interna venga proiettata sopra punti visibili dell'ambiente.
        safety_mask = ndimage.binary_dilation(occupied_bev, structure=np.ones((3, 3)))
        
        # Scansiona le scatole orientate di NuScenes per cercare i veicoli in primo piano
        for box in self._get_all_boxes():
            if not any(c in box.name.lower() for c in ["vehicle", "truck", "bus"]): continue
            
            center = box.center
            dist_A = np.linalg.norm(center[:2]) # Distanza del veicolo anteriore
            corners = box.corners()
            az_c = np.arctan2(corners[1], corners[0])
            if np.max(az_c) - np.min(az_c) > np.pi: az_c[az_c < 0] += 2*np.pi
            
            # 2. CONTRAZIONE DEL CUNEO ANGOLARE
            # Restringe il cuneo di 0.03 radianti per evitare che l'ombra strabordi sui fianchi del veicolo
            az_L, az_R = np.min(az_c) + 0.03, np.max(az_c) - 0.03
            az_C = (az_L + az_R) / 2 # Angolo bisettore centrale
            
            # Cerca la presenza di un ostacolo posteriore lungo la direttrice centrale del cuneo
            found_d_back = -1
            for d in np.linspace(dist_A + 4, GRID_RANGE, 40):
                tx, ty = d * np.cos(az_C), d * np.sin(az_C)
                gx, gy = int(tx/VOXEL_SIZE + GRID_DIM//2), int(ty/VOXEL_SIZE + GRID_DIM//2)
                if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM:
                    if occupied_bev[gx, gy]:
                        found_d_back = d
                        break
            
            # 3. CAMPIONAMENTO A TRE RAGGI (Interpolazione bilineare dell'ostacolo posteriore)
            # Se è presente un ostacolo posteriore ad almeno 5 metri di distanza:
            if found_d_back > 0 and (found_d_back - dist_A) > 5.0:
                dists_back = []
                # Misura la distanza di collisione lungo le tre direttrici esterne del cuneo (Sinistra, Centro, Destra)
                for angle in [az_L, az_C, az_R]:
                    d_edge = GRID_RANGE
                    for d in np.linspace(dist_A + 2, GRID_RANGE, 30):
                        tx, ty = d * np.cos(angle), d * np.sin(angle)
                        gx, gy = int(tx/VOXEL_SIZE + GRID_DIM//2), int(ty/VOXEL_SIZE + GRID_DIM//2)
                        if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM and occupied_bev[gx, gy]:
                            d_edge = d
                            break
                    dists_back.append(d_edge)
                
                d_L, d_C, d_R = dists_back
                x_c = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
                y_c = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
                
                # Proietta l'ombra sulle celle della griglia piana
                for ix in range(GRID_DIM):
                    for iy in range(GRID_DIM):
                        # Impedisce all'ombra di sovrapporsi ai punti LiDAR o alla loro area di dilatazione
                        if safety_mask[ix, iy]: continue
                        
                        xv, yv = x_c[ix], y_c[iy]
                        dv = np.sqrt(xv**2 + yv**2)
                        av = np.arctan2(yv, xv)
                        if av < az_L and az_R > np.pi: av += 2*np.pi
                        
                        # Interpolazione bilineare dell'ombra tra i tre raggi discreti proiettati
                        if az_L <= av <= az_R:
                            if av < az_C:
                                t = (av - az_L) / (az_C - az_L + 1e-6)
                                d_limit = d_L * (1-t) + d_C * t
                            else:
                                t = (av - az_C) / (az_R - az_C + 1e-6)
                                d_limit = d_C * (1-t) + d_R * t
                            
                            # Il cuneo parte a 2m dietro il veicolo e termina a 1m prima dell'ostacolo posteriore
                            if dv >= dist_A + 2.0 and dv < d_limit - 1.0:
                                self.internal_shadows[ix, iy, :] = 1
                                
        # 4. RIMOZIONE RUMORE DI DISCRETIZZAZIONE (FILTRO A COMPONENTI CONNESSE)
        # Elimina i cluster d'ombra isolati più piccoli di 20 voxel BEV
        shadow_bev = self.internal_shadows[:, :, 0]
        labeled, num_feat = ndimage.label(shadow_bev)
        if num_feat > 0:
            sizes = ndimage.sum(shadow_bev, labeled, range(num_feat + 1))
            mask_noise = (sizes < 20)
            shadow_bev[mask_noise[labeled]] = 0
            for iz in range(Z_DIM):
                self.internal_shadows[:, :, iz] = shadow_bev

        return self.internal_shadows

    # Esegue la logica definitiva di visibilità intersecando lo spazio noto con le ombre 3D ed applicando la chiusura morfologica
    def extract_occlusion_zones(self):
        if self.verbose:
            print("Calcolo visibilità finale con proiezione verticale...")
            
        # 1. Genera la griglia di conoscenza pulita sottraendo i cunei d'ombra interni (magenta)
        # Assicura che la zona nascosta tra due auto consecutive in fila non sia classificata come "visibile/nota"
        self.final_grid = self.grid.copy()
        self.final_grid[self.internal_shadows == 1] = 0
        
        # 2. Calcola la visibilità 2D a terra (Bird's Eye View) lungo l'altezza
        # Se il LiDAR ha attraversato anche solo un voxel di una colonna (x,y), l'intera colonna è considerata visibile
        bev_visibility = np.max(self.final_grid, axis=2)
        
        # 3. Sottrae la visibilità 2D dai coni d'ombra tridimensionali dei box
        # Rimuove l'ombra ovunque il LiDAR abbia registrato una linea di vista libera (terreno o aria)
        temp_occluded = self.box_shadows.copy().astype(bool)
        for iz in range(Z_DIM):
            temp_occluded[:, :, iz] &= (bev_visibility == 0)

        # 4. CHIUSURA MORFOLOGICA BEV (Kernel Quadrato 5x5)
        # Proietta l'ombra 3D su piano 2D e applica l'operazione morfologica di chiusura (dilatazione + erosione).
        # Tappa i buchi interni e salda le spaccature radiali provocate dall'inclinazione discreta dei fasci LiDAR (cerchi concentrici),
        # ricostruendo un poligono d'ombra continuo e solido. Un kernel 5x5 copre un'area reale di 2.0 x 2.0 metri.
        final_red_map = np.max(temp_occluded, axis=2)
        closed_red_map = ndimage.binary_closing(final_red_map, structure=np.ones((5, 5)))
        
        # 5. RIPROIEZIONE NELLO SPAZIO VOXEL 3D
        # Applica la mappa 2D chiusa su tutte le 24 fette verticali di altezza della griglia dei box
        self.occluded_final = np.zeros_like(self.box_shadows, dtype=bool)
        for iz in range(Z_DIM):
            self.occluded_final[:, :, iz] = self.box_shadows[:, :, iz] & closed_red_map
            
        return self.occluded_final

    # Converte la griglia voxel 3D delle occlusioni in poligoni vettoriali metrici ed esporta in formato JSON
    def save_occlusions_to_json(self, filename="occlusions_nuscenes.json"):
        if self.verbose:
            print(f"Esportazione dati in {filename} (Geometric-Ownership Mode)...")
        
        # Recupera i box dinamici e li combina con gli ostacoli statici già calcolati
        vehicles = [b for b in self._get_all_boxes() if any(c in b.name.lower() for c in CASTER_CATEGORIES)]
        all_casters = self.static_boxes + vehicles
        
        occlusions_output = []

        # 1. RAGGRUPPAMENTO PER OWNERSHIP (Assegnazione Pixel → Ostacolo)
        # Collassa la griglia 3D in mappa BEV 2D e associa ciascun pixel rosso al token dell'ostacolo proprietario
        final_red_map = np.max(self.occluded_final, axis=2).astype(np.uint8)
        
        object_occlusions = defaultdict(list)
        for ix in range(GRID_DIM):
            for iy in range(GRID_DIM):
                if final_red_map[ix, iy] > 0:
                    token = self.cone_ownership.get((ix, iy))
                    if token:
                        object_occlusions[token].append((ix, iy))
        
        for box in all_casters:
            intersection_pixels = object_occlusions.get(box.token)
            
            # 2. SOGLIA DINAMICA PER CATEGORIA
            # Pedoni/bici generano pochi pixel d'ombra (soglia bassa = 3px), veicoli ne richiedono di più (5px),
            # gli ostacoli statici tollerano soglie minime (2px) per catturare le ombre rasenti
            is_small = any(c in box.name.lower() for c in ["human", "bicycle", "motorcycle"])
            if "static" in box.name.lower():
                min_px = 2
            else:
                min_px = 3 if is_small else 5
            
            if not intersection_pixels:
                if self.verbose:
                    print(f"[DEBUG] {box.token} ({box.name}) saltato: 0 pixel di intersezione (totalmente coperto/sovrascritto da altri oggetti)")
                continue
                
            if len(intersection_pixels) < min_px:
                if self.verbose:
                    print(f"[DEBUG] {box.token} ({box.name}) saltato: {len(intersection_pixels)} pixel < soglia {min_px}")
                continue
            
            # 3. SHELL MAPPING (Scanning Radiale del Perimetro d'Ombra)
            # Converte gli indici voxel dei pixel d'ombra in coordinate polari (distanza, angolo) centrate sull'ego
            int_pixels_arr = np.array(intersection_pixels)
            r_idx, c_idx = int_pixels_arr[:, 0], int_pixels_arr[:, 1]
            
            xv_int = (r_idx - GRID_DIM//2) * VOXEL_SIZE
            yv_int = (c_idx - GRID_DIM//2) * VOXEL_SIZE
            az_int = np.arctan2(yv_int, xv_int)
            dist_int = np.sqrt(xv_int**2 + yv_int**2)
            
            if np.max(az_int) - np.min(az_int) > np.pi: az_int[az_int < 0] += 2*np.pi
            a_min, a_max = np.min(az_int), np.max(az_int)
            
            # Campiona 12 raggi equidistanti nell'intervallo angolare dell'ombra
            sample_angs = np.linspace(a_min, a_max, 12)
            inner_pts, outer_pts = [], []
            
            # Per ciascun raggio angolare, estrae il bordo interno (r_near) e il bordo esterno (r_far)
            for ang in sample_angs:
                m_ray = (np.abs(az_int - ang) < 0.05) | (np.abs(az_int + 2*np.pi - ang) < 0.05)
                if np.any(m_ray):
                    r_near = np.min(dist_int[m_ray])
                    r_far = np.max(dist_int[m_ray])
                    inner_pts.append([round(float(r_near * np.cos(ang)), 2), round(float(r_near * np.sin(ang)), 2)])
                    outer_pts.append([round(float(r_far * np.cos(ang)), 2), round(float(r_far * np.sin(ang)), 2)])
            
            if not inner_pts:
                if self.verbose:
                    print(f"[DEBUG] {box.token} ({box.name}) saltato: perimetro vuoto nello scanning radiale")
                continue
            # Unisce arco esterno + arco interno rovesciato per generare il poligono chiuso a "guscio"
            polygon_m = outer_pts + inner_pts[::-1]
            
            # 4. SANIFICAZIONE GEOMETRICA CON SHAPELY
            # Corregge auto-intersezioni da artefatti di discretizzazione e riduce i vertici superflui
            try:
                sh_poly = ShapelyPolygon(polygon_m)
                if not sh_poly.is_valid:
                    sh_poly = sh_poly.buffer(0) # Sana automaticamente le auto-intersezioni
                
                # Semplificazione con tolleranza di 0.2m (metà dimensione voxel) per ridurre i vertici
                sh_poly_simple = sh_poly.simplify(0.2, preserve_topology=True)
                if not sh_poly_simple.is_valid:
                    sh_poly_simple = sh_poly_simple.buffer(0)
                    
                polygon_m = [[round(x, 2), round(y, 2)] for x, y in sh_poly_simple.exterior.coords]
                # Shapely chiude il poligono ripetendo il primo punto alla fine; lo rimuoviamo per coerenza
                if len(polygon_m) > 1 and polygon_m[0] == polygon_m[-1]:
                    polygon_m = polygon_m[:-1]
            except Exception as e:
                pass
            
            # 5. SERIALIZZAZIONE JSON CON METADATI
            bbox = [float(np.min(xv_int)), float(np.min(yv_int)),
                    float(np.max(xv_int)), float(np.max(yv_int))]
            
            occlusions_output.append({
                "object_name": box.name,
                "object_token": box.token,
                "distance_m": round(float(np.linalg.norm(box.center[:2])), 2),
                "occlusion_bbox_m": bbox,
                "polygon_points_m": polygon_m,
                "area_sqm": round(len(intersection_pixels) * (VOXEL_SIZE**2), 2)
            })
            
        with open(filename, "w") as f:
            json.dump({"lidar_token": self.lidar_token, "occlusions": occlusions_output}, f, indent=4)
        
        if self.verbose:
            print(f"Esportazione terminata. Intersezione Coni-Risultati completata ({len(occlusions_output)} oggetti).")
        return len(occlusions_output)

    # Visualizzazione diagnostica BEV (Bird's Eye View) a 4 quadranti per debug visivo della pipeline
    def plot_bev(self):
        from matplotlib.colors import LinearSegmentedColormap
        
        # Proiezione verticale (max lungo Z) di ciascun buffer 3D per ottenere le mappe BEV 2D
        known_bev = np.max(self.grid, axis=2)            # Spazio noto originale (Space Carving)
        cleaned_known_bev = np.max((self.grid == 1) & (self.internal_shadows == 0), axis=2) # Noto meno cunei interni
        sim_bev = np.max(self.box_shadows, axis=2)       # Coni d'ombra simulati grezzi
        occluded_bev = np.max(self.occluded_final, axis=2) # Risultato finale (ombre validate)
        
        fig, axs = plt.subplots(2, 2, figsize=(18, 18), facecolor='#0f172a')
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
        
        # Colormap binarie semi-trasparenti personalizzate per ciascun quadrante
        cmap_gray = LinearSegmentedColormap.from_list('gray', [(0,0,0,0), (0.6, 0.6, 0.6, 0.8)], N=2)
        cmap_cyan = LinearSegmentedColormap.from_list('cyan', [(0,0,0,0), (0.0, 0.8, 1.0, 0.7)], N=2)
        cmap_yellow = LinearSegmentedColormap.from_list('yel', [(0,0,0,0), (1.0, 0.9, 0.0, 0.8)], N=2)
        cmap_red = LinearSegmentedColormap.from_list('red', [(0,0,0,0), (1.0, 0.0, 0.0, 0.9)], N=2)

        # Quadrante 1 (alto-sx, GRIGIO): Zona nota grezza calcolata dallo Space Carving dei raggi LiDAR
        axs[0,0].set_facecolor('#000000')
        axs[0,0].imshow(known_bev, extent=extent, origin='lower', cmap=cmap_gray)
        axs[0,0].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[0,0].set_title("1. ORIGINAL KNOWN ZONE (LIDAR)", color='white', fontsize=14)
        
        # Quadrante 2 (alto-dx, CIANO): Zona nota "pulita" dopo la sottrazione dei cunei d'ombra interni
        axs[0,1].set_facecolor('#000000')
        axs[0,1].imshow(cleaned_known_bev, extent=extent, origin='lower', cmap=cmap_cyan)
        axs[0,1].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[0,1].set_title("2. CLEANED KNOWN (KNOWN - INTERNAL)", color='white', fontsize=14)

        # Quadrante 3 (basso-sx, GIALLO): Coni d'ombra grezzi proiettati da tutti gli ostacoli
        axs[1,0].set_facecolor('#000000')
        axs[1,0].imshow(sim_bev, extent=extent, origin='lower', cmap=cmap_yellow)
        axs[1,0].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[1,0].set_title("3. SIMULATED BOX CONES", color='white', fontsize=14)

        # Quadrante 4 (basso-dx, ROSSO): Risultato finale dopo sottrazione della visibilità e chiusura morfologica
        axs[1,1].set_facecolor('#000000')
        axs[1,1].imshow(occluded_bev, extent=extent, origin='lower', cmap=cmap_red)
        axs[1,1].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[1,1].set_title("4. RESULT (CONES - INTERNAL_VALIDATED)", color='white', fontsize=14)

        for ax in axs.flat:
            ax.set_xlim(GRID_RANGE, -GRID_RANGE)
            ax.set_ylim(-GRID_RANGE, GRID_RANGE)
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

# Funzione di elaborazione batch: processa un intervallo di frame mediante l'Adapter e salva i JSON delle occlusioni
def batch_process_dataset(adapter, output_dir="extracted_occlusions", start_idx=0, end_idx=None):
    import time
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata la cartella di output: {output_dir}")
        
    if end_idx is None:
        end_idx = adapter.get_num_samples()
        
    print(f"=== AVVIO ELABORAZIONE BATCH (Sample {start_idx} a {end_idx}) ===")
    
    start_time = time.time()
    for idx in range(start_idx, end_idx):
        try:
            frame_data = adapter.get_sample_data(idx)
            sample_token = frame_data["sample_token"]
        except Exception as e:
            print(f"Errore caricamento dati per sample {idx}: {e}")
            continue
            
        output_path = os.path.join(output_dir, f"occlusion_sample_{idx:04d}_{sample_token}.json")
        
        # Riprendibilità: se il file JSON è già stato generato, salta al frame successivo
        if os.path.exists(output_path):
            print(f"[{idx+1}/{end_idx}] Saltato: {output_path} esiste già.")
            continue
            
        t0 = time.time()
        try:
            # Esegue l'intera pipeline per il singolo frame standardizzato
            caster = SOTARayCaster(frame_data, verbose=False)
            caster.generate_known_zone()
            caster.generate_box_shadows()
            caster.find_object_occlusion_wedges()
            caster.extract_occlusion_zones()
            num_occs = caster.save_occlusions_to_json(output_path)
            dt = time.time() - t0
            print(f"[{idx+1}/{end_idx}] Sample {idx:04d} (Token: {sample_token[:8]}...) elaborato in {dt:.3f}s | Oggetti occlusi salvati: {num_occs}")
        except Exception as e:
            print(f"Errore durante l'elaborazione del sample {idx}: {e}")
            
    total_dt = time.time() - start_time
    print(f"=== ELABORAZIONE BATCH COMPLETATA in {total_dt:.2f}s! ===")

# Punto di ingresso CLI
if __name__ == "__main__":
    import argparse
    from dataset_adapters import NuScenesAdapter
    
    parser = argparse.ArgumentParser(description="Pipeline di Estrazione Occlusioni LiDAR (SOTA) - NuScenes con Adapter")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch"],
                        help="Modalità d'uso: 'single' (esegue un singolo frame e mostra i grafici) o 'batch' (processa e salva un intervallo di frame)")
    parser.add_argument("--sample-idx", type=int, default=200, help="Indice del sample da processare in modalità 'single'")
    parser.add_argument("--start-idx", type=int, default=0, help="Indice iniziale per l'elaborazione batch")
    parser.add_argument("--end-idx", type=int, default=None, help="Indice finale per l'elaborazione batch (None = tutti)")
    parser.add_argument("--out-dir", type=str, default="extracted_occlusions", help="Cartella di output per i file JSON")
    
    args = parser.parse_args()
    
    # Istanziamo l'Adapter di NuScenes
    adapter = NuScenesAdapter(dataroot='./nuscenes', version='v1.0-mini')
    
    if args.mode == "single":
        print(f"Esecuzione in modalità singola sul sample {args.sample_idx}...")
        frame_data = adapter.get_sample_data(args.sample_idx)
        caster = SOTARayCaster(frame_data)
        caster.generate_known_zone()       # Fase 1: Space Carving
        caster.generate_box_shadows()      # Fase 2: Shadow Casting
        caster.find_object_occlusion_wedges() # Fase 3: Cunei d'ombra interni
        caster.extract_occlusion_zones()   # Fase 4: Fusione e chiusura morfologica
        
        # Fase 5: Esportazione JSON
        out_filename = "occlusions_nuscenes.json"
        caster.save_occlusions_to_json(out_filename)
        print(f"Mappa delle occlusioni salvata in {out_filename}")
        
        # Mostra la visualizzazione diagnostica interattiva a 4 quadranti
        caster.plot_bev()
    else:
        batch_process_dataset(adapter, output_dir=args.out_dir, start_idx=args.start_idx, end_idx=args.end_idx)

