import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import matplotlib
import os

# --- CONFIGURAZIONE ---
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE)
Z_DIM = 24 

class SOTARayCaster:
    def __init__(self, nusc, sample_idx):
        self.nusc = nusc
        self.sample = nusc.sample[sample_idx]
        self.lidar_token = self.sample['data']['LIDAR_TOP']
        
        # Caricamento punti LiDAR
        sd_record = nusc.get('sample_data', self.lidar_token)
        lidar_path = os.path.join(nusc.dataroot, sd_record['filename'])
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        self.pts = pc[:, :3]
        self.pc_full = pc # Salviamo tutto per i ring
        
        # Griglia 3D: 0=Sconosciuto, 1=Visto (Attraversato)
        self.grid = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        self.internal_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        self.box_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)

    def generate_known_zone(self):
        """
        Marca come NOTI tutti i blocchi attraversati dai raggi LiDAR.
        """
        print("Analisi dei blocchi attraversati (Ray-Tracing simulation)...")
        
        # 1. Creiamo una mappa di profondità sferica ad alta risoluzione
        AZ_RES, EL_RES = 1200, 400 
        depth_buffer = np.full((AZ_RES, EL_RES), -1.0)

        # Coordinate Sferiche dei punti LiDAR
        r = np.linalg.norm(self.pts, axis=1)
        az = np.arctan2(self.pts[:, 1], self.pts[:, 0])
        el = np.arcsin(np.clip(self.pts[:, 2] / (r + 1e-6), -1, 1))

        u = ((az + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(int)
        v = ((el + np.pi/2) / np.pi * (EL_RES - 1)).astype(int)

        # Riempimento del buffer con la distanza massima per ogni raggio
        for i in range(len(r)):
            if r[i] < GRID_RANGE:
                if r[i] > depth_buffer[u[i], v[i]]:
                    depth_buffer[u[i], v[i]] = r[i]

        # 1.5 GAP FILLING: Tappa i buchi tra i raggi LiDAR
        from scipy import ndimage
        print("Chiusura buchi tra i raggi (Interpolazione angolare)...")
        depth_buffer = ndimage.maximum_filter(depth_buffer, size=(3, 5))
        
        # Salviamo il buffer per riferimento
        self.depth_buffer_ref = depth_buffer

        # 2. Navigazione della griglia: un blocco è noto se un raggio lo ha 'passato'
        x = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z = (np.arange(Z_DIM) - Z_DIM//4) * VOXEL_SIZE 

        for iz in range(Z_DIM):
            zv = z[iz]
            for ix in range(GRID_DIM):
                xv = x[ix]
                for iy in range(GRID_DIM):
                    yv = y[iy]
                    
                    dist = np.sqrt(xv**2 + yv**2 + zv**2)
                    if dist > GRID_RANGE: continue
                    
                    # Direzione sferica del voxel rispetto al sensore
                    az_v = np.arctan2(yv, xv)
                    el_v = np.arcsin(zv / (dist + 1e-6))
                    u_v = int((az_v + np.pi) / (2 * np.pi) * (AZ_RES - 1))
                    v_v = int((el_v + np.pi/2) / np.pi * (EL_RES - 1))
                    
                    if 0 <= u_v < AZ_RES and 0 <= v_v < EL_RES:
                        d_limit = depth_buffer[u_v, v_v]
                        # Se il voxel è più vicino del punto di impatto, il raggio lo ha attraversato
                        if d_limit > 0 and dist < d_limit:
                            self.grid[ix, iy, iz] = 1

        return self.grid

    def generate_box_shadows(self):
        """
        FASE 2: Simulazione Geometrica dei Coni d'Ombra (Shadow Casting).
        
        Simula il comportamento fisico della luce (o del raggio LiDAR) che viene bloccato 
        dagli oggetti (veicoli, pedoni, etc.).
        
        Caratteristiche principali:
        - Shadow Casting: Proietta coni divergenti dietro ogni bounding box 3D.
        - Ownership Mapping: Assegna univocamente ogni pixel d'ombra al token dell'oggetto sorgente.
        - Surgical Wall Clipping: Utilizza il LiDAR per fermare l'ombra se incontra un muro reale.
        """
        print("Simulazione coni d'ombra in corso...")
        self.box_shadows = np.zeros_like(self.grid)
        self.cone_ownership = {} # (ix, iy) -> vehicle_token
        
        _, boxes, _ = self.nusc.get_sample_data(self.lidar_token)
        # Filtriamo veicoli, pedoni, biciclette e moto
        categories = ["vehicle", "truck", "bus", "trailer", "human.pedestrian", "bicycle", "motorcycle"]
        vehicles = [b for b in boxes if any(c in b.name.lower() for c in categories)]
        
        x = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        z = (np.arange(Z_DIM) - Z_DIM//4) * VOXEL_SIZE 

        # Voxelizzazione LiDAR temporanea per controllo visibilità
        lidar_occ = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        z_off = (Z_DIM // 4) * VOXEL_SIZE
        ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iz_p = ((self.pts[:, 2] + z_off) / VOXEL_SIZE).astype(int)
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM) & (iz_p >= 0) & (iz_p < Z_DIM)
        lidar_occ[ix_p[m], iy_p[m], iz_p[m]] = 1

        # Recuperiamo il buffer di profondità e le risoluzioni sferiche per il clipping
        depth_buffer = self.depth_buffer_ref
        AZ_RES = depth_buffer.shape[0]

        for box in vehicles:
            # 1. Filtro visibilità LiDAR (minimo un punto nel box)
            corners_3d = box.corners()
            min_c, max_c = np.min(corners_3d, axis=1), np.max(corners_3d, axis=1)
            r_r = np.clip([int((min_c[0]+GRID_RANGE)/VOXEL_SIZE), int((max_c[0]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            c_r = np.clip([int((min_c[1]+GRID_RANGE)/VOXEL_SIZE), int((max_c[1]+GRID_RANGE)/VOXEL_SIZE)], 0, GRID_DIM-1)
            z_r = np.clip([int((min_c[2]+z_off)/VOXEL_SIZE), int((max_c[2]+z_off)/VOXEL_SIZE)], 0, Z_DIM-1)
            
            if not np.any(lidar_occ[r_r[0]:r_r[1]+1, c_r[0]:c_r[1]+1, z_r[0]:z_r[1]+1]):
                continue

            # 2. Proiezione Cono Originale (Ripristinata)
            corners = box.corners()[:2, :].T
            r_min = np.min(np.linalg.norm(corners, axis=1))
            az_c = np.arctan2(corners[:, 1], corners[:, 0])
            if np.max(az_c) - np.min(az_c) > np.pi: az_c[az_c < 0] += 2 * np.pi
            az_min, az_max = np.min(az_c), np.max(az_c)
            
            for ix in range(GRID_DIM):
                for iy in range(GRID_DIM):
                    xv, yv = x[ix], y[iy]
                    dv = np.sqrt(xv**2 + yv**2)
                    if dv <= r_min: continue
                    
                    av = np.arctan2(yv, xv)
                    if av < az_min and az_max > np.pi: av += 2 * np.pi
                    
                    if az_min <= av <= az_max:
                        # 3. CONTROLLO MURO (Chirurgico)
                        av_orig = np.arctan2(yv, xv)
                        u_v = int((av_orig + np.pi) / (2 * np.pi) * (AZ_RES - 1))
                        u_v = np.clip(u_v, 0, AZ_RES - 1)
                        
                        d_hits_wall = depth_buffer[u_v, 70:] 
                        
                        # Un muro solido deve avere almeno 20 punti verticali e trovarsi oltre l'auto
                        wall_hits = d_hits_wall[d_hits_wall > (r_min + 5.0)]
                        if len(wall_hits) > 20:
                            d_wall = np.min(wall_hits)
                            if dv > d_wall:
                                continue 
                                
                        self.box_shadows[ix, iy, :] = 1
                        self.cone_ownership[(ix, iy)] = box.token

        return self.box_shadows

    def find_object_occlusion_wedges(self):
        """
        FASE 3: Identificazione Ombre Interne (Gap Stitching).
        
        Rileva i 'buchi' naturali nella nube di punti LiDAR per colmare i vuoti tra
        i punti e creare una mappa di 'Shadows' basata sui dati geometrici.
        """
        print("Pulizia e mascheramento cunei (Safety Buffer)...")
        self.internal_shadows = np.zeros_like(self.grid)
        
        # 1. Mappa dei punti occupati con Buffer di Sicurezza (Dilation)
        occupied_bev = np.zeros((GRID_DIM, GRID_DIM), dtype=np.uint8)
        ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM)
        occupied_bev[ix_p[m], iy_p[m]] = 1
        
        from scipy import ndimage
        # Espandiamo il divieto: il magenta deve stare lontano dai punti LiDAR
        safety_mask = ndimage.binary_dilation(occupied_bev, structure=np.ones((3, 3)))
        labeled_structs, num_s = ndimage.label(occupied_bev)
        
        _, boxes, _ = self.nusc.get_sample_data(self.lidar_token)
        
        for box in boxes:
            if not any(c in box.name.lower() for c in ["vehicle", "truck", "bus"]): continue
            
            center = box.center
            dist_A = np.linalg.norm(center[:2])
            corners = box.corners()
            az_c = np.arctan2(corners[1], corners[0])
            if np.max(az_c) - np.min(az_c) > np.pi: az_c[az_c < 0] += 2*np.pi
            
            # CONTRAZIONE: Stringiamo il cuneo di 2 gradi per lato per sicurezza
            az_L, az_R = np.min(az_c) + 0.03, np.max(az_c) - 0.03
            az_C = (az_L + az_R) / 2
            
            found_d_back = -1
            for d in np.linspace(dist_A + 4, GRID_RANGE, 40):
                tx, ty = d * np.cos(az_C), d * np.sin(az_C)
                gx, gy = int(tx/VOXEL_SIZE + GRID_DIM//2), int(ty/VOXEL_SIZE + GRID_DIM//2)
                if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM:
                    if occupied_bev[gx, gy]:
                        found_d_back = d
                        break
            
            if found_d_back > 0 and (found_d_back - dist_A) > 5.0:
                dists_back = []
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
                
                for ix in range(GRID_DIM):
                    for iy in range(GRID_DIM):
                        # RISPETTO RIGOROSO: Se c'è un punto o siamo vicini, NO magenta
                        if safety_mask[ix, iy]: continue
                        
                        xv, yv = x_c[ix], y_c[iy]
                        dv = np.sqrt(xv**2 + yv**2)
                        av = np.arctan2(yv, xv)
                        if av < az_L and az_R > np.pi: av += 2*np.pi
                        
                        if az_L <= av <= az_R:
                            if av < az_C:
                                t = (av - az_L) / (az_C - az_L + 1e-6)
                                d_limit = d_L * (1-t) + d_C * t
                            else:
                                t = (av - az_C) / (az_R - az_C + 1e-6)
                                d_limit = d_C * (1-t) + d_R * t
                            
                            # Il cuneo inizia dopo l'auto e finisce prima del muro
                            if dv >= dist_A + 2.0 and dv < d_limit - 1.0:
                                self.internal_shadows[ix, iy, :] = 1
                                
        # PULIZIA FINALE: Rimuoviamo puntini isolati
        from scipy import ndimage
        shadow_bev = self.internal_shadows[:, :, 0]
        labeled, num_feat = ndimage.label(shadow_bev)
        if num_feat > 0:
            sizes = ndimage.sum(shadow_bev, labeled, range(num_feat + 1))
            mask_noise = (sizes < 20)
            shadow_bev[mask_noise[labeled]] = 0
            for iz in range(Z_DIM):
                self.internal_shadows[:, :, iz] = shadow_bev

        return self.internal_shadows

    def generate_final_visibility(self):
        """
        FASE 4: Fusione e Validazione Finale (Intersection Logic).
        
        Combina la teoria (coni simulati) con la pratica (zona nota LiDAR) per estrarre 
        la mappa di occlusione definitiva (la Red Map del Grafico 4).
        
        Processo:
        - Calcola la zona 'Unknown' (dove il LiDAR non è passato).
        - Interseca l'ignoto con i coni simulati (Intersezione Pura).
        - Sottrae le ombre interne per isolare solo le occlusioni causate dagli oggetti.
        
        Returns:
            np.array: Voxel grid 3D delle occlusioni validate.
        """
        print("Sottrazione occlusioni e generazione mappa finale...")
        self.final_grid = self.grid.copy()
        # Operazione: Known = Known AND (NOT Internal_Occlusions)
        self.final_grid[self.internal_shadows == 1] = 0
        return self.final_grid

    def extract_occlusion_zones(self):
        """
        Logica Definitiva con proiezione verticale corretta:
        Se una colonna (x,y) è nota a qualsiasi altezza, non può essere occlusa.
        """
        print("Calcolo visibilità finale con proiezione verticale...")
        # 1. Generiamo la conoscenza 'pulita' (Known - Internal)
        self.final_grid = self.grid.copy()
        self.final_grid[self.internal_shadows == 1] = 0
        
        # 2. Calcoliamo la visibilità BEV (2D)
        # Se il LiDAR ha visto anche solo un punto in questa colonna, la colonna è 'nota'
        bev_visibility = np.max(self.final_grid, axis=2)
        
        # 3. Sottraiamo la visibilità 2D dai coni 3D
        self.occluded_final = self.box_shadows.copy().astype(bool)
        for iz in range(Z_DIM):
            # Cancelliamo il rosso ovunque il LiDAR abbia visto qualcosa (a terra o in aria)
            self.occluded_final[:, :, iz] &= (bev_visibility == 0)
            
        return self.occluded_final

    def save_occlusions_to_json(self, filename="occlusions_nuscenes.json"):
        """
        FASE 5: Estrazione Poligoni e Ownership Export (Purple Polygons).
        
        Converte la griglia di voxel in un formato vettoriale JSON compatibile con 
        i sistemi di pianificazione.
        
        Algoritmo Shell Mapping:
        - Raggruppa i pixel rossi per oggetto (Ownership).
        - Filtra per distanza (l'occlusione inizia dietro l'auto).
        - Esegue uno scanning radiale per estrarre il perimetro (interno ed esterno).
        - Genera poligoni a 5+ punti (Purple Style) con coordinate metriche.
        """
        import json
        from collections import defaultdict
        print(f"Esportazione dati in {filename} (Geometric-Ownership Mode)...")
        
        # 1. Recuperiamo i dati di tutto il traffico
        _, boxes, _ = self.nusc.get_sample_data(self.lidar_token)
        categories = ["vehicle", "truck", "bus", "trailer", "human.pedestrian", "bicycle", "motorcycle"]
        vehicles = [b for b in boxes if any(c in b.name.lower() for c in categories)]
        
        # 3. Estrazione per Intersezione Geometrica Rigorosa (Shell Mapping)
        occlusions_output = []
        # Mappa Rossa Finale (Grafico 4): La realtà validata
        final_red_map = np.max(self.occluded_final, axis=2).astype(np.uint8)
        
        lidar_occ = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        z_offset = (Z_DIM // 4) * VOXEL_SIZE
        ix_p = ((self.pts[:, 0] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iy_p = ((self.pts[:, 1] / VOXEL_SIZE) + GRID_DIM//2).astype(int)
        iz_p = ((self.pts[:, 2] + z_offset) / VOXEL_SIZE).astype(int)
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & (iy_p >= 0) & (iy_p < GRID_DIM) & (iz_p >= 0) & (iz_p < Z_DIM)
        lidar_occ[ix_p[m], iy_p[m], iz_p[m]] = 1

        x_coords = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_DIM)
        y_coords = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_DIM)
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='ij')
        dist_grid = np.sqrt(xv**2 + yv**2)
        az_grid = np.arctan2(yv, xv)
        
        # 1. Calcolo Ownership Finale (Intersezione Pura)
        # La mappa finale rossa (Grafico 4) contiene pixel che appartengono a diversi oggetti
        final_red_map = np.max(self.occluded_final, axis=2).astype(np.uint8)
        
        # Mappiamo ogni pixel rosso all'oggetto proprietario basandoci sulla simulazione
        object_occlusions = defaultdict(list)
        for ix in range(GRID_DIM):
            for iy in range(GRID_DIM):
                if final_red_map[ix, iy] > 0:
                    token = self.cone_ownership.get((ix, iy))
                    if token:
                        object_occlusions[token].append((ix, iy))
        
        for box in vehicles:
            intersection_pixels = object_occlusions.get(box.token)
            
            # Soglia dinamica: i pedoni sono piccoli, usiamo una soglia minima (3 pixel)
            is_small = any(c in box.name.lower() for c in ["human", "bicycle", "motorcycle"])
            min_px = 3 if is_small else 5
            
            if not intersection_pixels or len(intersection_pixels) < min_px: continue
            
            # 2. Trasformazione e Shell Mapping
            int_pixels_arr = np.array(intersection_pixels)
            r_idx, c_idx = int_pixels_arr[:, 0], int_pixels_arr[:, 1]
            
            xv_int = (r_idx - GRID_DIM//2) * VOXEL_SIZE
            yv_int = (c_idx - GRID_DIM//2) * VOXEL_SIZE
            az_int = np.arctan2(yv_int, xv_int)
            dist_int = np.sqrt(xv_int**2 + yv_int**2)
            
            if np.max(az_int) - np.min(az_int) > np.pi: az_int[az_int < 0] += 2*np.pi
            a_min, a_max = np.min(az_int), np.max(az_int)
            
            # Scansioniamo il perimetro dell'intersezione
            sample_angs = np.linspace(a_min, a_max, 12)
            inner_pts, outer_pts = [], []
            
            for ang in sample_angs:
                m_ray = (np.abs(az_int - ang) < 0.05) | (np.abs(az_int + 2*np.pi - ang) < 0.05)
                if np.any(m_ray):
                    r_near = np.min(dist_int[m_ray])
                    r_far = np.max(dist_int[m_ray])
                    inner_pts.append([round(float(r_near * np.cos(ang)), 2), round(float(r_near * np.sin(ang)), 2)])
                    outer_pts.append([round(float(r_far * np.cos(ang)), 2), round(float(r_far * np.sin(ang)), 2)])
            
            if not inner_pts: continue
            polygon_m = outer_pts + inner_pts[::-1]
            
            # Semplificazione geometrica con Shapely per ridurre il numero dei punti
            from shapely.geometry import Polygon as ShapelyPolygon
            try:
                sh_poly = ShapelyPolygon(polygon_m)
                # Semplificazione con tolleranza 0.2 metri (metà della dimensione del voxel)
                sh_poly_simple = sh_poly.simplify(0.2, preserve_topology=True)
                polygon_m = [[round(x, 2), round(y, 2)] for x, y in sh_poly_simple.exterior.coords]
                # Shapely chiude il poligono ripetendo il primo punto alla fine; lo rimuoviamo per coerenza
                if len(polygon_m) > 1 and polygon_m[0] == polygon_m[-1]:
                    polygon_m = polygon_m[:-1]
            except Exception as e:
                pass
            
            # Metadati
            bbox = [float(np.min(xv_int)), float(np.min(yv_int)),
                    float(np.max(xv_int)), float(np.max(yv_int))]
            
            occlusions_output.append({
                "object_name": box.name,
                "distance_m": round(float(np.linalg.norm(box.center[:2])), 2),
                "occlusion_bbox_m": bbox,
                "polygon_points_m": polygon_m,
                "area_sqm": round(len(intersection_pixels) * (VOXEL_SIZE**2), 2)
            })
            
        with open(filename, "w") as f:
            json.dump({"lidar_token": self.lidar_token, "occlusions": occlusions_output}, f, indent=4)
        
        print(f"Esportazione terminata. Intersezione Coni-Risultati completata ({len(occlusions_output)} oggetti).")
        return filename

    def plot_bev(self):
        from matplotlib.colors import LinearSegmentedColormap
        
        # Estraiamo i BEV necessari
        known_bev = np.max(self.grid, axis=2)
        # Quadrante 2: Known MINUS Internal (La zona nota 'pulita')
        cleaned_known_bev = np.max((self.grid == 1) & (self.internal_shadows == 0), axis=2)
        sim_bev = np.max(self.box_shadows, axis=2)
        occluded_bev = np.max(self.occluded_final, axis=2)
        
        fig, axs = plt.subplots(2, 2, figsize=(18, 18), facecolor='#0f172a')
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
        
        # Colormaps
        cmap_gray = LinearSegmentedColormap.from_list('gray', [(0,0,0,0), (0.6, 0.6, 0.6, 0.8)], N=2)
        cmap_cyan = LinearSegmentedColormap.from_list('cyan', [(0,0,0,0), (0.0, 0.8, 1.0, 0.7)], N=2)
        cmap_yellow = LinearSegmentedColormap.from_list('yel', [(0,0,0,0), (1.0, 0.9, 0.0, 0.8)], N=2)
        cmap_red = LinearSegmentedColormap.from_list('red', [(0,0,0,0), (1.0, 0.0, 0.0, 0.9)], N=2)

        # 1. TOP-LEFT: ZONA NOTA ORIGINALE
        axs[0,0].set_facecolor('#000000')
        axs[0,0].imshow(known_bev, extent=extent, origin='lower', cmap=cmap_gray)
        axs[0,0].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[0,0].set_title("1. ORIGINAL KNOWN ZONE (LIDAR)", color='white', fontsize=14)
        
        # 2. TOP-RIGHT: CLEANED KNOWN (KNOWN - INTERNAL)
        axs[0,1].set_facecolor('#000000')
        axs[0,1].imshow(cleaned_known_bev, extent=extent, origin='lower', cmap=cmap_cyan)
        axs[0,1].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[0,1].set_title("2. CLEANED KNOWN (KNOWN - INTERNAL)", color='white', fontsize=14)

        # 3. BOTTOM-LEFT: CONI SIMULATI (GIALLO)
        axs[1,0].set_facecolor('#000000')
        axs[1,0].imshow(sim_bev, extent=extent, origin='lower', cmap=cmap_yellow)
        axs[1,0].scatter(self.pts[:, 1], self.pts[:, 0], s=0.1, c='white', alpha=0.3)
        axs[1,0].set_title("3. SIMULATED BOX CONES", color='white', fontsize=14)

        # 4. BOTTOM-RIGHT: FINAL RESULT (CONES - (KNOWN & INTERNAL))
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

        for ax in axs.flat:
            ax.set_xlim(GRID_RANGE, -GRID_RANGE)
            ax.set_ylim(-GRID_RANGE, GRID_RANGE)
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    caster = SOTARayCaster(nusc, 87)
    caster.generate_known_zone()
    caster.generate_box_shadows() # Simulazione geometrica
    caster.find_object_occlusion_wedges() # Estrazione buchi interni
    caster.generate_final_visibility() # Pulizia mappa
    caster.extract_occlusion_zones() # Sottrazione finale
    
    # ESPORTAZIONE PER L'AGENTE
    caster.save_occlusions_to_json("occlusions_nuscenes.json")
    
    caster.plot_bev()
