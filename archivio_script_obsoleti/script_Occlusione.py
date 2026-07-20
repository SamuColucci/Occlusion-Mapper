import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import json
import cv2  # Per il processamento delle maschere di visibilità
from nuscenes.nuscenes import NuScenes

# ==============================================================================
# PARAMETRI
# ==============================================================================
SAMPLE_INDEX    = 87
MAX_RANGE_M     = 50.0 # Esteso a 50 metri
GAP_FACTOR      = 5
MIN_GAP_M       = 0.3
MAX_GAP_M       = 100.0  # Alzato per recuperare i gap distanti
INNER_RANGE_M   = 0

# ... (LidarLoader e GapDetector rimangono uguali)

# ==============================================================================
# CLASSE 1: LidarLoader
# ==============================================================================
class LidarLoader:
    """
    Gestisce il caricamento dei dati LiDAR e delle annotazioni da NuScenes.
    Converte le coordinate e prepara i dati per l'analisi.
    """
    def __init__(self, nusc: NuScenes, sample_index: int):
        self.nusc        = nusc
        self.sample      = nusc.sample[sample_index]
        self.lidar_token = self.sample['data']['LIDAR_TOP']
        # Mappatura per convertire gli indici numerici in nomi di classe (es. 24 -> 'car')
        self.idx2name    = {v: k for k, v in nusc.lidarseg_name2idx_mapping.items()}
        self._load_pointcloud()
        self._load_labels()
        self._load_boxes()
        self._extract_manmade_boxes() # Nuova funzione per strutture statiche

    def _load_pointcloud(self):
        """Carica il file binario della nuvola di punti e calcola le distanze."""
        path        = self.nusc.get_sample_data_path(self.lidar_token)
        pc          = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
        self.pts_xy = pc[:, :2] 
        self.rings  = pc[:, 4].astype(int) 
        self.dist2d = np.linalg.norm(self.pts_xy, axis=1) 
        print(f"[LidarLoader] {len(self.pts_xy)} punti caricati")

    def _load_labels(self):
        """Carica le etichette di segmentazione."""
        rec         = self.nusc.get('lidarseg', self.lidar_token)
        path        = os.path.join(self.nusc.dataroot, rec['filename'])
        self.labels = np.fromfile(path, dtype=np.uint8)

    def _load_boxes(self):
        """Recupera gli oggetti mobili (veicoli, pedoni)."""
        _, boxes, _ = self.nusc.get_sample_data(self.lidar_token)
        self.boxes = boxes
        self.boxes_corners = []
        for box in boxes:
            corners = box.bottom_corners()[:2, :].T 
            self.boxes_corners.append(corners)
        print(f"[LidarLoader] {len(self.boxes_corners)} box caricate")

    def _extract_manmade_boxes(self):
        """Identifica ammassi di punti 'manmade' e crea rettangoli virtuali."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("[LidarLoader] ATTENZIONE: scikit-learn non installato. Salto clustering manmade.")
            return
        # Classe 28 = static.manmade
        mask = (self.labels == 28) & (self.dist2d < MAX_RANGE_M)
        pts = self.pts_xy[mask]
        
        if len(pts) < 15: return
        
        # Clustering: raggruppiamo punti entro 2 metri
        clustering = DBSCAN(eps=2.0, min_samples=15).fit(pts)
        labels = clustering.labels_
        
        count = 0
        for cluster_id in np.unique(labels):
            if cluster_id == -1: continue
            
            cluster_pts = pts[labels == cluster_id]
            # Usiamo OpenCV per trovare il rettangolo minimo che racchiude i punti
            rect = cv2.minAreaRect(cluster_pts.astype(np.float32))
            box_points = cv2.boxPoints(rect)
            
            self.boxes_corners.append(box_points)
            # Creiamo un'annotazione fittizia per compatibilità
            class FakeBox:
                def __init__(self): self.name = "static.manmade"
                @property
                def wlh(self): return [5.0, 5.0, 5.0] # Dimensione dummy per il filtro
            self.boxes.append(FakeBox())
            count += 1
        print(f"[LidarLoader] {count} strutture manmade identificate")


# ==============================================================================
# CLASSE 2: GapDetector
# ==============================================================================
class GapDetector:
    """
    Analizza i punti LiDAR ring per ring per trovare 'salti' (gap) nella scansione.
    Questi gap rappresentano le zone dove il LiDAR perde la visibilità dietro un ostacolo.
    """
    # Classi lidarseg considerate superfici (dove cerchiamo i buchi)
    VALID_CLASSES = [24, 25, 26, 27]

    def __init__(self, loader: LidarLoader):
        self.loader      = loader
        self.all_breaks  = []

    def detect_gaps(self):
        """Scansiona ogni ring LiDAR e trova le discontinuità."""
        l         = self.loader
        # Maschera per filtrare i punti validi (classi ostacolo ed entro il range)
        mask      = (np.isin(l.labels, self.VALID_CLASSES)
                     & (l.dist2d > INNER_RANGE_M) & (l.dist2d < MAX_RANGE_M))
        pts_valid = l.pts_xy[mask]
        rings_v   = l.rings[mask]
        breaks    = []

        # Analizziamo ogni ring separatamente
        for ring_id in np.unique(rings_v):
            if ring_id < 3: continue # Salta i ring che puntano troppo verso il basso
            rm   = rings_v == ring_id
            rp   = pts_valid[rm]
            # Calcoliamo gli angoli per ordinare i punti in modo circolare
            rang = np.arctan2(rp[:, 1], rp[:, 0])
            if len(rp) < 4:
                continue
            si   = np.argsort(rang)
            rp   = rp[si]; rang = rang[si]
            
            # Calcoliamo la distanza tra ogni punto e il successivo nel ring
            n    = len(rp)
            diffs = np.linalg.norm(np.diff(rp, axis=0), axis=1)
            # Gestiamo la chiusura del cerchio (tra l'ultimo e il primo punto)
            wrap  = np.linalg.norm(rp[0] - rp[-1])
            all_d = np.append(diffs, wrap)
            
            # SOGLIA DINAMICA FISICA: 
            # In un LiDAR, la distanza tra i punti aumenta con la distanza dal sensore.
            # Usiamo la risoluzione angolare tipica (0.2 gradi) per calcolare lo spazio "normale".
            res_rad = np.radians(0.2)
            # Per ogni coppia di punti, calcoliamo la soglia massima tollerabile
            # prima di considerarlo un gap (occlusione).
            dist_punti = np.linalg.norm(rp, axis=1)
            thrs_fisici = dist_punti * np.sin(res_rad) * GAP_FACTOR
            
            for gi in range(n):
                li = gi; ri = (gi + 1) % n
                # Soglia adattiva per questo specifico punto del ring
                thr = max(thrs_fisici[li], MIN_GAP_M)

                if all_d[gi] > thr and all_d[gi] < MAX_GAP_M:
                    ang_stop  = float(rang[li])
                    # Calcoliamo l'angolo di inizio assicurandoci che sia coerente
                    ang_start = float(rang[ri] if ri > li else rang[ri] + 2*np.pi)
                    
                    # Salviamo tutti i dettagli del gap trovato
                    breaks.append({
                        'ring_id':   int(ring_id),
                        'P_stop':    rp[li].tolist(),   # Punto dove finisce la visibilità
                        'P_start':   rp[ri].tolist(),   # Punto dove ricomincia la visibilità
                        'mid':       ((rp[li] + rp[ri]) / 2.0).tolist(),
                        'ang_stop':  ang_stop,
                        'ang_start': ang_start,
                        'ang_mid':   (ang_stop + ang_start) / 2.0,
                        'gap_m':     float(all_d[gi]),
                    })

        print(f"[GapDetector] {len(breaks)} interruzioni trovate")
        self.all_breaks = breaks

    def run(self):
        """Esegue il rilevamento e lancia un errore se non trova nulla."""
        self.detect_gaps()
        if not self.all_breaks:
            raise RuntimeError("Nessun gap trovato!")

# ==============================================================================
# CLASSE 3: OccupancyMapper
# ==============================================================================
class OccupancyMapper:
    def __init__(self, loader: LidarLoader, res=0.1, size_m=60):
        self.loader = loader
        self.res = res
        self.size = int(size_m / res)
        self.center = self.size // 2
        self.zones = []

    def refine_with_gaps(self, gap_detector):
        """
        Analizza ogni oggetto rilevato e identifica i punti di ancoraggio 
        per le zone di occlusione basandosi sui gap LiDAR.
        """
        self.zones = []
        all_breaks = gap_detector.all_breaks
        if not all_breaks: return

        # Ciclo attraverso tutti gli oggetti (Bounding Boxes) presenti nel frame
        for i, corners in enumerate(self.loader.boxes_corners):
            box = self.loader.boxes[i]
            
            # 1. FILTRAGGIO OGGETTI (Sicurezza)
            # Calcoliamo la distanza euclidea di ogni spigolo dall'origine (Ego Vehicle)
            dists = np.linalg.norm(corners, axis=1)
            dist_min = np.min(dists)
            obj_name = self.loader.boxes[i].name 
            
            # FILTRI DI SICUREZZA:
            # - Escludiamo l'Ego Vehicle stesso (evita ombre che partono dal centro)
            # - Escludiamo oggetti troppo vicini (dist < 1.5m, zona cieca e riflessi scocca)
            # - Escludiamo oggetti oltre il range LiDAR (25m)
            if "ego" in obj_name.lower() or dist_min < 1.5 or dist_min > MAX_RANGE_M:
                continue
            
            # 2. ANALISI ANGOLARE (Shadow Cone)
            # Calcoliamo l'angolo di tutti e 4 gli spigoli rispetto al sensore
            all_angs = np.arctan2(corners[:, 1], corners[:, 0])
            
            # Gestione del "salto" a +-180 gradi (dietro il veicolo)
            if np.max(all_angs) - np.min(all_angs) > np.pi:
                all_angs = np.where(all_angs < 0, all_angs + 2*np.pi, all_angs)
            
            # FILTRO APERTURA: Se l'oggetto copre piu di 90 gradi, la geometria e sospetta.
            # NOTA: Per gli edifici (static.manmade) permettiamo aperture maggiori perché possono essere molto estesi.
            apertura_rad = np.max(all_angs) - np.min(all_angs)
            if "manmade" not in obj_name.lower():
                if apertura_rad > (np.pi / 2): # Limite 90 gradi solo per oggetti piccoli
                    continue
            
            if apertura_rad < np.radians(2.0): # Troppo sottile
                continue
            
            # FILTRO DIMENSIONE FISICA: Ignoriamo pali o oggetti troppo piccoli (area < 0.5 mq)
            # o oggetti classificati come detriti (debris)
            area_box = box.wlh[0] * box.wlh[1]
            if area_box < 0.5 or "debris" in obj_name.lower():
                continue
            
            # 3. IDENTIFICAZIONE ESTREMI
            # Troviamo gli indici dei punti con l'angolo più a destra (min) e più a sinistra (max)
            idx_right = np.argmin(all_angs)
            idx_left = np.argmax(all_angs)
            
            # Punti di ancoraggio per l'inizio dell'ombra
            p_right = corners[idx_right]
            p_left  = corners[idx_left]
            
            # Range angolare in cui l'oggetto proietta l'ombra
            ang_min = all_angs[idx_right]
            ang_max = all_angs[idx_left]
            
            # 4. DEBUG LOGGING
            # Recuperiamo il nome del tipo di oggetto (es. vehicle.car)
            obj_name = self.loader.boxes[i].name 
            print(f"[{i}] OGGETTO: {obj_name} | Distanza min: {dist_min:.2f}m")
            print(f"  -> SPIGOLO DX: {p_right} | Angolo: {np.degrees(ang_min):.2f}°")
            print(f"  -> SPIGOLO SX: {p_left} | Angolo: {np.degrees(ang_max):.2f}°")
            print("-" * 30)

            # --- COSTRUZIONE IBRIDA AVANZATA
            side_right = [p_right] 
            side_left  = [p_left]
            
            for ring in range(0, 32):
                gap_nel_ring = []
                for b in all_breaks:
                    if b['ring_id'] == ring:
                        g_ang = b['ang_mid']
                        if np.max(all_angs) > np.pi and g_ang < 0:
                            g_ang += 2*np.pi
                        if ang_min <= g_ang <= ang_max:
                            gap_nel_ring.append(b)
                
                if gap_nel_ring:
                    ref = (np.array(side_right[-1]) + np.array(side_left[-1])) / 2
                    ref_dist = np.linalg.norm(ref)
                    b = min(gap_nel_ring, key=lambda x: np.linalg.norm(np.array(x['mid']) - ref))
                    dist_nuovo = np.linalg.norm(b['mid'])
                    
                    # FILTRO CONTINUIT�: 
                    if dist_nuovo > ref_dist and (dist_nuovo - ref_dist) < 5.0:
                        side_right.append(b['P_stop'])
                        side_left.append(b['P_start'])
                    elif dist_nuovo > ref_dist:
                        break

            # 5. CHIUSURA E PROIEZIONE RAFFINATA
            # Se è una struttura manmade o una barriera, forziamo la proiezione al limite della mappa (50m)
            force_full = ("manmade" in obj_name.lower() or "barrier" in obj_name.lower())
            
            if len(side_right) == 1:
                dist_proiez = MAX_RANGE_M if force_full else min(dist_min + 15.0, MAX_RANGE_M)
                side_right.append((p_right / np.linalg.norm(p_right)) * dist_proiez)
                side_left.append((p_left / np.linalg.norm(p_left)) * dist_proiez)
            else:
                last_r = np.array(side_right[-1])
                last_l = np.array(side_left[-1])
                # Se è un edificio/barriera, spingiamo l'ultimo punto fino al bordo
                if force_full or np.linalg.norm(last_r) < MAX_RANGE_M:
                    side_right.append((last_r / np.linalg.norm(last_r)) * MAX_RANGE_M)
                    side_left.append((last_l / np.linalg.norm(last_l)) * MAX_RANGE_M)
                
            poly = np.array(side_right + side_left[::-1])
            self.zones.append(poly)


            

# ==============================================================================
# ==============================================================================
# ==============================================================================
class DebugPrinter:
    """Utility per stampare i dati dei gap rilevati in formato tabella nel terminale."""
    def __init__(self, gap_detector: GapDetector):
        self.gd = gap_detector
    def print_all_points(self):
        print("\n" + "="*85)
        print(f"{'RING':<6} | {'TIPO':<8} | {'X (m)':<9} | {'Y (m)':<9} | {'ANG ()':<8} | {'DIST (m)':<8}")
        print("-" * 85)
        for b in self.gd.all_breaks:
            p0, p1 = b['P_stop'], b['P_start']
            a0, a1 = np.degrees(b['ang_stop']), np.degrees(b['ang_start'])
            d0, d1 = np.linalg.norm(p0), np.linalg.norm(p1)
            # Stampiamo i due punti (fine visibilità e inizio nuova visibilità)
            print(f"{b['ring_id']:<6} | {'P_STOP':<8} | {p0[0]:>9.3f} | {p0[1]:>9.3f} | {a0:>8.2f} | {d0:>8.2f}")
            print(f"{b['ring_id']:<6} | {'P_START':<8} | {p1[0]:>9.3f} | {p1[1]:>9.3f} | {a1:>8.2f} | {d1:>8.2f}")
            print("-" * 85)
        print(f"Totale interruzioni elaborate: {len(self.gd.all_breaks)}")
        print("="*85 + "\n")

# ==============================================================================
# CLASSE 5: Exporter
# ==============================================================================
class Exporter:
    """Gestisce il salvataggio dei dati in formato JSON per usi esterni."""
    def __init__(self, gap_detector: GapDetector, mapper: OccupancyMapper):
        self.gd = gap_detector
        self.mapper = mapper
    def save_to_json(self, filename="occlusion_gaps.json"):
        import json
        data = {
            "gaps": self.gd.all_breaks,
            "occlusion_polygons": [z.tolist() for z in self.mapper.zones] # Convertiamo NumPy -> List
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[Exporter] Dati salvati in {filename}")

# ==============================================================================
# CLASSE 6: Visualizer
# ==============================================================================
class Visualizer:
    """Gestisce la rappresentazione grafica 2D dei dati e dei risultati."""
    def __init__(self, loader: LidarLoader, gap_detector: GapDetector, mapper: OccupancyMapper):
        self.loader  = loader
        self.gd      = gap_detector
        self.mapper  = mapper

    def plot(self):
        # Chiudiamo eventuali finestre residue e attiviamo la modalità interattiva
        plt.close('all')
        plt.ion() 
        l = self.loader
        
        # 1. Apriamo la panoramica multi-camera di NuScenes (Non bloccante grazie a ion)
        self.loader.nusc.render_sample(self.loader.sample['token'])

        # 2. Prepariamo i dati per la nostra mappa
        recon_pts = []
        for b in self.gd.all_breaks:
            p0, p1 = np.array(b['P_stop']), np.array(b['P_start'])
            if np.linalg.norm(p0 - p1) > MAX_GAP_M: continue
            
            d0, d1 = np.linalg.norm(p0), np.linalg.norm(p1)
            # Calcoliamo la differenza angolare corretta (tenendo conto del wrap 360)
            ang_stop, ang_start = b['ang_stop'], b['ang_start']
            gap_ang = ang_start - ang_stop
            
            # Aumentiamo la densità dei punti con la distanza per mantenere la curva fluida
            num_pts = max(15, int(np.degrees(gap_ang) * 3))
            for i in range(1, num_pts + 1):
                t = i / (num_pts + 1)
                ang = ang_stop + t * gap_ang
                # Interpolazione radiale: segue la curvatura del ring LiDAR
                dist = d0 + t * (d1 - d0)
                px, py = dist * np.cos(ang), dist * np.sin(ang)
                recon_pts.append([px, py])
        recon_pts = np.array(recon_pts) if recon_pts else np.empty((0, 2))

        # 3. Creiamo la SECONDA finestra (Mappa Occlusioni + Lidarseg)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        fig.patch.set_facecolor('#0a0a0f') 
        
        # --- SUBPLOT 1: Mappa Occlusioni ---
        # Ingrandiamo i limiti della vista: Orizzontale (-60, 60), Verticale (-60, 60)
        ax1.set_xlim(-60, 60); ax1.set_ylim(-60, 60)
        ax1.set_facecolor('#0a0a0f')
        ax1.set_aspect('equal')
        ax1.grid(True, color='#1a1a2f', linestyle='--', alpha=0.4)
        ax1.tick_params(colors='white')
        
        # Ruotiamo i punti di altri 180 gradi: X_plot = Y_lidar, Y_plot = -X_lidar
        ax1.scatter(l.pts_xy[:, 1], -l.pts_xy[:, 0], c='#ffffff', s=1, alpha=0.1, label='LiDAR Raw')
        if len(recon_pts) > 0:
            ax1.scatter(recon_pts[:, 1], -recon_pts[:, 0], s=4, c='#00ff88', alpha=0.6, label='Gap Ricostruiti')

        for corners in l.boxes_corners:
            # Trasformazione: (y, -x)
            rot_corners = np.column_stack((corners[:, 1], -corners[:, 0]))
            ax1.add_patch(mpatches.Polygon(rot_corners, closed=True, facecolor='none', edgecolor='#00ffff', linewidth=1.5, alpha=0.8))

        if self.gd.all_breaks:
            stops = np.array([b['P_stop'] for b in self.gd.all_breaks])
            starts = np.array([b['P_start'] for b in self.gd.all_breaks])
            ax1.scatter(stops[:, 1], -stops[:, 0], c='#ffaa00', s=25, marker='o', label='P_stop')
            ax1.scatter(starts[:, 1], -starts[:, 0], c='#ff00aa', s=25, marker='s', label='P_start')

        for i, poly in enumerate(self.mapper.zones):
            label = 'Zona Occlusa' if i == 0 else None
            # Trasformazione: (y, -x)
            rot_poly = np.column_stack((poly[:, 1], -poly[:, 0]))
            ax1.add_patch(mpatches.Polygon(rot_poly, closed=True, color='#ff0000', alpha=0.3, zorder=2, label=label))

        ax1.scatter(0, 0, s=300, c='#ffe600', marker='*', edgecolors='white', zorder=10)
        ax1.set_title("Mappa Occlusioni (Sincronizzata)", color='white', fontsize=12)
        ax1.legend(facecolor='#0a0a0f', labelcolor='white', loc='upper right', fontsize='small')

        # --- SUBPLOT 2: NuScenes Lidarseg ---
        self.loader.nusc.render_sample_data(self.loader.lidar_token,
                                            with_anns=False,
                                            show_lidarseg=True,
                                            show_lidarseg_legend=True,
                                            ax=ax2)
        ax2.set_title("NuScenes Lidarseg (Verità)", color='white', fontsize=12)
        
        plt.tight_layout()
        plt.ioff() # Disattiva modalità interattiva per bloccare le finestre aperte
        plt.show()

# ==============================================================================
# MAIN (Punto di ingresso dello script)
# ==============================================================================
if __name__ == '__main__':
    # Inizializziamo NuScenes (richiede il dataset nella cartella ./nuscenes)
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)

    # 1. Caricamento Dati
    loader = LidarLoader(nusc, SAMPLE_INDEX)

    # 2. Analisi dei Gap
    gap_detector = GapDetector(loader)
    gap_detector.run()

    # 3. Debug opzionale
    printer = DebugPrinter(gap_detector)
    # printer.print_all_points()

    # 4. Mappatura Occlusioni
    mapper = OccupancyMapper(loader)
    # Calcolo delle zone d'ombra
    mapper.refine_with_gaps(gap_detector)

    # 5. Esportazione Risultati
    exporter = Exporter(gap_detector, mapper)
    exporter.save_to_json()

    # 6. Visualizzazione Grafica
    Visualizer(loader, gap_detector, mapper).plot()
