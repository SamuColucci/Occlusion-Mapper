# Macro-Area 3D: Visualizzatore Comparativo Interattivo Affiancato (Bayes vs Per-Zone).
# Mostra contemporaneamente la Mappa Bayesiana e la Mappa Neurale Per-Zone in modo sincronizzato 1-a-1.

# Import dei moduli di sistema per la manipolazione dei percorsi di ricerca Python
import sys
import os

# Aggiunge la directory radice del progetto al sys.path per consentire l'importazione di moduli accessori
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import di glob per reperire l'elenco dei file di predizione in formato JSON
import glob
# Import di json per la lettura e deserializzazione delle predizioni salvate su disco
import json
# Import di numpy per operazioni di calcolo algebrico ed interpolazione vettoriale
import numpy as np
# Import del pacchetto Matplotlib per la visualizzazione grafica ed il rendering delle mappe
import matplotlib.pyplot as plt
# Import di Patch da matplotlib per la creazione degli elementi della legenda
from matplotlib.patches import Patch
# Import delle primitive geometriche di Shapely per le verifiche di contenimento punto-in-poligono
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
# Import del SDK nuScenes per la visualizzazione dello sfondo stradale HD Map e del sistema di riferimento
from nuscenes.nuscenes import NuScenes
# Import di Quaternion per le trasformazioni di rotazione 3D tra il sensore LiDAR e l'Ego Vehicle
from pyquaternion import Quaternion

# Classe principale per la gestione del visualizzatore comparativo affiancato Bayes vs Neurale Per-Zone
class ComparisonRuntimeVisualizer:
    def __init__(self):
        # Cartella contenente le occlusioni e le probabilità stimate dall'Agente Bayesiano
        self.dir_bayes = "extracted_occlusions_probabilities"
        # Cartella contenente le occlusioni e le probabilità stimate dall'Agente Neurale Per-Zone
        self.dir_pz = "extracted_occlusions_per_zone"
        # Se la cartella predefinita per-zone non esiste, utilizza la cartella alternativa per le predizioni neurali
        if not os.path.exists(self.dir_pz):
            self.dir_pz = "extracted_occlusions_neural"

        # Stampa dei messaggi di avvio e stato del caricamento dei componenti
        print("\nCaricamento dati per il Confronto Affiancato (Bayes vs Per-Zone)...")
        print("Inizializzazione NuScenes...")
        # Istanzia la classe SDK nuScenes selezionando la versione 'v1.0-mini'
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        
        # Recupera ed ordina tutti i file JSON delle predizioni bayesiane presenti nella cartella
        self.json_files = sorted(glob.glob(os.path.join(self.dir_bayes, "*.json")))
        # Se non viene trovato alcun file JSON, mostra un errore ed interrompe l'esecuzione
        if not self.json_files:
            print(f"[ERROR] Nessun file JSON trovato in '{self.dir_bayes}'!")
            return
            
        # Indice del file di fotogramma JSON attualmente selezionato per la visualizzazione
        self.current_file_idx = 0
        # Indice dell'occlusione singola selezionata all'interno del fotogramma corrente
        self.current_occ_idx = 0
        
        # Carica il contenuto dei file JSON sia per l'Agente Bayesiano che per l'Agente Neurale
        self.load_json_file()
        
        # Configurazione della figura Matplotlib: 3 sottografici (HUD a sinistra, Mappa Bayes al centro, Per-Zone a destra)
        self.fig, (self.ax_hud, self.ax_bayes, self.ax_pz) = plt.subplots(
            1, 3, figsize=(18, 9), facecolor='#0B0C10',
            gridspec_kw={'width_ratios': [1, 2, 2]}
        )
        # Impostazione dei margini e degli spazi interni per ottimizzare il layout grafico
        self.fig.subplots_adjust(left=0.02, right=0.82, top=0.92, bottom=0.04, wspace=0.06)
        # Impostazione dello sfondo scuro (dark mode) per ciascuno dei 3 pannelli di disegno
        self.ax_hud.set_facecolor('#0B0C10')
        self.ax_bayes.set_facecolor('#0B0C10')
        self.ax_pz.set_facecolor('#0B0C10')
        
        # Collegamento degli eventi della tastiera e del mouse alle rispettive funzioni callback di risposta
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Stampa delle istruzioni e dei comandi da tastiera e mouse per l'utente nel terminale
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI COMPARATIVI")
        print("=" * 50)
        print(" -> MOUSE            : Evidenzia simultaneamente l'ombra in entrambe le mappe")
        print(" -> FRECCIA SU/GIÙ   : Cambia fotogramma")
        print(" -> FRECCIA DX/SX    : Scorri le zone d'ombra")
        print("=" * 50 + "\n")
        
        # Genera il primo disegno comparativo a schermo e mostra la finestra interattiva
        self.plot_current()
        plt.show()

    def load_json_file(self):
        # Recupera il percorso completo del file JSON bayesiano corrente
        json_path_bayes = self.json_files[self.current_file_idx]
        # Estrae il nome base del file JSON per trovare il corrispondente file per-zone
        fname = os.path.basename(json_path_bayes)
        json_path_pz = os.path.join(self.dir_pz, fname)
        
        # Apre e deserializza il file JSON dell'Agente Bayesiano
        with open(json_path_bayes, 'r') as f:
            self.data_bayes = json.load(f)
            
        # Se esiste il file corrispondente dell'Agente Per-Zone, lo legge, altrimenti riutilizza quello bayesiano
        if os.path.exists(json_path_pz):
            with open(json_path_pz, 'r') as f:
                self.data_pz = json.load(f)
        else:
            self.data_pz = self.data_bayes

        # Estrae le liste di occlusioni per entrambi gli agenti dal relativo dizionario JSON
        self.occs_bayes = self.data_bayes.get('occlusions', [])
        self.occs_pz = self.data_pz.get('occlusions', [])
        # Resetta l'indice dell'occlusione selezionata all'inizio del nuovo fotogramma
        self.current_occ_idx = 0
        # Recupera il token identificativo del sensore LiDAR dal file JSON
        self.lidar_token = self.data_bayes.get('lidar_token')

    def get_unicode_bar(self, val, length=12):
        # Calcola quanti caratteri pieni disegnare in base al valore percentuale compreso tra 0 e 1
        filled = int(round(val * length))
        # Genera e restituisce la barra di avanzamento in formato testuale unicode
        return '█' * filled + '░' * (length - filled)

    def lidar_to_ego(self, pts_lidar):
        # Se l'array di punti è vuoto o non è presente il token del LiDAR, restituisce un array vuoto
        if len(pts_lidar) == 0 or not self.lidar_token:
            return np.zeros((0, 2))
        try:
            # Recupera le meta-informazioni del campione di dati LiDAR tramite l'SDK nuScenes
            sd = self.nusc.get('sample_data', self.lidar_token)
            # Recupera i dati di calibrazione del sensore rispetto all'Ego Vehicle
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            # Estrae la matrice di rotazione dal quaternione di calibrazione
            q_sensor = Quaternion(cs['rotation'])
            R_sensor = q_sensor.rotation_matrix[:2, :2]
            # Estrae il vettore di traslazione 2D (x, y) dal sensore
            t_sensor = np.array(cs['translation'])[:2]
            # Trasforma i punti dal sistema di riferimento LiDAR al sistema di riferimento dell'Ego Vehicle
            return pts_lidar @ R_sensor.T + t_sensor
        except Exception:
            # In caso di errore durante la trasformazione, restituisce i punti originali non trasformati
            return pts_lidar

    def _draw_map(self, ax, occs, title_str):
        # Pulisce gli elementi grafici precedenti dal pannello di disegno specificato
        ax.clear()
        # Nasconde gli assi graduati per rendere la mappa visivamente più pulita
        ax.axis('off')
        
        # 1. RENDER NATIVO UFFICIALE NUSCENES (Strada HD Map, 3D Boxes, LiDAR PointCloud)
        if self.lidar_token:
            try:
                # Disegna lo sfondo della mappa HD e le nuvole di punti LiDAR tramite l'SDK ufficiale nuScenes
                self.nusc.render_sample_data(self.lidar_token, ax=ax, underlay_map=True, verbose=False)
            except Exception as e:
                # Mostra un avviso se il rendering dello sfondo nuScenes fallisce
                print(f"[WARN] Impossibile renderizzare il background nuScenes: {e}")
                
        # Imposta il titolo identificativo della mappa in alto con font personalizzato stilizzato
        ax.set_title(title_str, color='#66FCF1', fontsize=11, fontweight='bold', pad=12)

        # Se la lista delle zone d'ombra è vuota, interrompe la procedura di disegno delle ombre
        if not occs:
            return

        # 2. Disegno delle occlusioni non selezionate in colore scuro trasparente
        for idx, occ in enumerate(occs):
            # Salta l'occlusione attualmente selezionata per disegnarla successivamente in evidenza
            if idx == self.current_occ_idx:
                continue
            # Estrae i vertici del poligono dell'occlusione espresso in metri
            poly_pts = np.array(occ.get('polygon_points_m', []))
            # Disegna il poligono solo se ha almeno 3 vertici validi
            if len(poly_pts) >= 3:
                # Chiude il poligono connettendo l'ultimo punto al primo punto della sequenza
                poly_closed = np.vstack([poly_pts, poly_pts[0]])
                # Trasforma le coordinate dal riferimento LiDAR a quello dell'Ego Vehicle
                poly_ego = self.lidar_to_ego(poly_closed)
                # Disegna il contorno sottile dell'occlusione secondaria
                ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#2C374E', linewidth=1.0, alpha=0.5, zorder=4)
                # Riempie l'area dell'occlusione con un colore scuro semitrasparente
                ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#1E293B', alpha=0.15, zorder=4)

        # 3. Disegno dell'occlusione attualmente selezionata dall'utente con evidenziazione in rosso acceso
        if self.current_occ_idx < len(occs):
            # Estrae l'occlusione corrispondente all'indice corrente
            occ = occs[self.current_occ_idx]
            poly = np.array(occ.get('polygon_points_m', []))
            # Verifica la validità geometrica del poligono
            if len(poly) >= 3:
                # Chiude la sequenza di punti del poligono
                poly_closed = np.vstack([poly, poly[0]])
                # Converte i vertici nel sistema di riferimento Ego
                poly_ego = self.lidar_to_ego(poly_closed)
                # Riempie l'occlusione attiva con un colore rosso scuro semitrasparente
                ax.fill(poly_ego[:, 0], poly_ego[:, 1], color='#8B0000', alpha=0.50, zorder=5)
                # Disegna il bordo evidenziato dell'occlusione in rosa/rosso brillante
                ax.plot(poly_ego[:, 0], poly_ego[:, 1], color='#FF0055', linewidth=2.2, zorder=5)
                
                # Calcola il baricentro (centro geometrico) dell'occlusione per posizionarvi laラベル con il numero dell'ombra
                cx = float(np.mean(poly_ego[:, 0]))
                cy = float(np.mean(poly_ego[:, 1]))
                # Stampa il numero identificativo dell'occlusione (#1, #2, ecc.) al centro del poligono
                ax.text(cx, cy, f"#{self.current_occ_idx+1}", color='white', fontsize=9, fontweight='bold', ha='center', va='center', zorder=8)

    def plot_current(self):
        # Estrae il nome del file JSON corrente per visualizzarlo nell'intestazione del pannello
        json_filename = os.path.basename(self.json_files[self.current_file_idx])
        
        # 1. Rendering del pannello centrale della Mappa Bayesiana
        self._draw_map(self.ax_bayes, self.occs_bayes, f"AGENTE BAYESIANO DINAMICO\nFrame: {json_filename}")
        
        # 2. Rendering del pannello di destra della Mappa Neurale Per-Zone
        ckpt_used = getattr(self, 'data_pz', {}).get("model_checkpoint_used", "N/A")
        self._draw_map(self.ax_pz, self.occs_pz, f"AGENTE NEURALE PER-ZONE\n(Checkpoint: {ckpt_used})")
        
        # 3. Costruzione del Pannello HUD Comparativo Affiancato a sinistra
        self.ax_hud.clear()
        self.ax_hud.axis('off')
        
        # Se non ci sono occlusioni o l'indice non è valido, aggiorna la tela e termina
        if not self.occs_bayes or self.current_occ_idx >= len(self.occs_bayes):
            self.fig.canvas.draw()
            return
            
        # Estrae l'occlusione selezionata sia dal dizionario Bayesiano che dal dizionario Per-Zone
        occ_b = self.occs_bayes[self.current_occ_idx]
        occ_pz = self.occs_pz[self.current_occ_idx] if self.current_occ_idx < len(self.occs_pz) else occ_b
        
        # Estrae le proprietà geometriche ed il tipo di terreno associato alla zona cieca
        dist = occ_b.get('distance_m', 0.0)
        area = occ_b.get('area_sqm', 0.0)
        terrain_str = occ_b.get('terrain_type', 'Sconosciuto')
        
        # Estrae i dizionari delle probabilità stimate dai due rispettivi agenti
        bayes_probs = occ_b.get('estimated_probabilities', {})
        pz_probs = occ_pz.get('estimated_probabilities', {})
        
        # Costruzione delle righe di testo da formattare ed inserire nel box dell'HUD
        hud_lines = []
        hud_lines.append(f"=== OCCLUSION #{self.current_occ_idx+1}/{len(self.occs_bayes)} ===")
        hud_lines.append(f"Distanza: {dist:.1f} m | Area: {area:.1f} m²")
        hud_lines.append(f"Terreno : {terrain_str}")
        hud_lines.append("=" * 30)
        hud_lines.append(" CONFRONTO PROBABILITÀ GLOBALI")
        hud_lines.append(" CATEGORIA   | BAYES  | PER-ZONE")
        hud_lines.append("-" * 30)
        
        # Scorre ciascuna delle 6 categorie principali ed impagina le probabilità affiancate a confronto
        for k in ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Barriera"]:
            pb = bayes_probs.get(k, 0.0) * 100
            ppz = pz_probs.get(k, 0.0) * 100
            hud_lines.append(f" {k:<11}: {pb:>5.1f}%  vs  {ppz:>5.1f}%")
            
        # Identifica la categoria con la probabilità più alta predetta dall'Agente Bayesiano e dall'Agente Per-Zone
        b_best = max(bayes_probs, key=bayes_probs.get) if bayes_probs else "N/A"
        pz_best = max(pz_probs, key=pz_probs.get) if pz_probs else "N/A"
        
        # Aggiunge all'HUD l'evidenziazione della predizione principale di entrambi i modelli
        hud_lines.append("=" * 30)
        hud_lines.append(f"BAYES   : {b_best} ({bayes_probs.get(b_best, 0.0)*100:.0f}%)")
        hud_lines.append(f"PER-ZONE: {pz_best} ({pz_probs.get(pz_best, 0.0)*100:.0f}%)")
        
        # Unisce tutte le righe in un'unica stringa formattata
        hud_box_str = "\n".join(hud_lines)
        
        # Disegna il box di testo dell'HUD nel pannello a sinistra con sfondo scuro e bordo azzurro neon
        self.ax_hud.text(
            0.05, 0.95, hud_box_str, color='white', fontsize=8.0, family='monospace',
            bbox=dict(facecolor='#1E293B', alpha=0.90, edgecolor='#00E5FF', boxstyle='round,pad=0.8'),
            va='top', ha='left', transform=self.ax_hud.transAxes
        )
        
        # Creazione dei rettangoli di legenda per identificare i colori delle classi e dei poligoni
        legend_elements = [
            Patch(facecolor='#8B0000', edgecolor='#FF0055', alpha=0.5, label='Occlusione Selezionata'),
            Patch(facecolor='#2C374E', edgecolor='#2C374E', alpha=0.4, label='Altre Occlusioni'),
            Patch(facecolor='#00E5FF', edgecolor='#00B0FF', alpha=0.6, label='Auto nuScenes'),
            Patch(facecolor='#FFD600', edgecolor='#FFAB00', alpha=0.6, label='Camion / Bus nuScenes'),
            Patch(facecolor='#FF1744', edgecolor='#D50000', alpha=0.6, label='Pedone nuScenes'),
        ]
        # Posiziona la legenda a destra della mappa neurale
        self.ax_pz.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), facecolor="#1E293B",
                          edgecolor="gray", fontsize=7.5, labelcolor="white")
        # Aggiorna la tela grafica con il nuovo contenuto renderizzato
        self.fig.canvas.draw()

    def on_key(self, event):
        # Gestisce la pressione della freccia destra per passare all'occlusione successiva
        if event.key == 'right':
            if self.occs_bayes:
                self.current_occ_idx = (self.current_occ_idx + 1) % len(self.occs_bayes)
                self.plot_current()
        # Gestisce la pressione della freccia sinistra per tornare all'occlusione precedente
        elif event.key == 'left':
            if self.occs_bayes:
                self.current_occ_idx = (self.current_occ_idx - 1) % len(self.occs_bayes)
                self.plot_current()
        # Gestisce la pressione della freccia giù per avanzare al fotogramma JSON successivo
        elif event.key == 'down':
            if len(self.json_files) > 1:
                self.current_file_idx = (self.current_file_idx + 1) % len(self.json_files)
                self.load_json_file()
                self.plot_current()
        # Gestisce la pressione della freccia su per tornare al fotogramma JSON precedente
        elif event.key == 'up':
            if len(self.json_files) > 1:
                self.current_file_idx = (self.current_file_idx - 1) % len(self.json_files)
                self.load_json_file()
                self.plot_current()

    def on_mouse_move(self, event):
        # Ignora i movimenti del mouse esterni ai due pannelli di mappa
        if event.inaxes not in [self.ax_bayes, self.ax_pz]:
            return
        # Verifica che le coordinate x, y del cursore siano valide
        if event.xdata is None or event.ydata is None:
            return
            
        # Crea un punto Shapely corrispondente alla posizione corrente del mouse
        point = ShapelyPoint(event.xdata, event.ydata)
        
        # Scorre tutte le occlusioni per verificare se il puntatore del mouse ricade all'interno di una zona cieca
        for idx, occ in enumerate(self.occs_bayes):
            poly_pts = np.array(occ.get('polygon_points_m', []))
            if len(poly_pts) > 2:
                # Converte i vertici del poligono nel sistema di riferimento dell'Ego Vehicle
                poly_ego = self.lidar_to_ego(poly_pts)
                # Istanzia un oggetto Poligono Shapely per la verifica di contenimento geometrico
                shp_poly = ShapelyPolygon(poly_ego)
                # Se il punto del mouse si trova all'interno dell'occlusione, seleziona l'occlusione corrispondente
                if shp_poly.contains(point):
                    if self.current_occ_idx != idx:
                        self.current_occ_idx = idx
                        # Aggiorna il disegno per evidenziare l'ombra selezionata dal mouse
                        self.plot_current()
                    break

# Blocco principale di esecuzione se lo script viene avviato direttamente da riga di comando
if __name__ == "__main__":
    # Istanzia ed avvia il visualizzatore comparativo affiancato
    ComparisonRuntimeVisualizer()
