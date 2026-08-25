# Visualizzatore Interattivo dei Canali d'Ingresso della Rete Neurale (visualizzatori/view_neural_inputs.py)
# Mostra la scomposizione grafica dettagliata degli 11 Canali BEV (LiDAR, Ombre RayCaster, Semantica HD Map e 6 Canali GT)
# per ciascun cono d'ombra estratto ed alimentato alla rete neurale PerZoneModel.

# Import dei moduli di sistema per la manipolazione dei percorsi di ambiente
import os
import sys
# Import di numpy per le operazioni matriciali ed algebriche sui canali
import numpy as np
# Import di matplotlib per il rendering grafico delle scomposizioni a canali
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Aggiunge la cartella radice del progetto al sys.path per l'importazione dei pacchetti interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import del caricatore del dataset OcclusionDatasetNeural
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetNeural

# Nomi descrittivi degli 11 canali BEV d'ingresso della rete neurale
CHANNEL_NAMES = [
    "1. Nuvola Punti LiDAR 2D",
    "2. Maschera Ombre RayCaster",
    "3. HD Map Carreggiata (Strada)",
    "4. HD Map Marciapiede",
    "5. HD Map Strisce Pedonali",
    "6. Ground Truth: Auto",
    "7. Ground Truth: Camion / Bus",
    "8. Ground Truth: Pedoni",
    "9. Ground Truth: Moto",
    "10. Ground Truth: Biciclette",
    "11. Ground Truth: Barriere / Coni"
]

# Mappa dei colori per la resa visiva dei canali
CHANNEL_CMAPS = [
    'cividis', 'Reds', 'Blues', 'copper', 'YlGn',
    'Blues', 'PuRd', 'Greens', 'Oranges', 'Purples', 'Greys'
]

# Classe principale per la gestione del visualizzatore dei canali d'ingresso neurali
class NeuralInputVisualizer:
    def __init__(self, dataroot="./nuscenes"):
        # Stampa l'intestazione di avvio del visualizzatore dei canali neurali
        print("\n" + "=" * 75)
        print("   INIZIALIZZAZIONE VISUALIZZATORE CANALI D'INGRESSO NEURALI (11 CANALI BEV)")
        print("=" * 75)
        
        # Inizializza il dataset neurale BEV sui 11 canali
        self.dataset = OcclusionDatasetNeural(dataset_name="nuscenes", dataroot=dataroot)
        self.total_samples = len(self.dataset)
        self.current_idx = 0
        
        # Configurazione della figura Matplotlib su griglia 3x4 (12 sottografici per 11 canali + 1 riepilogo)
        self.fig, self.axes = plt.subplots(3, 4, figsize=(16, 10), facecolor='#0B0C10')
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04, wspace=0.20, hspace=0.25)
        
        # Collegamento dell'evento da tastiera per la navigazione interattiva
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Stampa i controlli da tastiera per l'utente nel terminale
        print("\n" + "=" * 50)
        print("    CONTROLLI INTERATTIVI REGISTRATI (INPUT NEURALI)")
        print("=" * 50)
        print(" -> FRECCIA DX / SU  : Fotogramma Successivo")
        print(" -> FRECCIA SX / GIÙ : Fotogramma Precedente")
        print(" -> TASTO 'Q' / ESC  : Uscita dal Visualizzatore")
        print("=" * 50 + "\n")
        
        # Renderizza il primo fotogramma e mostra la finestra
        self.plot_current()
        plt.show()

    def on_key(self, event):
        # Freccia destra o su: passa al campione successivo
        if event.key in ['right', 'up']:
            self.current_idx = (self.current_idx + 1) % self.total_samples
            self.plot_current()
        # Freccia sinistra o giù: torna al campione precedente
        elif event.key in ['left', 'down']:
            self.current_idx = (self.current_idx - 1) % self.total_samples
            self.plot_current()
        # Tasto Q o ESC: chiude la finestra
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def plot_current(self):
        # Carica il tensore d'ingresso a 11 canali per il fotogramma corrente
        input_tensor, _ = self.dataset[self.current_idx]
        frame_data = self.dataset.adapter.get_sample_data(self.current_idx)
        sample_token = frame_data.get('sample_token', '')
        
        # Converti il tensore PyTorch in matrice NumPy (11, 200, 200)
        channels_np = input_tensor.numpy()
        
        # Appiattisce la matrice di sottografici 3x4 in una lista di 12 assi
        axes_flat = self.axes.flatten()
        
        # Renderizza ciascuno degli 11 canali nei primi 11 sottografici
        for c in range(11):
            ax = axes_flat[c]
            ax.clear()
            ax.set_facecolor('#0F172A')
            
            # Mostra la matrice 2D del canale con la mappa di colore corrispondente
            c_data = channels_np[c]
            ax.imshow(c_data, cmap=CHANNEL_CMAPS[c], origin='lower', extent=[-40, 40, -40, 40])
            ax.set_title(CHANNEL_NAMES[c], color='white', fontsize=8.5, fontweight='bold')
            ax.axis('off')

        # Il 12° sottografico (in basso a destra) mostra la mappa di riepilogo RGB combinata
        ax_summary = axes_flat[11]
        ax_summary.clear()
        ax_summary.set_facecolor('#0F172A')
        
        # Sovrapposizione combinata in RGB: Strada (Blu), Ombre (Rosso), LiDAR (Verde)
        rgb_img = np.zeros((200, 200, 3), dtype=np.float32)
        rgb_img[:, :, 0] = channels_np[1] * 0.7  # Canale Rosso: Ombre RayCaster
        rgb_img[:, :, 1] = channels_np[0] * 0.9  # Canale Verde: Punti LiDAR 2D
        rgb_img[:, :, 2] = channels_np[2] * 0.4  # Canale Blu: Carreggiata HD
        
        ax_summary.imshow(rgb_img, origin='lower', extent=[-40, 40, -40, 40])
        ax_summary.set_title("12. Composite Overlay BEV (RGB)", color='#66FCF1', fontsize=8.5, fontweight='bold')
        ax_summary.axis('off')
        
        # Titolo superiore della figura con il numero del fotogramma
        title_text = f"SCOMPOSIZIONE CANALI D'INGRESSO RETE NEURALE - FRAME [{self.current_idx + 1}/{self.total_samples}] | Sample: {sample_token[:16]}..."
        self.fig.suptitle(title_text, color='#66FCF1', fontsize=11, fontweight='bold', y=0.98)
        
        # Aggiorna la tela grafica
        self.fig.canvas.draw()

# Blocco principale di esecuzione se avviato da riga di comando
if __name__ == "__main__":
    NeuralInputVisualizer(dataroot="./nuscenes")
