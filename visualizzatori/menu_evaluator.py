# Menu Interattivo di Valutazione e Confronto Modelli (visualizzatori/menu_evaluator.py)
# Permette di consultare gli score dei 7 approcci del progetto e di effettuare confronti diretti affiancati

# Import delle librerie di sistema per la gestione dei percorsi e delle variabili d'ambiente
import os
import sys
# Import di glob per la ricerca di file tramite pattern di stringhe
import glob
# Import di json per la lettura e scrittura di file in formato JSON
import json
# Import di numpy per il calcolo numerico e la manipolazione di array
import numpy as np
# Import di torch per la gestione dei modelli di Deep Learning e dei sensori su GPU/CPU
import torch

# Aggiunge la cartella radice del progetto al percorso di sistema per consentire l'importazione dei moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dell'architettura neurale PerZoneModel dal pacchetto dedicato
from architettura_neurale.per_zone_model import PerZoneModel
# Import del caricatore del dataset OcclusionDatasetPerZone dal pacchetto adapter
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetPerZone

# Nomi descrittivi visualizzati nel menu per identificare ciascuno dei 7 modelli valutati
MODEL_NAMES = [
    "1. Agente Bayesiano Dinamico (Prior HD)",
    "2. Agente Neurale Baseline (BCE Standard)",
    "3. Agente Neurale BCE + Penalizzazione Semantica",
    "4. Agente Neurale Focal Loss Standard",
    "5. Agente Neurale Focal Loss + Penalizzazione (Picco Macro F1: 32.3%)",
    "6. Agente Neurale Contesto Esterno (Ring Semantics)",
    "7. Agente Neurale Asymmetric Loss - CVPR 2021 (Picco Pedoni Recall: 98.8%)"
]

# Percorsi dei file di checkpoint (.pth) per ciascuno dei modelli neurali (None indica l'Agente Bayesiano)
MODEL_CKPTS = [
    None,  # Agente Bayesiano (carica i file JSON anziché i pesi PyTorch)
    os.path.join("pesi_modelli", "per_zone_checkpoint.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_semantica.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_focal.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_focal_semantica.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_surrounding.pth"),
    os.path.join("pesi_modelli", "per_zone_checkpoint_asl.pth")
]

# Elenco delle 6 categorie di ostacoli predette dal sistema
CATEGORIES = ["Auto", "Camion/Bus", "Pedone", "Moto", "Bicicletta", "Barriera"]

# Classe principale per la gestione del menu interattivo e del calcolo delle metriche di valutazione
class ProjectEvaluatorMenu:
    def __init__(self):
        # Selezione automatica del dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Inizializzazione della variabile per il dataset (caricato in modo pigro/lazy)
        self.dataset = None
        # Dizionario per memorizzare in cache i risultati delle valutazioni già eseguite
        self.cached_results = {}
        # Contatore per il numero di zone non carrabili (marciapiedi e prato) presenti nel dataset
        self.non_driveable_count = 0

    def load_dataset_if_needed(self):
        # Carica il dataset nuScenes solo se non è già stato inizializzato in memoria
        if self.dataset is None:
            # Messaggio informativo per l'utente sull'avvio dell'estrazione delle feature
            print("\nInizializzazione Dataset nuScenes ed estrazione feature (18.682 coni d'ombra)...")
            # Istanzia il dataset caricando le occlusioni estratte da nuScenes
            self.dataset = OcclusionDatasetPerZone(dataset_name="nuscenes", dataroot="./nuscenes")
            
            # Calcola il numero totale di campioni che ricadono in zone non carrabili (marciapiede o terreno)
            self.non_driveable_count = sum(
                1 for s in self.dataset.samples 
                if (s['scalars'].numpy()[5] + s['scalars'].numpy()[8]) > 0.5
            )

    def evaluate_model_index(self, idx, threshold=0.30):
        # Se il modello specificato è già stato valutato, restituisce immediatamente il risultato salvato in cache
        if idx in self.cached_results:
            return self.cached_results[idx]

        # Assicura che il dataset sia stato caricato correttamente in memoria
        self.load_dataset_if_needed()
        # Stampa a schermo il nome del modello in fase di valutazione
        print(f"\nCalcolo metriche in corso per: {MODEL_NAMES[idx]}...")

        # Inizializzazione degli array per i Veri Positivi (TP), Falsi Positivi (FP) e Falsi Negativi (FN) per le 6 classi
        tp_arr, fp_arr, fn_arr = np.zeros(6), np.zeros(6), np.zeros(6)
        # Contatore per le violazioni della coerenza semantica (veicoli predetti su marciapiede/terreno)
        viol_count = 0

        # Ramo di esecuzione per l'Agente Bayesiano (indice 0)
        if idx == 0:
            # Recupera tutti i file JSON contenenti le probabilità stimate dall'Agente Bayesiano
            b_files = glob.glob(os.path.join("extracted_occlusions_probabilities", "*.json"))
            # Dizionario di cache per memorizzare le predizioni bayesiane indicizzate per token di campione
            b_cache = {}
            # Carica ciascun file JSON e lo inserisce nel dizionario di cache
            for fpath in b_files:
                # Estrae il token identificativo dal nome del file
                tok = os.path.basename(fpath).split("_")[-1].replace(".json", "")
                # Apre il file JSON e legge la lista di occlusioni
                with open(fpath, "r") as f:
                    b_cache[tok] = json.load(f).get("occlusions", [])

            # Scorre tutti i campioni presenti nel dataset
            for sample in self.dataset.samples:
                # Vettore binario di Ground Truth reale per le 6 classi
                target_vec = sample['target'].numpy()
                # Token univoco del campione nuScenes
                sample_token = sample['sample_token']
                # Punti del poligono dell'occlusione corrente
                raw_pts = sample['polygon_points']
                # Vettore dei descrittori scalari della mappa HD
                sc_np = sample['scalars'].numpy()
                # Flag che indica se l'occlusione si trova su marciapiede (indice 5) o terreno (indice 8)
                is_non_driveable = (sc_np[5] + sc_np[8]) > 0.5

                # Recupera la lista di occlusioni salvate in cache per il token corrente
                b_occs = b_cache.get(sample_token, [])
                b_dict = {}
                # Cerca l'occlusione bayesiana corrispondente tramite vicinanza dei baricentri
                for b_occ in b_occs:
                    b_pts = b_occ.get("polygon_points_m", [])
                    # Verifica che entrambi i poligoni abbiano almeno 3 vertici
                    if len(b_pts) >= 3 and len(raw_pts) >= 3:
                        # Se la distanza tra i baricentri è inferiore a 0.5 metri, associa le probabilità bayesiane
                        if np.linalg.norm(np.mean(b_pts, axis=0) - np.mean(raw_pts, axis=0)) < 0.5:
                            b_dict = b_occ.get("estimated_probabilities", {})
                            break

                # Costruisce il vettore delle probabilità bayesiane per le 6 categorie
                b_probs = np.array([
                    b_dict.get("Auto", 0.0),
                    max(b_dict.get("Camion", 0.0), b_dict.get("Bus", 0.0)),
                    b_dict.get("Pedone", 0.0),
                    b_dict.get("Moto", 0.0),
                    b_dict.get("Bicicletta", 0.0),
                    b_dict.get("Barriera", 0.0)
                ])
                # Genera la predizione binaria confrontando la probabilità con la soglia decisionale
                pred = (b_probs >= threshold).astype(np.float32)

                # Aggiorna i contatori TP, FP e FN per ciascuna delle 6 classi
                for c in range(6):
                    if target_vec[c] == 1.0 and pred[c] == 1.0: tp_arr[c] += 1
                    elif target_vec[c] == 0.0 and pred[c] == 1.0: fp_arr[c] += 1
                    elif target_vec[c] == 1.0 and pred[c] == 0.0: fn_arr[c] += 1

                # Incrementa le violazioni semantiche se viene predetto un veicolo a motore in zona non carrabile vuota
                if is_non_driveable and (pred[0] == 1 or pred[1] == 1 or pred[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0:
                    viol_count += 1
        else:
            # Ramo di esecuzione per gli Agenti Neurali (indici 1-6)
            ckpt_path = MODEL_CKPTS[idx]
            # Verifica se il file dei pesi del modello esiste sul disco
            if not os.path.exists(ckpt_path):
                print(f"[WARNING] Checkpoint '{ckpt_path}' non trovato sul disco! Esegui prima lo script di addestramento.")
                return None

            # Istanzia la rete neurale PerZoneModel e la sposta sul dispositivo selezionato (GPU o CPU)
            model = PerZoneModel(num_classes=6).to(self.device)
            # Carica lo stato dei pesi addestrati dal file di checkpoint
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
            # Imposta la rete in modalità di valutazione (disattiva dropout e batchnorm)
            model.eval()

            # Disabilita il calcolo dei gradienti per velocizzare l'inferenza e risparmiare memoria
            with torch.no_grad():
                # Scorre tutti i campioni del dataset
                for sample in self.dataset.samples:
                    # Vettore binario di Ground Truth reale
                    target_vec = sample['target'].numpy()
                    # Prepara il tensore della patch occupazionale aggiungendo la dimensione del batch
                    patch_tensor = sample['patch'].unsqueeze(0).to(self.device, dtype=torch.float32)
                    # Prepara il tensore dei descrittori scalari della mappa HD
                    scalar_tensor = sample['scalars'].unsqueeze(0).to(self.device, dtype=torch.float32)
                    sc_np = sample['scalars'].numpy()
                    # Flag per identificare le zone non carrabili (marciapiede o terreno)
                    is_non_driveable = (sc_np[5] + sc_np[8]) > 0.5

                    # Esegue il passaggio in avanti (forward pass) nella rete neurale per ottenere i logit
                    logits = model(patch_tensor, scalar_tensor)
                    # Applica la funzione Sigmoide per convertire i logit in probabilità
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    # Genera la predizione binaria applicando la soglia decisionale
                    pred = (probs >= threshold).astype(np.float32)

                    # Aggiorna i contatori TP, FP e FN per ciascuna classe
                    for c in range(6):
                        if target_vec[c] == 1.0 and pred[c] == 1.0: tp_arr[c] += 1
                        elif target_vec[c] == 0.0 and pred[c] == 1.0: fp_arr[c] += 1
                        elif target_vec[c] == 1.0 and pred[c] == 0.0: fn_arr[c] += 1

                    # Incrementa le violazioni semantiche in caso di veicolo a motore predetto in area non carrabile vuota
                    if is_non_driveable and (pred[0] == 1 or pred[1] == 1 or pred[3] == 1) and np.sum(target_vec[[0,1,3]]) == 0:
                        viol_count += 1

        # Liste per memorizzare le metriche calcolate per ciascuna classe
        prec_l, rec_l, f1_l, iou_l = [], [], [], []
        per_class = []

        # Calcola Precision, Recall, F1-Score e IoU per ciascuna delle 6 classi
        for c in range(6):
            tp, fp, fn = tp_arr[c], fp_arr[c], fn_arr[c]
            # Calcolo della Precision (TP / (TP + FP))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Calcolo della Recall (TP / (TP + FN))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # Calcolo dell'F1-Score (media armonica tra Precision e Recall)
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            # Calcolo dell'IoU (Intersection over Union: TP / (TP + FP + FN))
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            # Salva i valori calcolati nelle rispettive liste
            prec_l.append(prec); rec_l.append(rec); f1_l.append(f1); iou_l.append(iou)
            per_class.append({'prec': prec, 'rec': rec, 'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn})

        # Calcola la percentuale di coerenza semantica (100% meno la percentuale di violazioni)
        coherence_rate = (1.0 - (viol_count / max(1, self.non_driveable_count))) * 100

        # Costruisce il dizionario contenente tutti i risultati complessivi del modello
        result = {
            'name': MODEL_NAMES[idx],
            'per_class': per_class,
            'macro_prec': np.mean(prec_l),
            'macro_rec': np.mean(rec_l),
            'macro_f1': np.mean(f1_l),
            'coherence': coherence_rate,
            'viol_count': viol_count
        }

        # Salva il risultato nel dizionario di cache per evitare di ricalcolarlo in futuro
        self.cached_results[idx] = result
        return result

    def print_single_model_table(self, idx):
        # Esegue la valutazione del modello selezionato
        res = self.evaluate_model_index(idx)
        # Se il modello non è stato trovato o ha restituito un errore, interrompe la stampa
        if res is None:
            return

        # Stampa dell'intestazione decorativa della tabella dei risultati
        print("\n" + "=" * 90)
        print(f"  {res['name'].upper()} (Soglia Decisionale = 30%)")
        print("=" * 90)
        print(f" {'Classe':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'IoU':<12} {'TP/FP/FN'}")
        print("-" * 90)
        # Scorre e stampa i dettagli per ogni singola categoria di ostacolo
        for c in range(6):
            pc = res['per_class'][c]
            print(f" {CATEGORIES[c]:<14} {pc['prec']*100:6.1f}%       {pc['rec']*100:6.1f}%       {pc['f1']*100:6.1f}%       {pc['iou']*100:6.1f}%       {int(pc['tp'])}/{int(pc['fp'])}/{int(pc['fn'])}")
        print("-" * 90)
        # Stampa le medie macro complessive
        print(f" {'MEDIA MACRO':<14} {res['macro_prec']*100:6.1f}%       {res['macro_rec']*100:6.1f}%       {res['macro_f1']*100:6.1f}%")
        # Stampa il tasso di coerenza semantica e il numero assoluto di violazioni
        print(f"  TASSO DI COERENZA SEMANTICO TERRENO/MARCIAPIEDE: {res['coherence']:.1f}% ({res['viol_count']} violazioni su {self.non_driveable_count} zone)\n")
        print("=" * 90)
        # Attende la pressione del tasto INVIO per ritornare al menu principale
        input("\nPremi INVIO per tornare al menu...")

    def compare_two_models(self):
        # Stampa della schermata di selezione dei due modelli da mettere a confronto
        print("\n" + "=" * 60)
        print("     CONFRONTO DIRETTO ED AFFIANCATO TRA 2 MODELLI")
        print("=" * 60)
        # Mostra la lista enumerata dei modelli disponibili
        for i, name in enumerate(MODEL_NAMES):
            print(f"  {i+1}. {name}")
        print("-" * 60)

        # Gestione dell'input utente per la selezione dei due modelli
        try:
            m1_idx = int(input("Seleziona il PRIMO Modello (1-7): ")) - 1
            m2_idx = int(input("Seleziona il SECONDO Modello (1-7): ")) - 1
            # Verifica che entrambi gli indici inseriti rientrino nel range valido (0-6)
            if not (0 <= m1_idx < 7 and 0 <= m2_idx < 7):
                print("[ERROR] Indice modello non valido.")
                input("\nPremi INVIO per tornare al menu...")
                return
        except ValueError:
            print("[ERROR] Inserire un numero compreso tra 1 e 7.")
            input("\nPremi INVIO per tornare al menu...")
            return

        # Esegue o recupera dalla cache la valutazione dei due modelli selezionati
        res1 = self.evaluate_model_index(m1_idx)
        res2 = self.evaluate_model_index(m2_idx)

        # Se uno dei due modelli restituisce un errore, interrompe la procedura
        if res1 is None or res2 is None:
            input("\nPremi INVIO per tornare al menu...")
            return

        # Stampa dell'intestazione del confronto affiancato
        print("\n" + "=" * 95)
        print(f"  CONFRONTO: [M1] {res1['name']}  VS  [M2] {res2['name']}")
        print("=" * 95)
        print(f" {'Classe':<12} | {'F1 (M1)':<9} {'F1 (M2)':<9} {'Delta F1':<10} | {'Prec (M1)':<9} {'Prec (M2)':<9} | {'Rec (M1)':<9} {'Rec (M2)':<9}")
        print("-" * 95)

        # Scorre ciascuna classe e mostra i valori di F1-Score, Precision e Recall a confronto con il relativo delta
        for c in range(6):
            pc1 = res1['per_class'][c]
            pc2 = res2['per_class'][c]
            # Calcola la differenza di F1-Score tra il secondo ed il primo modello
            df1 = (pc2['f1'] - pc1['f1']) * 100
            d_str = f"{df1:+5.1f}%" if df1 != 0 else "  0.0%"
            print(f" {CATEGORIES[c]:<12} | {pc1['f1']*100:6.1f}%   {pc2['f1']*100:6.1f}%   {d_str:<10} | {pc1['prec']*100:6.1f}%   {pc2['prec']*100:6.1f}% | {pc1['rec']*100:6.1f}%   {pc2['rec']*100:6.1f}%")

        print("-" * 95)
        # Calcola il delta per la media macro dell'F1-Score
        d_macro_f1 = (res2['macro_f1'] - res1['macro_f1']) * 100
        df1_macro_str = f"{d_macro_f1:+5.1f}%" if d_macro_f1 != 0 else "  0.0%"
        # Stampa il riepilogo delle medie macro affiancate
        print(f" {'MEDIA MACRO':<12} | {res1['macro_f1']*100:6.1f}%   {res2['macro_f1']*100:6.1f}%   {df1_macro_str:<10} | {res1['macro_prec']*100:6.1f}%   {res2['macro_prec']*100:6.1f}% | {res1['macro_rec']*100:6.1f}%   {res2['macro_rec']*100:6.1f}%")
        print("-" * 95)
        # Stampa il confronto per la coerenza semantica ed il relativo guadagno/perdita percentuale
        print(f" COERENZA  | {res1['coherence']:6.1f}%   {res2['coherence']:6.1f}%   {(res2['coherence']-res1['coherence']):+5.1f}%")
        print("=" * 95 + "\n")
        # Attende l'interazione dell'utente per ritornare al menu
        input("\nPremi INVIO per tornare al menu...")

    def run_menu(self):
        # Ciclo principale del menu interattivo in riga di comando
        while True:
            # Pulisce la schermata del terminale (compatibile sia con Windows 'cls' che con Linux/Mac 'clear')
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 75)
            print("        MENU INTERATTIVO VALUTAZIONE E CONFRONTO METRICHE")
            print("=" * 75)
            print(" Seleziona l'opzione desiderata:")
            print(" " + "-" * 71)
            # Elenco delle opzioni di menu disponibili per l'utente
            print("  1. Mostra Score: 1. Agente Bayesiano Dinamico (Prior HD)")
            print("  2. Mostra Score: 2. Agente Neurale Baseline (BCE Standard)")
            print("  3. Mostra Score: 3. Agente Neurale BCE + Penalizzazione Semantica")
            print("  4. Mostra Score: 4. Agente Neurale Focal Loss Standard")
            print("  5. Mostra Score: 5. Agente Neurale Focal + Penalizzazione (Picco Macro F1: 32.3%)")
            print("  6. Mostra Score: 6. Agente Neurale Contesto Esterno (Ring Semantics)")
            print("  7. Mostra Score: 7. Agente Neurale Asymmetric Loss (Picco Pedoni Recall: 98.8%)")
            print(" " + "-" * 71)
            print("  8. CONFRONTO DIRETTO AFFIANCATO TRA 2 MODELLI A SCELTA")
            print("  9. ESCI")
            print("=" * 75)

            # Richiede all'utente di inserire una scelta
            choice = input(" Inserisci il numero dell'opzione (1-9): ").strip()
            # Se la scelta è tra 1 e 7, mostra la tabella dettagliata del singolo modello selezionato
            if choice in [str(i) for i in range(1, 8)]:
                self.print_single_model_table(int(choice) - 1)
            # Se la scelta è 8, avvia la schermata di confronto tra due modelli
            elif choice == "8":
                self.compare_two_models()
            # Se la scelta è 9, interrompe il ciclo ed esce dallo script
            elif choice == "9":
                print("\nUscita dal menu di valutazione. Arrivederci!\n")
                break
            # Gestione degli errori per input non validi
            else:
                input("\n[ERROR] Opzione non valida. Premi INVIO per riprovare...")

# Blocco principale di esecuzione dello script se avviato direttamente da riga di comando
if __name__ == "__main__":
    # Istanzia la classe del menu di valutazione
    menu = ProjectEvaluatorMenu()
    # Avvia il loop interattivo del menu
    menu.run_menu()
