# Script Principale di Estrazione Batch delle Zone Occluse LiDAR (estrazione_zone_occluse.py)
# Utilizza il RayCaster 3D per calcolare le ombre geometriche 2D BEV su tutti i fotogrammi nuScenes
# e salva le coordinate dei poligoni estratti nella cartella extracted_occlusions/ in formato JSON.

# Import dei moduli di sistema per la manipolazione dei percorsi
import os
import sys
# Import di json per la serializzazione e salvataggio dei poligoni d'ombra su disco
import json
# Import per la gestione degli argomenti da riga di comando
import argparse
# Import del garbage collector, necessario a preservare la memoria condivisa dopo il fork
import gc
# Import per la distribuzione dei fotogrammi su piu processi
from multiprocessing import Pool
# Import per silenziare la diagnostica del RayCaster nei processi figli
import contextlib
import io

# Inserisce la directory radice del progetto al primo posto in sys.path per consentire l'importazione dei pacchetti interni
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import della factory per la creazione dell'adattatore del dataset nuScenes
from dataset_adapter.factory_dataset import create_adapter
# Import del modulo principale RayCaster per la simulazione del fascio di raggi LiDAR
from raycaster.ray_caster import RayCaster

# Cartella di destinazione dei poligoni estratti
CARTELLA_OUTPUT = "extracted_occlusions"

# Se True i fotogrammi gia' presenti su disco vengono comunque rielaborati
RIGENERA = False

# Adattatore del dataset: viene creato dal processo padre e i processi figli lo
# ereditano al momento del fork, evitando di ricaricare i metadati una volta ciascuno
adapter = None


# Elabora un singolo fotogramma e ne salva i poligoni d'ombra su disco.
# I fotogrammi sono indipendenti fra loro, quindi l'ordine di esecuzione non influenza il risultato.
# Restituisce un numero di poligoni pari a None quando il fotogramma era gia' stato elaborato
def elabora_fotogramma(idx: int):
    # Il token si legge direttamente dai metadati, senza toccare LiDAR e mappe: permette
    # di riconoscere un fotogramma gia' completato prima di spendere tempo a rielaborarlo
    token = adapter.all_samples[idx]['token']
    out_path = os.path.join(CARTELLA_OUTPUT, f"{token}.json")

    if not RIGENERA and os.path.exists(out_path):
        return token, None

    # La diagnostica del RayCaster viene soppressa perche' con piu processi
    # attivi le righe si sovrapporrebbero rendendo l'avanzamento illeggibile
    with contextlib.redirect_stdout(io.StringIO()):
        # Recupero del dizionario dei dati del fotogramma corrente dall'adattatore
        frame_data = adapter.get_sample_data(idx)
        # Istanzia l'oggetto RayCaster passandogli i dati del fotogramma corrente
        caster = RayCaster(frame_data)
        # Generazione della maschera di occlusione binaria in coordinate polar-grid
        caster.get_occlusion_mask()
        # Estrazione dei poligoni d'ombra semplificati in metri rispetto al veicolo
        occlusions = caster.extract_polygons()

    # Salvataggio dei poligoni e del token LiDAR nel file JSON di output.
    # La scrittura avviene su un file temporaneo poi rinominato: se l'esecuzione viene
    # interrotta a meta' scrittura non resta un JSON troncato che una ripresa scambierebbe per valido
    tmp_path = f"{out_path}.parziale"
    with open(tmp_path, "w") as f:
        json.dump({"lidar_token": frame_data['lidar_token'], "occlusions": occlusions}, f, indent=4)
    os.replace(tmp_path, out_path)

    return token, len(occlusions)


def main():
    parser = argparse.ArgumentParser(description="Estrazione batch delle zone d'ombra LiDAR su nuScenes")
    parser.add_argument("--workers", type=int, default=8,
                        help="Numero di processi paralleli (default: 8). Usare 1 per l'esecuzione sequenziale")
    parser.add_argument("--rigenera", action="store_true",
                        help="Rielabora anche i fotogrammi gia' presenti in extracted_occlusions/ "
                             "(per impostazione predefinita vengono saltati, cosi da riprendere un'esecuzione interrotta)")
    args = parser.parse_args()

    global RIGENERA
    RIGENERA = args.rigenera

    # Scelta del dataset target da elaborare
    dataset_name = "nuscenes"
    # Definizione del percorso relativo della cartella dei dati grezzi nuScenes
    dataroot = "./nuscenes"

    global adapter
    print(f"Caricamento adattatore: {dataset_name}...")
    # Istanzia l'adattatore del dataset nuScenes tramite il Factory Pattern
    adapter = create_adapter(dataset_name, dataroot)
    # Recupera il numero totale di fotogrammi (sample) presenti nel dataset
    num_samples = adapter.get_num_samples()

    # Creazione della directory di output per il salvataggio dei JSON se non esiste già
    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    print("Precaricamento delle mappe vettoriali...")
    # Le mappe vengono costruite adesso, mentre il processo e' ancora unico, cosi i figli le ereditano
    adapter.precarica_mappe()

    # Sposta gli oggetti gia' allocati in una generazione permanente: senza questo il garbage
    # collector li attraverserebbe nei figli, duplicando gli 8 GB di metadati per ogni processo
    gc.freeze()

    print(f"Elaborazione di {num_samples} fotogrammi su {args.workers} processi...\n")
    elaborati = saltati = 0
    with Pool(processes=args.workers) as pool:
        for completati, (token, n_poligoni) in enumerate(
                pool.imap_unordered(elabora_fotogramma, range(num_samples), chunksize=8), start=1):
            if n_poligoni is None:
                saltati += 1
                esito = "gia' presente, saltato"
            else:
                elaborati += 1
                esito = f"{n_poligoni} poligoni"
            print(f"[{completati}/{num_samples}] {token}: {esito}")

    print(f"\nCompletato: {elaborati} fotogrammi elaborati, {saltati} gia' presenti e saltati.")


# Blocco principale di esecuzione se il file viene lanciato direttamente da riga di comando
if __name__ == "__main__":
    main()
