# Modulo Factory Pattern per la Creazione dinamica dell'Adapter Dataset (dataset_adapter/factory_dataset.py)
# Consentire un'interfaccia unica per istanziare l'adattatore del dataset (es. nuScenes, Waymo, CARLA).

def create_adapter(dataset_name: str, dataroot: str):
    """
    Funzione Factory che istanzia l'adattatore corretto in base al nome del dataset passato.
    
    Args:
        dataset_name (str): Nome del dataset ("nuscenes", ecc.)
        dataroot (str): Percorso della cartella contenente i dati grezzi del dataset
    Returns:
        BaseDatasetAdapter: Istanza dell'adattatore concreto selezionato.
    """
    # Verifica se il dataset richiesto è nuScenes
    if dataset_name == "nuscenes":
        # Import ritardata (Lazy Import) dell'adapter nuScenes per ottimizzare i tempi di avvio
        from dataset_adapter.nuscenes_dataset_adapter import NuscenesDatasetAdapter
        # Restituisce l'istanza dell'adapter nuScenes inizializzato con il dataroot
        return NuscenesDatasetAdapter(dataroot)
    else:
        # Solleva un'eccezione se il nome del dataset specificato non è supportato
        raise ValueError(f"Dataset '{dataset_name}' non supportato dalla Factory. Utilizzare 'nuscenes'.")