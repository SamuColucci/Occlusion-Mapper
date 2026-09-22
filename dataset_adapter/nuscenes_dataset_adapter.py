# Modulo Adattatore Concreto per il Dataset nuScenes (dataset_adapter/nuscenes_dataset_adapter.py)
# Implementa l'interfaccia AdapterDataset per convertire le strutture ed il formato nativo nuScenes
# in un dizionario normalizzato ed agnostico utilizzato per il raycasting e per l'addestramento neurale.

# Libreria per la gestione dei percorsi di file nel sistema operativo
from nuscenes import nuscenes
import os

# Libreria per la gestione dei dati numerici ed algebrici
import numpy as np

# Libreria per il caricamento principale del dataset NuScenes
from nuscenes.nuscenes import NuScenes

# Libreria per la gestione delle mappe HD vettoriali nuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Libreria per la gestione dei box degli oggetti 3D
from nuscenes.utils.data_classes import Box

# Libreria per la gestione delle matrici di rototraslazione tramite quaternioni
from pyquaternion import Quaternion

# Libreria per la gestione delle primitive geometriche
from shapely.geometry import Polygon, MultiPolygon

# Import dell'interfaccia astratta del pattern Adapter
from dataset_adapter.adapter_dataset import AdapterDataset, LocalBox

# Libreria per la gestione dei dizionari con valori predefiniti
from collections import defaultdict

# Classe concreta che estende l'interfaccia AdapterDataset per il dataset nuScenes
class NuscenesDatasetAdapter(AdapterDataset):
    def __init__(self, dataroot: str):
        # Invocazione del costruttore della classe padre astratta
        super().__init__(dataroot)

        # Rilevamento automatico della versione presente nel dataroot (trainval o mini)
        if os.path.exists(os.path.join(dataroot, "v1.0-trainval")):
            version = "v1.0-trainval"
        elif os.path.exists(os.path.join(dataroot, "v1.0-mini")):
            version = "v1.0-mini"
        else:
            version = "v1.0-trainval"

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Controllo di sicurezza sull'inizializzazione corretta del dataset
        if self.nusc is None:
            raise ValueError("Errore nell'inizializzazione del dataset NuScenes")
        
        # Recupero dell'elenco completo dei campioni (sample) presenti nel dataset
        self.all_samples = self.nusc.sample

        # Le mappe vettoriali distinte sono solo 4 per l'intero dataset, mentre i fotogrammi
        # sono decine di migliaia: vengono conservate qui per non riparsarne il JSON ad ogni frame
        self._map_cache = {}

    # Costruisce in anticipo tutte le mappe vettoriali citate dai log del dataset.
    # Va invocato prima di distribuire il lavoro su piu processi: cosi le mappe sono
    # gia in memoria al momento del fork e i figli le condividono invece di ricostruirle
    def precarica_mappe(self) -> None:
        for log in self.nusc.log:
            map_name = log['location']
            if map_name not in self._map_cache:
                self._map_cache[map_name] = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

    # Metodo per ottenere il numero totale di campioni del dataset
    def get_num_samples(self) -> int:
        return len(self.all_samples)

    # Restituisce la lunghezza della lista dei campioni
    def __len__(self) -> int:
        return len(self.all_samples)
        
    # Estrae gli indici dei campioni raggruppati per scena ed ordinati in modo cronologico
    def get_scene_indices(self) -> dict:
        scenes = defaultdict(list)
        # Ciclo per ogni indice e campione presente nel dataset
        for idx, sample in enumerate(self.all_samples):
            # Ottenimento del token univoco della scena
            scene_token = sample['scene_token']
            # Aggiunta della coppia (indice, timestamp) alla lista della scena
            scenes[scene_token].append((idx, sample['timestamp']))

        # Ordinamento dei fotogrammi di ciascuna scena per timestamp crescente
        for scene_token, idx_ts_list in scenes.items():
            # Ordinamento basato sul valore del timestamp (secondo elemento)
            idx_ts_list.sort(key=lambda x: x[1])
            risultato = []
            # Estrae solo l'indice numerico del campione scartando il timestamp
            for x in idx_ts_list:
                risultato.append(x[0])
            # Aggiorna il dizionario con gli indici temporali ordinati
            scenes[scene_token] = risultato
        
        return scenes

    # Estrae i dati normalizzati del fotogramma corrispondente all'indice specificato
    def get_sample_data(self, idx: int) -> dict:
        # Recuperiamo il campione dalla lista
        sample = self.all_samples[idx]
        # Estraiamo il token del Lidar dal record del sample
        lidar_token = sample['data']['LIDAR_TOP']
        # Estraiamo il record del Lidar
        sd_record = self.nusc.get('sample_data', lidar_token)
        # Estraiamo il percorso del Lidar
        lidar_path = os.path.join(self.dataroot, sd_record['filename'])
        # Caricamento del Lidar, leggendo il flusso di byte dal file fisico in un array numpy
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        # Estraiamo i punti x,y,z dal Lidar, ignorando il quarto punto che rappresenta l'intensità
        pts_local = pc[:, :3]

        # Ottenimento dei box 3D orientati dal record del Lidar
        risultato = self.nusc.get_sample_data(lidar_token)
        # Estraiamo le scatole orientate dal record del Lidar, prendendo il secondo campo ovvero boxes_ego
        boxes_ego = risultato[1]

        local_box = []

        # Ciclo su ogni box presente nel record del Lidar
        for box in boxes_ego:
            # Ottenimento del record dell'annotazione, instance_token è l'id persistente dell'oggetto
            annotation = self.nusc.get('sample_annotation', box.token)
            # Estrazione del token dell'istanza
            instance_token = annotation['instance_token']
            # Estrazione della visibilità
            visibility = annotation['visibility_token']

            # Istanzia l'oggetto LocalBox con i dati dell'ostacolo
            lbox = LocalBox(
                center=box.center,
                wlh=box.wlh,
                corners_3d=box.corners(),
                name=box.name,
                token=instance_token,
                visibility=visibility
            )
            local_box.append(lbox)

        # Estrazione della posa dell'ego-vehicle dal record del sample
        ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        # Conversione della rotazione dell'ego-vehicle in quaternion
        q = Quaternion(ego_pose['rotation'])

        # Ottenimento della rotazione inversa dell'ego-vehicle
        R_inv = q.inverse.rotation_matrix
        
        # Estrazione della traslazione dell'ego-vehicle
        tx, ty, tz = ego_pose['translation']

        # Estrazione del token della scena globale
        scene = self.nusc.get('scene', sample['scene_token'])
        # Estrazione del token del log
        log = self.nusc.get('log', scene['log_token'])
        
        # Estrazione del nome della mappa
        map_name = log['location']

        # Salviamo in nusc_map l'oggetto MappaVettoriale associata alla scena corrente
        if map_name not in self._map_cache:
            self._map_cache[map_name] = NuScenesMap(dataroot=self.dataroot, map_name=map_name)
        nusc_map = self._map_cache[map_name]

        # Angolo di direzione (yaw) dell'auto robot in gradi
        yaw_deg = np.degrees(q.yaw_pitch_roll[0])
        
        # Patch rettangolare di 80m x 80m centrato sull'auto robot
        patch_box = (tx, ty, 80, 80)
        layer_names = ['drivable_area', 'walkway', 'carpark_area', 'ped_crossing']
        
        # Generazione diretta delle maschere binarie 2D (200x200) traslate e ruotate perfettamente
        map_masks = nusc_map.get_map_mask(patch_box, yaw_deg, layer_names, canvas_size=(200, 200))
        
        # Costruzione del dizionario delle maschere semantiche del terreno
        semantic_map = {
            'drivable_area': map_masks[0],
            'walkway': map_masks[1],
            'carpark_area': map_masks[2],
            'ped_crossing': map_masks[3]
        }

        # Preparazione e restituzione del dizionario contenente i dati del sample
        return {
            "lidar_token": lidar_token,
            "sample_token": sample['token'],
            "scene_token": sample['scene_token'],
            "points": pts_local,
            "boxes": local_box,
            "semantic_map": semantic_map
        }