# Libreria per i percorsi dei file
from nuscenes import nuscenes
import os

# Libreria per la gestione dei dati numerici
import numpy as np

# Libreria per il caricamento dei dati del dataset
from nuscenes.nuscenes import NuScenes

# Libreria per la gestione delle mappe HD
from nuscenes.map_expansion.map_api import NuScenesMap

# Libreria per la gestione dei box degli oggetti 3D
from nuscenes.utils.data_classes import Box

# Libreria per la gestione delle matrici di rototraslazione
from pyquaternion import Quaternion

# Libreria per la gestione dei poligoni
from shapely.geometry import Polygon, MultiPolygon

# Import del pattern Adapter
from adapter_dataset import AdapterDataset
# Import della classe LocalBox
from adapter_dataset import LocalBox

# Libreria per la gestione dei dizionari defaultdict
from collections import defaultdict

class NuscenesDatasetAdapter(AdapterDataset):
    def __init__(self, dataroot: str):
        # Costruttore del pattern Adapter
        super().__init__(dataroot)

        # Inizializzazione del dataset NuScenes
        self.nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)

        # Controllo che l'inizializzazione sia avvenuta correttamente
        if self.nusc is None:
            raise ValueError("Errore nell'inizializzazione del dataset NuScenes")
        
        # Ottenimento di tutti i campioni del dataset
        self.all_samples = self.nusc.sample

    # Metodo per ottenere il numero di campioni del dataset
    def get_num_samples(self) -> int:
        return len(self.all_samples)
        
    def get_scene_indices(self) -> dict:
        scenes = defaultdict(list)
        # Ciclo per ogni indice del sample e per ogni sample presente nel dataset
        # Sample contiene i dati del frame, l'ID della scena, il timestamp, il token del lidar, ecc. analizzato in quel momento
        for idx, sample in enumerate(self.all_samples):
            # Ottenimento del token della scena
            scene_token = sample['scene_token']
            # Aggiunta dell'indice del sample e del timestamp al dizionario delle scene
            scenes[scene_token].append((idx,sample['timestamp']))

        # Ordinamento delle scene per timestamp
        for scene_token,idx_ts_list in scenes.items():
            # Ordinamento crescente del timestamp
            idx_ts_list.sort(key=lambda x:x[1])
            # Aggiornamento del dizionario con gli indici ordinati
            risultato = []
            # Ciclo per ogni indice del sample e per ogni timestamp presente nel dataset
            for x in idx_ts_list:
                # Aggiunta dell'indice del sample al dizionario, al fine di rimuovere il timestamp e salvare solo l'indice
                risultato.append(x[0])
            # Aggiornamento del dizionario con gli indici ordinati
            scenes[scene_token] = risultato
        
        return scenes

    def get_sample_data(self, idx: int) -> dict:
        # Recuperiamo il campione dalla lista
        sample = self.all_samples[idx]
        # Estraiamo il token del Lidar dal record del sample
        lidar_token = sample['data']['LIDAR_TOP']
        # Estraiamo il record del Lidar
        sd_record = self.nusc.get('sample_data',lidar_token)
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

        local_box= []

        # Ciclo su ogni box presente nel record del Lidar
        for box in boxes_ego:
            # Ottenimento del record dell'annotazione, instance_token è l'id persistente dell'oggetto in modo indipendente dai frame 
            annotation = self.nusc.get('sample_annotation', box.token)
            # Estrazione del token dell'istanza
            instance_token = annotation['instance_token']
            # Estrazione della visibilità
            visibility = annotation['visibility_token']

            lbox = LocalBox(
                center = box.center,
                wlh = box.wlh,
                corners_3d = box.corners(),
                name = box.name,
                token = instance_token,
                visibility = visibility
            )
            local_box.append(lbox)

        # Estrazione della posa dell'ego-vehicle dal record del sample
        ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        # Conversione della rotazione dell'ego-vehicle in quaternion
        # Permette di adattare le coordinate dei box dall'ego-vehicle al mondo globale
        # Quaternione è un modo matematico per rappresentare le rotazioni in uno spazio 4D
        # Il quaternione è un vettore 4D [w, x, y, z]
        q = Quaternion(ego_pose['rotation'])

        # Ottenimento della rotazione inversa dell'ego-vehicle
        R_inv = q.inverse.rotation_matrix
        
        # Estrazione della traslazione dell'ego-vehicle
        tx,ty,tz = ego_pose['translation']

        # Estrazione del token della scena globale
        scene = self.nusc.get('scene',sample['scene_token'])
        # Estrazione del token del log
        log = self.nusc.get('log',scene['log_token'])
        
        # Estrazione del nome della mappa
        map_name = log['location']

        # Salviamo in nusc_map l'oggetto MappaVettoriale associata alla scena corrente
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        # Impostiamo la distanza massima dal ego vehicle (nostro centro di interesse) per la quale vogliamo analizzare l'occlusione
        range_m = 50

        # Definiamo l'area di interesse, quadrata, centrata sul ego vehicle
        box_coords = (tx- range_m, ty- range_m, tx+ range_m, ty+ range_m)

        # Creiamo un dizionario con i 4 layer semantici del dataset NuScenes, per separare le varie parti della mappa
        # 'drivable_area': area percorribile dai veicoli
        # 'walkway': area pedonale
        # 'carpark_area': area di parcheggio
        # 'ped_crossing': attraversamento pedonale
        semantic_map = {
            'drivable_area': [],
            'walkway': [],
            'carpark_area': [],
            'ped_crossing': []
        }

        # Estrazione dei record di ogni layer semantico
        for layer in ['drivable_area', 'walkway', 'carpark_area', 'ped_crossing']:
            # records è un dizionario
            # Estrazione del record di ogni layer semantico all'interno del patch di interesse
            records = nusc_map.get_records_in_patch(box_coords, layer_names=[layer], mode = 'intersect')
            # Estrazione dei token dei record
            tokens = records.get(layer, [])

            # Per ogni token, estraiamo i poligoni associati
            for token in tokens:
                # Estrazione del record
                record = nusc_map.get(layer, token)
                # Estrazione dei token dei poligoni (drivable_area usa la lista polygon_tokens, gli altri usano polygon_token singolo)
                polygon_tokens = record.get('polygon_tokens', [])
                # (drivable_area usa la lista polygon_tokens, gli altri usano polygon_token singolo)
                if 'polygon_token' in record:
                    polygon_tokens.append(record['polygon_token'])

                for polygon_token in polygon_tokens:
                    # Estrazione del poligono
                    polygon = nusc_map.extract_polygon(polygon_token)
                    # Coordinate del poligono, formato da una lista di coordinate 
                    coords = np.array(polygon.exterior.coords)

                    # Prendiamo solo le coordinate X, Y (ignoriamo Z se presente)
                    coords_2d = coords[:, :2]
                    # Traslazione: sottraiamo la posizione dell'auto
                    coords_traslate = coords_2d - np.array([tx, ty])
                    # Rotazione inversa: allineiamo al sistema locale dell'auto, per far coincidere asse X con asse frontale del veicolo
                    coords_locali = coords_traslate @ R_inv[:2, :2].T

                    # Aggiunta del poligono al dizionario semantico
                    semantic_map[layer].append(coords_locali)
        # Preparazione del dizionario contenente i dati del sample
        return {
            "lidar_token": lidar_token,
            "sample_token": sample['token'],
            "scene_token": sample['scene_token'],
            "points": pts_local,
            "boxes": local_box,
            "semantic_map": semantic_map
        }




                    
                
            
            

        

        

        

        
        
            


        
         

    

         

    
    