# Questo file contiene il design Pattern Adapter al fine di essere indipendenti da uno specifico dataset
# In questo modo, possiamo cambiare facilmente dataset senza dover riscrivere tutto il codice

# Questo import permette di definire classi astratte
from typing import TypedDict, List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

# Definiamo il contratto rigoroso per i dati che l'Adapter deve restituire
class FrameData(TypedDict):
    # Token identificativo del Lidar
    lidar_token: str
    # Token identificativo del campione
    sample_token: str
    # Token identificativo della scena
    scene_token: str
    # Punti 3D del Lidar
    points: np.ndarray
    # Box 3D degli oggetti
    boxes: List[Any]
    # Informazioni relative all'ambiente circostante il veicolo
    semantic_map: Dict[str, List[np.ndarray]]

#Definizione della classe astratta
class AdapterDataset(ABC):
    # Costruttore della classe a cui passiamo il dataset presente in una specifica cartella del progetto
    def __init__(self, dataroot: str):
        self.dataroot = dataroot
    
    # Funzione per ottenere il numero di campioni del dataset
    @abstractmethod
    def get_num_samples(self) -> int:
        pass

    # Funzione per ottenere gli indici delle scene del dataset
    @abstractmethod
    def get_scene_indices(self) -> dict:
        pass

    # Funzione per ottenere i dati del dataset
    @abstractmethod
    def get_sample_data(self, idx: int) -> FrameData:
        pass


# Classe per rappresentare i box degli oggetti 3D presenti nel dataset
class LocalBox():
    # Costruttore della classe
    # center: lista contenente le coordinate del centro del box (x, y, z)
    # wlh: lista contenente la larghezza, lunghezza e altezza del box
    # corners_3d: lista contenente i vertici 3D del box, 4 anteriori e 4 inferiori
    # name: nome dell'oggetto
    # token: token identificativo dell'oggetto
    # visibility: visibilità dell'oggetto
    def __init__(self, center: list, wlh: list, corners_3d: list, name: str, token: str, visibility: str = "1"):
        self.center = center
        self.wlh = wlh
        self.corners_3d = corners_3d
        self.name = name
        self.token = token
        self.visibility = visibility

    # Metodo che restituisce i vertici 3D del box in formato array
    def corners(self):
        return self.corners_3d
    

    