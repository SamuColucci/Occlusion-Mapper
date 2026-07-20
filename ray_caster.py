import numpy as np
from scipy import ndimage

# Raggio di analisi attorno all'auto (40 metri in tutte le direzioni)
GRID_RANGE= 40

# Risoluzione spaziale: ogni "cubetto" del nostro mondo virtuale misurerà 40 cm
VOXEL_SIZE=0.4

# Calcoliamo le dimensioni effettive della griglia (in numero di celle)
# Moltiplichiamo per 2 perché copriamo sia l'asse positivo che negativo
GRID_DIM = int((GRID_RANGE *2) / VOXEL_SIZE)

# Altezza del nostro mondo virtuale in cubetti (da -2m a +2m, cioè 24 cubetti, ovvero 9,6 m)
Z_DIM= 24

class RayCaster:
    # Costruttore della classe RayCaster
    # frame_data: dizionario contenente i dati del frame
    def __init__(self, frame_data):
        # Token identificativo del Lidar
        self.lidar_token = frame_data['lidar_token']
        # Punti 3D del Lidar
        self.pts = frame_data['points']
        # Box 3D degli oggetti
        self._all_boxes = frame_data['boxes']

        # Creazione delle griglie 3D vuote (Voxel). Inizialmente tutto è 0.
        # Suddivide lo spazio 3D in una griglia di cubetti (voxel).
        # Usiamo tre griglie al fine di non sovrapporre i punti del Lidar e di calcolare correttamente le zone occluse, come sottrazioni o addizioni di valori.
        # La griglia di punti memorizza i punti del Lidar
        # La griglia delle ombre memorizza le ombre esterne dei box
        # La griglia delle ombre interne memorizza le ombre interne dei box

        # Inizializziamo la griglia dei punti
        self.grid = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Inizializziamo la griglia delle ombre esterne
        self.box_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)
        # Inizializziamo la griglia delle ombre interne
        self.internal_shadows = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)

    # Metodo per convertire le coordinate dei punti del Lidar in coordinate della griglia
    # Al fine di insirire nella griglia tutti i punti del lidar 
    def _get_lidar_occupancy(self):
        
        # Griglia da riempire con i punti del Lidar
        lidar_occupancy = np.zeros((GRID_DIM, GRID_DIM, Z_DIM), dtype=np.uint8)

        # Offsett verticale per centrare i punti del lidar nell'area utile della griglia (da -2m a +2m)
        z_off = (Z_DIM // 4)* VOXEL_SIZE

        # Trasformazione dei punti da metri a cubetti per poter essere inseriti nella griglia
        # Nota: Con : prendiamo tutti i punti e con le colonne prendiamo x,y,z con rispettivamente 0,1,2
        # Asse X
        ix_p = (self.pts[:,0] / VOXEL_SIZE) + (GRID_DIM // 2)
        # Conversione in interi per indicizzare la griglia
        ix_p = ix_p.astype(int)
        # Asse Y
        iy_p = (self.pts[:,1] / VOXEL_SIZE) + (GRID_DIM // 2)
        # Conversione in interi per indicizzare la griglia
        iy_p = iy_p.astype(int)
        # Asse Z
        iz_p = (self.pts[:,2] + z_off) / VOXEL_SIZE
        # Conversione in interi per indicizzare la griglia
        iz_p = iz_p.astype(int)

        # Filtro per evitare di considerare punti che superano i 40 metri e che quindi non sono nel range utile per il nostro scopo
        m = (ix_p >= 0) & (ix_p < GRID_DIM) & \
            (iy_p >= 0) & (iy_p < GRID_DIM) & \
            (iz_p >= 0) & (iz_p < Z_DIM)

        # Inserimento dei punti del Lidar nella griglia, segnando con 1 i cubetti che contengono i punti
        lidar_occupancy[ix_p[m], iy_p[m], iz_p[m]] = 1

        # Restituzione della griglia dei punti
        return lidar_occupancy
    
    # Metodo per generare la griglia delle zone note sfruttando il concetto di raycasting e space carving
    # Lanciamo dei raggi dal sensore verso ogni punto del lidar per segnare le zone note, assegnado 1 ai cubetti che vengono attraversati dal raggio.
    def generate_known_zone(self):

        # Calcoliamo la griglia dei punti del lidar
        lidar_occupancy = self._get_lidar_occupancy()
        
        # Otteniamo le coordinate dei punti che sono stati colpiti dal lidar
        pts_x,pts_y,pts_z = np.nonzero(lidar_occupancy)

        # Caso limite: Se non ci sono punti lidar, non possiamo fare nulla
        if(len(pts_x) == 0):
            return
        
        # Risoluzione angolare del lidar, creazione di un Depth Buffer Sferico (DBR) 
        # Utilizziamo questa tecnica per ridurre il numero di raggi da sparare e quindi il calcolo computazionale
        # Sulla bolla in ogni direzione scriviamo un solo numero, poi prendiamo i cubetti rappresentanti i punti del lidar
        # Se il cubetto del lidar si trova tra il sensore e il numero scritto sulla bolla, allora quel cubetto è noto
        # In questo modo non dobbiamo sparare un raggio per ogni cubetto
        # Altrimenti (il cubetto è dietro) quel cubetto non e' noto
        AZ_RES= 1200
        EL_RES= 400

        # Riempiamo la bolla con -1 valore tale da permettere l'analisi e la gestione dei raggi che non colpiscono nessun punto
        # Essendo la distanza sempre positiva, il valore -1 ci permette di distinguerli facilmente
        depth_buffer= np.full((AZ_RES, EL_RES), -1.0)

        # Spostiamo l'origine della griglia al centro del volume 3D per facilitare il calcolo degli angoli e delle distanze
        # Questo ci permette di lavorare con coordinate relative al sensore, eliminando la necessità di gestire offset complessi
        x_centered = pts_x - (GRID_DIM//2)
        y_centered = pts_y - (GRID_DIM//2)
        z_centerd = pts_z - (Z_DIM//4)

        # Calcolo delle distanze in metri dei punti del lidar dal sensore
        # Per il calcolo della distanza si utiliza il teorema di Pitagora
        r = np.sqrt(x_centered**2 + y_centered**2 + z_centerd**2) * VOXEL_SIZE

        # Angolo azimutale (orientamento orizzontale del raggio)
        az= np.arctan2(y_centered, x_centered)
        # Angolo di elevazione (inclinazione verticale del raggio)
        # Usiamo r + 1e-6 per evitare divisioni per zero nel caso in cui il raggio colpisca un punto molto vicino al sensore.
        # Usiamo np.clip per evitare che il valore di el sia maggiore di 1 o minore di -1, limitandolo all'intervallo [-1, 1]
        # Infine calcoliamo l'arcoseno per ottenere l'angolo di elevazione
        el= np.arcsin(np.clip((z_centerd * VOXEL_SIZE) / (r + 1e-6), -1, 1))

        # Convertiamo gli angoli (che sono in radianti) in indici interi della bolla (da 0 a 1199 e da 0 a 399)
        az_idx = ((az + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(int)
        el_idx = ((el + np.pi / 2) / np.pi * (EL_RES - 1)).astype(int)

        # Riempiamo la bolla con le distanze dei punti del lidar
        # Se due punti colpiscono lo stesso angolo, prendiamo solo il più lontano
        for i in range(len(r)):
            if r[i] <= GRID_RANGE:
                if r[i] > depth_buffer[az_idx[i], el_idx[i]]:
                    depth_buffer[az_idx[i], el_idx[i]] = r[i]

        # Utilizziamo la maximum_filter per espandere le zone visibili in modo da compensare la bassa risoluzione del lidar
        # Altrimenti avremmo delle zone visibili non collegate tra loro, non permettendo la corretta identificazione delle zone occuluse
        # In questo modo riusciamo a "unire" le zone visibili e a creare un unico muro di zone note
        depth_buffer = ndimage.maximum_filter(depth_buffer, size=(3, 5))

        # Creazione delle coordinate cartesiane dei punti della griglia in metri
        # Adattiamo gli indici per essere simmetrici attorno allo zero
        # Coordinate x della griglia: da -40m a +40m con passo 0.4m
        x= (np.arange(GRID_DIM)- (GRID_DIM//2)) * VOXEL_SIZE
        # Coordinate y della griglia: da -40m a +40m con passo 0.4m
        y= (np.arange(GRID_DIM)- (GRID_DIM//2)) * VOXEL_SIZE
        # Coordinate z della griglia: da -2m a +2m con passo 0.4m
        z= (np.arange(Z_DIM)- (Z_DIM//4)) * VOXEL_SIZE

        # Meshgrid per ottenere tutte le combinazioni di coordinate (x,y,z)
        # indexing='ij' fa sì che la prima dimensione sia l'indice x, la seconda y e la terza z
        xv, yv, zv = np.meshgrid(x,y,z, indexing='ij')

        # Calcolo delle distanze in metri dei punti della griglia dal sensore
        # Per il calcolo della distanza si utiliza il teorema di Pitagora
        dist = np.sqrt(xv**2 + yv**2 + zv**2)

        # Lavoriamo solo sui cubetti che si trovano entro i 40 metri di raggio (evitiamo gli angoli del quadrato)
        valid_mask = dist <= GRID_RANGE 

        # Calcoliamo l'angolo di ogni singolo cubetto per capire dove guardare sulla Bolla
        az_v = np.arctan2(yv, xv)
        el_v = np.arcsin(np.clip(zv / (dist + 1e-6), -1.0, 1.0))

        # Troviamo il "pixel" corrispondente sulla bolla
        # Conversione degli angoli (che sono in radianti) in indici interi della bolla (da 0 a 1199 e da 0 a 399)
        u_v = ((az_v + np.pi) / (2 * np.pi) * (AZ_RES - 1)).astype(int)
        v_v = ((el_v + np.pi / 2) / np.pi * (EL_RES - 1)).astype(int)
        # Filtro per evitare di considerare indici fuori range
        u_v = np.clip(u_v, 0, AZ_RES - 1)
        v_v = np.clip(v_v, 0, EL_RES - 1)

        # Indichiamo la profondita' massima del lidar per quel raggio
        d_limit = depth_buffer[u_v,v_v]

        # La mappa conosciuta e' data dai cubetti che sono nel range del lidar, non sono oltre il limite di profondita'
        known_mask = valid_mask & (d_limit > 0) & (dist < d_limit)

        # Impostiamo il valore 1 per i cubetti che sono nel range del lidar, indicando che conosciamo il loro stato
        self.grid[known_mask] = 1

        return self.grid

    # Metodo per proiettari i coni d'ombra di un singolo ostacolo dinamico 
    def _add_box_shadow(self, cornes_3d):
        # Lavoriamo in 2D
        # Per le ombre dei veicoli a terra, ci basta guardare la scena dall'alto.
        # Prendiamo solo le coordinate X e Y dei 4 angoli della base del veicolo.
        cornes_2d = cornes_3d[:2, :]

        # Prima avevamo due vettori separati (uno per le X e uno per le Y).
        # Dopo la trasposta otteniamo una lista di 8 "Punti" (ogni punto è una coppia [X, Y]).
        # 4 spigoli anteriori e 4 spigoli posteriori
        cornes_2d = cornes_2d.T 

        # Calcoliamo la distanza degli 8 punti dal sensore
        # np.linalg.norm calcola la distanza di tutti gli 8 punti rispetto al sensore.
        # np.min sceglie il punto più vicino a noi: da quel punto in poi inizierà l'ombra
        r_min = np.min(np.linalg.norm(cornes_2d, axis=1))

        # Calcoliamo gli estremi del cono d'ombra al fine di capire la direzione di ognuno degli 8 spigoli della macchina
        az_c = np.arctan2(cornes_2d[:, 1], cornes_2d[:, 0])
       
        # Se un veicolo si trova a cavallo della linea posteriore (-180° / +180°),
        # Evita che vengano create ombre di 360 gradi attorno al veicolo
        if np.max(az_c) - np.min(az_c) > np.pi: 
            az_c[az_c < 0] += 2 * np.pi
            
        # Troviamo da che angolo a che angolo si estende l'ostacolo
        az_min = np.min(az_c)
        az_max = np.max(az_c)

        # Creiamo la griglia 2D per trovare i cubi che si trovano all'interno del cono d'ombra
        x_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        y_coords = (np.arange(GRID_DIM) - GRID_DIM//2) * VOXEL_SIZE
        # Meshgrid per ottenere tutte le combinazioni di coordinate (x,y)
        # indexing='ij' fa sì che la prima dimensione sia l'indice x, la seconda y
        xv_2d, yv_2d = np.meshgrid(x_coords, y_coords, indexing='ij')


        

        

        

        


        
        
            
        



 
        

        