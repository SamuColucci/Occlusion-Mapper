# Script di utilità isolato e temporaneo per visualizzare i frame originali NuScenes (visualizzatori/view_nuscenes_sample.py).
# Questo script mostra le immagini delle telecamere e la proiezione 3D del LiDAR usando i metodi di rendering nativi dell'SDK.

# Import del modulo os per la gestione dei percorsi su disco
import os
# Import di matplotlib per il rendering grafico a schermo
import matplotlib.pyplot as plt
# Import della classe NuScenes dall'SDK ufficiale per caricare i campioni di dati
from nuscenes.nuscenes import NuScenes

def main():
    # Stampa il messaggio di avvio dell'inizializzazione del dataset
    print("Inizializzazione NuScenes...")
    # Carica il dataset NuScenes mini situato nella cartella corrente ./nuscenes
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=True)
    
    # Elabora in sequenza tutti i campioni (sample) presenti nel dataset nuScenes mini
    for sample_idx in range(len(nusc.sample)):
        # Estrae l'oggetto sample corrente dalla lista
        sample = nusc.sample[sample_idx]
        # Recupera il token univoco del campione
        sample_token = sample['token']
        
        # Stampa le informazioni d'intestazione del fotogramma corrente nel terminale
        print(f"\n============================================================")
        print(f"Visualizzazione Frame #{sample_idx}")
        print(f"Sample Token: {sample_token}")
        print(f"============================================================")
        
        # Estrae il token di tracciamento per il sensore LiDAR superiore (LIDAR_TOP)
        lidar_token = sample['data']['LIDAR_TOP']
        print(f"Sto renderizzando i dati del sensore LIDAR_TOP (Token: {lidar_token})...")
        print("Premi la 'X' in alto a destra sulla finestra o premi 'Q' per passare al frame successivo.")
        
        # Il metodo nativo render_sample_data mostra la mappa LiDAR con i bounding box 3D annotati
        nusc.render_sample_data(
            lidar_token,
            with_anns=True,                     # Mostra i bounding box 3D degli ostacoli reali
            underlay_map=True,                  # Renderizza la mappa stradale HD di sfondo
            use_flat_vehicle_coordinates=True   # Allinea le coordinate al piano piano dell'Ego Vehicle
        )
        # Imposta il titolo superiore della finestra con il numero del campione corrente
        plt.title(f"NuScenes - Dati LiDAR BEV - Sample #{sample_idx}", color='black', fontsize=12, fontweight='bold')
        
        # Mostra la finestra a schermo e sospende l'esecuzione fino alla chiusura dell'utente
        plt.show()

# Blocco principale di esecuzione da riga di comando
if __name__ == "__main__":
    main()
