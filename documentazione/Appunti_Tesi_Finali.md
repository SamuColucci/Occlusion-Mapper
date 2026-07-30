- Utilizzo del design pattern adapter per utilizzare più di un dataset senza modificare l'intero codice 
- Utilizzo del design pattern factory al fine di modificare solo il main del file quando chiamiano il dataset da usare e la funzione create adpater per la scelta del dataset nel caso in cui fosse uno di quello non previsti di default dal codice

- adapter_dataset.py: file contenente la classe astratta AdapterDataset e la classe concerta NuscenesDatasetAdapter
- nuscenes_dataset_adapter.py: file contenente l'adapter per il dataset NuScenes
- factory_dataset.py: file contenente la factory per la creazione dell'adapter


- ray_casting.py: file contenente la logica di ray casting al fine di trovare le zone note, occluse

- estrazione_zone_occluse.py: file principale che utilizza l'adapter per estrarre le zone occluse dal dataset NuScenes e salvare i dati in formato JSON

- conditional_probability_occlusion_zone.py: file per calcolare la probabilità condizionata di trovare una determinata categoria di ostacoli in base al tipo di superficie semantica del terreno

- conditional_probability_dataset.py: file per analizzare i frame estratti in precedenza al fine di calcolare la probabilità condizionata secondo le regole scritte in conditional_probability_occlusion_zone.py

- unet_model.py: file contenente il modello UNet 2D per la predizione delle mappe di calore

- neural_occlusion_agent.py: file contenente l'agente neurale per la predizione delle mappe di calore

- train_unet.py: file per addestrare il modello UNet 2D

- per_zone_model.py: rete neurale ibrida multimodale per la classificazione probabilistica diretta delle singole zone d'ombra

- dataset_generator_per_zone.py: generatore dei patch visivi (64x64) e delle feature scalari per-zone

- per_zone_occlusion_agent.py: agente neurale di inferenza per la stima e salvataggio dei dati probabilistici per-zone

- train_per_zone.py: script per addestrare il modello ibrido PerZoneModel

- visualizzatori/verify_runtime_neural_heatmap.py: visualizzatore della mappa di colore continua (heatmap 2D pixel-wise 200x200) generata dalla UNet 2D

- visualizzatori/verify_runtime_triple_comparison.py: visualizzatore comparativo sincronizzato in parallelo tra i 3 approcci (Bayesiano, UNet 2D Heatmap e Rete Neurale Per-Zone)


Fonti usate nel file estrazione_zone_occluse.py:
- High_resolution_maps_from_wide_angle_sonar.pdf
    - Sulle Occupancy Grids (Griglie di Occupazione) e Ray Casting applicato ai sensori: Il concetto di dividere lo spazio in "Voxel" (o in una griglia 2D/3D) e lanciare raggi per capire cosa è "libero" e cosa è "occupato" è stato inventato da due giganti della robotica
- A_1008191222954.pdf 
    - Sullo Space Carving (Scolpire lo spazio noto): L'idea di partire da un blocco pieno e "scolpirlo" togliendo lo spazio attraversato dai raggi si chiama Space Carving.