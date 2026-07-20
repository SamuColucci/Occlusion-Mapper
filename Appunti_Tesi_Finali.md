- Utilizzo del design pattern adapter per utilizzare più di un dataset senza modificare l'intero codice 
- Utilizzo del design pattern factory al fine di modificare solo il main del file quando chiamiano il dataset da usare e la funzione create adpater per la scelta del dataset nel caso in cui fosse uno di quello non previsti di default dal codice

- adapter_dataset.py: file contenente la classe astratta AdapterDataset e la classe concerta NuscenesDatasetAdapter
- nuscenes_dataset_adapter.py: file contenente l'adapter per il dataset NuScenes
- factory_dataset.py: file contenente la factory per la creazione dell'adapter


- ray_casting.py: file contenente la logica di ray casting al fine di trovare le zone note, occluse

- estrazione_zone_occluse.py: file principale che utilizza l'adapter per estrarre le zone occluse dal dataset NuScenes

Fonti usate nel file estrazione_zone_occluse.py:
- High_resolution_maps_from_wide_angle_sonar.pdf
    - Sulle Occupancy Grids (Griglie di Occupazione) e Ray Casting applicato ai sensori: Il concetto di dividere lo spazio in "Voxel" (o in una griglia 2D/3D) e lanciare raggi per capire cosa è "libero" e cosa è "occupato" è stato inventato da due giganti della robotica
- A_1008191222954.pdf 
    - Sullo Space Carving (Scolpire lo spazio noto): L'idea di partire da un blocco pieno e "scolpirlo" togliendo lo spazio attraversato dai raggi si chiama Space Carving.