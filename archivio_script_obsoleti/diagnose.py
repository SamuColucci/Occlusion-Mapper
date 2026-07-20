import json
import numpy as np
from occ3d_occlusion_explorer import SOTARayCaster
from collections import defaultdict
from nuscenes.nuscenes import NuScenes

def run_diagnostics():
    print("=== AVVIO DIAGNOSTICA OCCLUSIONI ===")
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    caster = SOTARayCaster(nusc, 87)
    
    # Eseguiamo la pipeline fino all'estrazione finale
    caster.generate_known_zone()
    caster.generate_box_shadows()
    caster.find_object_occlusion_wedges()
    caster.generate_final_visibility()
    caster.extract_occlusion_zones()
    
    # 1. Recuperiamo la mappa rossa finale
    occluded_final = caster.occluded_final
    final_red_map = np.max(occluded_final, axis=2).astype(np.uint8)
    
    # 2. Contiamo i pixel rossi per ogni token in cone_ownership
    tally = defaultdict(int)
    for ix in range(final_red_map.shape[0]):
        for iy in range(final_red_map.shape[1]):
            if final_red_map[ix, iy] > 0:
                token = caster.cone_ownership.get((ix, iy))
                if token:
                    tally[token] += 1
                else:
                    tally["NESSUN_PROPRIETARIO"] += 1
                    
    # 3. Esportiamo e carichiamo il JSON effettivamente generato
    caster.save_occlusions_to_json('occlusions_nuscenes.json')
    with open('occlusions_nuscenes.json', 'r') as f:
        json_data = json.load(f)
    exported_names = [o['object_name'] for o in json_data['occlusions']]
    
    print("\n--- RIEPILOGO PIXEL ROSSI ATTRIBUITI ---")
    all_casters = caster.static_boxes + [b for b in nusc.get_sample_data(caster.lidar_token)[1] if any(c in b.name.lower() for c in ["vehicle", "truck", "bus", "trailer", "human.pedestrian", "bicycle", "motorcycle"])]
    token_to_name = {b.token: b.name for b in all_casters}
    token_to_dist = {b.token: np.linalg.norm(b.center[:2]) for b in all_casters}
    
    # Ordiniamo per numero di pixel decrescente
    for token, count in sorted(tally.items(), key=lambda x: x[1], reverse=True):
        name = token_to_name.get(token, "Sconosciuto")
        dist = token_to_dist.get(token, 0.0)
        
        # Controlliamo se questo specifico oggetto è finito nel JSON
        # Poiché nel JSON non salviamo il token, facciamo un controllo per nome e distanza approssimativa
        in_json = "NO"
        for o in json_data['occlusions']:
            if o['object_name'] == name and abs(o['distance_m'] - dist) < 1.0:
                in_json = "SI"
                break
                
        print(f"Token: {token:<35} | Nome: {name:<25} | Dist: {dist:5.2f}m | Pixel Rossi: {count:<4} | Esportato nel JSON? {in_json}")
        
if __name__ == "__main__":
    run_diagnostics()
