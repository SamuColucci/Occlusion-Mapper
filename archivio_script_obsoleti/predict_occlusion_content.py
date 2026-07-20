import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from occ3d_occlusion_explorer import SOTARayCaster

# --- CONFIGURAZIONE ---
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE)

class TemporalPredictor:
    def __init__(self, nusc, sample_idx=200):
        self.nusc = nusc
        self.sample_idx = sample_idx
        self.sample_t = nusc.sample[sample_idx]
        self.lidar_token_t = self.sample_t['data']['LIDAR_TOP']
        
        # Carichiamo i dati di calibrazione ed ego pose per il tempo T
        self.sd_record_t = nusc.get('sample_data', self.lidar_token_t)
        self.calib_t = nusc.get('calibrated_sensor', self.sd_record_t['calibrated_sensor_token'])
        self.ego_pose_t = nusc.get('ego_pose', self.sd_record_t['ego_pose_token'])
        
        # Inizializziamo il RayCaster per calcolare le occlusioni correnti
        print(f"Esecuzione Ray-Casting per il sample corrente {sample_idx}...")
        self.caster = SOTARayCaster(nusc, sample_idx)
        self.caster.generate_known_zone()
        self.caster.generate_box_shadows()
        self.caster.find_object_occlusion_wedges()
        self.caster.generate_final_visibility()
        self.caster.extract_occlusion_zones()
        
        # Salviamo il file JSON locale per comodità
        self.json_filename = "occlusions_nuscenes.json"
        self.caster.save_occlusions_to_json(self.json_filename)
        
        with open(self.json_filename, 'r') as f:
            self.occlusion_data = json.load(f)
        self.occlusions = self.occlusion_data.get('occlusions', [])
        
    def transform_points_to_t(self, pts_future, sd_future):
        """
        Trasforma i punti LiDAR da un frame futuro (sd_future) al frame del sensore LiDAR a tempo T (self.sd_record_t).
        """
        # 1. Recupero calibrazione ed ego-pose del frame futuro
        calib_f = self.nusc.get('calibrated_sensor', sd_future['calibrated_sensor_token'])
        ego_pose_f = self.nusc.get('ego_pose', sd_future['ego_pose_token'])
        
        # 2. Future Sensor -> Future Ego
        rot_calib_f = Quaternion(calib_f['rotation']).rotation_matrix
        trans_calib_f = np.array(calib_f['translation'])
        pts_ego_f = np.dot(pts_future, rot_calib_f.T) + trans_calib_f
        
        # 3. Future Ego -> Global
        rot_ego_f = Quaternion(ego_pose_f['rotation']).rotation_matrix
        trans_ego_f = np.array(ego_pose_f['translation'])
        pts_global = np.dot(pts_ego_f, rot_ego_f.T) + trans_ego_f
        
        # 4. Global -> Current Ego
        rot_ego_t = Quaternion(self.ego_pose_t['rotation']).rotation_matrix
        trans_ego_t = np.array(self.ego_pose_t['translation'])
        pts_ego_t = np.dot(pts_global - trans_ego_t, rot_ego_t) # Moltiplicazione per R^T
        
        # 5. Current Ego -> Current Sensor
        rot_calib_t = Quaternion(self.calib_t['rotation']).rotation_matrix
        trans_calib_t = np.array(self.calib_t['translation'])
        pts_sensor_t = np.dot(pts_ego_t - trans_calib_t, rot_calib_t)
        
        return pts_sensor_t

    def get_box_in_t(self, box_global):
        """
        Trasforma un Box NuScenes (in coordinate globali) nel frame del sensore LiDAR a tempo T.
        """
        box = box_global.copy()
        
        # 1. Global -> Current Ego
        box.translate(-np.array(self.ego_pose_t['translation']))
        box.rotate(Quaternion(self.ego_pose_t['rotation']).inverse)
        
        # 2. Current Ego -> Current Sensor
        box.translate(-np.array(self.calib_t['translation']))
        box.rotate(Quaternion(self.calib_t['rotation']).inverse)
        
        return box

    def analyze_future_reveals(self, steps_ahead=[1, 3, 5, 8]):
        """
        Scorre i frame futuri nel dataset NuScenes per verificare quali oggetti e punti
        compaiono all'interno delle zone occluse precedentemente calcolate.
        """
        print(f"\n=== ANALISI TEMPORALE DELLE OCCLUSIONI (Self-Supervised Future Reveals) ===")
        results = []
        
        # Per ciascuna occlusione, teniamo traccia delle predizioni probabilistiche
        for occ_idx, occ in enumerate(self.occlusions):
            poly_pts = np.array(occ['polygon_points_m'])
            path = Path(poly_pts)
            name = occ['object_name']
            dist = occ['distance_m']
            area = occ['area_sqm']
            
            print(f"\nAnalisi Occlusione #{occ_idx + 1} | Oggetto Causa: '{name}' | Distanza: {dist}m | Area: {area}mq")
            
            occurrences = {} # class_name -> list of (step, num_points)
            total_future_points = 0
            
            # Scorriamo i passi futuri nella scena
            curr_sample = self.sample_t
            for step in range(1, max(steps_ahead) + 1):
                next_token = curr_sample['next']
                if not next_token:
                    print(f"  [Fine Scena] Nessun altro frame disponibile al passo +{step}.")
                    break
                curr_sample = self.nusc.get('sample', next_token)
                
                if step in steps_ahead:
                    # 1. Carichiamo i punti LiDAR del frame futuro
                    lidar_token_f = curr_sample['data']['LIDAR_TOP']
                    sd_record_f = self.nusc.get('sample_data', lidar_token_f)
                    lidar_path_f = os.path.join(self.nusc.dataroot, sd_record_f['filename'])
                    pc_f = np.fromfile(lidar_path_f, dtype=np.float32).reshape(-1, 5)[:, :3]
                    
                    # 2. Proiettiamo i punti nel frame del tempo T
                    pts_projected = self.transform_points_to_t(pc_f, sd_record_f)
                    
                    # 3. Verifichiamo quali punti cadono dentro il poligono dell'occlusione (BEV)
                    pts_bev = pts_projected[:, :2]
                    inside_mask = path.contains_points(pts_bev)
                    pts_inside = pts_projected[inside_mask]
                    
                    # 4. Otteniamo i Box reali del frame futuro per capire che classe c'è lì
                    # Nota: per ottenere i box globali a tempo T_future usiamo sample_annotation
                    boxes_global = [self.nusc.get_box(ann) for ann in curr_sample['anns']]
                    boxes_projected = [self.get_box_in_t(b) for b in boxes_global]
                    
                    # Se ci sono punti, vediamo a quali box appartengono
                    if len(pts_inside) > 0:
                        total_future_points += len(pts_inside)
                        for pt in pts_inside:
                            matched_class = "free_space"
                            for box in boxes_projected:
                                # Controlliamo se il punto 3D cade dentro il Box (semplificato in BEV + Z)
                                corners = box.corners()
                                min_c = np.min(corners, axis=1)
                                max_c = np.max(corners, axis=1)
                                if (min_c[0] <= pt[0] <= max_c[0]) and \
                                   (min_c[1] <= pt[1] <= max_c[1]) and \
                                   (min_c[2] <= pt[2] <= max_c[2]):
                                    matched_class = box.name
                                    break
                            
                            if matched_class not in occurrences:
                                occurrences[matched_class] = 0
                            occurrences[matched_class] += 1
                            
                        print(f"  Passo +{step}: Rilevati {len(pts_inside)} punti proiettati.")
            
            # Calcolo delle probabilità finali per questa zona occlusa
            probabilities = {}
            if total_future_points > 0:
                for cls, count in occurrences.items():
                    probabilities[cls] = round(count / total_future_points, 3)
            else:
                probabilities["unknown"] = 1.0
                
            print(f"  --> Probabilità di presenza stimate:")
            for cls, p in probabilities.items():
                print(f"      * {cls}: {p * 100:.1f}%")
                
            results.append({
                "occlusion_id": occ_idx,
                "causal_object": name,
                "distance_m": dist,
                "area_sqm": area,
                "polygon_points_m": occ['polygon_points_m'],
                "predictions": probabilities
            })
            
        # Salviamo i risultati predetti
        output_filename = "predicted_occlusions.json"
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nRisultati predetti salvati con successo in '{output_filename}'.")
        return results

    def plot_predictions(self, results):
        """
        Visualizza le zone occluse colorandole in base alla classe predetta più probabile.
        """
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        
        # Disegniamo i punti LiDAR di background
        ax.scatter(self.caster.pts[:, 1], self.caster.pts[:, 0], s=0.2, c='white', alpha=0.3, label='LiDAR Background (T)')
        
        for res in results:
            poly = np.array(res['polygon_points_m'])
            predictions = res['predictions']
            
            # Troviamo la classe più probabile (escludendo "free_space" o considerandola)
            best_class = max(predictions, key=predictions.get)
            prob = predictions[best_class]
            
            # Assegniamo un colore in base alla classe predetta
            if "vehicle" in best_class.lower():
                color = "#06b6d4" # Ciano per veicoli
                label = f"Vehicle ({prob*100:.0f}%)"
            elif "human" in best_class.lower() or "pedestrian" in best_class.lower():
                color = "#10b981" # Verde per pedoni
                label = f"Pedestrian ({prob*100:.0f}%)"
            elif "free" in best_class.lower():
                color = "#eab308" # Giallo per spazio vuoto/strada libera
                label = f"Free Space ({prob*100:.0f}%)"
            else:
                color = "#64748b" # Grigio per sconosciuto/altro
                label = f"Unknown ({prob*100:.0f}%)"
                
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5)
            ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color, alpha=0.4, label=label)
            
            # Aggiungiamo un testo con la classe e la percentuale
            centroid = np.mean(poly, axis=0)
            ax.text(centroid[1], centroid[0], f"{best_class.split('.')[-1]}\n{prob*100:.0f}%", 
                    color='white', fontsize=9, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="white", lw=0.5, alpha=0.8))

        # Ego vehicle
        ax.plot(0, 0, 'ro', markersize=8, label='Ego Vehicle')
        
        ax.set_xlim(40, -40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"TEMPORAL REVEAL PREDICTION\nWhat's behind the occlusions? (Sample {self.sample_idx})", 
                     color='white', fontsize=14, fontweight='bold', pad=15)
        
        # Rimuoviamo i duplicati nella legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", facecolor="#1e293b", labelcolor="white", edgecolor="#334155")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    predictor = TemporalPredictor(nusc, 200)
    results = predictor.analyze_future_reveals(steps_ahead=[1, 2, 4, 6])
    predictor.plot_predictions(results)
