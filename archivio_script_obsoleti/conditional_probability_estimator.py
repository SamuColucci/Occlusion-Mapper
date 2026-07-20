import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from occ3d_occlusion_explorer import SOTARayCaster

# --- CONFIGURAZIONE ---
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE)

class ConditionalProbabilityEstimator:
    def __init__(self, nusc):
        self.nusc = nusc
        # Tabella per contare le occorrenze: (causa_class, dist_bin) -> { outcome_class: count }
        self.counts = defaultdict(lambda: defaultdict(int))
        self.prob_table = {}

    def get_distance_bin(self, dist):
        if dist < 12.0:
            return "near"
        elif dist < 25.0:
            return "medium"
        else:
            return "far"

    def transform_points(self, pts, sd_src, sd_dst):
        """ Trasforma punti dal sensore di sd_src a quello di sd_dst """
        calib_src = self.nusc.get('calibrated_sensor', sd_src['calibrated_sensor_token'])
        ego_src = self.nusc.get('ego_pose', sd_src['ego_pose_token'])
        calib_dst = self.nusc.get('calibrated_sensor', sd_dst['calibrated_sensor_token'])
        ego_dst = self.nusc.get('ego_pose', sd_dst['ego_pose_token'])
        
        # Src Sensor -> Src Ego -> Global
        r_c_src = Quaternion(calib_src['rotation']).rotation_matrix
        t_c_src = np.array(calib_src['translation'])
        pts_ego = np.dot(pts, r_c_src.T) + t_c_src
        
        r_e_src = Quaternion(ego_src['rotation']).rotation_matrix
        t_e_src = np.array(ego_src['translation'])
        pts_global = np.dot(pts_ego, r_e_src.T) + t_e_src
        
        # Global -> Dst Ego -> Dst Sensor
        r_e_dst = Quaternion(ego_dst['rotation']).rotation_matrix
        t_e_dst = np.array(ego_dst['translation'])
        pts_ego_dst = np.dot(pts_global - t_e_dst, r_e_dst)
        
        r_c_dst = Quaternion(calib_dst['rotation']).rotation_matrix
        t_c_dst = np.array(calib_dst['translation'])
        pts_sensor_dst = np.dot(pts_ego_dst - t_c_dst, r_c_dst)
        
        return pts_sensor_dst

    def get_box_in_dst(self, box_global, sd_dst):
        """ Porta un Box globale nel sensore di sd_dst """
        box = box_global.copy()
        ego_dst = self.nusc.get('ego_pose', sd_dst['ego_pose_token'])
        calib_dst = self.nusc.get('calibrated_sensor', sd_dst['calibrated_sensor_token'])
        
        box.translate(-np.array(ego_dst['translation']))
        box.rotate(Quaternion(ego_dst['rotation']).inverse)
        box.translate(-np.array(calib_dst['translation']))
        box.rotate(Quaternion(calib_dst['rotation']).inverse)
        return box

    def train_probability_table(self, start_idx=100, end_idx=130):
        """
        Fase di Addestramento: Scorre una serie di campioni per popolare la tabella
        delle probabilità condizionate P(Stato reale | Categoria ostacolo, Fascia di distanza)
        """
        print(f"=== INIZIO FASE DI CONTEGGIO STATISTICO (Sample {start_idx} a {end_idx}) ===")
        
        for idx in range(start_idx, end_idx):
            if idx >= len(self.nusc.sample): break
            sample = self.nusc.sample[idx]
            next_token = sample['next']
            if not next_token: continue # Salta se è l'ultimo frame della scena
            
            sample_next = self.nusc.get('sample', next_token)
            lidar_token_t = sample['data']['LIDAR_TOP']
            sd_t = self.nusc.get('sample_data', lidar_token_t)
            
            # Eseguiamo il RayCaster per ottenere le occlusioni a tempo T
            caster = SOTARayCaster(self.nusc, idx)
            caster.generate_known_zone()
            caster.generate_box_shadows()
            caster.find_object_occlusion_wedges()
            caster.generate_final_visibility()
            caster.extract_occlusion_zones()
            
            # Esportiamo le occlusioni del tempo T
            temp_json = f"temp_occ_{idx}.json"
            caster.save_occlusions_to_json(temp_json)
            with open(temp_json, 'r') as f:
                data = json.load(f)
            os.remove(temp_json) # Pulizia file temporaneo
            
            occlusions = data.get('occlusions', [])
            if not occlusions: continue
            
            # Carichiamo i dati del futuro T+1
            lidar_token_f = sample_next['data']['LIDAR_TOP']
            sd_f = self.nusc.get('sample_data', lidar_token_f)
            pc_f = np.fromfile(os.path.join(self.nusc.dataroot, sd_f['filename']), dtype=np.float32).reshape(-1, 5)[:, :3]
            
            # Proiettiamo i punti e i box del futuro nel frame del tempo T
            pts_projected = self.transform_points(pc_f, sd_f, sd_t)
            boxes_global = [self.nusc.get_box(ann) for ann in sample_next['anns']]
            boxes_projected = [self.get_box_in_dst(b, sd_t) for b in boxes_global]
            
            # Per ogni occlusione, raccogliamo la statistica
            for occ in occlusions:
                poly = np.array(occ['polygon_points_m'])
                path = Path(poly)
                
                # Chi ha causato l'occlusione e a che distanza?
                causal_name = occ['object_name']
                # Categorizziamo la classe causa (veicolo o pedone)
                causal_cat = "vehicle" if "vehicle" in causal_name or "truck" in causal_name or "bus" in causal_name else "pedestrian"
                dist = occ['distance_m']
                dist_bin = self.get_distance_bin(dist)
                
                key = (causal_cat, dist_bin)
                
                # Troviamo i punti futuri che cadono dentro
                inside_mask = path.contains_points(pts_projected[:, :2])
                pts_inside = pts_projected[inside_mask]
                
                if len(pts_inside) == 0:
                    self.counts[key]["unknown"] += 1
                    continue
                
                # Identifichiamo di che classe sono i punti proiettati
                for pt in pts_inside:
                    matched_class = "free_space"
                    for box in boxes_projected:
                        corners = box.corners()
                        min_c = np.min(corners, axis=1)
                        max_c = np.max(corners, axis=1)
                        if (min_c[0] <= pt[0] <= max_c[0]) and \
                           (min_c[1] <= pt[1] <= max_c[1]) and \
                           (min_c[2] <= pt[2] <= max_c[2]):
                            matched_class = "vehicle" if "vehicle" in box.name or "truck" in box.name else "pedestrian"
                            break
                    self.counts[key][matched_class] += 1
        
        # Calcolo delle probabilità condizionate finali
        print("\n=== TABELLA DELLE PROBABILITA CONDIZIONATE CALCOLATA ===")
        print(f"{'CONDIZIONE (Causa, Distanza)':<35} | {'P(Free Space)':<15} | {'P(Vehicle)':<12} | {'P(Pedestrian)':<15}")
        print("-" * 85)
        for key, outcomes in self.counts.items():
            total = sum(outcomes.values())
            self.prob_table[key] = {
                "free_space": outcomes["free_space"] / total if total > 0 else 0.0,
                "vehicle": outcomes["vehicle"] / total if total > 0 else 0.0,
                "pedestrian": outcomes["pedestrian"] / total if total > 0 else 0.0
            }
            print(f"({key[0]:<10}, {key[1]:<6}) | {self.prob_table[key]['free_space']*100:12.1f}% | {self.prob_table[key]['vehicle']*100:10.1f}% | {self.prob_table[key]['pedestrian']*100:13.1f}%")
        print("========================================================\n")

    def predict_sample(self, test_idx=200):
        """
        Applica la tabella delle probabilità condizionate alle occlusioni del frame di test.
        """
        print(f"=== INFERENZA PROBABILISTICA SUL SAMPLE {test_idx} ===")
        sample = self.nusc.sample[test_idx]
        caster = SOTARayCaster(self.nusc, test_idx)
        caster.generate_known_zone()
        caster.generate_box_shadows()
        caster.find_object_occlusion_wedges()
        caster.generate_final_visibility()
        caster.extract_occlusion_zones()
        
        temp_json = "temp_test.json"
        caster.save_occlusions_to_json(temp_json)
        with open(temp_json, 'r') as f:
            data = json.load(f)
        os.remove(temp_json)
        
        occlusions = data.get('occlusions', [])
        predictions_output = []
        
        for occ in occlusions:
            causal_name = occ['object_name']
            causal_cat = "vehicle" if "vehicle" in causal_name or "truck" in causal_name or "bus" in causal_name else "pedestrian"
            dist = occ['distance_m']
            dist_bin = self.get_distance_bin(dist)
            
            key = (causal_cat, dist_bin)
            
            # Se la condizione è presente in tabella, usiamo la probabilità appresa, altrimenti prior standard
            if key in self.prob_table:
                probs = self.prob_table[key]
            else:
                # Prior di fallback uniforme/ragionevole se la combinazione non è stata vista nel training
                probs = {"free_space": 0.70, "vehicle": 0.25, "pedestrian": 0.05}
                
            predictions_output.append({
                "causal_object": causal_name,
                "distance_m": dist,
                "polygon_points_m": occ['polygon_points_m'],
                "probabilities": probs
            })
            
            print(f"Occlusione causata da: '{causal_name}' a {dist:.1f}m ({dist_bin})")
            print(f"  -> Probabilità Stimate: P(Libero)={probs['free_space']*100:.1f}%, P(Veicolo)={probs['vehicle']*100:.1f}%, P(Pedone)={probs['pedestrian']*100:.1f}%")
            
        self.plot_predictions(caster, predictions_output)

    def plot_predictions(self, caster, predictions):
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        
        # Background punti LiDAR originali
        ax.scatter(caster.pts[:, 1], caster.pts[:, 0], s=0.2, c='white', alpha=0.3, label='LiDAR Background')
        
        for pred in predictions:
            poly = np.array(pred['polygon_points_m'])
            probs = pred['probabilities']
            
            # Troviamo lo stato più probabile
            best_state = max(probs, key=probs.get)
            prob_val = probs[best_state]
            
            if best_state == "vehicle":
                color = "#06b6d4" # Ciano
                label = f"Vehicle ({prob_val*100:.0f}%)"
            elif best_state == "pedestrian":
                color = "#10b981" # Verde
                label = f"Pedestrian ({prob_val*100:.0f}%)"
            else:
                color = "#eab308" # Giallo per spazio libero
                label = f"Free Space ({prob_val*100:.0f}%)"
                
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5)
            ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color, alpha=0.35)
            
            # Scrittura della stima sull'area
            centroid = np.mean(poly, axis=0)
            text_str = f"P(Free): {probs['free_space']*100:.0f}%\nP(Veh): {probs['vehicle']*100:.0f}%"
            ax.text(centroid[1], centroid[0], text_str, color='white', fontsize=8, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="white", lw=0.5, alpha=0.85))
            
        ax.plot(0, 0, 'ro', markersize=8, label='Ego Vehicle')
        ax.set_xlim(40, -40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("PREDIZIONE TRAMITE PROBABILITÀ CONDIZIONATA\nStima basata sul contesto geometrico-spaziale", 
                     color='white', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    estimator = ConditionalProbabilityEstimator(nusc)
    # Addestramento della tabella su 30 frame per avere una statistica robusta
    estimator.train_probability_table(start_idx=100, end_idx=130)
    # Inferenza sul frame di test 200
    estimator.predict_sample(test_idx=200)
