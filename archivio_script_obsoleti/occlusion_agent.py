import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Importiamo il raycaster dall'explorer esistente
from nuscenes.nuscenes import NuScenes
from occ3d_occlusion_explorer import SOTARayCaster

# Configurazione della griglia coerente con occ3d_occlusion_explorer
GRID_RANGE = 40.0
VOXEL_SIZE = 0.4
GRID_DIM = int((GRID_RANGE * 2) / VOXEL_SIZE)

class OcclusionPredictiveAgent:
    def __init__(self, model_path="occlusion_agent_model.pkl"):
        self.model_path = model_path
        # Random Forest Classifier per predire: 0 (Free Space), 1 (Vehicle), 2 (Pedestrian)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.is_trained = False
        
        # Mappatura delle categorie causali in ID numerici
        self.causal_mapping = {
            "static": 0,
            "car": 1,
            "truck": 2,
            "bus": 3,
            "trailer": 4,
            "pedestrian": 5,
            "bicycle": 6,
            "motorcycle": 7
        }
        
    def _extract_features(self, occlusion_dict):
        """
        Estrae feature numeriche da un dizionario di occlusione del JSON.
        """
        # 1. Feature geometriche
        area = float(occlusion_dict.get("area_sqm", 0.0))
        distance = float(occlusion_dict.get("distance_m", 0.0))
        
        bbox = occlusion_dict.get("occlusion_bbox_m", [0.0, 0.0, 0.0, 0.0])
        dx = abs(float(bbox[2] - bbox[0]))
        dy = abs(float(bbox[3] - bbox[1]))
        aspect_ratio = dx / (dy + 1e-5)
        
        # 2. Categoria dell'oggetto causa (Label Encoding basato su parole chiave)
        causal_name = occlusion_dict.get("causal_object", occlusion_dict.get("object_name", "unknown")).lower()
        causal_id = 8  # Classe 'other' di fallback
        for key, val in self.causal_mapping.items():
            if key in causal_name:
                causal_id = val
                break
                
        return np.array([area, distance, dx, dy, aspect_ratio, causal_id])

    def prepare_dataset(self, dataset_path):
        """
        Carica i dati dal dataset JSON generato dalla pipeline di reveal temporale.
        Filtra le zone non rivelate ('unknown').
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset non trovato in: {dataset_path}")
            
        with open(dataset_path, "r") as f:
            data = json.load(f)
            
        X = []
        y = []
        
        for idx, item in enumerate(data):
            predictions = item.get("predictions", {})
            if not predictions:
                continue
                
            # Se la zona è rimasta totalmente sconosciuta (nessun raggio LiDAR futuro c'è passato),
            # non abbiamo una etichetta di ground truth affidabile, quindi saltiamo.
            if "unknown" in predictions and predictions["unknown"] >= 0.99:
                continue
                
            # Determiniamo la classe target reale (quella a frequenza maggiore nel futuro)
            best_class = max(predictions, key=predictions.get)
            
            # Mapping target label:
            # 0 = Free Space (Spazio vuoto)
            # 1 = Vehicle (Auto, camion, autobus, trailer)
            # 2 = Pedestrian (Pedoni)
            if "vehicle" in best_class.lower() or "truck" in best_class.lower() or "bus" in best_class.lower() or "trailer" in best_class.lower():
                label = 1
            elif "human" in best_class.lower() or "pedestrian" in best_class.lower():
                label = 2
            elif "free" in best_class.lower():
                label = 0
            else:
                # Altre classi minori o non categorizzate le saltiamo o le associamo a spazio libero
                label = 0
                
            feats = self._extract_features(item)
            X.append(feats)
            y.append(label)
            
        return np.array(X), np.array(y)

    def train_model(self, dataset_path):
        """
        Esegue il training dell'agente sul dataset auto-supervisionato.
        """
        X, y = self.prepare_dataset(dataset_path)
        print(f"Dataset caricato: {len(X)} campioni validi trovati.")
        
        if len(X) < 10:
            print("❌ Errore: Troppi pochi campioni nel dataset per addestrare il modello.")
            return False
            
        # Divisione in Train e Test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        print(f"Fase di fitting del modello Random Forest su {len(X_train)} campioni...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Validazione prestazioni
        preds = self.model.predict(X_test)
        print("\n" + "="*40)
        print("=== STATISTICHE DI VALIDAZIONE AGENTE ===")
        print("="*40)
        target_names = ["Free Space", "Vehicle", "Pedestrian"]
        # Filtriamo i nomi delle classi effettivamente presenti nel test set per evitare errori nel report
        present_classes = np.unique(np.concatenate((y_test, preds)))
        labels_names = [target_names[i] for i in present_classes]
        
        print(classification_report(y_test, preds, labels=present_classes, target_names=labels_names))
        
        # Salva il modello su disco
        joblib.dump(self.model, self.model_path)
        print(f"Modello dell'agente salvato con successo in: {self.model_path}")
        return True

    def predict_risk(self, occlusion_dict):
        """
        Esegue l'inferenza su una singola zona occlusa e calcola il punteggio di rischio.
        """
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            else:
                raise ValueError("Il modello dell'agente non è addestrato e non è stato trovato alcun checkpoint salvato.")
                
        feats = self._extract_features(occlusion_dict).reshape(1, -1)
        
        # Predizione delle probabilità
        probs = self.model.predict_proba(feats)[0]
        
        # Mappatura delle probabilità delle classi (gestendo il caso in cui il modello conosca meno classi nel training)
        classes = self.model.classes_
        prob_dict = {"free_space": 0.0, "vehicle": 0.0, "pedestrian": 0.0}
        
        for idx, cls_id in enumerate(classes):
            if cls_id == 0:
                prob_dict["free_space"] = float(probs[idx])
            elif cls_id == 1:
                prob_dict["vehicle"] = float(probs[idx])
            elif cls_id == 2:
                prob_dict["pedestrian"] = float(probs[idx])
                
        # Calcolo del Risk Score dell'Agente:
        # I pedoni invisibili rappresentano la minaccia più critica (peso 3x) rispetto ai veicoli (peso 1x)
        risk_score = prob_dict["vehicle"] * 1.0 + prob_dict["pedestrian"] * 3.0
        
        return prob_dict, risk_score

    def plot_predictions(self, caster, predictions_list):
        """
        Visualizza le zone occluse colorandole in base alla classe predetta più probabile dall'Agente ML.
        """
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        
        # Background: Punti LiDAR originali a tempo T
        ax.scatter(caster.pts[:, 1], caster.pts[:, 0], s=0.2, c='white', alpha=0.3, label='LiDAR Background')
        
        for item in predictions_list:
            poly = np.array(item['polygon_points_m'])
            probs = item['probabilities']
            risk = item['risk_score']
            
            # Troviamo lo stato più probabile
            best_state = max(probs, key=probs.get)
            prob_val = probs[best_state]
            
            if best_state == "vehicle":
                color = "#06b6d4"  # Ciano per veicoli
                label = f"Vehicle ({prob_val*100:.0f}%)"
            elif best_state == "pedestrian":
                color = "#10b981"  # Verde per pedoni
                label = f"Pedestrian ({prob_val*100:.0f}%)"
            else:
                color = "#eab308"  # Giallo per spazio libero
                label = f"Free Space ({prob_val*100:.0f}%)"
                
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 1], poly_closed[:, 0], color=color, linewidth=2.5)
            ax.fill(poly_closed[:, 1], poly_closed[:, 0], color=color, alpha=0.35)
            
            # Centroide per scrivere il testo
            centroid = np.mean(poly, axis=0)
            text_str = f"P(Free): {probs['free_space']*100:.0f}%\nP(Veh): {probs['vehicle']*100:.0f}%\nP(Ped): {probs['pedestrian']*100:.0f}%\nRisk: {risk:.2f}"
            ax.text(centroid[1], centroid[0], text_str, color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="white", lw=0.5, alpha=0.85))
            
        # Ego vehicle
        ax.plot(0, 0, 'ro', markersize=8, label='Ego Vehicle')
        
        ax.set_xlim(GRID_RANGE, -GRID_RANGE)
        ax.set_ylim(-GRID_RANGE, GRID_RANGE)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("STIMA DELLE OCCLUSIONI - AGENTE ML PREDECTIVE\nClassificatore Random Forest su Feature Geometrico-Causali", 
                     color='white', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Agente Predittivo delle Zone Occluse basato su Machine Learning")
    parser.add_argument("--mode", type=str, default="predict", choices=["train", "predict"],
                        help="Modalità operativa dell'agente: 'train' (addestra il classificatore) o 'predict' (esegue l'inferenza)")
    parser.add_argument("--dataset", type=str, default="predicted_occlusions.json",
                        help="File JSON contenente i dati di addestramento auto-supervisionati")
    parser.add_argument("--model-path", type=str, default="occlusion_agent_model.pkl",
                        help="Percorso di salvataggio/caricamento del modello serializzato")
    parser.add_argument("--sample-idx", type=int, default=200,
                        help="Indice del frame NuScenes da predire in modalità 'predict'")
    
    args = parser.parse_args()
    
    agent = OcclusionPredictiveAgent(model_path=args.model_path)
    
    if args.mode == "train":
        print("=== AVVIO FASE DI ADDESTRAMENTO DELL'AGENTE ===")
        success = agent.train_model(args.dataset)
        if success:
            print("=== ADDESTRAMENTO AGENTE COMPLETATO CON SUCCESSO! ===")
        else:
            print("❌ Addestramento fallito.")
            
    elif args.mode == "predict":
        print(f"=== AVVIO FASE DI PREDIZIONE SU SAMPLE {args.sample_idx} ===")
        
        # Inizializziamo NuScenes per estrarre le occlusioni correnti
        if not os.path.exists("./nuscenes"):
            print("❌ Errore: Cartella NuScenes non trovata in ./nuscenes.")
            return
            
        nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
        caster = SOTARayCaster(nusc, args.sample_idx)
        caster.generate_known_zone()
        caster.generate_box_shadows()
        caster.find_object_occlusion_wedges()
        caster.generate_final_visibility()
        caster.extract_occlusion_zones()
        
        # Esportiamo le occlusioni del tempo T per caricarle nel formato corretto
        temp_json = "temp_agent_predict.json"
        caster.save_occlusions_to_json(temp_json)
        
        with open(temp_json, "r") as f:
            data = json.load(f)
        os.remove(temp_json)
        
        occlusions = data.get("occlusions", [])
        if not occlusions:
            print("Nessuna zona occlusa rilevata in questo frame.")
            return
            
        predictions_list = []
        print(f"\nTrovate {len(occlusions)} zone occluse. Esecuzione inferenza dell'agente...")
        print("-" * 85)
        
        for idx, occ in enumerate(occlusions):
            try:
                probs, risk = agent.predict_risk(occ)
                predictions_list.append({
                    "object_name": occ["object_name"],
                    "distance_m": occ["distance_m"],
                    "polygon_points_m": occ["polygon_points_m"],
                    "probabilities": probs,
                    "risk_score": risk
                })
                
                print(f"Occlusione #{idx+1:<2} | Causa: {occ['object_name']:<25} | Dist: {occ['distance_m']:5.2f}m | Area: {occ['area_sqm']:5.2f}m²")
                print(f"  -> P(Free): {probs['free_space']*100:5.1f}% | P(Veh): {probs['vehicle']*100:5.1f}% | P(Ped): {probs['pedestrian']*100:5.1f}%")
                print(f"  -> RISK SCORE: {risk:5.2f}")
                print("-" * 85)
            except Exception as e:
                print(f"❌ Errore nell'inferenza per l'occlusione #{idx+1}: {e}")
                
        # Mostra i grafici con le predizioni dell'agente
        agent.plot_predictions(caster, predictions_list)

if __name__ == "__main__":
    main()
