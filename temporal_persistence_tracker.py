"""
Analisi Temporale delle Occlusioni: Persistenza e Risk Score.

Questo script ripercorre la sequenza cronologica dei frame per ogni scena NuScenes
al fine di misurare il tempo di persistenza continuo (in frame e in secondi)
di ciascuna zona d'ombra associata a un oggetto occludente (caster).

Modello di rischio applicato:
La probabilità di penetrazione (ingresso di un terzo elemento invisibile nell'ombra)
cresce all'aumentare della durata dello stato di cecità (persistenza dell'occlusione)
secondo una legge esponenziale cumulativa:
    P(presenza) = 1 - exp(-lambda * t_persistenza)

Output:
I file JSON in 'extracted_occlusions/' vengono riscritti inserendo tre nuovi campi:
  - "persistence_frames": numero di frame consecutivi di persistenza
  - "persistence_seconds": durata temporale in secondi (basata sul dt di campionamento)
  - "risk_score": valore probabilistico normalizzato tra 0.0 (nessun rischio) e 1.0 (rischio massimo)
"""
import os
import glob
import json
import math
import numpy as np
from collections import defaultdict
from nuscenes.nuscenes import NuScenes

# Parametro di crescita lambda: modella la velocità con cui sale il fattore di rischio.
# Con LAMBDA = 0.15, dopo 7 keyframe consecutivi (3.5 secondi di cecità) il rischio supera 0.65 (65%)
LAMBDA = 0.15

# Frequenza temporale dei keyframe nel dataset NuScenes (campionati a 2 Hz, quindi 0.5s tra frame consecutivi)
FRAME_DT = 0.5  # secondi

# Funzione per calcolare il punteggio di rischio (Risk Score)
def risk_score(t_pers):
    """
    Applica il modello matematico cumulativo esponenziale:
    P(presenza) = 1 - e^(-lambda * t)
    """
    return 1.0 - math.exp(-LAMBDA * t_pers)

def main():
    print("Inizializzazione NuScenes...")
    # Istanziamo il devkit NuScenes (v1.0-mini) per estrarre la cronologia delle scene
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    
    # Cerchiamo tutti i JSON delle occlusioni estratti precedentemente dalla pipeline geometrica
    json_files = sorted(glob.glob("extracted_occlusions/*.json"))
    if not json_files:
        print("Nessun JSON trovato. Genera prima le occlusioni.")
        return
    
    # Creiamo un dizionario di lookup (lidar_token -> percorso file JSON) per velocizzare il caricamento durante il loop
    token_to_json = {}
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
        lt = data.get("lidar_token")
        if lt:
            token_to_json[lt] = fpath
    
    print(f"Mappati {len(token_to_json)} file JSON ai loro lidar_token.\n")
    
    # Raggruppiamo i sample del dataset in base alla scena di appartenenza
    scenes = defaultdict(list)
    for sample in nusc.sample:
        scenes[sample['scene_token']].append(sample)
    
    # Ordiniamo i sample all'interno di ciascuna scena in ordine cronologico (usando il timestamp)
    for scene_token in scenes:
        scenes[scene_token].sort(key=lambda s: s['timestamp'])
    
    # Liste per raccogliere statistiche complessive utili al report finale
    all_persistences = []
    all_risks = []
    persistence_distribution = defaultdict(int)
    enriched_count = 0
    
    print(f"Analisi temporale su {len(scenes)} scene...\n")
    
    # Ciclo principale su ciascuna scena del dataset
    for scene_idx, (scene_token, samples) in enumerate(scenes.items()):
        scene_record = nusc.get('scene', scene_token)
        scene_name = scene_record['name']
        
        # Mappa attiva per questa scena: memorizza (object_token -> contatore di frame di persistenza)
        active_occlusions = {}  
        
        # Iteriamo cronologicamente lungo la sequenza di sample
        for frame_idx, sample in enumerate(samples):
            lidar_token = sample['data']['LIDAR_TOP']
            json_path = token_to_json.get(lidar_token)
            
            # Se un frame intermedio non possiede il file JSON delle occlusioni, azzeriamo la persistenza attiva
            if not json_path:
                active_occlusions.clear()
                continue
            
            with open(json_path, "r") as f:
                data = json.load(f)
            
            occlusions = data.get("occlusions", [])
            
            # Set per tracciare quali oggetti (tokens) generano un'occlusione in questo specifico frame
            current_tokens = set()
            
            for occ in occlusions:
                obj_token = occ.get("object_token", "")
                if not obj_token:
                    continue
                
                current_tokens.add(obj_token)
                
                # Se l'ostacolo era già presente nel frame precedente, incrementiamo la persistenza.
                # Altrimenti, creiamo una nuova voce con persistenza iniziale = 1.
                if obj_token in active_occlusions:
                    active_occlusions[obj_token] += 1
                else:
                    active_occlusions[obj_token] = 1
                
                t_pers = active_occlusions[obj_token]
                r_score = risk_score(t_pers)
                
                # Arricchiamo l'oggetto occlusione nel JSON con i nuovi attributi temporali
                occ["persistence_frames"] = t_pers
                occ["persistence_seconds"] = round(t_pers * FRAME_DT, 1)
                occ["risk_score"] = round(r_score, 4)
                
                # Salviamo i valori per il calcolo delle statistiche globali
                all_persistences.append(t_pers)
                all_risks.append(r_score)
                persistence_distribution[t_pers] += 1
            
            # Rileviamo ed eliminiamo dalla mappa gli ostacoli che non generano più occlusioni in questo frame
            # (es. l'oggetto è uscito dal campo visivo o la sua ombra è stata completamente scoperta)
            disappeared = [t for t in active_occlusions if t not in current_tokens]
            for t in disappeared:
                del active_occlusions[t]
            
            # Sovrascriviamo il file JSON arricchito con le nuove metriche temporali
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            enriched_count += 1
        
        # Stampa a console dello stato finale della scena per debug/monitoraggio
        max_pers = max(active_occlusions.values()) if active_occlusions else 0
        print(f"  Scena {scene_idx+1}/{len(scenes)}: {scene_name} | "
              f"{len(samples)} frame | "
              f"Max persistenza alla fine: {max_pers} frame ({max_pers * FRAME_DT:.1f}s)")
    
    # === GENERAZIONE E STAMPA DEL REPORT STATISTICO GLOBALE ===
    print("\n" + "=" * 70)
    print("REPORT ANALISI TEMPORALE")
    print("=" * 70)
    print(f"File JSON arricchiti:      {enriched_count}")
    print(f"Occlusioni totali tracciate: {len(all_persistences)}")
    
    if all_persistences:
        arr = np.array(all_persistences)
        print(f"\nPersistenza (in frame):")
        print(f"  Media:   {arr.mean():.2f} frame ({arr.mean() * FRAME_DT:.1f}s)")
        print(f"  Mediana: {np.median(arr):.0f} frame ({np.median(arr) * FRAME_DT:.1f}s)")
        print(f"  Max:     {arr.max()} frame ({arr.max() * FRAME_DT:.1f}s)")
        
        risk_arr = np.array(all_risks)
        print(f"\nRisk Score (0 = sicuro, 1 = massimo rischio):")
        print(f"  Media:   {risk_arr.mean():.4f}")
        print(f"  Mediana: {np.median(risk_arr):.4f}")
        print(f"  Max:     {risk_arr.max():.4f}")
        
        # Distribuzione statistica dettagliata della persistenza
        print(f"\nDistribuzione della Persistenza:")
        print(f"  {'Frame':<8} {'Secondi':<10} {'Conteggio':<12} {'Risk Score':<12} {'%':<8}")
        print(f"  {'-'*50}")
        total = len(all_persistences)
        for t in sorted(persistence_distribution.keys()):
            c = persistence_distribution[t]
            pct = c / total * 100
            rs = risk_score(t)
            print(f"  {t:<8} {t*FRAME_DT:<10.1f} {c:<12} {rs:<12.4f} {pct:<8.1f}")
    
    print(f"\nI file JSON in extracted_occlusions/ sono stati arricchiti con:")
    print(f"  - 'persistence_frames': frame consecutivi di occlusione")
    print(f"  - 'persistence_seconds': secondi di cecità")
    print(f"  - 'risk_score': P(presenza) = 1 - exp(-{LAMBDA} * t)")

if __name__ == "__main__":
    main()
