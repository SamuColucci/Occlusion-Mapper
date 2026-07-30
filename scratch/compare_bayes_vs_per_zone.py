"""
Script di Analisi e Confronto: Agente Bayesiano vs Agente Neurale Per-Zone (scratch/compare_bayes_vs_per_zone.py).

Confronta le probabilità stimate per ciascuna zona d'ombra dai due approcci:
1. Calcola il tasso di convergenza/accordo sulla categoria dominante (#1).
2. Calcola lo scarto medio percentuale (Mean Absolute Difference) per ogni classe.
3. Stampa una tabella comparativa affiancata per le zone d'ombra più significative.
"""

import os
import glob
import json
import numpy as np

def main():
    dir_bayes = "extracted_occlusions_bayes" if os.path.exists("extracted_occlusions_bayes") else "extracted_occlusions_probabilities"
    dir_per_zone = "extracted_occlusions_per_zone" if os.path.exists("extracted_occlusions_per_zone") else "extracted_occlusions_neural"

    print("=" * 80)
    print("      CONFRONTO RISULTATI: AGENTE BAYESIANO vs AGENTE NEURALE PER-ZONE")
    print("=" * 80)
    print(f" Cartella Bayes    : {os.path.abspath(dir_bayes)}")
    print(f" Cartella Per-Zone : {os.path.abspath(dir_per_zone)}")
    print("-" * 80)

    files_bayes = sorted(glob.glob(os.path.join(dir_bayes, "*.json")))
    if not files_bayes:
        print("[ERRORE] Nessun file JSON Bayesiano trovato! Esegui prima conditional_probability_dataset.py.")
        return

    total_matched_occlusions = 0
    top1_agreements = 0
    categories = ["Auto", "Pedone", "Camion", "Bicicletta", "Moto", "Barriera"]
    
    diffs = {c: [] for c in categories}
    comparison_table = []

    for f_bayes in files_bayes:
        token = os.path.basename(f_bayes).split("_")[-1].replace(".json", "")
        # Cerca il corrispettivo file per-zone
        match_files = glob.glob(os.path.join(dir_per_zone, f"*{token}.json"))
        if not match_files:
            continue
            
        f_pz = match_files[0]
        
        with open(f_bayes, "r") as fb, open(f_pz, "r") as fp:
            data_bayes = json.load(fb)
            data_pz = json.load(fp)
            
        occs_b = data_bayes.get("occlusions", [])
        occs_pz = data_pz.get("occlusions", [])
        
        num_occ = min(len(occs_b), len(occs_pz))
        for i in range(num_occ):
            ob = occs_b[i]
            opz = occs_pz[i]
            
            pb = ob.get("estimated_probabilities", {})
            ppz = opz.get("estimated_probabilities", {})
            
            if not pb or not ppz:
                continue
                
            total_matched_occlusions += 1
            
            # 1. Verifico accordo categoria dominante (#1)
            dom_b = max(pb, key=pb.get)
            dom_pz = max(ppz, key=ppz.get)
            
            if dom_b == dom_pz:
                top1_agreements += 1
                
            # 2. Calcolo scarti per categoria
            for c in categories:
                val_b = pb.get(c, 0.0)
                val_pz = ppz.get(c, 0.0)
                diffs[c].append(abs(val_b - val_pz))
                
            # Salviamo campioni significativi per la tabella
            if len(comparison_table) < 10 and (pb.get("Auto", 0) > 0.3 or pb.get("Pedone", 0) > 0.4):
                comparison_table.append({
                    "sample": token[:8],
                    "idx": i + 1,
                    "area": ob.get("area_sqm", 0.0),
                    "surface": ob.get("dominant_surface", "Generico"),
                    "dom_b": f"{dom_b} ({pb[dom_b]*100:.0f}%)",
                    "dom_pz": f"{dom_pz} ({ppz[dom_pz]*100:.0f}%)",
                    "auto_b": f"{pb.get('Auto', 0)*100:.1f}%",
                    "auto_pz": f"{ppz.get('Auto', 0)*100:.1f}%",
                    "ped_b": f"{pb.get('Pedone', 0)*100:.1f}%",
                    "ped_pz": f"{ppz.get('Pedone', 0)*100:.1f}%",
                })

    if total_matched_occlusions == 0:
        print("[WARNING] Nessun cono d'ombra accoppiato trovato per il confronto.")
        return

    agreement_pct = (top1_agreements / total_matched_occlusions) * 100

    print(f"\n--- STATISTICHE DI CONVERGENZA SU {total_matched_occlusions} ZONE D'OMBRA ---")
    print(f" -> Accordo Categoria Dominante (#1) : {agreement_pct:.1f}% ({top1_agreements}/{total_matched_occlusions} ombre)")
    print("\n -> Scarto Medio Assoluto per Categoria (|Bayes - PerZone|):")
    for c in categories:
        mean_diff = np.mean(diffs[c]) * 100
        print(f"    - {c:<12s}: {mean_diff:.1f}% di scarto medio")

    print("\n" + "=" * 85)
    print("  TABELLA COMPARATIVA SAMPLE: BAYES vs PER-ZONE NEURAL (TOP 10 ZONE D'OMBRA)")
    print("=" * 85)
    print(f" {'Sample':<9} {'#':<3} {'Area':<6} {'Terreno':<15} {'Bayes Dominante':<18} {'Per-Zone Dominante':<18} {'Auto (B/PZ)':<14} {'Ped (B/PZ)':<14}")
    print("-" * 85)
    for row in comparison_table:
        print(f" {row['sample']:<9} #{row['idx']:<2} {row['area']:<5.1f}m² {row['surface']:<15} {row['dom_b']:<18} {row['dom_pz']:<18} {row['auto_b']}/{row['auto_pz']:<7} {row['ped_b']}/{row['ped_pz']:<7}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
