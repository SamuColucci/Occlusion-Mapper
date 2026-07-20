"""
batch_regression_check.py — confronta i JSON esistenti in extracted_occlusions/ 
con quelli prodotti dal codice refactored, sample per sample.

Usage:
  python batch_regression_check.py                     # tutti i 170 sample
  python batch_regression_check.py --max-samples 10    # solo i primi 10 (veloce)

Per ogni sample: rigenera il JSON col nuovo codice, confronta numero di oggetti, 
token, area e poligoni con il file originale. Segnala differenze.

ponytail: nessun framework, nessuna dipendenza nuova. Confronto strutturale JSON.
"""
import argparse
import json
import os
import sys
import glob


def compare_jsons(ref_path, cur_data):
    """Confronta il JSON di riferimento con i dati rigenerati. Ritorna lista di differenze."""
    with open(ref_path) as f:
        ref = json.load(f)
    
    diffs = []
    
    # Stesso lidar_token?
    if ref.get("lidar_token") != cur_data.get("lidar_token"):
        diffs.append(f"lidar_token diverso: {ref.get('lidar_token')} vs {cur_data.get('lidar_token')}")
        return diffs  # inutile continuare
    
    ref_occs = {o["object_token"]: o for o in ref.get("occlusions", [])}
    cur_occs = {o["object_token"]: o for o in cur_data.get("occlusions", [])}
    
    ref_tokens = set(ref_occs.keys())
    cur_tokens = set(cur_occs.keys())
    
    added = cur_tokens - ref_tokens
    removed = ref_tokens - cur_tokens
    
    if added:
        diffs.append(f"  +{len(added)} oggetti nuovi: {list(added)[:3]}...")
    if removed:
        diffs.append(f"  -{len(removed)} oggetti mancanti: {list(removed)[:3]}...")
    
    # Per gli oggetti in comune, confronta area e polygon
    common = ref_tokens & cur_tokens
    area_diffs = 0
    poly_diffs = 0
    for token in common:
        r, c = ref_occs[token], cur_occs[token]
        if abs(r["area_sqm"] - c["area_sqm"]) > 0.01:
            area_diffs += 1
        if r["polygon_points_m"] != c["polygon_points_m"]:
            poly_diffs += 1
    
    if area_diffs:
        diffs.append(f"  {area_diffs}/{len(common)} oggetti con area diversa")
    if poly_diffs:
        diffs.append(f"  {poly_diffs}/{len(common)} oggetti con poligoni diversi")
    
    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", default="extracted_occlusions", help="Cartella con i JSON di riferimento")
    p.add_argument("--max-samples", type=int, default=None, help="Limita il numero di sample da controllare")
    args = p.parse_args()
    
    ref_files = sorted(glob.glob(os.path.join(args.ref_dir, "occlusion_sample_*.json")))
    if not ref_files:
        sys.exit(f"Nessun JSON trovato in {args.ref_dir}")
    
    if args.max_samples:
        ref_files = ref_files[:args.max_samples]
    
    print(f"Confronto {len(ref_files)} sample con il codice refactored...")
    print("(Ogni sample richiede ~30-60s per la pipeline completa)\n")
    
    # Import pesanti solo qui
    from nuscenes.nuscenes import NuScenes
    from occ3d_occlusion_explorer import SOTARayCaster
    
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=False)
    
    total_ok = 0
    total_warn = 0
    total_fail = 0
    
    for i, ref_path in enumerate(ref_files):
        # Estraiamo l'indice dal nome file (occlusion_sample_XXXX_...)
        basename = os.path.basename(ref_path)
        idx = int(basename.split("_")[2])
        
        print(f"[{i+1}/{len(ref_files)}] Sample {idx}...", end=" ", flush=True)
        
        try:
            caster = SOTARayCaster(nusc, idx)
            caster.generate_known_zone()
            caster.generate_box_shadows()
            caster.find_object_occlusion_wedges()
            caster.extract_occlusion_zones()
            
            # Salviamo in un file temporaneo per caricare il JSON
            tmp_path = os.path.join(args.ref_dir, "_tmp_check.json")
            caster.save_occlusions_to_json(tmp_path)
            
            with open(tmp_path) as f:
                cur_data = json.load(f)
            os.remove(tmp_path)
            
            diffs = compare_jsons(ref_path, cur_data)
            
            if not diffs:
                print("[OK] identico")
                total_ok += 1
            else:
                print("[DIFF]")
                for d in diffs:
                    print(f"    {d}")
                total_warn += 1
                
        except Exception as e:
            print(f"[ERR] {e}")
            total_fail += 1
    
    print(f"\n--- RISULTATI ---")
    print(f"OK:   {total_ok}/{len(ref_files)}")
    print(f"DIFF: {total_warn}/{len(ref_files)}")
    print(f"ERR:  {total_fail}/{len(ref_files)}")
    
    if total_warn == 0 and total_fail == 0:
        print("\n[PASS] Tutti i JSON sono identici al codice precedente.")
    elif total_fail > 0:
        print("\n[FAIL] Ci sono errori da investigare.")
    else:
        print("\n[WARN] Ci sono differenze, controlla i dettagli sopra.")
    
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
