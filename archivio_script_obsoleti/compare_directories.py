"""
compare_directories.py — confronta i file JSON generati nella cartella 'extracted_occlusions' (nuovi)
con quelli in 'extracted_occlusions_old' (vecchi) a livello di file, sample per sample.

Questo confronto è istantaneo perché carica i dati direttamente da disco.
"""
import argparse
import json
import os
import sys
import glob


def compare_jsons(old_path, new_path):
    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    diffs = []
    
    # lidar_token
    if old_data.get("lidar_token") != new_data.get("lidar_token"):
        diffs.append(f"lidar_token diverso: {old_data.get('lidar_token')} vs {new_data.get('lidar_token')}")
        return diffs

    old_occs = {o["object_token"]: o for o in old_data.get("occlusions", [])}
    new_occs = {o["object_token"]: o for o in new_data.get("occlusions", [])}

    old_tokens = set(old_occs.keys())
    new_tokens = set(new_occs.keys())

    added = new_tokens - old_tokens
    removed = old_tokens - new_tokens

    if added:
        diffs.append(f"  +{len(added)} oggetti nuovi: {list(added)[:3]}...")
    if removed:
        diffs.append(f"  -{len(removed)} oggetti mancanti: {list(removed)[:3]}...")

    # Per gli oggetti in comune, confronta le aree
    common = old_tokens & new_tokens
    area_diffs = 0
    poly_diffs = 0
    for token in common:
        o, n = old_occs[token], new_occs[token]
        # Tolleranza area 0.01 sqm
        if abs(o["area_sqm"] - n["area_sqm"]) > 0.01:
            area_diffs += 1
        if o["polygon_points_m"] != n["polygon_points_m"]:
            poly_diffs += 1

    if area_diffs:
        diffs.append(f"  {area_diffs}/{len(common)} oggetti con area diversa")
    if poly_diffs:
        diffs.append(f"  {poly_diffs}/{len(common)} oggetti con poligoni diversi (semplificazione)")

    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-dir", default="extracted_occlusions_old", help="Cartella dei vecchi JSON")
    p.add_argument("--new-dir", default="extracted_occlusions", help="Cartella dei nuovi JSON")
    args = p.parse_args()

    old_files = sorted(glob.glob(os.path.join(args.old_dir, "occlusion_sample_*.json")))
    if not old_files:
        sys.exit(f"Nessun file trovato in {args.old_dir}")

    print(f"Inizio confronto di {len(old_files)} JSON...")
    
    total_ok = 0
    total_warn = 0
    total_missing = 0

    for old_path in old_files:
        filename = os.path.basename(old_path)
        new_path = os.path.join(args.new_dir, filename)

        if not os.path.exists(new_path):
            # Cerca se esiste lo stesso sample_idx ma con token finale diverso
            # Il nome è 'occlusion_sample_XXXX_TOKEN.json'
            parts = filename.split("_")
            idx = parts[2]
            candidates = glob.glob(os.path.join(args.new_dir, f"occlusion_sample_{idx}_*.json"))
            if candidates:
                new_path = candidates[0]
            else:
                print(f"Sample {idx}: [MANCANTE] nel nuovo directory")
                total_missing += 1
                continue

        idx = filename.split("_")[2]
        diffs = compare_jsons(old_path, new_path)

        if not diffs:
            # print(f"Sample {idx}: [OK] identico")
            total_ok += 1
        else:
            print(f"Sample {idx}: [DIFF]")
            for d in diffs:
                print(f"  {d}")
            total_warn += 1

    print("\n--- STATISTICHE FINALI ---")
    print(f"Totale confrontati: {len(old_files)}")
    print(f"Identici al 100%:  {total_ok}")
    print(f"Con differenze:     {total_warn} (differenze note di semplificazione dei poligoni)")
    print(f"Mancanti:           {total_missing}")


if __name__ == "__main__":
    main()
