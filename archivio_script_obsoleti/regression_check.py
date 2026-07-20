"""
regression_check.py — snapshot & compare for occ3d_occlusion_explorer refactors.

Usage:
  1. BEFORE refactoring:   python regression_check.py --snapshot
  2. AFTER refactoring:    python regression_check.py --check

Both run the pipeline on sample 0 (cheapest in mini) and compare every
intermediate grid + the final JSON output byte-for-byte.

ponytail: uses sample 0 because it's the first available in nuscenes-mini.
If your dataset starts elsewhere, pass --sample-idx N.
"""
import argparse
import json
import os
import sys
import numpy as np

SNAP_DIR = os.path.join(os.path.dirname(__file__), "scratch", "regression_snapshot")

GRIDS = ["grid", "box_shadows", "internal_shadows", "occluded_final"]


def _run_pipeline(sample_idx, use_old=False):
    from nuscenes.nuscenes import NuScenes
    if use_old:
        from old_explorer import SOTARayCaster
    else:
        from occ3d_occlusion_explorer import SOTARayCaster

    nusc = NuScenes(version="v1.0-mini", dataroot="./nuscenes", verbose=False)
    c = SOTARayCaster(nusc, sample_idx)
    c.generate_known_zone()
    c.generate_box_shadows()
    c.find_object_occlusion_wedges()
    c.extract_occlusion_zones()
    c.save_occlusions_to_json(os.path.join(SNAP_DIR, "_tmp_occlusions.json"))
    return c


def snapshot(sample_idx):
    os.makedirs(SNAP_DIR, exist_ok=True)
    c = _run_pipeline(sample_idx, use_old=True)

    for name in GRIDS:
        arr = getattr(c, name)
        # ponytail: cast bool grids to uint8 so dtype never mismatches
        np.save(os.path.join(SNAP_DIR, f"{name}.npy"), np.asarray(arr, dtype=np.uint8))

    # JSON: re-dump sorted so key order doesn't matter
    with open(os.path.join(SNAP_DIR, "_tmp_occlusions.json")) as f:
        data = json.load(f)
    with open(os.path.join(SNAP_DIR, "occlusions_ref.json"), "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    os.remove(os.path.join(SNAP_DIR, "_tmp_occlusions.json"))
    print(f"[OK] Snapshot salvato in {SNAP_DIR}")


def check(sample_idx):
    if not os.path.isdir(SNAP_DIR):
        sys.exit(f"[FAIL] Nessuno snapshot trovato in {SNAP_DIR}. Lancia prima --snapshot.")

    c = _run_pipeline(sample_idx)
    ok = True

    for name in GRIDS:
        ref = np.load(os.path.join(SNAP_DIR, f"{name}.npy"))
        cur = np.asarray(getattr(c, name), dtype=np.uint8)
        if ref.shape != cur.shape:
            print(f"[FAIL] {name}: shape diversa -- ref {ref.shape} vs cur {cur.shape}")
            ok = False
        elif not np.array_equal(ref, cur):
            diff = np.count_nonzero(ref != cur)
            total = ref.size
            print(f"[FAIL] {name}: {diff}/{total} voxel diversi ({diff/total*100:.3f}%)")
            ok = False
        else:
            print(f"[OK] {name}: identico")

    # JSON comparison
    with open(os.path.join(SNAP_DIR, "occlusions_ref.json")) as f:
        ref_json = json.load(f)
    with open(os.path.join(SNAP_DIR, "_tmp_occlusions.json")) as f:
        cur_json = json.load(f)
    os.remove(os.path.join(SNAP_DIR, "_tmp_occlusions.json"))

    if json.dumps(ref_json, sort_keys=True) == json.dumps(cur_json, sort_keys=True):
        print("[OK] JSON output: identico")
    else:
        print("[FAIL] JSON output: differenze trovate")
        # mostra quali oggetti mancano/cambiano
        ref_tokens = {o["object_token"] for o in ref_json.get("occlusions", [])}
        cur_tokens = {o["object_token"] for o in cur_json.get("occlusions", [])}
        added = cur_tokens - ref_tokens
        removed = ref_tokens - cur_tokens
        if added:
            print(f"   + aggiunti: {added}")
        if removed:
            print(f"   - rimossi: {removed}")
        if not added and not removed:
            print("   (stessi oggetti, ma valori diversi in qualche campo)")
        ok = False

    if ok:
        print("\n[PASS] Tutti i risultati sono identici. Refactor sicuro.")
    else:
        print("\n[WARN] Regressione rilevata. Controlla le differenze sopra.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="store_true", help="Salva i risultati di riferimento")
    p.add_argument("--check", action="store_true", help="Confronta con lo snapshot salvato")
    p.add_argument("--sample-idx", type=int, default=0)
    args = p.parse_args()

    if args.snapshot == args.check:
        sys.exit("Specifica --snapshot OPPURE --check")

    if args.snapshot:
        snapshot(args.sample_idx)
    else:
        check(args.sample_idx)
