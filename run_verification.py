import os
import sys
import subprocess
import traceback

def run_test(name, cmd_list):
    print(f"\n==================================================", flush=True)
    print(f" [TESTING] {name}", flush=True)
    print(f"==================================================", flush=True)
    try:
        res = subprocess.run([sys.executable] + cmd_list, text=True, check=True)
        print(f"[SUCCESS] {name} completato con esito positivo!", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Errore durante {name}: exit code {e.returncode}")
        if e.stdout:
            print("--- STDOUT ---")
            print(e.stdout[-1000:])
        if e.stderr:
            print("--- STDERR ---")
            print(e.stderr[-1000:])
        return False
    except Exception as ex:
        print(f"[FAIL] Eccezione durante {name}: {ex}")
        traceback.print_exc()
        return False

def main():
    print("===============================================================", flush=True)
    print("   AVVIO RIGENERAZIONE E VERIFICA COMPLETA DELL'INTERO PROGETTO", flush=True)
    print("===============================================================", flush=True)
    
    tests = [
        ("Compilazione Modulo Architettura Neurale", ["-m", "py_compile", "architettura_neurale/per_zone_model.py", "architettura_neurale/loss_functions.py"]),
        ("Compilazione Modulo Dataset Adapter", ["-m", "py_compile", "dataset_adapter/dataset_generator_per_zone.py", "dataset_adapter/dataset_generator_surrounding.py", "dataset_adapter/nuscenes_dataset_adapter.py", "dataset_adapter/adapter_dataset.py", "dataset_adapter/factory_dataset.py"]),
        ("Compilazione Modulo Ground Truth", ["-m", "py_compile", "ground_truth/ground_truth_extractor.py", "ground_truth/ground_truth_extractor_synthetic.py"]),
        ("Compilazione Modulo RayCaster 3D", ["-m", "py_compile", "raycaster/ray_caster.py"]),
        ("Compilazione Modulo Inferenza Agenti", ["-m", "py_compile", "inferenza_agenti/neural_agent.py", "inferenza_agenti/bayes_agent.py", "inferenza_agenti/bayes_zone_calculator.py"]),
        ("Compilazione Moduli Addestramento", ["-m", "py_compile", "addestramento/train_per_zone.py", "addestramento/train_per_zone_semantica.py", "addestramento/train_per_zone_focal.py", "addestramento/train_per_zone_focal_semantica.py", "addestramento/train_per_zone_surrounding.py", "addestramento/train_per_zone_asl.py"]),
        
        ("1/6 Addestramento Modellazione BCE Baseline", ["addestramento/train_per_zone.py"]),
        ("2/6 Addestramento Modello BCE + Penalizzazione Semantica", ["addestramento/train_per_zone_semantica.py"]),
        ("3/6 Addestramento Modello Focal Loss Standard", ["addestramento/train_per_zone_focal.py"]),
        ("4/6 Addestramento Modello Focal Loss + Penalizzazione Semantica", ["addestramento/train_per_zone_focal_semantica.py"]),
        ("5/6 Addestramento Modello Contesto Esterno (Ring Semantics)", ["addestramento/train_per_zone_surrounding.py"]),
        ("6/6 Addestramento Modello Asymmetric Loss (CVPR 2021)", ["addestramento/train_per_zone_asl.py"]),
        
        ("Inferenza Agente Neurale su tutti i Fotogrammi", ["inferenza_agenti/neural_agent.py"]),
        ("Benchmark Metrica GT (7 Modelli a Confronto)", ["valutazione/evaluate_focal_comparison.py"]),
        ("Stress Test Ground Truth Sintetica (6 Classi)", ["valutazione/evaluate_synthetic_gt_comparison.py"]),
        ("Generazione Tabella Risultati Completa Markdown", ["valutazione/export_results_to_md.py"]),
    ]
    
    failed_tests = []
    for name, cmd in tests:
        ok = run_test(name, cmd)
        if not ok:
            failed_tests.append(name)
            
    print("\n" + "=" * 65, flush=True)
    if not failed_tests:
        print("[SUCCESS] TUTTI I TEST E L'ADDESTRAMENTO DEL PROGETTO SONO PASSATI AL 100% SENZA NESSUN ERRORE!", flush=True)
    else:
        print(f"[FAIL] TEST FALLITI ({len(failed_tests)}):")
        for ft in failed_tests:
            print(f" - {ft}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
