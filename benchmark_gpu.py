"""
benchmark_gpu.py - Pipeline completa non-interattiva per test GPU.
Esegue: RayCaster → Training (Focal+Sem, il miglior modello) → Inferenza → Tabella Risultati
Misura il tempo di ogni fase per confrontarlo con la baseline CPU.
"""
import sys, os, time, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
PY  = os.path.join(BASE, ".venv", "Scripts", "python.exe")

steps = [
    ("RayCaster (estrazione occlusioni LiDAR)",
     [PY, "estrazione_zone_occluse.py"]),

    ("Training Focal Loss + Penalizzazione Semantica (GPU)",
     [PY, "addestramento/train_per_zone_semantica.py"]),

    ("Inferenza Agente Neurale Per-Zone",
     [PY, "inferenza_agenti/neural_agent.py"]),

    ("Valutazione comparativa (tutti gli agenti)",
     [PY, "valutazione/evaluate_focal_comparison.py"]),
]

print("\n" + "=" * 70)
print("  BENCHMARK GPU - Pipeline Completa")
print("=" * 70)

total_start = time.time()
times = []
for label, cmd in steps:
    print(f"\n>>> [{label}]")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    times.append((label, elapsed, result.returncode == 0))
    status = "OK" if result.returncode == 0 else "ERRORE"
    print(f"    [{status}] Completato in {elapsed:.1f}s ({elapsed/60:.1f} min)")

total = time.time() - total_start
print("\n" + "=" * 70)
print("  RIEPILOGO TEMPI")
print("=" * 70)
for label, elapsed, ok in times:
    stato = "[OK]" if ok else "[FAIL]"
    print(f"  {stato:<6} {label:<50} {elapsed:6.1f}s")
print(f"\n  TOTALE PIPELINE:  {total:.1f}s  ({total/60:.1f} minuti)")
print("=" * 70)
