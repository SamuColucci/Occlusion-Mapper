# Script di Benchmark e Misurazione dei Tempi di Esecuzione su GPU (benchmark_gpu.py)
# Esegue in sequenza e senza interazione utente l'intera pipeline del progetto:
#   1. RayCaster (estrazione geometrica delle zone d'ombra LiDAR)
#   2. Addestramento Rete Neurale Per-Zone su GPU (Focal Loss + Penalizzazione Semantica)
#   3. Inferenza Batch dell'Agente Neurale Per-Zone
#   4. Valutazione Comparativa e calcolo delle metriche di prestazione
# Misura i tempi di calcolo di ciascuna fase per quantificare lo speedup rispetto alla baseline CPU.

# Import dei moduli di sistema per la gestione dei processi figlio e tracciamento del tempo
import sys
import os
import time
import subprocess

# Determina il percorso della directory radice del progetto
BASE = os.path.dirname(os.path.abspath(__file__))
# Identifica l'eseguibile Python all'interno dell'ambiente virtuale .venv
PY  = os.path.join(BASE, ".venv", "Scripts", "python.exe")

# Definizione dei 4 passaggi sequenziali della pipeline di benchmark da cronometrare
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

# Stampa dell'intestazione principale del Benchmark su GPU
print("\n" + "=" * 70)
print("  BENCHMARK GPU - Pipeline Completa")
print("=" * 70)

# Avvia il timer per la misurazione della durata totale della pipeline
total_start = time.time()
times = []

# Scorre ed esegue in sequenza ciascuno dei passaggi definiti nella pipeline
for label, cmd in steps:
    print(f"\n>>> [{label}]")
    t0 = time.time()
    # Esegue il processo figlio attendendone il completamento
    result = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    # Memorizza la durata dell'esecuzione e lo stato di uscita (successo/errore)
    times.append((label, elapsed, result.returncode == 0))
    status = "OK" if result.returncode == 0 else "ERRORE"
    print(f"    [{status}] Completato in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Calcola il tempo totale cumulativo di esecuzione dell'intera pipeline
total = time.time() - total_start

# Stampa a schermo la tabella di riepilogo finale dei tempi di calcolo
print("\n" + "=" * 70)
print("  RIEPILOGO TEMPI")
print("=" * 70)
for label, elapsed, ok in times:
    stato = "[OK]" if ok else "[FAIL]"
    print(f"  {stato:<6} {label:<50} {elapsed:6.1f}s")
print(f"\n  TOTALE PIPELINE:  {total:.1f}s  ({total/60:.1f} minuti)")
print("=" * 70)
