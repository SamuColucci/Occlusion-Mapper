# ==============================================================================
# LAUNCHER PARALLELO: AVVIO CONTEMPORANEO DEI DUE VISUALIZZATORI TESI
# File: visualizzatori/avvia_entrambi.py
#
# Apre contemporaneamente in due finestre separate e indipendenti:
# 1. Visualizzatore Raycasting Occlusioni (Paper Scientifico)
# 2. Visualizzatore Ufficiale nuScenes (Devkit Explorer)
# ==============================================================================

import os
import sys
import argparse
import threading
import subprocess

def terminal_listener(root_dir):
    """Ascolta l'input dal terminale per permettere di digitare qualsiasi frame in tempo reale."""
    sync_file = os.path.join(root_dir, "scratch", "sync_frame.txt")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            val = int(line)
            target = (val - 1) if val >= 1 else val
            os.makedirs(os.path.dirname(sync_file), exist_ok=True)
            with open(sync_file, "w") as f:
                f.write(f"{target},terminal")
            print(f"\n>>> [SALTO FRAME] Entrambi i visualizzatori sincronizzati al frame: {val} (indice {target})\n")
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Avvia contemporaneamente i due visualizzatori tesi in parallelo")
    parser.add_argument("frame", nargs="?", type=int, default=None, help="Numero del frame iniziale (1-404, es. 17)")
    parser.add_argument("--frame", "-f", type=int, default=None, dest="frame_opt", help="Numero del frame iniziale")
    parser.add_argument("--gt", action="store_true", help="Usa il visualizzatore Ground Truth (vis_ground_truth_occlusioni.py)")
    parser.add_argument("--neural", action="store_true", help="Usa il visualizzatore Inferenza Neurale Live (vis_inferenza_neurale.py)")
    parser.add_argument("--inputs", "--input", action="store_true", dest="inputs_mode", help="Usa il visualizzatore Input Multimodali & Rete Ausiliaria (vis_input_rete_neurale.py)")
    parser.add_argument("--bayes", action="store_true", help="Usa il visualizzatore Probabilità Bayesiana (vis_probabilita_bayes.py)")
    parser.add_argument("--eval", "--metrics", action="store_true", dest="eval_mode", help="Usa la Dashboard Prestazioni & Metriche (vis_valutazione_prestazioni.py)")
    parser.add_argument("--viewer", choices=["raycasting", "gt", "neural", "inputs", "bayes", "eval"], default=None, help="Scegli quale visualizzatore BEV affiancare a nuScenes official")
    args = parser.parse_args()

    init_frame = args.frame if args.frame is not None else args.frame_opt
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_exe = sys.executable

    if args.eval_mode or args.viewer == "eval":
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_valutazione_prestazioni.py")
    elif args.inputs_mode or args.viewer == "inputs":
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_input_rete_neurale.py")
    elif args.neural or args.viewer == "neural":
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_inferenza_neurale.py")
    elif args.bayes or args.viewer == "bayes":
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_probabilita_bayes.py")
    elif args.gt or args.viewer == "gt":
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_ground_truth_occlusioni.py")
    else:
        script_bev = os.path.join(root_dir, "visualizzatori", "vis_raycasting_occlusioni.py")

    script_official = os.path.join(root_dir, "visualizzatori", "vis_official_nuscenes.py")

    cmd1 = [python_exe, script_bev]
    cmd2 = [python_exe, script_official]
    if init_frame is not None:
        cmd1.append(str(init_frame))
        cmd2.append(str(init_frame))

    print("\n" + "=" * 80)
    print("   AVVIO PARALLELO VISUALIZZATORI TESI (SINCRONIZZATI)")
    print("=" * 80)
    print(f">>> Frame iniziale impostato: {init_frame if init_frame is not None else 1}")
    print(f">>> 1. Lancio: {os.path.basename(script_bev)} ...")
    p1 = subprocess.Popen(cmd1, cwd=root_dir)

    print(f">>> 2. Lancio: {os.path.basename(script_official)} ...")
    p2 = subprocess.Popen(cmd2, cwd=root_dir)

    print("\n" + "-" * 80)
    print("[SISTEMA DI SALTO FRAME RAPIDO ATTIVO]")
    print("  * DA QUESTO TERMINALE: digita un numero (es. 17) e premi INVIO per saltare")
    print("  * DALLA FINESTRA GUI:  clicca sulla casella 'Vai al frame' in basso e premi INVIO")
    print("  * Entrambe le finestre sono sincronizzate in tempo reale!")
    print("  * Premi Ctrl+C qui per chiuderle entrambe contemporaneamente.")
    print("-" * 80 + "\n")

    # Avvio thread in ascolto del terminale
    t = threading.Thread(target=terminal_listener, args=(root_dir,), daemon=True)
    t.start()

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        print("\nChiusura dei visualizzatori in corso...")
        p1.terminate()
        p2.terminate()

if __name__ == "__main__":
    main()
