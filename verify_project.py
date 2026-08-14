# Dashboard Centrale di Controllo e Verifica del Progetto Occlusion-Mapper.
import os
import sys
import subprocess
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(cmd_list):
    try:
        subprocess.run([sys.executable] + cmd_list)
    except Exception as e:
        print(f"\n[ERROR] Impossibile avviare lo script: {e}")
        input("\nPremi INVIO per tornare al menu...")

def main():
    while True:
        clear_screen()
        print("=" * 75)
        print("       DASHBOARD DI VALIDAZIONE PROGETTO: GESTIONE OCCLUSIONI")
        print("=" * 75)
        print(" Seleziona lo strumento di verifica da eseguire:")
        print(" " + "-" * 71)
        print("  1. Visualizzatore Interattivo Zone Occluse Grezze (RayCaster Output)")
        print("     (Mostra a schermo la mappa nuScenes e i coni d'ombra 2D BEV estratti)")
        print("")
        print("  2. Visualizzatore Interattivo Agente Bayesiano (Prior Condizionate CSV)")
        print("     (Mostra a schermo la mappa nuScenes nativa, sub-zone e probabilità)")
        print("")
        print("  3. Visualizzatore Interattivo Agente Per-Zone (CNN + 9 Scalari)")
        print("     (Mostra a schermo le stime probabilistiche del modello PerZoneModel)")
        print("")
        print("  4. Valutazione Metriche Ground Truth (Recall, Precision, F1-Score, IoU)")
        print("     (Valuta le predizioni rispetto agli ostacoli reali nuScenes 3D GT)")
        print("")
        print("  5. Esegui Calcolo Probabilità Condizionate Bayesiane (Tutti i frame)")
        print("     (Rigenera i file JSON probabilistici dell'Agente Bayesiano)")
        print("")
        print("  6. Esegui Inferenza Agente Per-Zone su Tutti i Frame")
        print("     (Rigenera i file JSON probabilistici della rete PerZoneModel)")
        print("")
        print("  7. Visualizzatore Comparativo Affiancato (Bayes vs Per-Zone HUD)")
        print("     (Confronta le due mappe contemporaneamente a schermo)")
        print("")
        print("  8. 📊 Menu Interattivo Valutazione e Confronto 2 Modelli (Score e Delta)")
        print("")
        print("  9. 🎯 Visualizzatore Interattivo Ground Truth (Reale nuScenes vs Sintetica)")
        print("     (Mostra a schermo gli ostacoli reali 3D GT e quelli sintetici dello Stress Test)")
        print("")
        print("  10. ⚡ Esegui Benchmark Prestazioni GPU (Misura Tempi Pipeline)")
        print("      (Esegue in sequenza ed analizza i tempi di: RayCaster -> Training -> Inferenza -> Valutazione)")
        print(" " + "-" * 71)
        print("  11. Esci")
        print("=" * 75)
        
        choice = input(" Inserisci la tua scelta [1-11]: ").strip()
        
        if choice == '1':
            print("\nAvvio Visualizzatore Interattivo Zone Occluse Grezze...")
            run_script([os.path.join("visualizzatori", "verify_runtime_bayes.py"), "extracted_occlusions"])
        elif choice == '2':
            print("\nAvvio Visualizzatore Interattivo Bayesiano...")
            run_script([os.path.join("visualizzatori", "verify_runtime_bayes.py")])
        elif choice == '3':
            print("\nAvvio Visualizzatore Interattivo Agente Per-Zone...")
            run_script([os.path.join("visualizzatori", "verify_runtime_per_zone.py")])
        elif choice == '4':
            print("\nAvvio Valutazione Metriche Ground Truth...")
            run_script([os.path.join("valutazione", "evaluate_focal_comparison.py")])
            input("\nPremi INVIO per tornare al menu...")
        elif choice == '5':
            print("\nAvvio Calcolo Probabilità Condizionate Bayesiane...")
            run_script([os.path.join("inferenza_agenti", "bayes_agent.py")])
            input("\nPremi INVIO per tornare al menu...")
        elif choice == '6':
            print("\nAvvio Inferenza Agente Per-Zone su tutti i frame...")
            run_script([os.path.join("inferenza_agenti", "neural_agent.py")])
            input("\nPremi INVIO per tornare al menu...")
        elif choice == '7':
            print("\nAvvio Visualizzatore Comparativo Affiancato...")
            run_script([os.path.join("visualizzatori", "verify_runtime_comparison.py")])
        elif choice == '8':
            print("\nAvvio Menu Interattivo di Valutazione e Confronto...")
            run_script([os.path.join("visualizzatori", "menu_evaluator.py")])
        elif choice == '9':
            print("\nAvvio Visualizzatore Interattivo Ground Truth (Reale vs Sintetica)...")
            run_script([os.path.join("visualizzatori", "verify_runtime_ground_truth.py")])
        elif choice == '10':
            print("\nAvvio Benchmark Prestazioni GPU e Misura Tempi...")
            run_script(["benchmark_gpu.py"])
            input("\nBenchmark completato. Premi INVIO per tornare al menu...")
        elif choice == '11':
            print("\nUscita dalla Dashboard. Arrivederci!\n")
            sys.exit(0)
        else:
            print("\n[WARNING] Scelta non valida! Inserisci un numero da 1 a 11.")
            time.sleep(1.5)

if __name__ == "__main__":
    main()
