"""
Dashboard Centrale di Controllo e Verifica del Progetto.

Questo script funge da menu interattivo centralizzato per avviare 
qualunque strumento di visualizzazione o validazione del progetto
senza dover digitare comandi complessi da terminale.
"""
import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(cmd_list):
    try:
        # Avvia lo script come sottoprocesso mantenendo il terminale interattivo
        subprocess.run([sys.executable] + cmd_list)
    except Exception as e:
        print(f"\n[ERROR] Impossibile avviare lo script: {e}")
        input("\nPremi INVIO per tornare al menu...")

def main():
    while True:
        clear_screen()
        print("=" * 70)
        print("       DASHBOARD DI VALIDAZIONE PROGETTO: GESTIONE OCCLUSIONI")
        print("=" * 70)
        print(" Seleziona lo strumento di verifica da eseguire:")
        print(" " + "-" * 66)
        print("  1. Visualizzatore Geometrico delle Occlusioni BEV")
        print("     (Mostra le zone d'ombra LiDAR classificate in 2D)")
        print("")
        print("  2. Test di Integrità Geometrico Automatico")
        print("     (Analisi di parsing, auto-intersezioni e sovrapposizioni errate)")
        print("")
        print("  3. Validazione degli Hit Reali / Near-Miss (Ground Truth)")
        print("     (Mostra le situazioni reali in cui un oggetto era vicino all'ombra)")
        print("")
        print("  4. Agente Bayesiano a Runtime")
        print("     (Visualizza le stime Bayes + Rischio Temporale stocastico)")
        print("")
        print("  5. Agente Neurale a Runtime")
        print("     (Visualizza le stime multi-classe predette dalla UNet PyTorch)")
        print("")
        print("  6. Confronto Sincronizzato Affiancato (Bayes vs UNet)")
        print("     (La visualizzazione comparativa con calcolo dei Delta a terminale)")
        print("")
        print("  7. Visualizza Report Storico Cumulativo (CSV in Browser)")
        print("     (Apre un'interfaccia web interattiva con ricerca e ordinamento)")
        print(" " + "-" * 66)
        print("  8. Esci")
        print("=" * 70)
        
        choice = input(" Inserisci la tua scelta [1-8]: ").strip()
        
        if choice == '1':
            print("\nAvvio Visualizzatore Geometrico...")
            run_script([os.path.join("visualizzatori", "verify_geometry_pipeline.py"), "--mode", "visual"])
        elif choice == '2':
            print("\nAvvio Test di Integrità...")
            run_script([os.path.join("visualizzatori", "verify_geometry_pipeline.py"), "--mode", "check"])
            input("\nPremi INVIO per tornare al menu...")
        elif choice == '3':
            print("\nAvvio Validazione Hit Reali...")
            run_script([os.path.join("visualizzatori", "verify_bayesian_pipeline.py")])
        elif choice == '4':
            print("\nAvvio Agente Bayesiano...")
            run_script([os.path.join("visualizzatori", "verify_runtime_bayes.py")])
        elif choice == '5':
            print("\nAvvio Agente Neurale...")
            run_script([os.path.join("visualizzatori", "verify_runtime_neural.py")])
        elif choice == '6':
            print("\nAvvio Confronto Sincronizzato...")
            run_script([os.path.join("visualizzatori", "verify_runtime_comparison.py")])
        elif choice == '7':
            print("\nApertura Report Storico nel browser...")
            run_script([os.path.join("visualizzatori", "apri_storico.py")])
            import time
            time.sleep(1.0)
        elif choice == '8':
            print("\nChiusura della dashboard.")
            break
        else:
            print("\n[WARNING] Scelta non valida! Inserisci un numero da 1 a 8.")
            time_sleep = 1.5
            import time
            time.sleep(time_sleep)

if __name__ == "__main__":
    main()
