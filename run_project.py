# Pannello di Controllo ed Esecuzione della Pipeline (Run).
# Questo script funge da menu interattivo centralizzato per avviare
# i calcoli geometrici del RayCaster, l'Agente Bayesiano, l'addestramento UNet 2D e l'Agente Per-Zone.

import os
import sys
import time
import subprocess
import shutil

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(cmd_list, description=""):
    if description:
        print(f"\n>>> Avvio fase: {description}...")
    try:
        start_t = time.time()
        res = subprocess.run([sys.executable] + cmd_list, check=True)
        if res.returncode == 0:
            print(f"[SUCCESS] Fase completata in {time.time() - start_t:.1f} secondi.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Errore critico durante l'esecuzione del comando: {e}")
        input("\nPremi INVIO per tornare al menu...")
        return False
    except Exception as e:
        print(f"\n[ERROR] Impossibile avviare il processo: {e}")
        input("\nPremi INVIO per tornare al menu...")
        return False

def clean_data():
    print("\nPulizia delle cartelle dei risultati in corso...")
    folders = [
        "extracted_occlusions",
        "extracted_occlusions_bayes",
        "extracted_occlusions_neural",
        "extracted_occlusions_per_zone"
    ]
    for f in folders:
        if os.path.exists(f):
            try:
                shutil.rmtree(f)
                print(f"  -> Rimossa cartella temporanea: {f}")
            except Exception as e:
                print(f"  -> Impossibile rimuovere {f}: {e}")

    for ckpt in ["unet_occlusion_checkpoint.pth", "per_zone_checkpoint.pth"]:
        if os.path.exists(ckpt):
            try:
                os.remove(ckpt)
                print(f"  -> Rimosso checkpoint '{ckpt}'")
            except Exception as e:
                print(f"  -> Impossibile rimuovere {ckpt}: {e}")
    print("Pulizia completata con successo.")
    input("\nPremi INVIO per tornare al menu...")

def main():
    while True:
        clear_screen()
        print("=" * 75)
        print("      PANNELLO DI CONTROLLO PIPELINE: CALCOLI ED ELABORAZIONE DATI")
        print("=" * 75)
        print(" Seleziona l'elaborazione o la fase da lanciare:")
        print(" " + "-" * 71)
        print("  1. Estrazione Geometrica delle Zone Occluse (LiDAR RayCasting)")
        print("     (Genera i coni d'ombra 2D BEV nei JSON a partire da NuScenes)")
        print("")
        print("  2. Esegui Calcolo Probabilità Condizionate Bayesiane")
        print("     (Genera/aggiorna i JSON probabilistici dell'Agente Bayesiano)")
        print("")
        print("  3. Addestra Rete Neurale UNet 2D & Esegui Inferenza (Pixel-wise BEV)")
        print("     (Addestra il modello UNet 2D e genera le predizioni neurali JSON)")
        print("")
        print("  4. Addestra Rete Neurale Per-Zone & Esegui Inferenza (Patch 64x64 + Scalari)")
        print("     (Addestra il modello ibrido PerZoneModel e genera le predizioni JSON)")
        print("")
        print("  5. Esegui Pipeline Completa (RUN ALL)")
        print("     (Esegue in sequenza: RayCasting -> Probabilità Bayes -> Training UNet 2D & Per-Zone)")
        print(" " + "-" * 71)
        print("  6. RESET TOTALE: Pulisci Risultati Precedenti")
        print("     (Elimina le cartelle di output ed i checkpoint per ripartire da zero)")
        print("  7. Esci")
        print("=" * 75)
        
        choice = input(" Inserisci la tua scelta [1-7]: ").strip()
        
        if choice == '1':
            clear_screen()
            print("ESTRAZIONE OCCLUSIONI GEOMETRICHE DA NUSCENES...")
            run_script(["estrazione_zone_occluse.py"], "RayCasting e Space Carving LiDAR")
            input("\nElaborazione completata. Premi INVIO per tornare al menu...")
            
        elif choice == '2':
            clear_screen()
            print("AVVIO CALCOLO PROBABILITÀ CONDIZIONATE BAYESIANE...")
            run_script(["conditional_probability_dataset.py"], "Calcolo Bayesiano Probabilità Condizionate")
            input("\nCalcolo Bayesiano completato con successo. Premi INVIO per tornare al menu...")
            
        elif choice == '3':
            clear_screen()
            print("ADDESTRAMENTO RETE NEURALE UNET 2D & GENERAZIONE PREDIZIONI...")
            if run_script(["train_neural.py"], "Addestramento Rete Neurale UNet 2D"):
                run_script(["neural_occlusion_agent.py"], "Generazione Inferenza Neurale UNet 2D")
            input("\nAddestramento ed Inferenza UNet 2D completati con successo. Premi INVIO per tornare al menu...")
            
        elif choice == '4':
            clear_screen()
            print("ADDESTRAMENTO RETE NEURALE PER-ZONE (PATCH 64x64 + SCALARI)...")
            if run_script(["train_per_zone.py"], "Addestramento Modello Ibrido Per-Zone"):
                run_script(["per_zone_occlusion_agent.py"], "Generazione Inferenza Neurale Per-Zone")
            input("\nAddestramento ed Inferenza Per-Zone completati con successo. Premi INVIO per tornare al menu...")
            
        elif choice == '5':
            clear_screen()
            print("AVVIO PIPELINE DI RUN ALL COMPLETA...")
            if run_script(["estrazione_zone_occluse.py"], "RayCasting LiDAR"):
                print("\n--- [2/4] Approccio Bayesiano ---")
                run_script(["conditional_probability_dataset.py"], "Calcolo Probabilità Condizionate")
                print("\n--- [3/4] Approccio Neurale UNet 2D ---")
                if run_script(["train_neural.py"], "Addestramento UNet 2D"):
                    run_script(["neural_occlusion_agent.py"], "Inferenza UNet 2D")
                print("\n--- [4/4] Approccio Neurale Per-Zone ---")
                if run_script(["train_per_zone.py"], "Addestramento Per-Zone Model"):
                    run_script(["per_zone_occlusion_agent.py"], "Inferenza Per-Zone Model")
                    
            input("\nPipeline RUN ALL completata con successo! Premi INVIO per tornare al menu...")
            
        elif choice == '6':
            clear_screen()
            clean_data()
            
        elif choice == '7':
            print("\nChiusura del pannello di esecuzione.")
            break
        else:
            print("\n[WARNING] Scelta non valida! Inserisci un numero da 1 a 7.")
            time.sleep(1.5)

if __name__ == "__main__":
    main()
