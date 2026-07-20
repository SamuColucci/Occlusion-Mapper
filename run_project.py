"""
Pannello di Controllo ed Esecuzione della Pipeline (Run).

Questo script funge da menu interattivo centralizzato per avviare
i calcoli geometrici, addestrare il modello ed esportare le probabilità 
dei due agenti in modo sequenziale automatizzato.
"""
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
        # Esegue il comando attendendo il completamento
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
        "extracted_occlusions_probabilities",
        "extracted_occlusions_probabilities_near",
        "extracted_occlusions_neural",
        "extracted_occlusions_neural_near"
    ]
    for f in folders:
        if os.path.exists(f):
            try:
                shutil.rmtree(f)
                print(f"  -> Rimossa cartella temporanea: {f}")
            except Exception as e:
                print(f"  -> Impossibile rimuovere {f}: {e}")
    # Rimuove il modello
    if os.path.exists("best_model.pth"):
        try:
            os.remove("best_model.pth")
            print("  -> Rimosso file dei pesi del modello 'best_model.pth'")
        except Exception as e:
            print(f"  -> Impossibile rimuovere best_model.pth: {e}")
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
        print("  1. Estrazione Geometrica delle Occlusioni (LiDAR batch completo)")
        print("     (Genera i coni d'ombra 2D BEV nei JSON a partire da NuScenes)")
        print("")
        print("  2. Esegui Pipeline Completa APPROCCIO A (Agente Bayesiano)")
        print("     (Esegue in sequenza: Estrazione Prior CSV -> Tracker Persistenza -> Agente)")
        print("")
        print("  3. Esegui Pipeline Completa APPROCCIO B (Agente Neurale)")
        print("     (Esegue in sequenza: Addestramento UNet in RAM -> Esportazione Batch Neurale)")
        print("")
        print("  4. Addestra Rete Neurale UNet (Solo Addestramento)")
        print("     (Rigenera il file con i pesi 'best_model.pth' in base al dataset in RAM)")
        print("")
        print("  5. Esporta Mappe di Probabilità Neurale (Solo Inferenza)")
        print("     (Genera la cartella dei JSON probabilistici della UNet)")
        print("")
        print("  6. Esegui Pipeline Completa (RUN ALL)")
        print("     (Esegue in sequenza e genera TUTTI i dataset geometrici e probabilistici)")
        print(" " + "-" * 71)
        print("  7. RESET TOTALE: Pulisci Risultati Precedenti")
        print("     (Elimina le cartelle di output e il modello addestrato per ripartire)")
        print("  8. Esci")
        print("=" * 75)
        
        choice = input(" Inserisci la tua scelta [1-8]: ").strip()
        
        if choice == '1':
            clear_screen()
            print("ESTRAZIONE OCCLUSIONI GEOMETRICHE DA NUSCENES...")
            run_script(["occ3d_occlusion_explorer.py", "--mode", "batch"], "Space Carving LiDAR")
            input("\nElaborazione completata. Premi INVIO per tornare al menu...")
            
        elif choice == '2':
            clear_screen()
            print("AVVIO PIPELINE COMPLETA APPROCCIO A (BAYES)...")
            # 1. Estrae le priorità CSV
            if not run_script(["bayesian_prior_extractor.py"], "Estrazione Priorità Condizionate CSV"):
                continue
            # 2. Esegue il tracciamento della persistenza
            if not run_script(["temporal_persistence_tracker.py"], "Tracking Temporale Persistenza Ombre"):
                continue
            # 3. Esegue l'agente probabilistico finale (Diretto)
            if not run_script(["bayesian_occlusion_agent.py", "--mode", "batch"], "Generazione Dataset Bayes Diretto"):
                continue
            # 4. Esegue l'agente probabilistico finale (Near-Miss)
            if not run_script(["bayesian_occlusion_agent.py", "--mode", "batch", "--use-near-miss"], "Generazione Dataset Bayes Near-Miss"):
                continue
            input("\nApproccio A (Diretto + Near-Miss) completato con successo. Premi INVIO per tornare al menu...")
            
        elif choice == '3':
            clear_screen()
            print("AVVIO PIPELINE COMPLETA APPROCCIO B (NEURALE)...")
            # 1. Addestra il modello
            if not run_script(["train_neural_agent.py"], "Training della Rete UNet"):
                continue
            # 2. Esporta i risultati nei JSON (Diretto)
            if not run_script(["neural_occlusion_agent.py", "--mode", "batch"], "Esportazione Batch Inferenza UNet Diretta"):
                continue
            # 3. Esporta i risultati nei JSON (Near-Miss)
            if not run_script(["neural_occlusion_agent.py", "--mode", "batch", "--use-near-miss"], "Esportazione Batch Inferenza UNet Near-Miss"):
                continue
            input("\nApproccio B (Diretto + Near-Miss) completato con successo. Premi INVIO per tornare al menu...")
            
        elif choice == '4':
            clear_screen()
            print("AVVIO ADDESTRAMENTO ISOLATO DELLA RETE...")
            run_script(["train_neural_agent.py"], "Training della Rete UNet")
            input("\nElaborazione completata. Premi INVIO per tornare al menu...")
            
        elif choice == '5':
            clear_screen()
            print("AVVIO ESPORTAZIONE ISOLATA INFERENZA NEURALE...")
            run_script(["neural_occlusion_agent.py", "--mode", "batch"], "Esportazione Batch Inferenza UNet Diretta")
            run_script(["neural_occlusion_agent.py", "--mode", "batch", "--use-near-miss"], "Esportazione Batch Inferenza UNet Near-Miss")
            input("\nElaborazione completata. Premi INVIO per tornare al menu...")
            
        elif choice == '6':
            clear_screen()
            print("AVVIO PIPELINE DI RUN ALL COMPLETA...")
            # 1. Estrazione geometrica
            if run_script(["occ3d_occlusion_explorer.py", "--mode", "batch"], "Space Carving LiDAR"):
                # 2. Pipeline Bayes (Prior + Tracker + Agente Bayes Diretto + Agente Bayes Near-Miss)
                print("\n--- [2/3] Approccio A (Bayes) ---")
                if run_script(["bayesian_prior_extractor.py"], "Estrazione Priorità Condizionate CSV"):
                    if run_script(["temporal_persistence_tracker.py"], "Tracking Temporale Persistenza Ombre"):
                        run_script(["bayesian_occlusion_agent.py", "--mode", "batch"], "Generazione Dataset Bayes Diretto")
                        run_script(["bayesian_occlusion_agent.py", "--mode", "batch", "--use-near-miss"], "Generazione Dataset Bayes Near-Miss")
                
                # 3. Pipeline Neurale (Training UNet + Inferenza Diretta + Inferenza Near-Miss)
                print("\n--- [3/3] Approccio B (Neurale) ---")
                if run_script(["train_neural_agent.py"], "Training della Rete UNet"):
                    run_script(["neural_occlusion_agent.py", "--mode", "batch"], "Esportazione Batch Inferenza UNet Diretta")
                    run_script(["neural_occlusion_agent.py", "--mode", "batch", "--use-near-miss"], "Esportazione Batch Inferenza UNet Near-Miss")
                    
            input("\nPipeline RUN ALL completata con successo! Premi INVIO per tornare al menu...")
            
        elif choice == '7':
            clear_screen()
            clean_data()
            
        elif choice == '8':
            print("\nChiusura della dashboard di esecuzione.")
            break
        else:
            print("\n[WARNING] Scelta non valida! Inserisci un numero da 1 a 8.")
            time_sleep = 1.5
            import time
            time.sleep(time_sleep)

if __name__ == "__main__":
    main()
