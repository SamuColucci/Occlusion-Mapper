# Guida all'Esecuzione Sequenziale della Pipeline

Questa guida illustra l'ordine corretto in cui eseguire gli script per calcolare la mappatura geometrica, estrarre le priorità probabilistiche dal dataset, tracciare la persistenza temporale e visualizzare in 2D BEV i profili di rischio stimati dai due agenti.

---

## ⚡ METODO RAPIDO CONSIGLIATO (Dashboard Interattiva)

Per semplificare l'elaborazione dei dati ed evitare di dover digitare comandi complessi, è stato creato uno script orchestratore centrale che gestisce l'intero ciclo di vita del progetto (calcoli, addestramento, esportazione e reset):
```powershell
python run_project.py
```
Eseguendo questo comando potrai lanciare l'intera pipeline o le singole fasi selezionando semplicemente un'opzione numerica a schermo.

In particolare:
*   **Opzione 6 - RUN ALL**: Avvia in sequenza l'intera pipeline ed esporta in automatico sia i risultati **Diretti** sia **Near-Miss** per entrambi gli approcci (Bayes e UNet), creando tutte e 4 le cartelle di output probabilistico.
*   **Opzione 7 - RESET TOTALE**: Pulisce ed elimina le 5 cartelle di risultati generati e i pesi del modello addestrato per ripartire in modo pulito.

Se invece preferisci comprendere o eseguire i singoli moduli passo-passo in modo manuale, segui l'ordine sequenziale descritto di seguito.

---

## 🗺️ Flusso di Esecuzione (Ordine Sequenziale Manuale)

Per ottenere il dataset probabilistico finale completo, esegui i moduli nell'ordine indicato di seguito:

### 1. Estrazione delle Occlusioni Geometriche (Batch)
Genera la griglia voxel 3D dello spazio noto ed estrae i poligoni d'ombra BEV per ogni frame di NuScenes.
```powershell
python occ3d_occlusion_explorer.py --mode batch
```
*   **Input**: Dataset NuScenes (`v1.0-mini`)
*   **Output**: Cartella `extracted_occlusions/` contenente i file JSON con la geometria delle ombre.

### 2. Estrazione delle Probabilità Bayesiane Storiche (Frequentista)
Incrocia le ombre geometriche estratte al passo 1 con le annotazioni reali (Ground Truth) di NuScenes per calcolare con quale frequenza statistica un oggetto si nasconde (o si trova in prossimità) dietro a un altro.
```powershell
python bayesian_prior_extractor.py
```
*   **Input**: Cartella `extracted_occlusions/` + Annotazioni NuScenes
*   **Output**: File [prior_bayesiane.csv](file:///c:/Users/samue/Desktop/Tirocinio/prior_bayesiane.csv) contenente la tabella delle probabilità condizionate statiche (Dirette e Near-Miss).

### 3. Tracciamento Temporale di Persistenza e Rischio
Ripercorre le scene in ordine cronologico calcolando la persistenza dell'occlusione (in secondi) e il conseguente *risk score* esponenziale cumulativo.
```powershell
python temporal_persistence_tracker.py
```
*   **Input**: Cartella `extracted_occlusions/` (legge e sovrascrive i JSON)
*   **Output**: JSON arricchiti con i campi `persistence_frames`, `persistence_seconds` e `risk_score`.

### 4. Generazione del Dataset Probabilistico di Runtime (Agente Bayesiano)
Fonde i dati storici (passo 2) e temporali (passo 3) tramite l'agente Bayesiano per stimare la probabilità di presenza semantica in ogni istante.
*   **Per generare le stime Dirette**:
    ```powershell
    python bayesian_occlusion_agent.py --mode batch
    ```
    *   **Output**: Cartella `extracted_occlusions_probabilities/`
*   **Per generare le stime Near-Miss**:
    ```powershell
    python bayesian_occlusion_agent.py --mode batch --use-near-miss
    ```
    *   **Output**: Cartella `extracted_occlusions_probabilities_near/`

### 5. Addestramento e Inferenza (Agente Neurale UNet)
*   **Addestramento della UNet**:
    ```powershell
    python train_neural_agent.py
    ```
    *   **Output**: File con i pesi `best_model.pth`.
*   **Esportazione Inferenza Diretta**:
    ```powershell
    python neural_occlusion_agent.py --mode batch
    ```
    *   **Output**: Cartella `extracted_occlusions_neural/`
*   **Esportazione Inferenza Near-Miss**:
    ```powershell
    python neural_occlusion_agent.py --mode batch --use-near-miss
    ```
    *   **Output**: Cartella `extracted_occlusions_neural_near/`

---

## 🔍 Strumenti di Visualizzazione e Test (Dashboard Unificata)

Per visualizzare e validare le stime a schermo, lancia la dashboard dei risultati:
```powershell
python verify_project.py
```

### Strumenti principali disponibili:

*   **Opzione 4 - Agente Bayesiano a Runtime** ed **Opzione 5 - Agente Neurale a Runtime**: all'avvio ti verrà chiesto interattivamente a console se caricare la visualizzazione dei **Diretti** o dei **Near-Miss**. I visualizzatori mostrano graficamente e in tempo reale la mappa delle superfici semantiche (Strada, Marciapiede, Prato) ed applicano la modulazione del rischio e delle probabilità di classe.
*   **Opzione 6 - Confronto Sincronizzato Affiancato**: avvia la visualizzazione comparativa definitiva a due colonne che confronta contemporaneamente sia i risultati Diretti sia quelli Near-Miss per ogni ombra e classe stradale, calcolando a runtime le frazioni di occupazione di Strada, Marciapiede e Prato ($\alpha_{road}$, $\alpha_{side}$, $\alpha_{terr}$).

---

## 📝 Riproduzione e Rigenerazione da Zero (Manuale PowerShell)
Se desideri pulire le cartelle temporanee e rigenerare l'intero processo da capo manualmente con un unico blocco di comandi:
```powershell
# 1. Rimuove tutte le cartelle dei risultati precedenti
Remove-Item -Recurse -Force extracted_occlusions, extracted_occlusions_probabilities, extracted_occlusions_probabilities_near, extracted_occlusions_neural, extracted_occlusions_neural_near, best_model.pth -ErrorAction SilentlyContinue

# 2. Riesegue l'intera pipeline
python occ3d_occlusion_explorer.py --mode batch
python bayesian_prior_extractor.py
python temporal_persistence_tracker.py
python bayesian_occlusion_agent.py --mode batch
python bayesian_occlusion_agent.py --mode batch --use-near-miss
python train_neural_agent.py
python neural_occlusion_agent.py --mode batch
python neural_occlusion_agent.py --mode batch --use-near-miss
```
