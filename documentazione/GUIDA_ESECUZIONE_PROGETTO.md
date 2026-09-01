# 🚀 Guida Ufficiale di Esecuzione della Pipeline e degli Agenti (Tesi)

Questa guida illustra la procedura completa per riprodurre, addestrare, valutare e visualizzare tutti i moduli del sistema di stima delle occlusioni sviluppato per la tesi di laurea.

---

## 🛠️ Prerequisiti e Ambiente Virtuale
Tutti i comandi utilizzano l'ambiente virtuale dedicato configurato con supporto CUDA per GPU NVIDIA (RTX 3060):
```powershell
# Esegui i comandi dalla radice del progetto
.\.venv\Scripts\python.exe <percorso_script.py>
```

---

## 📦 1. Estrazione Geometrica delle Zone d'Ombra (Raycasting LiDAR)
Esegue la simulazione del fascio LiDAR 3D polar-grid e calcola i poligoni 2D Bird's Eye View (BEV) delle zone cieche per tutti i 404 frame nuScenes, salvandoli in `extracted_occlusions/`:
```powershell
.\.venv\Scripts\python.exe estrazione_zone_occluse.py
```

---

## 🎯 2. Visualizzatori Interattivi Ufficiali

### A. Visualizzatore Ufficiale del Modello Finale (`verify_runtime_attention.py`)
Esegue l'**inferenza live su GPU** del modello `AttentionPerZoneModel` mostrando a schermo:
* La mappa BEV HD con i coni d'ombra classificati per stato ($TP, FP, FN, TN$).
* Toggle istantaneo tra **Ground Truth Sintetica Neurosimbolica** e **Ground Truth Reale nuScenes** (Tasto `M` o `G`).
* Conteggi di frazione per classe: `Trovati (TP / Totale Scena)` (es. `10/11`), metriche di classe e barre di probabilità `[████░░░]`.
```powershell
.\.venv\Scripts\python.exe visualizzatori/verify_runtime_attention.py
```
* **Comandi**: Frecce SX/DX (naviga frame), Tasto M/G (cambia Ground Truth), Tasto R (raggio 20m/25m), Click Mouse (ispezione zona).

### B. Visualizzatore Probabilità Condizionata Bayesiana (`verify_runtime_bayes.py`)
Mostra l'aggiornamento bayesiano spaziale guidato dalle superfici della mappa HD e dai vincoli OBB delle zone occluse:
```powershell
.\.venv\Scripts\python.exe visualizzatori/verify_runtime_bayes.py
```

### C. Visualizzatore Confronto Ground Truth (`verify_runtime_ground_truth.py`)
Confronta affiancate in tempo reale la **Ground Truth 3D nuScenes** e la **Ground Truth Neurosimbolica con regole fisiche**:
```powershell
.\.venv\Scripts\python.exe visualizzatori/verify_runtime_ground_truth.py
```

### D. Visualizzatore Benchmark Architetture (`verify_runtime_model_comparison.py`)
Visualizza il confronto affiancato 1-a-1 tra la **Baseline Neurale (CNN Semplice)** e il **Modello Finale ad Attenzione (SE+FiLM)**:
```powershell
.\.venv\Scripts\python.exe visualizzatori/verify_runtime_model_comparison.py
```

---

## 🧠 3. Addestramento del Modello

### A. Addestramento Ufficiale su Ground Truth Sintetica Neurosimbolica
Addestra l'architettura con **Channel Attention (Squeeze-and-Excitation)** e **Modulazione FiLM con Mappa HD** per 20 epoche su GPU, salvando i pesi in `pesi_modelli/per_zone_checkpoint_attention_neuro.pth`:
```powershell
.\.venv\Scripts\python.exe addestramento/train_per_zone_attention.py
```

### B. Addestramento di Baseline su Ground Truth Reale nuScenes (Ablation Study)
Addestra la medesima architettura supervisionata unicamente dai box 3D annotati fisicamente visibili:
```powershell
.\.venv\Scripts\python.exe addestramento/train_per_zone_real_gt.py
```

---

## 📊 4. Valutazione e Benchmark Ufficiale

### A. Valutazione del Modello Ufficiale su Dataset Completo
Calcola le metriche certificate ($TP, FP, FN$, Precision, Recall, F1-Score, IoU) su tutti i 404 fotogrammi nuScenes per entrambe le Ground Truth e per entrambi i raggi operativi (20m e 25m):
```powershell
.\.venv\Scripts\python.exe valutazione/evaluate_final_official.py
```

### B. Studio di Ablazione sulla Supervisione (GT Reale vs GT Neurosimbolica)
Confronta in un unico report le metriche dei due modelli addestrati a 25 metri:
```powershell
.\.venv\Scripts\python.exe valutazione/evaluate_training_supervision_ablation.py
```

---

## 🤖 5. Inferenza Batch dell'Agente Neurale
Genera i file JSON probabilistici di tutte le zone occluse per tutti i fotogrammi del dataset utilizzando il modello caricato da `pesi_modelli/`:
```powershell
.\.venv\Scripts\python.exe inferenza_agenti/neural_agent.py
```
