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

I visualizzatori si trovano in `visualizzatori/`. Possono essere avviati singolarmente oppure affiancati al viewer ufficiale nuScenes tramite il launcher `avvia_entrambi.py`.

### Avvio Singolo (frame opzionale 1-404)
```powershell
.\.venv\Scripts\python.exe visualizzatori/<nome_vis>.py [frame]
```

### Avvio Affiancato con nuScenes Explorer (sincronizzazione bidirezionale)
```powershell
# Sintassi generale
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py [frame] [--FLAG]

# Esempi rapidi
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17               # raycasting (default)
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17 --inputs       # input rete neurale
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17 --neural       # inferenza neurale
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17 --bayes        # probabilita bayesiana
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17 --gt           # ground truth
.\.venv\Scripts\python.exe visualizzatori/avvia_entrambi.py 17 --eval         # valutazione prestazioni
```
> Digita un numero di frame nel terminale del launcher per sincronizzare entrambe le finestre.

---

### A. Raycasting Occlusioni (`vis_raycasting_occlusioni.py`)
Mappa BEV con coni d'ombra LiDAR, strutture statiche (muri/edifici), icone ostacoli e tooltip:
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_raycasting_occlusioni.py [frame]
```
**Comandi**: `</`>`/->` naviga frame, `T` toggle aux, `Click` ispeziona zona, `S` salva HD

---

### B. Input Multimodali Rete Neurale (`vis_input_rete_neurale.py`) - Figure 6 Tesi
Dashboard che mostra esattamente cosa riceve la rete neurale per ogni frame:
- **Sinistra**: Mappa BEV 25m con HD-Map, LiDAR, ombre, strutture statiche e icone ostacoli
- **Destra**: Griglia degli **11 canali tensoriali** `(11, 200, 200)` + pesi Channel Attention (SE)
- **Toggle "Rete Ausiliaria"**: ispezione del flusso 9 scalari -> MLP -> parametri FiLM gamma/beta
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_input_rete_neurale.py [frame]
```
**Comandi**: `</`>`/->` naviga, `T/Spazio/M` toggle rete ausiliaria, `Z/X` scorri zone ombra, `Click` ispeziona scalari FiLM, `S` salva HD

---

### C. Inferenza Neurale Live (`vis_inferenza_neurale.py`)
Inferenza live su GPU con classificazione TP/FP/FN/TN e barre di probabilita per classe:
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_inferenza_neurale.py [frame]
```
**Comandi**: `</`>`/->` naviga, `M/G` cambia GT (Sintetica/Reale), `R` raggio 20m/25m, `Click` ispeziona

---

### D. Probabilita Bayesiana Condizionata (`vis_probabilita_bayes.py`)
Approccio alternativo senza rete neurale: aggiornamento bayesiano guidato da HD Map e OBB:
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_probabilita_bayes.py [frame]
```

---

### E. Ground Truth Occlusioni (`vis_ground_truth_occlusioni.py`)
Confronto affiancato GT 3D nuScenes vs GT Neurosimbolica con regole fisiche:
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_ground_truth_occlusioni.py [frame]
```

---

### F. Dashboard Valutazione Prestazioni (`vis_valutazione_prestazioni.py`)
Report interattivo con metriche Precision/Recall/F1/IoU per tutti i 404 frame:
```powershell
.\.venv\Scripts\python.exe visualizzatori/vis_valutazione_prestazioni.py [frame]
```

---

## 🧠 3. Addestramento del Modello

### A. Addestramento Ufficiale su Ground Truth Sintetica Neurosimbolica
Addestra `AttentionPerZoneModel` con Channel Attention (SE) e Modulazione FiLM per 20 epoche su GPU. Salva in `pesi_modelli/per_zone_checkpoint_attention_neuro_hybrid.pth`:
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
