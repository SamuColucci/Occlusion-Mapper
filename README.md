# Occlusion-Mapper: Stima Probabilistica degli Ostacoli nelle Zone Occluse per Guida Autonoma

**Occlusion-Mapper** è una piattaforma software per l'estrazione geometrica, la modellazione probabilistica e la classificazione neurale delle **zone d'ombra e sub-zone occluse** nel contesto della guida autonoma urbana (dataset **nuScenes**).

---

### 🌟 Caratteristiche Principali

- **Ray-Casting 2D/3D & Estrazione Sub-Zone**: Identificazione geometrica dei coni d'ombra generati da ostacoli fisici ed edifici.
- **Rappresentazione BEV Multi-Canale (11 Canali)**: Integrazione di dati LiDAR, mappe HD (strada, marciapiedi, strisce pedonali, parcheggi, terreno) e ground truth 3D GT.
- **7 Varianti di Agenti e Funzioni di Loss a Confronto**:
  1. **Agente Bayesiano Dinamico** (Prior HD analitiche).
  2. **Agente Neurale Baseline (BCE Standard)**.
  3. **Agente Neurale BCE + Penalizzazione Semantica**.
  4. **Agente Neurale Focal Loss Standard**.
  5. **Agente Neurale Focal Loss + Penalizzazione Semantica** 🏆 (*Picco Macro F1: 32.3%, Coerenza: 99.0%*).
  6. **Agente Neurale Contesto Esterno (Ring Semantics)** (*Studio di Ablazione*).
  7. **Agente Neurale Asymmetric Loss - CVPR 2021** 🚨 (*Picco Recall Pedoni: 98.8%, Macro Recall: 72.4%*).
- **Stress Test Sintetico ad Alta Densità (3.102 Ostacoli Inseriti)**: Valutazione della capacità di generalizzazione Zero-Shot dei modelli neurali.
- **Dashboard & Visualizzatori Interattivi Runtime**: Interfaccia HUD per l'ispezione grafica e la comparazione affiancata.

---

### 📂 Struttura del Repository

```text
Occlusion-Mapper/
├── verify_project.py                   # Dashboard principale di controllo e validazione
├── per_zone_occlusion_agent.py         # Agente Neurale principale
├── conditional_probability_occlusion_zone.py # Agente Bayesiano Dinamico
├── loss_functions.py                   # Modulo Loss (Weighted BCE, Focal Loss, Asymmetric Loss)
├── per_zone_model.py                   # Architettura della Rete Neurale (CNN BEV + Scalari)
├── dataset_generator_per_zone.py       # Estrazione ed elaborazione dataset Per-Zone
├── requirements.txt                    # Elenco delle dipendenze di ambiente
├── documentazione/                     # Documentazione approfondita e appunti di Tesi
├── valutazione/                        # Script di benchmark metriche GT e Stress Test
├── pesi_modelli/                       # Checkpoint dei pesi neurali addestrati (.pth)
└── visualizzatori/                     # Visualizzatori grafici ed HUD interattivi
```

---

### 🚀 Guida Rapida all'Esecuzione

#### 1. Configurazione Ambiente Virtuale
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Avvio Dashboard Principale
```powershell
python verify_project.py
```

#### 3. Esecuzione Benchmark 7 Modelli o Menu Interattivo
```powershell
python valutazione/evaluate_focal_comparison.py
python visualizzatori/menu_evaluator.py
```

---

### 📚 Documentazione Dettagliata
Tutti i documenti di approfondimento, le guide per gli agenti e la stesura dei capitoli della Tesi sono disponibili nella cartella **[`documentazione/`](documentazione/)**.
