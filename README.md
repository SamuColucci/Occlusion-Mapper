# Occlusion-Mapper: Stima Probabilistica ed Apprendimento Neurale degli Ostacoli nelle Zone Occluse per Guida Autonoma

**Occlusion-Mapper** è una piattaforma software per l'estrazione geometrica, la modellazione probabilistica analitica e la classificazione neurale delle **zone d'ombra e sub-zone occluse** nel contesto della guida autonoma urbana (dataset **nuScenes**).

---

### 🌟 Caratteristiche Principali

- **Ray-Casting 2D/3D & Estrazione Sub-Zone**: Identificazione geometrica dei coni d'ombra generati da ostacoli fisici ed edifici con decomposizione vettoriale in sotto-zone.
- **Rappresentazione BEV Multi-Canale**: Integrazione di dati LiDAR, mappe HD (strada, marciapiedi, strisce pedonali, parcheggi, terreno) e ground truth 3D GT.
- **10 Varianti di Agenti e Funzioni di Loss a Confronto**:
  1. **Agente Bayesiano Dinamico**: Baseline analitica senza rete (*Prior HD + Memoria Temporale*).
  2. **Agente Neurale Baseline (BCE Standard)**: Baseline neurale grezza (*Macro F1: 29.5%*).
  3. **Agente Neurale BCE + Penalizzazione Semantica**: Bilanciamento intermedio (*Macro F1: 28.8%, Coerenza: 78.2%*).
  4. **Agente Neurale Focal Loss Standard**: Massima sensitivity senza vincoli (*Macro Recall: 82.1%*).
  5. **Agente Neurale Focal Loss + Penalizzazione Semantica** 🏆 (*Vincente F1 / Motion Planning: Macro F1 33.8%, Coerenza: 88.4%*).
  6. **Agente Neurale Focal Loss Neurosimbolica Completa**: Inibizione + VRU (*Macro F1: 32.3%, Coerenza: 89.3%*).
  7. **Agente Neurale Asymmetric Loss Standard (CVPR 2021)**: ASL originale (*Macro Recall: 99.3%*).
  8. **Agente Neurale Asymmetric Loss + Penalizzazione Semantica** 🚨 (*Vincente Frenata d'Emergenza AEB: Recall Pedoni 99.9%, Macro Recall 79.6%, Coerenza: 85.2%*).
  9. **Agente Neurale Asymmetric Loss Neurosimbolica Completa**: Inibizione + VRU (*Macro Recall: 77.2%, Coerenza: 86.1%*).
  10. **Agente Neurale Contesto Esterno (Ring Semantics)**: Studio di ablazione contesto esterno (*Macro F1: 11.5%*).
- **Stress Test Sintetico ad Alta Densità (3.102 Ostacoli Inseriti)**: Valutazione della capacità di generalizzazione Zero-Shot dei modelli neurali.
- **Dashboard & Visualizzatori Interattivi Runtime**: Interfaccia HUD per l'ispezione grafica e la comparazione affiancata in tempo reale.

---

### 📂 Struttura del Repository

```text
Occlusion-Mapper/
├── verify_project.py                   # Dashboard principale di controllo e validazione
├── estrazione_zone_occluse.py          # Script di estrazione delle zone d'ombra
├── requirements.txt                    # Elenco delle dipendenze di ambiente
├── raycaster/                          # Modulo RayCaster 2D/3D per il campionamento LiDAR
├── dataset_adapter/                    # Generazione dataset BEV e ground truth
├── inferenza_agenti/                   # Agenti Bayesiani e predittori neurali
├── architettura_neurale/               # Definizione rete OcclusionPredictor e loss functions
├── addestramento/                      # Script di training per i 10 modelli
├── valutazione/                        # Script di benchmark metriche GT (F1, Recall, Coerenza)
├── pesi_modelli/                       # Checkpoint dei pesi neurali addestrati (.pth)
├── documentazione/                     # Documentazione approfondita e appunti di Tesi
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

#### 3. Esecuzione Benchmark 10 Modelli o Menu Interattivo
```powershell
python valutazione/export_results_to_md.py
python visualizzatori/menu_evaluator.py
```

---

### 📚 Documentazione Dettagliata
Tutti i documenti di approfondimento, le guide per gli agenti e la stesura dei capitoli della Tesi sono disponibili nella cartella **[`documentazione/`](documentazione/)**.

