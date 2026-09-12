# 🧹 Registro Pulizia e Refactoring Futuro (Cleanup TODO)

Questo documento traccia in modo puntuale tutti i collegamenti, fallback legacy e porzioni di codice obsoleto che potranno essere ripuliti o semplificati non appena l'architettura definitiva (`AttentionPerZoneModel`) sarà consolidata.

---

## 📋 Lista Elementi da Rivedere / Rimuovere

### 1. `inferenza_agenti/neural_agent.py`
* **Righe**: 204 - 227
* **Elemento Attuale**:
  * Lista di checkpoint legacy (`per_zone_checkpoint_asl_standard.pth`, `per_zone_checkpoint_surrounding.pth`, `per_zone_checkpoint.pth`).
  * Blocco condizionale `if "attention" in checkpoint_path.lower(): ... else: from architettura_neurale.per_zone_model import PerZoneModel`.
* **Cosa Cambiare in Futuro**:
  * Rimuovere la lista di candidati legacy e mantenere solo il caricamento del checkpoint ufficiale (`per_zone_checkpoint_attention_neuro.pth` o `best_per_zone_attention.pth`).
  * Rimuovere il ramo `else` con `PerZoneModel` e importare/istanziare direttamente `AttentionPerZoneModel`.
* **Motivazione / Perché**:
  * Con `AttentionPerZoneModel` confermato come modello finale ufficiale ad alte prestazioni, il supporto a runtime alla vecchia baseline senza attenzione diventa superfluo.

### 2. Rinomina File: `inferenza_agenti/neural_agent.py` -> `inferenza_agenti/inference_pipeline.py`
* **File Coinvolti**: `inferenza_agenti/neural_agent.py`, `verify_complete_pipeline.py`, documentazione
* **Cosa Cambiare in Futuro**:
  * Rinominare fisicamente il file `inferenza_agenti/neural_agent.py` in `inferenza_agenti/inference_pipeline.py` (o `neural_inference_pipeline.py`).
  * Rinominare la classe `PerZoneOcclusionAgent` in `PerZoneInferencePipeline` (o `NeuralInferencePipeline`).
* **Motivazione / Perché**:
  * Nel SOTA della Guida Autonoma e della Computer Vision, questo modulo svolge la funzione di "Pipeline / Engine di Inferenza a Runtime" (caricamento dati, ritaglio patch, esecuzione modello ed esportazione JSON), rendendo la nomenclatura molto più chiara e formale per la tesi.

---

### 3. `architettura_neurale/loss_functions.py`
* **Righe**: 1 - 305
* **Elemento Attuale**:
  * Il file contiene 8 diverse formulazioni storiche di loss usate durante le varie fasi sperimentali (`WeightedBCESemanticLoss`, `FocalLoss`, varianti con penalizzazioni semantiche sperimentali, `AsymmetricLoss_neurosimbolica_completa`, ecc.).
* **Cosa Cambiare in Futuro**:
  * Mantenere solo la classe ufficiale vincente **`AsymmetricLoss`** (righe 158-200) e l'eventuale variante con vincoli topologici `AsymmetricLoss_penalizzazione_zona_semantica`.
  * Rimuovere o spostare in un archivio legacy le vecchie varianti BCE pesate e Focal Loss preliminari che non vengono più utilizzate negli script ufficiali di addestramento.
* **Motivazione / Perché**:
  * Snellire il file mantenendo solo la funzione di loss ufficiale dello stato dell'arte (`AsymmetricLoss`) con parametri ottimizzati ($\gamma_- = 4.0, \gamma_+ = 1.0, m = 0.05$).

---

### 4. `architettura_neurale/per_zone_model.py`
* **Righe**: 1 - 5 (e relativa cartella `.prove/per_zone_model.py`)
* **Elemento Attuale**:
  * File stub contenente solo `from .prove.per_zone_model import PerZoneModel` per retrocompatibilità con vecchi script di test preliminari.
* **Cosa Cambiare in Futuro**:
  * Eliminare il file `architettura_neurale/per_zone_model.py` e rimuovere `PerZoneModel` dall'`__init__.py` del modulo `architettura_neurale`.
* **Motivazione / Perché**:
  * Il modello ufficiale definitivo e certificato del progetto è esclusivamente `AttentionPerZoneModel` (definito in `architettura_neurale/attention_per_zone_model.py`). Mantenere lo stub di retrocompatibilità della vecchia baseline non è più necessario a sistema consolidato.

---
