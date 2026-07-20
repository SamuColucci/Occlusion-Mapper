# Script Obsoleti — Archivio

File spostati qui il 01/07/2026 durante il refactoring ponytail del progetto. Non sono più necessari per la pipeline attiva.

## Versioni precedenti dell'explorer

- **`prova.py`** — Prima bozza dell'explorer (510 righe). Stessa classe `SOTARayCaster` ma senza `PseudoBox`, senza clustering manmade, senza batch mode. Superata da `occ3d_occlusion_explorer.py`.

- **`script_Occlusione.py`** — Versione ancora piu' vecchia (478 righe). Usa `cv2` e parametri diversi (`MAX_RANGE=50m`, `GAP_FACTOR=5`). Approccio completamente diverso dalla pipeline SOTA finale.

## Moduli sostituiti

- **`conditional_probability_estimator.py`** — Stimatore frequenzista basato su tabelle di co-occorrenza (causa, distanza) -> outcome. Sostituito da `occlusion_agent.py` che usa un Random Forest Classifier con feature geometriche.

- **`diagnose.py`** — Script di debug one-shot che eseguiva la pipeline su sample 87 e stampava statistiche sui pixel rossi, ownership e coverage. Chiamava `generate_final_visibility()` che e' stata rimossa nel refactor ponytail, quindi e' rotto.

## Utility di verifica refactoring (create il 01/07/2026)

- **`regression_check.py`** — Snapshot e confronto delle 4 griglie 3D + JSON prima/dopo il refactor ponytail. Ha verificato che il codice refactored produce risultati identici bit-per-bit. Lavoro completato.

- **`batch_regression_check.py`** — Confronto batch dei JSON in `extracted_occlusions/` con l'output del codice refactored. Ha confermato che le differenze nei poligoni erano pre-esistenti (non introdotte dal refactor).

- **`compare_polygons.py`** — Visualizzatore matplotlib interattivo per confrontare i poligoni vecchi vs nuovi sovrapposti. Usato per confermare visivamente che le zone coperte sono identiche.

## File JSON obsoleti

- **`occlusions_nuscenes.json`** — File di output temporaneo sovrascritto a ogni esecuzione in modalità singola (`--mode single`).
- **`occlusion_gaps.json`** — Vecchia esportazione di gap angolari sui ring LiDAR. Inutilizzata nel codice attivo.
- **`occlusion_query.json`** — Vecchio file di query geometriche con parametri griglia superati. Inutilizzato.
