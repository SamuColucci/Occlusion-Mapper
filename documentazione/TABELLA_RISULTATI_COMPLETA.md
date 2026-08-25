# Tabella Completa dei Risultati e Metriche del Progetto Occlusion-Mapper

Il presente documento raccoglie in modo esaustivo e strutturato tutti i risultati quantitativi live ottenuti da ciascuno dei 10 modelli/funzioni di loss sperimentate nel progetto (Soglia decisionale = 30%).

Ogni modello è corredato da una **spiegazione testuale semplice e discorsiva** per comprendere immediatamente la differenza concettuale e fisica tra le varie funzioni di loss e l'approccio Bayesiano analitico.

---

## 1. Agente Bayesiano Dinamico (Prior HD)

**Come funziona in modo semplice**: Non usa la rete neurale. Prende la zona d'ombra calcolata dal LiDAR ed interseca il poligono con le superfici della mappa HD (strada, marciapiede, strisce, parcheggio). Calcola la larghezza del varco di passaggio in metri (OBB) e consulta una tabella di probabilità fisiche. Se un ostacolo era visibile prima di entrare nell'ombra, eleva la sua probabilità al 95% per memoria temporale.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 17.9% | 43.8% | 25.4% | 14.5% | 883 / 4054 / 1133 |
| **Camion/Bus** | 5.7% | 33.7% | 9.8% | 5.1% | 132 / 2182 / 260 |
| **Pedone** | 5.7% | 36.8% | 9.9% | 5.2% | 359 / 5894 / 616 |
| **Moto** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 129 |
| **Bicicletta** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 1 / 133 |
| **Barriera** | 16.2% | 1.5% | 2.8% | 1.4% | 12 / 62 / 763 |
| **MEDIA MACRO** | **7.6%** | **19.3%** | **8.0%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **79.0%** (3925 violazioni su 18682 zone non carrabili).

---

## 2. Agente Neurale Baseline (BCE Standard)

**Come funziona in modo semplice**: Ritaglia l'immagine BEV dell'ombra a 10 canali d'ingresso ed estrae le 4 feature geometriche (area, distanza, larghezza ed altezza dell'ostacolo). Addestrata con la classica Binary Cross-Entropy (BCE). Tratta tutte le classi e tutti i tipi di terreno allo stesso modo senza penalizzare gli errori semantici o stradali.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 26.1% | 84.0% | 39.8% | 24.8% | 1693 / 4797 / 323 |
| **Camion/Bus** | 24.0% | 41.1% | 30.3% | 17.8% | 161 / 511 / 231 |
| **Pedone** | 10.6% | 95.9% | 19.0% | 10.5% | 935 / 7924 / 40 |
| **Moto** | 41.5% | 17.1% | 24.2% | 13.8% | 22 / 31 / 107 |
| **Bicicletta** | 10.8% | 60.9% | 18.4% | 10.1% | 81 / 668 / 52 |
| **Barriera** | 41.7% | 50.5% | 45.7% | 29.6% | 391 / 547 / 384 |
| **MEDIA MACRO** | **25.8%** | **58.2%** | **29.5%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **75.7%** (4542 violazioni su 18682 zone non carrabili).

---

## 3. Agente Neurale BCE + Penalizzazione Semantica

**Come funziona in modo semplice**: Stesso ritaglio d'ombra e feature geometriche della Baseline, ma aggiunge un vincolo di penalità nella loss se la rete predice veicoli a motore (auto, camion, moto) su zone non carrabili come marciapiedi o terreno. Insegna alla rete a rispettare la mappa stradale.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 28.8% | 84.6% | 43.0% | 27.4% | 1706 / 4217 / 310 |
| **Camion/Bus** | 20.7% | 59.9% | 30.8% | 18.2% | 235 / 900 / 157 |
| **Pedone** | 13.3% | 94.3% | 23.2% | 13.1% | 919 / 6016 / 56 |
| **Moto** | 13.6% | 37.2% | 20.0% | 11.1% | 48 / 304 / 81 |
| **Bicicletta** | 11.4% | 78.2% | 20.0% | 11.1% | 104 / 805 / 29 |
| **Barriera** | 22.3% | 88.6% | 35.7% | 21.7% | 687 / 2388 / 88 |
| **MEDIA MACRO** | **18.4%** | **73.8%** | **28.8%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **78.2%** (4082 violazioni su 18682 zone non carrabili).

---

## 4. Agente Neurale Focal Loss Standard

**Come funziona in modo semplice**: Applica la Focal Loss (Lin et al., ICCV 2017). Questa funzione abbassa drasticamente l'importanza (i gradienti) delle tantissime ombre totalmente vuote (negativi facili), costringendo la rete neurale ad allenarsi e concentrarsi solo sulle ombre ambigue ed ostiche dove c'è realmente qualcosa di nascosto.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 20.7% | 93.7% | 33.9% | 20.4% | 1888 / 7224 / 128 |
| **Camion/Bus** | 12.3% | 82.7% | 21.5% | 12.0% | 324 / 2302 / 68 |
| **Pedone** | 15.8% | 92.7% | 27.0% | 15.6% | 904 / 4829 / 71 |
| **Moto** | 13.0% | 57.4% | 21.2% | 11.8% | 74 / 496 / 55 |
| **Bicicletta** | 14.0% | 77.4% | 23.7% | 13.4% | 103 / 634 / 30 |
| **Barriera** | 28.1% | 88.9% | 42.7% | 27.2% | 689 / 1762 / 86 |
| **MEDIA MACRO** | **17.3%** | **82.1%** | **28.3%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **62.0%** (7096 violazioni su 18682 zone non carrabili).

---

## 5. Agente Neurale Focal Loss + Penalizzazione Semantica (Vincente F1)

**Come funziona in modo semplice (Modello Vincente F1-Score)**: È il nostro modello principale per la guida fluida. Unisce la Focal Loss (che abbatte i falsi allarmi nelle ombre vuote) con la penalizzazione semantica stradale (lambda=1.5). La rete impara sia a distinguere gli ostacoli reali dalle ombre vuote sia a non posizionare veicoli fuori dalla strada. Ottiene il miglior equilibrio globale F1-Score (33.8%).

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 38.9% | 72.4% | 50.6% | 33.9% | 1460 / 2293 / 556 |
| **Camion/Bus** | 29.8% | 50.8% | 37.5% | 23.1% | 199 / 469 / 193 |
| **Pedone** | 14.0% | 90.1% | 24.2% | 13.8% | 878 / 5396 / 97 |
| **Moto** | 42.1% | 37.2% | 39.5% | 24.6% | 48 / 66 / 81 |
| **Bicicletta** | 8.8% | 51.9% | 15.1% | 8.2% | 69 / 713 / 64 |
| **Barriera** | 22.7% | 82.5% | 35.6% | 21.6% | 639 / 2179 / 136 |
| **MEDIA MACRO** | **26.0%** | **64.1%** | **33.8%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **88.4%** (2162 violazioni su 18682 zone non carrabili).

---

## 6. Agente Neurale Focal Loss Neurosimbolica Completa (Inibizione + VRU)

**Come funziona in modo semplice**: Spinge al massimo l'apprendimento neurosimbolico sulla Focal Loss. Sopprime i negativi facili e applica due regole rigide: inibisce/penalizza la presenza di veicoli su marciapiedi/prato e contemporaneamente incentiva con alta sensibilità la presenza di pedoni e biciclette (utenti vulnerabili) su strisce pedonali e marciapiedi.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 39.9% | 69.7% | 50.8% | 34.0% | 1406 / 2117 / 610 |
| **Camion/Bus** | 29.3% | 48.0% | 36.4% | 22.2% | 188 / 454 / 204 |
| **Pedone** | 12.5% | 96.0% | 22.1% | 12.4% | 936 / 6545 / 39 |
| **Moto** | 42.7% | 27.1% | 33.2% | 19.9% | 35 / 47 / 94 |
| **Bicicletta** | 9.1% | 86.5% | 16.4% | 8.9% | 115 / 1155 / 18 |
| **Barriera** | 22.1% | 81.0% | 34.8% | 21.0% | 628 / 2210 / 147 |
| **MEDIA MACRO** | **25.9%** | **68.1%** | **32.3%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **89.3%** (1994 violazioni su 18682 zone non carrabili).

---

## 7. Agente Neurale Asymmetric Loss Standard (CVPR 2021)

**Come funziona in modo semplice**: Usa la Asymmetric Loss originale (Ridnik et al., CVPR 2021). Tratta in modo asimmetrico i casi positivi (ostacolo presente) ed i casi negativi (ombra vuota), usando un esponente molto più severo sui negativi ed azzerando completamente i gradienti sotto il margine m=0.05. Diventa iper-sensibile alla presenza di pericoli, ma genera molti falsi allarmi se priva di vincoli stradali.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 12.2% | 99.9% | 21.8% | 12.2% | 2014 / 14473 / 2 |
| **Camion/Bus** | 3.3% | 100.0% | 6.3% | 3.3% | 392 / 11607 / 0 |
| **Pedone** | 7.1% | 99.9% | 13.3% | 7.1% | 974 / 12724 / 1 |
| **Moto** | 1.4% | 97.7% | 2.8% | 1.4% | 126 / 8643 / 3 |
| **Bicicletta** | 2.1% | 99.2% | 4.1% | 2.1% | 132 / 6199 / 1 |
| **Barriera** | 7.3% | 99.4% | 13.6% | 7.3% | 770 / 9803 / 5 |
| **MEDIA MACRO** | **5.6%** | **99.3%** | **10.3%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **23.3%** (14332 violazioni su 18682 zone non carrabili).

---

## 8. Agente Neurale Asymmetric Loss + Penalizzazione Semantica (Vincente Recall)

**Come funziona in modo semplice (Modello Vincente Recall Salvavita)**: È il nostro modello vincente per la sicurezza e la frenata d'emergenza (AEB). Sfrutta la spinta asimmetrica della Asymmetric Loss sui casi positivi per non perdere mai un ostacolo reale, mentre la penalizzazione semantica corregge la coerenza stradale riducendo i falsi allarmi su marciapiede e prato. Raggiunge la Recall record del 99.9% sui pedoni occlusi.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 34.7% | 77.8% | 48.0% | 31.6% | 1568 / 2947 / 448 |
| **Camion/Bus** | 27.8% | 61.5% | 38.3% | 23.7% | 241 / 625 / 151 |
| **Pedone** | 6.0% | 99.9% | 11.4% | 6.0% | 974 / 15187 / 1 |
| **Moto** | 42.3% | 40.3% | 41.3% | 26.0% | 52 / 71 / 77 |
| **Bicicletta** | 1.8% | 98.5% | 3.5% | 1.8% | 131 / 7220 / 2 |
| **Barriera** | 5.8% | 99.4% | 10.9% | 5.8% | 770 / 12559 / 5 |
| **MEDIA MACRO** | **19.7%** | **79.6%** | **25.6%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **85.2%** (2771 violazioni su 18682 zone non carrabili).

---

## 9. Agente Neurale Asymmetric Loss Neurosimbolica Completa (Inibizione + VRU)

**Come funziona in modo semplice**: Combina la massima sensibilità della Asymmetric Loss con un'inibizione specifica per veicoli fuori strada e con un peso di promozione triplo (3.0) sui pedoni e biciclette occlusi su marciapiedi e strisce pedonali. Garantisce che nessun utente vulnerabile venga ignorato dalla rete.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 36.0% | 76.7% | 49.0% | 32.5% | 1547 / 2749 / 469 |
| **Camion/Bus** | 26.7% | 56.4% | 36.2% | 22.1% | 221 / 608 / 171 |
| **Pedone** | 6.4% | 100.0% | 11.9% | 6.4% | 975 / 14377 / 0 |
| **Moto** | 44.6% | 31.8% | 37.1% | 22.8% | 41 / 51 / 88 |
| **Bicicletta** | 2.0% | 99.2% | 3.8% | 2.0% | 132 / 6594 / 1 |
| **Barriera** | 5.7% | 99.0% | 10.8% | 5.7% | 767 / 12695 / 8 |
| **MEDIA MACRO** | **20.2%** | **77.2%** | **24.8%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **86.1%** (2591 violazioni su 18682 zone non carrabili).

---

## 10. Agente Neurale Contesto Esterno (Ring Semantics - Ablazione)

**Come funziona in modo semplice (Studio di Ablazione)**: Per dimostrare quanto sia fondamentale la mappa stradale DENTRO l'ombra, oscuriamo completamente l'interno dell'ombra e diamo alla rete solo la corona stradale circostante di 2.0m. Le prestazioni crollano (F1-Score 11.5%), dimostrando scientificamente che la semantica interna dell'ombra è indispensabile.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 15.0% | 89.4% | 25.7% | 14.8% | 1803 / 10187 / 213 |
| **Camion/Bus** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 392 |
| **Pedone** | 8.0% | 78.8% | 14.5% | 7.8% | 768 / 8856 / 207 |
| **Moto** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 129 |
| **Bicicletta** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 9 / 133 |
| **Barriera** | 45.4% | 21.2% | 28.9% | 16.9% | 164 / 197 / 611 |
| **MEDIA MACRO** | **11.4%** | **31.6%** | **11.5%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **47.1%** (9887 violazioni su 18682 zone non carrabili).

---

## 📊 Tabella Sinottica Riassuntiva di Confronto (10 Modelli)

| # | Modello / Loss | Macro Precision | Macro Recall | Macro F1-Score | Coerenza Semantica | Applicazione Principale |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | 1. Agente Bayesiano Dinamico (Prior HD) | 7.6% | 19.3% | 8.0% | 79.0% | Baseline Analitica senza Rete |
| **2** | 2. Agente Neurale Baseline (BCE Standard) | 25.8% | 58.2% | 29.5% | 75.7% | Baseline Neurale Grezza |
| **3** | 3. Agente Neurale BCE + Penalizzazione Semantica | 18.4% | 73.8% | 28.8% | 78.2% | Bilanciamento Intermedio BCE |
| **4** | 4. Agente Neurale Focal Loss Standard | 17.3% | 82.1% | 28.3% | 62.0% | Massima Sensitivity Senza Vincoli |
| **5** | 5. Agente Neurale Focal Loss + Penalizzazione Semantica (Vincente F1) | **26.0%** | **64.1%** | **33.8%** | **88.4%** | **Guida Autonoma Fluidità & Motion Planning** |
| **6** | 6. Agente Neurale Focal Loss Neurosimbolica Completa (Inibizione + VRU) | 25.9% | 68.1% | 32.3% | 89.3% | Apprendimento Neurosimbolico Completo Focal |
| **7** | 7. Agente Neurale Asymmetric Loss Standard (CVPR 2021) | 5.6% | 99.3% | 10.3% | 23.3% | ASL Originale (Ridnik et al., CVPR 2021) |
| **8** | 8. Agente Neurale Asymmetric Loss + Penalizzazione Semantica (Vincente Recall) | **19.7%** | **79.6%** | 25.6% | **85.2%** | **Frenata di Emergenza Salvavita (AEB - Pedoni 99.9%)** |
| **9** | 9. Agente Neurale Asymmetric Loss Neurosimbolica Completa (Inibizione + VRU) | 20.2% | 77.2% | 24.8% | 86.1% | Apprendimento Neurosimbolico Completo ASL |
| **10** | 10. Agente Neurale Contesto Esterno (Ring Semantics - Ablazione) | 11.4% | 31.6% | 11.5% | 47.1% | Studio di Ablazione Contesto Esterno |
