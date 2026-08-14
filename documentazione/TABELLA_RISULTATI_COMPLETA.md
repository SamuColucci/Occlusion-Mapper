# Tabella Completa dei Risultati e Metriche del Progetto Occlusion-Mapper

Il presente documento raccoglie in modo esaustivo e strutturato tutti i risultati quantitativi live ottenuti da ciascun modello/funzione di loss sperimentata nel progetto (Soglia decisionale = 30%).

---

## 1. Agente Bayesiano Dinamico (Prior HD)

- **Descrizione**: Approccio analitico basato sulle probabilità condizionate storiche estratte dalle mappe HD di nuScenes.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 17.9% | 44.6% | 25.5% | 14.6% | 904 / 4145 / 1125 |
| **Camion/Bus** | 5.7% | 34.3% | 9.8% | 5.2% | 135 / 2215 / 259 |
| **Pedone** | 5.8% | 38.3% | 10.0% | 5.3% | 376 / 6143 / 606 |
| **Moto** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 129 |
| **Bicicletta** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 1 / 134 |
| **Barriera** | 20.5% | 2.1% | 3.7% | 1.9% | 16 / 62 / 762 |
| **MEDIA MACRO** | **8.3%** | **19.9%** | **8.2%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **79.3%** (2036 violazioni su 9824 zone non carrabili).

---

## 2. Agente Neurale Baseline (BCE Standard)

- **Descrizione**: Rete CNN Per-Zone addestrata con funzione di Loss pesata Binary Cross-Entropy (BCE) standard.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 25.3% | 82.0% | 38.7% | 24.0% | 1664 / 4901 / 365 |
| **Camion/Bus** | 12.4% | 68.5% | 20.9% | 11.7% | 270 / 1914 / 124 |
| **Pedone** | 11.8% | 96.3% | 21.0% | 11.7% | 946 / 7081 / 36 |
| **Moto** | 7.3% | 48.1% | 12.7% | 6.8% | 62 / 789 / 67 |
| **Bicicletta** | 12.3% | 71.6% | 20.9% | 11.7% | 96 / 687 / 38 |
| **Barriera** | 21.3% | 92.4% | 34.7% | 21.0% | 719 / 2652 / 59 |
| **MEDIA MACRO** | **15.1%** | **76.5%** | **24.8%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **86.0%** (1374 violazioni su 9824 zone non carrabili).

---

## 3. Agente Neurale BCE + Penalizzazione Semantica

- **Descrizione**: Rete CNN Per-Zone addestrata con BCE Loss integrata con la penalizzazione dei veicoli su Marciapiede/Terreno.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 24.5% | 87.7% | 38.3% | 23.7% | 1779 / 5486 / 250 |
| **Camion/Bus** | 15.4% | 67.5% | 25.0% | 14.3% | 266 / 1464 / 128 |
| **Pedone** | 13.1% | 95.1% | 23.1% | 13.0% | 934 / 6181 / 48 |
| **Moto** | 20.6% | 39.5% | 27.1% | 15.7% | 51 / 196 / 78 |
| **Bicicletta** | 14.1% | 61.2% | 22.9% | 12.9% | 82 / 500 / 52 |
| **Barriera** | 21.7% | 93.4% | 35.3% | 21.4% | 727 / 2617 / 51 |
| **MEDIA MACRO** | **18.2%** | **74.1%** | **28.6%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **79.0%** (2064 violazioni su 9824 zone non carrabili).

---

## 4. Agente Neurale Focal Loss Standard

- **Descrizione**: Rete CNN Per-Zone addestrata con Focal Loss (gamma=2.0, alpha=0.75) per abbattere i gradienti delle ombre vuote.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 20.9% | 96.5% | 34.4% | 20.8% | 1958 / 7405 / 71 |
| **Camion/Bus** | 12.6% | 83.8% | 22.0% | 12.3% | 330 / 2279 / 64 |
| **Pedone** | 17.8% | 92.8% | 29.9% | 17.6% | 911 / 4206 / 71 |
| **Moto** | 16.9% | 73.6% | 27.5% | 16.0% | 95 / 466 / 34 |
| **Bicicletta** | 17.1% | 86.6% | 28.5% | 16.6% | 116 / 564 / 18 |
| **Barriera** | 31.3% | 97.4% | 47.3% | 31.0% | 758 / 1667 / 20 |
| **MEDIA MACRO** | **19.4%** | **88.4%** | **31.6%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **56.4%** (4282 violazioni su 9824 zone non carrabili).

---

## 5. Agente Neurale Focal Loss + Penalizzazione Semantica (Vincente F1)

- **Descrizione**: Modello vincente del progetto. Focal Loss (gamma=2.0) combinata con penalizzazione semantica bilanciata (lambda=1.5).

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 39.0% | 60.0% | 47.3% | 31.0% | 1218 / 1906 / 811 |
| **Camion/Bus** | 27.7% | 43.1% | 33.8% | 20.3% | 170 / 443 / 224 |
| **Pedone** | 14.3% | 90.3% | 24.6% | 14.0% | 887 / 5336 / 95 |
| **Moto** | 36.0% | 27.9% | 31.4% | 18.7% | 36 / 64 / 93 |
| **Bicicletta** | 10.4% | 61.9% | 17.8% | 9.8% | 83 / 713 / 51 |
| **Barriera** | 22.6% | 94.3% | 36.5% | 22.3% | 734 / 2509 / 44 |
| **MEDIA MACRO** | **25.0%** | **62.9%** | **31.9%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **98.9%** (105 violazioni su 9824 zone non carrabili).

---

## 6. Agente Neurale Contesto Esterno (Ring Semantics)

- **Descrizione**: Studio di ablazione. Rete addestrata oscurando la semantica interna dell'ombra e fornendo solo la corona circostante di 2.0m.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 15.0% | 89.6% | 25.6% | 14.7% | 1818 / 10329 / 211 |
| **Camion/Bus** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 394 |
| **Pedone** | 8.1% | 78.5% | 14.7% | 7.9% | 771 / 8747 / 211 |
| **Moto** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 / 129 |
| **Bicicletta** | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 9 / 134 |
| **Barriera** | 51.0% | 20.4% | 29.2% | 17.1% | 159 / 153 / 619 |
| **MEDIA MACRO** | **12.3%** | **31.4%** | **11.6%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **47.1%** (5194 violazioni su 9824 zone non carrabili).

---

## 7. Agente Neurale Asymmetric Loss (ASL - CVPR 2021) (Vincente Recall)

- **Descrizione**: Modello vincente per la Sicurezza Salvavita. Asymmetric Loss (gamma_+=1.0, gamma_-=4.0, m=0.05) + Penalizzazione Semantica.

| Classe | Precision | Recall | F1-Score | IoU | TP / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auto** | 34.7% | 66.5% | 45.6% | 29.6% | 1349 / 2536 / 680 |
| **Camion/Bus** | 22.7% | 53.3% | 31.8% | 18.9% | 210 / 715 / 184 |
| **Pedone** | 7.8% | 98.2% | 14.4% | 7.8% | 964 / 11421 / 18 |
| **Moto** | 23.3% | 34.1% | 27.7% | 16.1% | 44 / 145 / 85 |
| **Bicicletta** | 3.0% | 91.8% | 5.8% | 3.0% | 123 / 3964 / 11 |
| **Barriera** | 10.6% | 98.7% | 19.1% | 10.6% | 768 / 6495 / 10 |
| **MEDIA MACRO** | **17.0%** | **73.8%** | **24.1%** | - | - |

- 🛡️ **Tasso di Coerenza Semantico (Terreno/Marciapiede)**: **98.1%** (182 violazioni su 9824 zone non carrabili).

---

## 📊 Tabella Sinottica Riassuntiva di Confronto

| # | Modello / Loss | Macro Precision | Macro Recall | Macro F1-Score | Coerenza Semantica | Applicazione Principale |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | 1. Agente Bayesiano Dinamico (Prior HD) | 8.3% | 19.9% | 8.2% | 79.3% | Baseline Analitica senza Rete |
| **2** | 2. Agente Neurale Baseline (BCE Standard) | 15.1% | 76.5% | 24.8% | 86.0% | Baseline Neurale Grezza |
| **3** | 3. Agente Neurale BCE + Penalizzazione Semantica | 18.2% | 74.1% | 28.6% | 79.0% | Bilanciamento Intermedio BCE |
| **4** | 4. Agente Neurale Focal Loss Standard | 19.4% | 88.4% | 31.6% | 56.4% | Massima Sensitivity Senza Vincoli |
| **5** | 5. Agente Neurale Focal Loss + Penalizzazione Semantica (Vincente F1) | **25.0%** | **62.9%** | **31.9%** | **98.9%** | **Guida Autonoma Fluidità & Motion Planning** |
| **6** | 6. Agente Neurale Contesto Esterno (Ring Semantics) | 12.3% | 31.4% | 11.6% | 47.1% | Studio di Ablazione Contesto |
| **7** | 7. Agente Neurale Asymmetric Loss (ASL - CVPR 2021) (Vincente Recall) | **17.0%** | **73.8%** | 24.1% | **98.1%** | **Frenata di Emergenza Salvavita (AEB - Pedoni 98.8%)** |
