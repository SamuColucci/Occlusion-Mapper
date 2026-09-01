# CONFRONTO 1-A-1 TRA MODELLO BASE E MODELLO AVANZATO AD ATTENZIONE

## 1. ASL con GT Reale nuScenes

### Raggio Distanza: 20 Metri

| Modello Architetturale | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Modello Base (CNN Semplice)** | **20.9%** | **68.2%** | **29.1%** | **17.2%** |
| **Modello Avanzato (SE-Attention + FiLM)** | **78.2%** | **70.8%** | **71.7%** | **58.4%** |

#### Dettaglio Per-Classe: Modello Base (CNN Semplice)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 32.3% | 43.4% | 37.0% | 22.7% |
| **Camion/Bus** | 21.1% | 39.3% | 27.5% | 15.9% |
| **VRU (Pedoni/Bici/Moto)** | 16.8% | 93.9% | 28.5% | 16.6% |
| **Barriera** | 13.4% | 96.1% | 23.6% | 13.4% |


#### Dettaglio Per-Classe: Modello Avanzato (SE-Attention + FiLM)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 84.1% | 45.0% | 58.6% | 41.5% |
| **Camion/Bus** | 70.3% | 42.6% | 53.1% | 36.1% |
| **VRU (Pedoni/Bici/Moto)** | 76.2% | 95.8% | 84.9% | 73.8% |
| **Barriera** | 82.2% | 99.8% | 90.2% | 82.1% |

### Raggio Distanza: 25 Metri

| Modello Architetturale | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Modello Base (CNN Semplice)** | **18.2%** | **66.6%** | **25.8%** | **14.9%** |
| **Modello Avanzato (SE-Attention + FiLM)** | **78.0%** | **69.0%** | **70.0%** | **56.6%** |

#### Dettaglio Per-Classe: Modello Base (CNN Semplice)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 29.5% | 41.9% | 34.6% | 20.9% |
| **Camion/Bus** | 17.6% | 34.9% | 23.4% | 13.2% |
| **VRU (Pedoni/Bici/Moto)** | 14.4% | 93.5% | 24.9% | 14.2% |
| **Barriera** | 11.3% | 95.9% | 20.2% | 11.2% |


#### Dettaglio Per-Classe: Modello Avanzato (SE-Attention + FiLM)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 84.2% | 43.3% | 57.2% | 40.1% |
| **Camion/Bus** | 71.5% | 37.6% | 49.3% | 32.7% |
| **VRU (Pedoni/Bici/Moto)** | 75.3% | 95.5% | 84.2% | 72.7% |
| **Barriera** | 81.1% | 99.7% | 89.4% | 80.9% |

## 2. ASL con GT Sintetica Neurosimbolica

### Raggio Distanza: 20 Metri

| Modello Architetturale | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Modello Base (CNN Semplice)** | **45.6%** | **67.6%** | **48.5%** | **33.3%** |
| **Modello Avanzato (SE-Attention + FiLM)** | **83.6%** | **36.1%** | **48.9%** | **33.4%** |

#### Dettaglio Per-Classe: Modello Base (CNN Semplice)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 81.2% | 64.9% | 72.1% | 56.4% |
| **Camion/Bus** | 43.4% | 29.9% | 35.4% | 21.5% |
| **VRU (Pedoni/Bici/Moto)** | 30.6% | 80.5% | 44.4% | 28.5% |
| **Barriera** | 27.2% | 95.2% | 42.3% | 26.8% |


#### Dettaglio Per-Classe: Modello Avanzato (SE-Attention + FiLM)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 92.3% | 29.4% | 44.6% | 28.7% |
| **Camion/Bus** | 77.2% | 17.3% | 28.2% | 16.4% |
| **VRU (Pedoni/Bici/Moto)** | 80.8% | 47.8% | 60.1% | 42.9% |
| **Barriera** | 84.0% | 49.9% | 62.6% | 45.5% |

### Raggio Distanza: 25 Metri

| Modello Architetturale | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Modello Base (CNN Semplice)** | **47.1%** | **66.5%** | **49.7%** | **34.4%** |
| **Modello Avanzato (SE-Attention + FiLM)** | **83.5%** | **29.5%** | **42.4%** | **27.6%** |

#### Dettaglio Per-Classe: Modello Base (CNN Semplice)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 82.8% | 65.9% | 73.4% | 58.0% |
| **Camion/Bus** | 44.8% | 29.8% | 35.8% | 21.8% |
| **VRU (Pedoni/Bici/Moto)** | 27.4% | 76.4% | 40.3% | 25.3% |
| **Barriera** | 33.3% | 93.9% | 49.2% | 32.6% |


#### Dettaglio Per-Classe: Modello Avanzato (SE-Attention + FiLM)

| Categoria | Precision (%) | Recall (%) | F1-Score (%) | IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Auto** | 92.5% | 26.7% | 41.4% | 26.1% |
| **Camion/Bus** | 78.0% | 13.8% | 23.4% | 13.2% |
| **VRU (Pedoni/Bici/Moto)** | 80.2% | 43.5% | 56.4% | 39.3% |
| **Barriera** | 83.3% | 33.9% | 48.2% | 31.8% |

