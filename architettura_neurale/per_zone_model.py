# PerZoneModel per la predizione delle occlusioni per-zone in coordinate BEV
# Rete neurale ibrida (CNN + MLP) con 10 canali d'ingresso BEV + 4 feature scalari numeriche e 5 canali di output
# Ingresso: 
#   Patch Visivo 2D (10, 64, 64):
#     Canale 0: Nuvola di punti LiDAR 2D
#     Canale 1: Zone d'Ombra del RayCaster
#     Canale 2: Layer Mappa (strada)
#     Canale 3: Layer Mappa (marciapiede)
#     Canale 4: Layer Mappa (strisce pedonali)
#     Canale 5: Oggetti visibili Auto
#     Canale 6: Oggetti visibili Camion/Bus
#     Canale 7: Oggetti visibili Moto/Bici
#     Canale 8: Oggetti visibili Pedone
#     Canale 9: Oggetti visibili Barriera/Struttura
#   Feature Scalari Numeriche (4,):
#     Scalare 0: Area della zona d'ombra in metri quadri (area_sqm)
#     Scalare 1: Distanza dal sensore ego in metri (distance_m)
#     Scalare 2: Larghezza dell'occlusore in metri (occluder_width_m)
#     Scalare 3: Altezza dell'occlusore in metri (occluder_height_m)
# Output: 
#   Vettore di 5 Logits / Probabilità Sigmoid:
#     Canale 0: Probabilità di auto occlusa
#     Canale 1: Probabilità di camion/bus occluso
#     Canale 2: Probabilità di moto/bici occlusa
#     Canale 3: Probabilità di pedone occluso
#     Canale 4: Probabilità di barriera/struttura occlusa

import torch
import torch.nn as nn
import torch.nn.functional as F

# Architettura Neurale Ibrida Per-Zone che combina un Encoder Convoluzionale 2D (CNN) ed un Classificatore MLP (Multi-Layer Perceptron)
class PerZoneModel(nn.Module):
    # in_channels: numero di canali d'ingresso del patch visivo (10 canali BEV)
    # num_scalars: numero di feature numeriche scalari dell'ombra (9 scalari: 4 geometrici + 5 One-Hot semantici del terreno)
    # num_classes: numero di classi target d'uscita (5 classi semantiche)
    def __init__(self, in_channels=11, num_scalars=9, num_classes=6):
        super(PerZoneModel, self).__init__()
        
        # L'encoder convoluzionale applica 3 stadi di estrazione visiva (Conv2D + BatchNorm + ReLU + MaxPool)
        # che riducono progressivamente la risoluzione spaziale incrementando la profondità dei filtri
        # da 10 a 128, fino ad estrarre un vettore compatto di 512 caratteristiche visive ad alto livello della zona d'ombra
        # Encoder Convoluzionale per estrarre le caratteristiche visive dal patch 2D dell'ombra (10, 64, 64)
        self.conv_net = nn.Sequential(
            # Conv2d (Convolutional 2D Layer): applica filtri convoluzionali 2D per estrarre caratteristiche di contorno e forma (10 canali -> 32 filtri)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d (Batch Normalization 2D): normalizza il mini-batch per stabilizzare l'addestramento e velocizzare la convergenza dei gradienti
            nn.BatchNorm2d(32),
            # ReLU (Rectified Linear Unit): funzione di attivazione non lineare che azzera i valori negativi e mantiene quelli positivi f(x) = max(0, x)
            nn.ReLU(inplace=True),
            # MaxPool2d (Max Pooling 2D): sottocampionamento spaziale che riduce le dimensioni da 64x64 a 32x32 conservando i valori massimi
            nn.MaxPool2d(2, 2),
            
            # Secondo strato convoluzionale (Conv2d: 32 -> 64 filtri)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d (Batch Normalization 2D)
            nn.BatchNorm2d(64),
            # ReLU (Rectified Linear Unit)
            nn.ReLU(inplace=True),
            # MaxPool2d (Max Pooling 2D): riduce le dimensioni da 32x32 a 16x16
            nn.MaxPool2d(2, 2),
            
            # Terzo strato convoluzionale (Conv2d: 64 -> 128 filtri)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d (Batch Normalization 2D)
            nn.BatchNorm2d(128),
            # ReLU (Rectified Linear Unit)
            nn.ReLU(inplace=True),
            # AdaptiveAvgPool2d (Adaptive Average Pooling 2D): riduce la mappa visiva ad una dimensione fissa (2x2) per 128 filtri = 512 feature totali
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Linear (Fully Connected Linear Layer): strato di proiezione lineare per ridurre le feature visive da 512 a 256 numeri
        self.fc_vis = nn.Linear(128 * 2 * 2, 256)
        
        # Classificatore Ibrido Multimodale MLP (256 Feature Visive + 4 Feature Scalari Numeriche = 260 Input Totali)
        self.classifier = nn.Sequential(
            # Linear Layer: concatena feature visive e scalari numeriche (260 -> 128 neuroni)
            nn.Linear(256 + num_scalars, 128),
            # ReLU (Rectified Linear Unit): attivazione non lineare
            nn.ReLU(inplace=True),
            # Dropout (Dropout Regularization): spegne casualmente il 20% dei neuroni durante l'addestramento per prevenire l'overfitting (memorizzazione a memoria)
            nn.Dropout(0.2),
            # Linear Layer finale: produce i 5 Logits grezzi di output per ciascuna classe semantica
            nn.Linear(128, num_classes)
        )


    # Il metodo forward definisce la pipeline d'inferenza:
    # La CNN estrae 512 feature visive dal patch 2D che vengono proiettate a 256 e concatenate con i 4 scalari geometrici dell'ombra
    # Il vettore multimodale risultante di 260 elementi viene infine elaborato dal classificatore MLP per produrre i 5 logits di presenza per classe
    # Passaggio Forward nella rete neurale ibrida
    # patch_tensor: tensore 4D (Batch, 10, 64, 64) contenente i patch visivi 2D
    # scalar_features: tensore 2D (Batch, 4) contenente [area_sqm, distance_m, occluder_w, occluder_h]
    def forward(self, patch_tensor, scalar_features):
        # Estrazione delle caratteristiche visive dal patch 2D mediante l'encoder convoluzionale
        x_vis = self.conv_net(patch_tensor)
        # Flatten: appiattimento della mappa 2D in un vettore 1D per ogni campione della batch (512 feature visive)
        x_vis = torch.flatten(x_vis, start_dim=1)
        # Proiezione lineare e attivazione ReLU (Rectified Linear Unit: 256 feature)
        x_vis = F.relu(self.fc_vis(x_vis))
        
        # Concatenazione delle 256 feature visive con i 4 scalari numerici (260 feature totali)
        x_combined = torch.cat([x_vis, scalar_features], dim=1)
        
        # Classificazione finale ed estrazione dei 5 Logits di output
        logits = self.classifier(x_combined)
        return logits


if __name__ == "__main__":
    # Self-test dell'architettura neurale per-zone
    model = PerZoneModel()
    dummy_patch = torch.randn(4, 10, 64, 64)
    dummy_scalars = torch.randn(4, 4)
    out_logits = model(dummy_patch, dummy_scalars)
    out_probs = torch.sigmoid(out_logits)
    print("Self-Test PerZoneModel Completo!")
    print(f"  Input Patch Shape: {dummy_patch.shape}")
    print(f"  Input Scalars Shape: {dummy_scalars.shape}")
    print(f"  Output Logits Shape: {out_logits.shape} | Probs Range: [{out_probs.min():.3f}, {out_probs.max():.3f}]")
