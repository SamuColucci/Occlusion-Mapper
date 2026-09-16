import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    Blocco Channel Attention (Squeeze-and-Excitation):
    Ricalibra dinamicamente l'importanza degli 11 canali di input:
      - Canale 0: Punti LiDAR 3D proiettati su griglia BEV a terra
      - Canale 1: Maschera binaria del cono d'ombra / zona occlusa (dal Raycaster)
      - Canale 2: Mappa HD dell'area carrabile / asfalto (Drivable Area + Parcheggi)
      - Canale 3: Mappa HD dei marciapiedi e camminamenti pedonali (Walkway)
      - Canale 4: Mappa HD delle strisce pedonali e attraversamenti (Ped Crossing)
      - Canale 5: Maschera spaziale delle Auto già visibili nei dintorni
      - Canale 6: Maschera spaziale dei Camion e Bus visibili nei dintorni
      - Canale 7: Maschera spaziale dei Pedoni visibili nei dintorni
      - Canale 8: Maschera spaziale dei Ciclisti visibili nei dintorni
      - Canale 9: Maschera spaziale dei Motociclisti visibili nei dintorni
      - Canale 10: Maschera spaziale delle Barriere / Guardrail / Muri visibili
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Dimensione del collo di bottiglia per ridurre il numero di parametri
        reduced = max(4, channels // reduction)
        # Ricevi 11 canali in input e ne emetti 'reduced' canali in uscita
        self.fc1 = nn.Linear(channels, reduced)
        # Ricevi 'reduced' canali in input e ne emetti 'channels' canali in uscita
        self.fc2 = nn.Linear(reduced, channels)

    # Funzione che permette di dare importanze diverse ai vari canali in base alla loro presenza nella zona d'ombra 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estrae le dimensioni del tensore 4D: b=batch, c=canali (11), altezza (64), larghezza (64)
        # Tensore: griglia di numeri organizzati in 4 dimensioni (batch, canali, altezza, larghezza)
        b, c, _, _ = x.shape
        # Riduciamo le 11 mappe di input a un vettore di 11 numeri medi riassuntivi
        y = x.view(b, c, -1).mean(dim=2)
        # Azzeriamo i valori negativi, sfruttando la fc1 che usa 4 canali
        # Permettendo di ignorare le mappe di input che non sono rilevanti per la zona considerata
        y = F.relu(self.fc1(y))

        # Calcoliamo gli 11 valori per ciascuno degli 11 canali di input
        # questi valori saranno compresi tra 0 e 1, attraverso la sigmoide 
        # che li tarsforma in percentuali
        # applicando con la view lo stesso peso ai 64x64 pixel di quel canale considerato
        # essendo gli 11 valori dei numeri singoli
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Moltiplichiamo ogni canale di input per il peso calcolato al fine di capire quali canali considerare

        return x * y

class ResidualSEBlock(nn.Module):
    """
    Blocco Convoluzionale Residuo.
    Sfrutta due percorsi paralleli:
        - principale: Fa due convoluzioni 2D, normalizza i valori e applica Squeeze Excitation
        - shortcut: Prende l'immagine originale e la porta alla fine senza toccarla
    Successivamente, i due percorsi vengono sommati tra loro in modo da non perdere le informazioni originali
    """

    # in_c: numero di canali in input
    # out_c: numero di canali in output
    # stride: passo con cui l'immagine viene ridotta
    # stride=1: stessa risoluzione dell'input
    # stride=2: dimezza la risoluzione
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        # Primo strato di convoluzione
        # Scorriamo out_c per estrarre bordi e forme geometriche di base 
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        # Normalizza i valori ottenuti in conv1
        self.bn1 = nn.BatchNorm2d(out_c)
        
        # Secondo strato di convoluzione
        # Combina le forma semplice estratte in precedenza per costruire dettagli più complessi
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        # Normalizza i valori ottenuti in conv2
        self.bn2 = nn.BatchNorm2d(out_c)
        # Applica Squeeze Excitation per calibrare l'importanza dei canali
        self.se = SqueezeExcitation(out_c)

        # Prendiamo l'immagine di partenza e la portiamo alla fine senza toccarla, ovvero gli 11 canali
        # di input sovrapposti
        
        self.shortcut = nn.Sequential()
        # Controlliamo se le dimensioni dei canali sono cambiate 
        if stride != 1 or in_c != out_c:
            # Se ci sono state modifiche, applichiamo una convoluzione 1x1 per riportare i canali alla dimensione corretta
            self.shortcut = nn.Sequential(
                # Trasforma i canali da 11 a 32, dato che le due convoluzione producono come 
                # output 32 canali di 32x32 pixel
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                # Normalizza i valori in output della convoluzione della shortcut
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Salviamo in res la copia dell'input adattata (stessa dimensione di out)
        res = self.shortcut(x)
        # Primo strato di convoluzione con normalizzazione
        # ReLU permette di considerare solo i valori positivi, azzerando il rumore
        out = F.relu(self.bn1(self.conv1(x)))
        # Secondo strato di convoluzione con normalizzazione e Squeeze Excitation
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        # Sommiamo res a out e applichiamo ReLU
        # Permette che informazioni importanti del layer precedente vengano conservate
        out = F.relu(out + res)
        return out

# Modello per calcolare la probabilità che sia presente un oggetto nell'area occlusa considerata
# Prende gli 11 canali di input, applica Squeeze-and-Excitation per calibrare l'importanza dei canali
# Estrae le sagome geometriche attraverso 4 blocchi convoluzionali residui
# in_channels: numero di canali in input (11)
# num_classes: numero di classi di output (6)
# num_scalars: numero di scalari in input (9) (area, distanza, larghezza, lunghezza, asfalto, marciapiede, strisce, dinamico, terreno)
class AttentionPerZoneModel(nn.Module):
    def __init__(self, in_channels: int = 11, num_classes: int = 6, num_scalars: int = 9):
        super().__init__()
        self.num_classes = num_classes

        # Applichiamo Squeeze Excitation agli 11 canali di input per pesare l'importanza di ciascun canale
        self.input_se = SqueezeExcitation(in_channels, reduction=2)

        # Trasforma gli 11 canali 64x64 in 32 canali 32x32 applicando due blocchi con stride 2
        # Trova i bordi e i contorni più semplici
        self.stage1 = ResidualSEBlock(in_channels, 32, stride=2)  # 64x64 -> 32x32
        # Trasforma i 32 canali 32x32 in 64 canali 16x16
        # Estrae forme più complesse combinando i contorni trovati in precedenza
        self.stage2 = ResidualSEBlock(32, 64, stride=2)           # 32x32 -> 16x16
        # Trasforma i 64 canali 16x16 in 128 canali 8x8
        # Riconosce texture complesse e pattern geometrici
        self.stage3 = ResidualSEBlock(64, 128, stride=2)          # 16x16 -> 8x8
        # Trasforma i 128 canali 8x8 in 128 canali 4x4
        # Comprensione semantica della scena
        self.stage4 = ResidualSEBlock(128, 128, stride=2)         # 8x8 -> 4x4

        #Trasforma la griglia finale in un unico vettore di 128 numeri
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Trasforma i 9 numeri scalari num_scalars in un vettore di 128 numeri
        self.scalar_mlp = nn.Sequential(
            # Prende i 9 numeri scalari e li trasforma in 64 numeri per combinare i dati geometrici con i dati visivi
            nn.Linear(num_scalars, 64),
            # Normalizza i 64 numeri
            nn.LayerNorm(64),
            # Applica ReLU, azzera i negativi per eliminare informazioni irrilevanti
            nn.ReLU(),
            # Prende i 64 numeri e li trasforma in 128 numeri
            nn.Linear(64, 128),
            # Normalizza i 128 numeri
            nn.LayerNorm(128),
            # Applica ReLU, azzera i negativi per eliminare informazioni irrilevanti
            nn.ReLU()
        )

        # Prende il vettore dei 9 scalari elaborato (128 numeri) e genera 256 numeri suddivisi in gamma e beta
        # gamma: Moltiplica il segnale visivo per aumentarlo o diminuirlo in base alle esigenze
        # beta: Somma un valore costante di partenza al fine di aggiungere una spinta di base in alcune zone ad esempio sulle strisce pedonali per i pedoni
        self.film_gen = nn.Linear(128, 128 * 2)
        

        # Prende le informazioni raccolte ed emette i punteggi finali per le 6 classi di pericoli considerate
        self.classifier = nn.Sequential(
            # Fonde visione e mappa, comprimendo il tutto a 128 elementi
            nn.Linear(128 + 128, 128),
            # Normalizza i 128 numeri
            nn.LayerNorm(128),
            # Applica ReLU, azzera i negativi per eliminare informazioni irrilevanti
            nn.ReLU(),
            # Ogni ciclo spegno il 25% dei neuroni per evitare alla rete di imparare a memoria le scene del dataset
            nn.Dropout(0.25),
            # Prende i 128 numeri e li riduce a 64 feature decisionali
            nn.Linear(128, 64),
            # Applica ReLU, azzera i negativi per eliminare informazioni irrilevanti
            nn.ReLU(),
            # Emette i 6 punteggi finali (Logit) per le 6 classi di pericolo
            nn.Linear(64, num_classes)
        )

    def forward(self, patch: torch.Tensor, scalars: torch.Tensor, occluder_mask: torch.Tensor = None) -> torch.Tensor:
        # Regola i volumi iniziali degli 11 canali di input
        x = self.input_se(patch)

        # Estrazione dei contorni e delle sagome geometriche
        x = self.stage1(x)
        # Estrazione di forme più complesse combinando i contorni trovati in precedenza
        x = self.stage2(x)
        # Riconoscimento di texture complesse e pattern geometrici
        x = self.stage3(x)
        # Comprensione semantica della scena
        x = self.stage4(x)
        # Trasforma la griglia finale in un unico vettore di 128 numeri
        vis_feat = self.global_pool(x).flatten(1) # [B, 128]

        # Elaborazione dei 9 dati fisici della mappa nel vettore sc_feat
        sc_feat = self.scalar_mlp(scalars)        # [B, 128]

        # Crea i parametri gamma e beta usati per modulare i filtri visivi
        film_params = self.film_gen(sc_feat)
        # Divide i parametri gamma e beta
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        # Modula i filtri visivi con i parametri gamma e beta per aumentare o diminuire l'importanza delle informazioni
        modulated_vis = vis_feat * (1.0 + torch.tanh(gamma)) + beta

        # Uniamo il vettore dei 128 numeri visivi modificato con il vettore dei 128 numeri scalari
        combined = torch.cat([modulated_vis, sc_feat], dim=1) # [B, 256]

        # Applica il classificatore ai dati combinati per ottenere i logit finali (end-to-end)
        logits = self.classifier(combined)

        # Modulo Neuro-Simbolico Integrato (Logit Masking Differenziabile):
        # Se viene fornito il vincolo dell'occludore, applica la maschera ontologica direttamente ai logit
        if occluder_mask is not None:
            if occluder_mask.dim() == 1:
                occluder_mask = occluder_mask.unsqueeze(0)
            logits = logits + (1.0 - occluder_mask.to(logits.device)) * (-10000.0)

        return logits

    @staticmethod
    def build_compatibility_mask(
        occluder_name=None,
        occluder_wlh=None,
        road_f=None,
        roadside_f=None,
        device=None
    ) -> torch.Tensor:
        """
        Modulo Simbolico Integrato (Ontologia Fisico-Spaziale + Affordance HD-Map):
        Genera il tensore di consistenza fisica M in {0, 1}^6:
        - Pedone / Ciclista / Sagoma stretta (w < 0.95m): ammessi solo VRU (Pedoni/Bici, indici 2 e 4)
        - Auto standard (h < 1.50m, non pesante, non muro): escluso Camion/Bus (indice 1)
        - Affordance HD-Map: Veicoli (Auto, Camion, Moto) ammessi SOLO se la zona contiene carreggiata
          o si trova a bordo strada entro 2.5m (road_f >= 0.05 o roadside_f >= 0.15).
          In zone 100% terreno o pedonali isolate da strade, Auto, Camion e Moto vengono categoricamente escluse.
        """
        # [Auto(0), Camion(1), Pedone(2), Moto(3), Bici(4), Barriera(5)]
        mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if occluder_name is not None or occluder_wlh is not None:
            name_l = str(occluder_name).lower() if occluder_name is not None else ""
            occ_h = float(occluder_wlh[2]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None
            occ_w = float(occluder_wlh[0]) if (occluder_wlh is not None and len(occluder_wlh) >= 3) else None

            is_human_or_bike = any(k in name_l for k in ["human", "pedestrian", "bicycle", "motorcycle"])
            is_narrow = (occ_w is not None and occ_w < 0.95)
            is_heavy = any(k in name_l for k in ["truck", "bus", "trailer", "construction"])
            is_manmade = ("static.manmade" in name_l or "building" in name_l or "wall" in name_l)
            is_tall = (occ_h is not None and occ_h >= 1.50) # Calibrato: veicoli >= 1.50m (SUV, crossover, van, furgoni) possono celare mezzi commerciali leggeri

            if is_human_or_bike or is_narrow:
                # Sagoma stretta: ammessi solo Pedone (2) e Bici (4)
                mask = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
            elif not (is_heavy or is_manmade or is_tall):
                # Berlina molto bassa (h < 1.50m): quota LiDAR a 1.84m esclude fisicamente Camion/Bus (1)
                mask = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        # Vincolo di Affordance Semantica (HD-Map): Veicoli ammessi solo su asfalto o accosto al cordolo
        if road_f is not None and roadside_f is not None:
            if road_f < 0.05 and roadside_f < 0.15:
                mask[0] = 0.0  # Auto
                mask[1] = 0.0  # Camion
                mask[3] = 0.0  # Moto

        tensor_m = torch.tensor(mask, dtype=torch.float32)
        if device is not None:
            tensor_m = tensor_m.to(device)
        return tensor_m

    def forward_vru(self, patch: torch.Tensor, scalars: torch.Tensor, occluder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Output calibrato a 4 macro-classi con vincolo neuro-simbolico incorporato:
        [0: Auto, 1: Camion/Bus, 2: VRU (Pedoni+Moto+Bici), 3: Barriera]
        """
        # Esegue il forward pass e ottiene i punteggi finali mascherati
        logits_6 = self.forward(patch, scalars, occluder_mask=occluder_mask)
        # Applica la sigmoide per ottenere le probabilità percentuali
        probs_6 = torch.sigmoid(logits_6)
        # Prende le probabilità delle 3 classi VRU e ne fa il massimo per ottenere un'unica probabilità per la macro-classe VRU
        probs_vru = torch.max(probs_6[:, 2:5], dim=1, keepdim=True)[0]
        # Unisce le probabilità delle 4 macro-classi
        probs_4 = torch.cat([probs_6[:, 0:2], probs_vru, probs_6[:, 5:6]], dim=1)
        return probs_4
