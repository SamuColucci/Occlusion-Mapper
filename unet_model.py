"""
Definizione dell'Architettura UNet per la Predizione delle Occlusioni in BEV.

Questo script implementa una rete neurale convoluzionale Encoder-Decoder di tipo UNet.
La rete riceve in input una griglia Bird's Eye View (BEV) a 4 canali (dimensioni: Bx4x200x200)
e restituisce in output una heatmap a 1 canale (dimensioni: Bx1x200x200) che rappresenta
la probabilità di occupazione nascosta stramite inferenza neurale.
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Blocco convoluzionale doppio: (Convoluzione 2D -> Batch Normalization -> ReLU) x2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetBEV(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[32, 64, 128]):
        super(UNetBEV, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Discesa): Riduzione spaziale del BEV ed estrazione del contesto semantico
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Decoder (Salita): Ripristino della risoluzione spaziale (200x200) tramite up-sampling
        for feature in reversed(features):
            # Convoluzione trasposta per l'up-sampling
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck (Il punto più profondo a bassa risoluzione)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Convoluzione finale (Output a singolo canale con attivazione Sigmoide per probabilità)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        # Fase di Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Inversione dell'ordine delle skip connections per il decoder
        skip_connections = skip_connections[::-1]

        # Fase di Decoder
        for idx in range(0, len(self.ups), 2):
            # Up-sample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Concatenazione lungo la dimensione dei canali (Skip Connection)
            concat_x = torch.cat((skip_connection, x), dim=1)
            # Doppio blocco convoluzionale
            x = self.ups[idx + 1](concat_x)

        return self.sigmoid(self.final_conv(x))

if __name__ == "__main__":
    # Test rapido di validità strutturale della rete
    model = UNetBEV(in_channels=4, out_channels=3)
    x = torch.randn((1, 4, 200, 200)) # Batch di 1 immagine a 4 canali 200x200
    preds = model(x)
    print("STRUTTURA UNET VERIFICATA:")
    print("Input shape: ", x.shape)
    print("Output shape:", preds.shape)
