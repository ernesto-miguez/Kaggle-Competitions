"""
Arquitecturas de modelos para cell tracking.

Contiene la definición del 3D U-Net y la función de pérdida personalizada.
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Bloque de dos convoluciones 3D con BatchNorm y ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net optimizado para detección de centroides celulares.
    
    Arquitectura:
    - Codificador: 2 bloques de convolución + pooling
    - Cuello de botella: 1 bloque de convolución
    - Decodificador: 2 bloques de convolución transpuesta
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        
        # Encoder (downsampling)
        self.enc1 = DoubleConv3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(f * 2, f * 4)

        # Decoder (upsampling)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(f * 4, f * 2)  # f*4 porque se concatena con enc2
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(f * 2, f)      # f*2 porque se concatena con enc1

        # Output layer
        self.out_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        # Decoding path
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class HeatmapMSELoss(nn.Module):
    """
    Función de pérdida personalizada para detección de centroides con mapa de calor.
    
    Amplifica el error en zonas donde hay célula (target=1) para darle más peso
    a las regiones de interés.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicciones sin activación del modelo (shape: [B, 1, Z, Y, X])
            targets: Ground truth (shape: [B, 1, Z, Y, X])
        
        Returns:
            Loss escalado por la presencia de células
        """
        # Convertir logits a probabilidades [0, 1] mediante sigmoid
        probs = torch.sigmoid(logits)
        
        # Pesar más el error en las zonas con célula (target=1)
        # weight = 1.0 en background, weight = 51.0 en foreground (células)
        weight = 1.0 + 50.0 * targets
        
        # MSE ponderado
        loss = weight * (probs - targets) ** 2
        return loss.mean()
