import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self, x):
        return x + self.block(x)
    
class GeneratorResNet(nn.Module):
    def __init__(self, input_channels: int, n_residual_blocks: int = 15):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3,padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks)])

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3,padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        res = self.res_blocks(enc3)
        dec1 = self.decoder1(res) + enc2  # Skip connection
        dec2 = self.decoder2(dec1) + enc1  # Skip connection
        return self.output_layer(dec2)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )
    
    def forward(self, img):
        return self.model(img)
