import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import MRIDataset

def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNET(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2, init_features = 32):
        super(UNET, self).__init__()

        # Descending part of the U : the encoding part
        self.encoder1 = encoder_block(in_channels, init_features)
        self.encoder2 = encoder_block(init_features, init_features*2)
        self.encoder3 = encoder_block(init_features*2, init_features*4)
        self.encoder4 = encoder_block(init_features*4, init_features*8)

        # The bootleneck part
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=init_features*8, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=init_features*16, out_channels=init_features*16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = decoder_block(init_features*16, init_features*8)
        self.decoder2 = decoder_block(init_features*8, init_features*4)
        self.decoder3 = decoder_block(init_features*4, init_features*2)
        self.decoder4 = decoder_block(init_features*2, out_channels)

        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding part
        e1 = self.encoder1(x)
        e2 = self.encoder2(e2)
        e3 = self.encoder3(e3)
        e4 = self.encoder4(e3)

        # Bottom of the U : bottleneck
        b = self.bottleneck(e4)

        # Decoding part
        d1 = self.decoder1(b)
        d1 = torch.cat((d1, e4), dim=1) # concatenate the output of the encoder with the input of the incoder
        d2 = self.decoder2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d4 = self.decoder4(d3)
        d4 = torch.cat((d4, e1), dim=1)

        output = self.final_conv(d4)
        output = torch.sigmoid(output) # because want a binary output : 0 no aliasing and 1 aliasing

        return output