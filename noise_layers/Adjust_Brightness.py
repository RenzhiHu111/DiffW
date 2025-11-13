import torch
import torch.nn as nn
from kornia.enhance import AdjustBrightness

class Adjust_Brightness(nn.Module):
    def __init__(self, factor):
        super(Adjust_Brightness, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = noised_and_cover[0]
        encoded = AdjustBrightness(brightness_factor=self.factor)(encoded)
        return encoded
