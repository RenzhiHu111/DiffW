import torch
import torch.nn as nn
from kornia.enhance import AdjustContrast

class Adjust_Contrast(nn.Module):
    def __init__(self, factor):
        super(Adjust_Contrast, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        encoded = noised_and_cover[0]
        encoded = AdjustContrast(contrast_factor=self.factor)(encoded)
        return encoded
