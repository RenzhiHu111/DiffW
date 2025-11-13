import torch
import torch.nn as nn


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return image
