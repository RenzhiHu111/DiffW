from noise_layers import *
import numpy as np
import torch.nn as nn

class Noise(nn.Module):

	def __init__(self):
		super(Noise, self).__init__()
		layers = ["Combined([JpegMask(50),Jpeg(10),Identity(),GF(2),Crop(0.15, 0.15)])"]
		# layers = ["GF(2)"]
		# layers = ["GF(3)"]
		# layers = ["GF(4)"]
		# layers = ["JpegTest(50)"]
		# layers = ["JpegTest(40)"]
		# layers = ["JpegTest(30)"]
		# layers = ["Identity()"]
		# layers = ["Dropout(0.7)"]
		# layers = ["Dropout(0.2)"]
		# layers = ["Dropout(0.1)"]
		# p=0.035
		# layers = ["Crop(0.1871, 0.1871)"]
		# p = 0.045
		# layers = ["Crop(0.21213, 0.21213)"]
		# p = 0.055
		# layers = ["Crop(0.23452, 0.23452)"]
		# layers = ["Cropout(0.8367, 0.8367)"]
		# layers = ["Cropout(0.5477, 0.5477)"]
		# layers = ["Cropout(0.4472, 0.4472)"]
		# layers = ["SP(0.3)"]
		# layers = ["SP(0.2)"]
		# layers = ["SP(0.1)"]
		# layers = ["MF(3)"]
		# layers = ["MF(5)"]
		# layers = ["MF(7)"]
		# layers = ["MF(9)"]
		# layers = ["GN(0.1)"]
		# layers = ["GN(0.2)"]
		# layers = ["GN(0.3)"]
		# layers = ["Adjust_Brightness(1.1)"]
		# layers = ["Adjust_Brightness(1.3)"]
		# layers = ["Adjust_Brightness(1.5)"]
		# layers = ["Adjust_Contrast(1.1)"]
		# layers = ["Adjust_Contrast(1.3)"]
		# layers = ["Adjust_Contrast(1.5)"]
		# layers = ["Adjust_hue(0.1)"]
		# layers = ["Adjust_hue(0.2)"]
		# layers = ["Adjust_hue(0.3)"]
		# layers = ["WebP(50)"]
		# layers = ["WebP(30)"]
		# layers = ["WebP(10)"]

		for i in range(len(layers)):
			layers[i] = eval(layers[i])
		self.noise = nn.Sequential(*layers)

	def forward(self, image_and_cover):
		noised_image = self.noise(image_and_cover)
		return noised_image
