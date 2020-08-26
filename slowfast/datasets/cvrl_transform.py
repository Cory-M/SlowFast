import torch
import numpy as np
import slowfast.datasets.functional as F
from torch import Tensor
import torch.nn as nn
import math
import pdb
from . import transform as transform

def color_jitter_and_greyscale(images, cfg):
	# C, T, H, W
	# Temporal consistent augmentation
	if np.random.uniform() < cfg.COLOR_JITTER_PROB:		
		images = transform.color_jitter(images,
					img_brightness = cfg.BRIGHTNESS_FACTOR,
					img_contrast = cfg.CONTRAST_FACTOR,
					img_saturation = cfg.SATURATION_FACTOR,
					img_hue = cfg.HUE_FACTOR)
	if np.random.uniform() < cfg.GREYSCALE_PROB:
		images = grayscale_jitter(images)
	return images

def grayscale_jitter(images):
	# C, T, H, W
	# RGB order
	images = 0.299 * images[0] + 0.587 * images[1] + 0.114 * images[2]
	images = images.unsqueeze(0).expand(3, -1, -1, -1)
	return images

class GaussianBlur(nn.Module):
	def __init__(self, cfg):
		super(GaussianBlur, self).__init__()
		crop_size = cfg.DATA.TRAIN_CROP_SIZE
		channels = cfg.DATA.INPUT_CHANNEL_NUM[0]

		kernel_size = crop_size // 10
		radius = kernel_size // 2
		kernel_size = radius * 2 + 1 # to odd kernel size

		x_coord = torch.arange(-radius, radius+1)
		x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
		y_grid = x_grid.t()
		self.xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
		# process the gaussian kernel along x/y-axis simutanously
		
		self.filter = nn.Conv2d(in_channels=channels, out_channels=channels,
				kernel_size=kernel_size, groups=channels, bias=False, padding=radius)
#		self.filter.weight.requires_grad = False

		self.kernel_size = kernel_size
		self.channels = channels
	def random_update_sigma(self):
		sigma = np.random.uniform(0.1, 2.0)
		variance = sigma**2
		gaussian_kernel = (1./(2.*math.pi*variance)) *\
						  torch.exp(-torch.sum(self.xy_grid**2., dim=-1) /\
									  (2*variance))
		gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
	
		gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
		gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1).cuda()

		self.filter.weight.data = gaussian_kernel

	def forward(self, x):
		for i in range(x.size(0)):
			self.random_update_sigma()
			x[i] = self.filter(x[i].permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
		return x

def build_GaussianBlur(cfg):
	model = GaussianBlur(cfg)

	cur_device = torch.cuda.current_device()
	model = model.cuda(device=cur_device)

	if cfg.NUM_GPUS > 1:
		model = torch.nn.parallel.DistributedDataParallel(
#		model = torch.nn.parallel.DataParallel(
					module=model, device_ids=[cur_device], output_device=cur_device
				)
	return model
