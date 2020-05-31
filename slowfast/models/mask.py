import torch
import torch.nn as nn
import pdb

class Trainable_Mask(nn.Module):

	def __init__(self, cfg):
		super(Trainable_Mask, self).__init__()
		self.mask = nn.Parameter(torch.randn(cfg.MODEL.NUM_FEATURES))

	def forward(self, x, mask):
#		self.mask = nn.Parameter(self.mask / torch.norm(self.mask))
		x[mask] += self.mask
		return x

def build_mask(cfg):
	model = Trainable_Mask(cfg)
	cur_device = torch.cuda.current_device()
	# Transfer the model to the current GPU device
	model = model.cuda(device=cur_device)
	# Use multi-process data parallel model in the multi-gpu setting
	if cfg.NUM_GPUS > 1:
		# Make model replica operate on the current device
		model = torch.nn.parallel.DistributedDataParallel(
			module=model, device_ids=[cur_device], output_device=cur_device
		)
	return model


