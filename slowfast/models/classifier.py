import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):

	def __init__(self, cfg):
		super(SimpleClassifier, self).__init__()
		self.net = nn.Sequential(
					nn.Linear(cfg.MODEL.NUM_FEATURES, cfg.MODEL.NUM_CLASSES)
		)
		self.K = cfg.SLOWFAST.K
	def forward(self, x):
		x, _ = torch.topk(x, self.K, dim=1)
		return self.net(x.mean(dim=1))

	
def build_classifier(cfg):
	model = SimpleClassifier(cfg)
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

