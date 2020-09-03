import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SimpleClassifier', 'LinearProbe']

class SimpleClassifier(nn.Module):

	def __init__(self, cfg, num_hidden=2000, dropout=0.2):
		super(SimpleClassifier, self).__init__()
		self.net = nn.Sequential(
					nn.Linear(cfg.MODEL.NUM_FEATURES, num_hidden),
					nn.Dropout(dropout),
					nn.BatchNorm1d(num_hidden),
					nn.ReLU(),
					nn.Linear(num_hidden, cfg.MODEL.NUM_CLASSES))
	def forward(self, x):
		return self.net(x)

class LinearProbe(nn.Module):
	def __init__(self, cfg):
		super(LinearProbe, self).__init__()
		input_size = cfg.MODEL.NUM_FEATURE
		self.fc = nn.Linear(input_size, cfg.MODEL.NUM_CLASSES)
		# TODO
		# self.fc = nn.Linear(cfg.MODEL.NUM_FEATURE, cfg.MODEL.NUM_CLASSES)
	def forward(self, x):
		return self.fc(x)

	
def build_classifier(cfg):
#	model = SimpleClassifier(cfg)
	model = globals()[cfg.MODEL.CLASSIFIER](cfg)
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

