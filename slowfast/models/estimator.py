#-----------------------------------------------------------------------------
# model definition
#-----------------------------------------------------------------------------
import pdb
import torch, torchvision, torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class LocalDiscriminator(nn.Module):
	def __init__(self, M_channels=128, V_channels=64, interm_channels=512):
		super().__init__()

		in_channels = V_channels + M_channels
		self.c0 = torch.nn.Conv2d(in_channels, interm_channels, 
			  kernel_size=1, stride=1, bias=False)
		self.c1 = torch.nn.Conv2d(interm_channels, interm_channels, 
			  kernel_size=1, stride=1, bias=False)
		self.c2 = torch.nn.Conv2d(interm_channels, 1, 
			  kernel_size=1, stride=1, bias=False)

	def forward(self, x):

		score = F.relu(self.c0(x))
		score = F.relu(self.c1(score))
		score = self.c2(score)

		return score


class Estimator(nn.Module):

	def __init__(self, cfg):
		super().__init__()

		self.get_models(M_channels=cfg.MI.M_CHANNEL, 
			  V_channels=cfg.MI.V_CHANNEL, 
			  interm_dim=cfg.MI.INTERM_DIM)

	def get_models(self, M_channels, V_channels, interm_dim):

		self.local_D = LocalDiscriminator(M_channels, V_channels, interm_dim)

	def forward(self, Y, M, M_fake):
		'''
			Y: [B, V_CHANNEL]
			M/M_fake: [B, M_CHANNEL, T, H, W]
		'''
		M = torch.cat(torch.split(M, 1, dim=2), dim=3).squeeze()
		M_fake = torch.cat(torch.split(M_fake, 1, dim=2), dim=3).squeeze()
		H, W = M.size(2), M.size(3)

		Y_replicated = Y.unsqueeze(-1).unsqueeze(-1)
		Y_replicated = Y_replicated.expand(-1, -1, H, W)
		
		Y_cat_M = torch.cat((M, Y_replicated), dim=1)
		Y_cat_M_fake = torch.cat((M_fake, Y_replicated), dim=1)

		# local loss
		Ej = -F.softplus(-self.local_D(Y_cat_M)).mean()
		Em = F.softplus(self.local_D(Y_cat_M_fake)).mean()
		local_loss = -(Ej - Em)

		return local_loss

def build_estimator(cfg):
	model = Estimator(cfg)
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

