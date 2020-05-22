import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

	def __init__(self, cfg, nhead=1, nhid=512, nlayers=2, dropout=0.5):
		super(TransformerModel, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer
		self.ninp = cfg.MODEL.NUM_FEATURES
		encoder_layers = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#		self.encoder = nn.Embedding(ntoken, ninp)
#		self.ninp = cfg.MODEL.NUM_CLASSES
#		self.decoder = nn.Linear(ninp, ntoken)
		
#		self.init_weights()

#	def _generate_square_subsequent_mask(self, sz):
#		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#		return mask
	
#	def init_weights(self):
#		initrange = 0.1
#		self.encoder.weight.data.uniform_(-initrange, initrange)
#		self.decoder.bias.data.zero_()
#		self.decoder.weight.data.uniform_(-initrange, initrange)
	
	def forward(self, src):
#		if self.src_mask is None or self.src_mask.size(0) != len(src):
#			device = src.device
#			mask = self._generate_square_subsequent_mask(len(src)).to(device)
#			self.src_mask = mask
		src = src * math.sqrt(self.ninp)
#		src = self.pos_encoder(src)
		output = self.transformer_encoder(src)#, self.src_mask)
		# Mask is inplemented on inputs, here is bi-directional transformer
		# without mask
#		output = self.decoder(output)
		return output

def build_transformer(cfg):
	model = TransformerModel(cfg)
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

