import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import pdb

class MemoryMoCo(nn.Module):
	"""Fixed-size queue with momentum encoder"""

	def __init__(self, cfg):
		super(MemoryMoCo, self).__init__()
		self.inputSize = cfg.MODEL.NUM_EMBEDDING
		self.queueSize = cfg.NCE.K
		self.T = cfg.NCE.T
		self.index = 0
		self.thresh = cfg.NCE.THRESH

		stdv = 1. / math.sqrt(self.inputSize / 3)
		self.register_buffer('memory', torch.rand(self.queueSize, self.inputSize).mul_(2 * stdv).add_(-stdv))
		self.register_buffer('relation_memory', torch.rand(self.queueSize, self.inputSize).mul_(2 * stdv).add_(-stdv))

	def forward(self, q, k, o):
		batchSize = q.shape[0]
		k = k.detach()
		o = o.detach()

		# pos logit
		l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
		l_pos = l_pos.view(batchSize, 1) # [bs, 1]
		# neg logit
		queue = self.memory.clone()
		l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
		l_neg = l_neg.transpose(0, 1)
		# correlation mining
		relation_queue = self.relation_memory.clone()
		similarity = torch.mm(relation_queue.detach(), o.transpose(1, 0))
		similarity = similarity.transpose(0, 1) # [bs, queue_size]
		target = similarity.ge(self.thresh).long()  # [bs, queue_size], 0/1 long

		out = torch.cat((l_pos, l_neg), dim=1) # [bs, 1 + queue_size]

		out = torch.div(out, self.T)
		out = out.squeeze().contiguous()

		# # update memory and relation memory
		with torch.no_grad():
			out_ids = torch.arange(batchSize).cuda()
			out_ids += self.index
			out_ids = torch.fmod(out_ids, self.queueSize)
			out_ids = out_ids.long()
			# index_copy_: (dim, index, tensor)
			self.memory.index_copy_(0, out_ids, k)
			self.relation_memory.index_copy_(0, out_ids, o)

			self.index = (self.index + batchSize) % self.queueSize

		return out, target


def build_moco_nce(cfg):
	model = MemoryMoCo(cfg)
	# Transfer the model to the current GPU device
	model = model.cuda()
	# Use multi-process data parallel model in the multi-gpu setting
	# if cfg.NUM_GPUS > 1:
	#	  # Make model replica operate on the current device
	#	  model = torch.nn.parallel.DistributedDataParallel(
	#		  module=model, device_ids=[cur_device], output_device=cur_device
	#	  )
	return model
