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
		stdv = 1. / math.sqrt(self.inputSize / 3)
		self.register_buffer('memory', torch.rand(self.queueSize, self.inputSize).mul_(2 * stdv).add_(-stdv))
		self.register_buffer('relation_memory', torch.rand(self.queueSize, self.inputSize).mul_(2 * stdv).add_(-stdv))
		self.register_buffer('ground_truth', torch.zeros(self.queueSize).long())

		self.thresh = cfg.NCE.THRESH
		self.topk = cfg.NCE.TOPK
		self.selection = cfg.NCE.SELECTION
		self.do_QE = cfg.NCE.QE
		if self.do_QE:
			self.qe_num = cfg.NCE.QE_NUM

	def forward(self, q, k, o, labels):
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

#		Correlation Mining
		"""
			Available Input: 
			  relation_queue ([queue_size, feature_size])
			  o (current original image query)
			Output:
			  target ([batch_size, queue_size], multi-hot labels (0/1),
					type: Float )
		"""
		relation_queue = self.relation_memory.clone()
		similarity = torch.mm(relation_queue.detach(), o.transpose(1, 0))
		similarity = similarity.transpose(0, 1) # [bs, queue_size]

		if self.do_QE:
			topk_idx = torch.topk(similarity, dim=1, k=self.qe_num)[1] #[bs, qe_num]
			topk_feats = relation_queue[topk_idx.view(-1)].detach() #[bs*qe_num, f_size]
			topk_feats = topk_feats.view(-1, self.qe_num, self.inputSize) #[bs, qe_num, f_size]
			concat_feats = torch.cat([o.view(-1, 1, self.inputSize), topk_feats], dim=1) #[bs, qe_num+1, f_size] 
			weights = torch.arange(self.qe_num+1, 0, -1).repeat(o.size(0)).view(-1, self.qe_num+1, 1).cuda() # [bs, qe_num+1, f_size]
			val_feats = torch.mean(weights * concat_feats, axis=1) # [bs, f_size]
			val_feats = val_feats / torch.norm(val_feats, dim=1, keepdim=True)
			similarity = torch.mm(relation_queue.detach(), val_feats.transpose(1, 0)).transpose(0, 1) # [bs, queue_size]

		if self.selection == 'thresh':
			target = similarity.ge(self.thresh).float()  # [bs, queue_size]
		elif self.selection == 'topk':
			target = torch.topk(similarity, dim=1, k=self.topk)[1]
			target = torch.zeros_like(similarity).scatter_(
						dim=1, index=target, value=1) # convert to multi-hot
		elif self.selection == 'ground_truth':
			gt_label = self.ground_truth.expand(batchSize, -1)
			cur_label = labels.view(-1, 1).expand(-1, self.queueSize)
			target = (gt_label == cur_label).float() # [bs, queue_size]
		else:
			raise NotImplementedError('selection method {} is not supported'.format(self.selection))

		# Computing mininig acc
		q_label_idx, k_label_idx = torch.nonzero(target, as_tuple=True)
		q_label = labels[q_label_idx]
		k_label = self.ground_truth[k_label_idx]
		mining_num = torch.sum(q_label == k_label).float()

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
			self.ground_truth.index_copy_(0, out_ids, labels)
			self.index = (self.index + batchSize) % self.queueSize
		
		return out, target, mining_num


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
