import torch
import pdb
import random

def flatten(tensor, cfg):
	CF, H = cfg.DATA.CHUNK_FRAMES, cfg.DATA.TRAIN_CROP_SIZE
	if isinstance(tensor, (list,)):
		tensor[0] = tensor[0].view(-1, cfg.DATA.INPUT_CHANNEL_NUM[0], 
					CF // cfg.SLOWFAST.ALPHA, H, H)
		tensor[1] = tensor[1].view(-1, cfg.DATA.INPUT_CHANNEL_NUM[1],
					CF, H, H)
	else:
		tensor = tensor.view(-1, cfg.DATA.INPUT_CHANNEL_NUM, CF, H, H)
	return tensor
			
def unflatten(tensor, cfg):
	return tensor.view(-1, 
				cfg.DATA.NUM_FRAMES//cfg.DATA.CHUNK_FRAMES, 
				cfg.MODEL.NUM_FEATURES)


def maskout(tensor, cfg):
	B, CF, FN = tensor.size()
#	N = int(cfg.TRANSFORMER.RATIO * CF)
#	k = random.randint(0, CF-1)
	k = 2
	mask = torch.zeros(B, CF)
	for i in range(B):
		mask[i, (i+k)%CF] = 1
#	mask[[[i, (i+k)%CF] for i in range(B)]] = 1 
	mask = (mask == 1).cuda()
	if cfg.MODEL.TRAINABLE_MASK:
		return tensor.masked_fill(mask.view(B, CF, 1), 0), mask	
	else:
		return tensor.masked_fill(mask.view(B, CF, 1), 1), mask

#	bool_mask = torch.rand(B, CF).le(cfg.TRANSFORMER.RATIO).cuda()
#	int_mask = torch.ones(B, CF, FN).cuda()
#	int_mask[bool_mask] = 0
#	return tensor*int_mask, bool_mask

def compute_score(mask, feature, preds):
	N = feature[mask].size(0)
	score = torch.matmul(preds[mask], feature[mask].transpose(0,1))
	target = torch.eye(N).argmax(dim=1).cuda()
	return score, target

