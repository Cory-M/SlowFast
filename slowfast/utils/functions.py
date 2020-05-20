import torch

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
	return tensor.view(cfg.TRAIN.BATCH_SIZE, -1, cfg.MODEL.NUM_CLASSES)








