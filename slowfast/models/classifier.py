import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):

    def __init__(self, cfg):
        super(SimpleClassifier, self).__init__()

        self.net = nn.Sequential(
                    nn.Linear(cfg.MI.V_CHANNEL, cfg.CLSF.NUM_HIDDEN),
#					nn.Dropout(cfg.CLSF.DROPOUT),
                    nn.BatchNorm1d(cfg.CLSF.NUM_HIDDEN),
                    nn.ReLU(),
                    nn.Linear(cfg.CLSF.NUM_HIDDEN, cfg.MODEL.NUM_CLASSES))

    def forward(self, x):
        return self.net(x)

    
def build_classifier(cfg):
    model = SimpleClassifier(cfg)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model

