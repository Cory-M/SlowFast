import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from slowfast.models.transformer_simplified import *

class TransformerV2Model(nn.Module):

    def __init__(self, cfg, nhead=1, nhid=512, nlayers=2, dropout=0.5):
        super(TransformerV2Model, self).__init__()
        self.ninp = cfg.MODEL.NUM_FEATURES
        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, self.ninp)
        ff = PositionwiseFeedForward(self.ninp, nhid, dropout)
        self.position = PositionalEncoding(self.ninp, self.dropout)
        self.transformer_encoder = Encoder(EncoderLayer(self.ninp, c(attn), c(ff), dropout), nlayers)


    def forward(self, src):
        src = src * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, None)  # , self.src_mask)
        return output


def build_transformer(cfg):
    model = TransformerV2Model(cfg)
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