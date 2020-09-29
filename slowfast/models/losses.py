#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import log_softmax
import pdb

class MultiLabelCCE(_WeightedLoss):
	def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
		super(MultiLabelCCE, self).__init__(weight, size_average, reduce, reduction)
		self.ignore_index = ignore_index
		self.reduction = reduction
		self.softmax = nn.Softmax(dim=1)
	
	def forward(self, input, target):
		"""
		input: N x C
		target: N x C, multi-hot label
		""" 
		scale_factor = torch.sum(target, dim=1).float()
		scaled_target = target / scale_factor.view(-1, 1)
		
		log_sftmax_input = log_softmax(input, 1)
#		assert (torch.sum(scaled_target) == input.size(0))
		if self.reduction == 'mean':
			return  torch.mean(torch.sum(-scaled_target * log_sftmax_input, dim=1))
		elif self.reduction == 'sum':
			return torch.sum(-scaled_target * log_sftmax_input)
		else:
			raise NotImplementedError("{} reduction is not supported for mlCCE loss".format(
							self.reduction))

			
_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
	"multi_label_cce": MultiLabelCCE,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

