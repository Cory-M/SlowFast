#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode

def add_custom_config(_C):
	# Add your own customized configs.
	_C.NCE = CfgNode()
	_C.NCE.K = 16384
	_C.NCE.T = 0.07
	_C.NCE.M = 0.5
	_C.NCE.ALPHA = 0.999
	_C.MODEL.FEATURE_L2 = True

	_C.METRIC = CfgNode()
	_C.METRIC.TRAIN = ['top1_err', 'top5_err', 'loss']
	_C.METRIC.EVAL = ['top1_err', 'top5_err', 'loss']
	_C.METRIC.EVAL_MIN = ['top1_err', 'top5_err']

	_C.DATA.PATH_TO_DATA_FILE = "./meta/"

	_C.MODEL.NUM_VERB_CLASSES = 125
	_C.MODEL.NUM_NOUN_CLASSES = 331

	pass
