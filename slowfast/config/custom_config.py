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
	_C.NCE.THRESH = 0.99
	_C.NCE.TOPK = 5
	_C.NCE.SELECTION = 'thresh'
	pass
