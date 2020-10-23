#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import os
import pickle
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
import pdb

import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.utils.c2_model_loading import get_name_convert_func

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
	"""
	Creates the checkpoint directory (if not present already).
	Args:
		path_to_job (string): the path to the folder of the current job.
	"""
	checkpoint_dir = os.path.join(path_to_job, "checkpoints")
	# Create the checkpoint dir from the master process
	if du.is_master_proc() and not PathManager.exists(checkpoint_dir):
		try:
			PathManager.mkdirs(checkpoint_dir)
		except Exception:
			pass
	return checkpoint_dir


def get_checkpoint_dir(path_to_job):
	"""
	Get path for storing checkpoints.
	Args:
		path_to_job (string): the path to the folder of the current job.
	"""
	return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch):
	"""
	Get the full path to a checkpoint file.
	Args:
		path_to_job (string): the path to the folder of the current job.
		epoch (int): the number of epoch for the checkpoint.
	"""
	name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
	return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
	"""
	Get the last checkpoint from the checkpointing folder.
	Args:
		path_to_job (string): the path to the folder of the current job.
	"""

	d = get_checkpoint_dir(path_to_job)
	names = PathManager.ls(d) if PathManager.exists(d) else []
	names = [f for f in names if "checkpoint" in f]
	assert len(names), "No checkpoints found in '{}'.".format(d)
	# Sort the checkpoints by epoch.
	name = sorted(names)[-1]
	return os.path.join(d, name)


def has_checkpoint(path_to_job):
	"""
	Determines if the given directory contains a checkpoint.
	Args:
		path_to_job (string): the path to the folder of the current job.
	"""
	d = get_checkpoint_dir(path_to_job)
	files = PathManager.ls(d) if PathManager.exists(d) else []
	return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
	"""
	Determine if a checkpoint should be saved on current epoch.
	Args:
		cur_epoch (int): current number of epoch of the model.
		checkpoint_period (int): the frequency of checkpointing.
	"""
	return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_job, model_dict, optimizer, epoch, cfg):
	"""
	Save a checkpoint.
	Args:
		model (model): model to save the weight to the checkpoint.
		optimizer (optim): optimizer to save the historical state.
		epoch (int): current number of epoch of the model.
		cfg (CfgNode): configs to save.
	"""
	# Save checkpoints only from the master process.
	if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
		return
	# Ensure that the checkpoint dir exists.
	PathManager.mkdirs(get_checkpoint_dir(path_to_job))
	# Omit the DDP wrapper in the multi-gpu setting.
	sd_dict = {}
	for k, v in model_dict.items():
		try:
			sd_dict[k] = v.module.state_dict() if cfg.NUM_GPUS > 1 else v.state_dict()
		except:
			sd_dict[k] = v.state_dict()
	# Record the state.
	checkpoint = {
		"epoch": epoch,
		"model_state": sd_dict,
		"optimizer_state": optimizer.state_dict(),
		"cfg": cfg.dump(),
	}
	# Write the checkpoint.
	path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
	with PathManager.open(path_to_checkpoint, "wb") as f:
		torch.save(checkpoint, f)
	return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
	"""
	Inflate 2D model weights in state_dict_2d to the 3D model weights in
	state_dict_3d. The details can be found in:
	Joao Carreira, and Andrew Zisserman.
	"Quo vadis, action recognition? a new model and the kinetics dataset."
	Args:
		state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
		state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
	Returns:
		state_dict_inflated (OrderedDict): a dict of inflated parameters.
	"""
	# v3d: C_out, C_in, K1, K2, K3
	# v2d: C_out, C_in, K_h, K_w
	state_dict_inflated = OrderedDict()
	for k, v2d in state_dict_2d.items():
		try:
			assert k in state_dict_3d.keys()
		except:
			logger.info(
				"skip {} during inflation".format(k))
			continue
		v3d = state_dict_3d[k]
		# Inflate the weight of 2D conv to 3D conv.
		if len(v2d.shape) == 4 and len(v3d.shape) == 5:
			logger.info(
				"Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
			)
			# Dimension need to be match.
			assert v2d.shape[-2:] == v3d.shape[-2:]
			assert v2d.shape[:2] == v3d.shape[:2]
			v3d = (
				v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
			)
		if v2d.shape == v3d.shape:
			v3d = v2d
		state_dict_inflated[k] = v3d.clone()
	return state_dict_inflated


def load_checkpoint(
	path_to_checkpoint,
	model_dict,
	data_parallel=True,
	optimizer=None,
	inflation=False,
	convert_from_caffe2=False,
):
	"""
	Load the checkpoint from the given file. If inflation is True, inflate the
	2D Conv weights from the checkpoint to 3D Conv.
	Args:
		path_to_checkpoint (string): path to the checkpoint to load.
		model (model): model to load the weights from the checkpoint.
		data_parallel (bool): if true, model is wrapped by
		torch.nn.parallel.DistributedDataParallel.
		optimizer (optim): optimizer to load the historical state.
		inflation (bool): if True, inflate the weights from the checkpoint.
		convert_from_caffe2 (bool): if True, load the model from caffe2 and
			convert it to pytorch.
	Returns:
		(int): the number of training epoch of the checkpoint.
	"""
	assert PathManager.exists(
		path_to_checkpoint
	), "Checkpoint '{}' not found".format(path_to_checkpoint)
	# Account for the DDP wrapper in the multi-gpu setting.

	if convert_from_caffe2:
		ms = model_dict['model'].module if data_parallel else model_dict['model'] 
		with PathManager.open(path_to_checkpoint, "rb") as f:
			caffe2_checkpoint = pickle.load(f, encoding="latin1")
		state_dict = OrderedDict()
		name_convert_func = get_name_convert_func()
		for key in caffe2_checkpoint["blobs"].keys():
			converted_key = name_convert_func(key)
			converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
			if converted_key in ms.state_dict():
				c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
				model_blob_shape = ms.state_dict()[converted_key].shape
				# Load BN stats to Sub-BN.
				if (
					len(model_blob_shape) == 1
					and len(c2_blob_shape) == 1
					and model_blob_shape[0] > c2_blob_shape[0]
					and model_blob_shape[0] % c2_blob_shape[0] == 0
				):
					caffe2_checkpoint["blobs"][key] = np.concatenate(
						[caffe2_checkpoint["blobs"][key]]
						* (model_blob_shape[0] // c2_blob_shape[0])
					)
					c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

				if c2_blob_shape == tuple(model_blob_shape):
					state_dict[converted_key] = torch.tensor(
						caffe2_checkpoint["blobs"][key]
					).clone()
					logger.info(
						"{}: {} => {}: {}".format(
							key,
							c2_blob_shape,
							converted_key,
							tuple(model_blob_shape),
						)
					)
				else:
					logger.warn(
						"!! {}: {} does not match {}: {}".format(
							key,
							c2_blob_shape,
							converted_key,
							tuple(model_blob_shape),
						)
					)
			else:
				if not any(
					prefix in key for prefix in ["momentum", "lr", "model_iter"]
				):
					logger.warn(
						"!! {}: can not be converted, got {}".format(
							key, converted_key
						)
					)
		ms.load_state_dict(state_dict, strict=False)
		epoch = -1
		return epoch

	# Load the checkpoint on CPU to avoid GPU mem spike.
	ms_dict = {}
	for k, v in model_dict.items():
		try:
			ms_dict[k] = v.module if data_parallel and v.module else v
		except:
			ms_dict[k] = v

	with PathManager.open(path_to_checkpoint, "rb") as f:
		checkpoint = torch.load(f, map_location="cpu")
	if inflation:
		# Try to inflate the model.
#		model_state_dict_3d = (
#			model.module.state_dict()
#			if data_parallel
#			else model.state_dict()
#		)
		model_state_dict_3d = ms_dict['model'].state_dict()
		inflated_model_dict = inflate_weight(
			checkpoint["model_state"], model_state_dict_3d
		)
		ms_dict['model'].load_state_dict(inflated_model_dict, strict=False)
	else:
		for k, ms in ms_dict.items():
			ms.load_state_dict(checkpoint["model_state"][k], strict=False)
		# Load the optimizer state (commonly not done when fine-tuning)
		if optimizer:
			optimizer.load_state_dict(checkpoint["optimizer_state"])
	if "epoch" in checkpoint.keys():
		epoch = checkpoint["epoch"]
	else:
		epoch = -1
	return epoch

def c2_normal_to_sub_bn(key, model_keys):
	"""
	Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
	Args:
		key (OrderedDict): source dict of parameters.
		mdoel_key (OrderedDict): target dict of parameters.
	Returns:
		new_sd (OrderedDict): converted dict of parameters.
	"""
	if "bn.running_" in key:
		if key in model_keys:
			return key

		new_key = key.replace("bn.running_", "bn.split_bn.running_")
		if new_key in model_keys:
			return new_key
	else:
		return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
	"""
	Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
	Args:
		checkpoint_sd (OrderedDict): source dict of parameters.
		model_sd (OrderedDict): target dict of parameters.
	Returns:
		new_sd (OrderedDict): converted dict of parameters.
	"""
	for key in model_sd:
		if key not in checkpoint_sd:
			if "bn.split_bn." in key:
				load_key = key.replace("bn.split_bn.", "bn.")
				bn_key = key.replace("bn.split_bn.", "bn.bn.")
				checkpoint_sd[key] = checkpoint_sd.pop(load_key)
				checkpoint_sd[bn_key] = checkpoint_sd[key]

	for key in model_sd:
		if key in checkpoint_sd:
			model_blob_shape = model_sd[key].shape
			c2_blob_shape = checkpoint_sd[key].shape

			if (
				len(model_blob_shape) == 1
				and len(c2_blob_shape) == 1
				and model_blob_shape[0] > c2_blob_shape[0]
				and model_blob_shape[0] % c2_blob_shape[0] == 0
			):
				before_shape = checkpoint_sd[key].shape
				checkpoint_sd[key] = torch.cat(
					[checkpoint_sd[key]]
					* (model_blob_shape[0] // c2_blob_shape[0])
				)
				logger.info(
					"{} {} -> {}".format(
						key, before_shape, checkpoint_sd[key].shape
					)
				)
	return checkpoint_sd

