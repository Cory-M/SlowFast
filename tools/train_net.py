#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import pdb
import os

import numpy as np
import pprint
import torch
import torch.nn.functional as F
import torch.distributed as dist
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import random
from torch.utils.tensorboard import SummaryWriter

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.utils.functions as func

import slowfast.datasets.cvrl_transform as ct

from slowfast.datasets import loader
from slowfast.models import build_model, build_classifier
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.misc import *
from sklearn.metrics import average_precision_score as aps

from torchvision.utils import save_image, make_grid


logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, classifier, optimizer, train_meter, cur_epoch, cfg, tb_logger, gaussian_blur):
	"""
	Perform the video training for one epoch.
	Args:
		train_loader (loader): video training loader.
		model (model): the video model to train.
		optimizer (optim): the optimizer to perform optimization on the model's
			parameters.
		train_meter (TrainMeter): training meters to log the training performance.
		cur_epoch (int): current epoch of training.
		cfg (CfgNode): configs. Details can be found in
			slowfast/config/defaults.py
	"""
	# Enable train mode.
	model.train()
	classifier.train()

	def set_bn_train(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.train()


	train_meter.iter_tic()
	data_size = len(train_loader)

	for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
		# Transfer the data to the current GPU device.
		clip_q, clip_k = inputs

		# clip_q and clip_k will be forwarded with the same backbone
		clip_q_k = [torch.cat([clip_q[0], clip_k[0]], dim=0)]

		clip_q_k = [x.cuda(non_blocking=True) for x in clip_q_k]
		del inputs

		if cfg.DATA.CVRL_AUG:
			with torch.no_grad():
				clip_q_k = [gaussian_blur(x) for x in clip_q_k]

		labels = labels.cuda()

		# Update the learning rate.
		lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
		optim.set_lr(optimizer, lr)

		
		if cfg.TRAIN.EVAL_FEATURE:
			feat_q_k, feature = model(clip_q_k)
		else:
			feat_q_k = model(clip_q_k)
			feature = feat_q_k
		
		feat_q, feat_k = torch.chunk(feat_q_k, 2, dim=0)

		# Compute the simCLR loss
		# Aggregate feat_q and feat_k over all nodes
		batch_size = feat_q.size(0)
		feat_q_large, feat_k_large = du.all_gather([feat_q, feat_k])
		enlarged_batch_size = feat_q_large.size(0)
		nce_labels = ((dist.get_rank() * batch_size) + torch.arange(batch_size)).cuda().long()
		masks = F.one_hot(nce_labels, enlarged_batch_size).cuda()
		
		logits_aa = torch.matmul(feat_q, feat_q_large.t()) / cfg.NCE.T
		logits_aa = logits_aa - masks * 1e9
		logits_bb = torch.matmul(feat_k, feat_k_large.t()) / cfg.NCE.T
		logits_bb = logits_bb - masks * 1e9

		logits_ab = torch.matmul(feat_q, feat_k_large.t()) / cfg.NCE.T
		logits_ba = torch.matmul(feat_k, feat_q_large.t()) / cfg.NCE.T

		loss_func_a = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
		loss_func_b = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
		loss_a = loss_func_a(torch.cat([logits_ab, logits_aa], dim=1), nce_labels)
		loss_b = loss_func_b(torch.cat([logits_ba, logits_bb], dim=1), nce_labels)
		loss = loss_a + loss_b

		# inference loss for the on-going classfier on the features
		feature = feature[:batch_size]
		inf_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
		inf_cls = classifier(feature.detach())
		inf_loss = inf_loss_fun(inf_cls, labels)

		# Compute the loss.
		Loss = loss + 0.1 * inf_loss

		# check Nan Loss.
		misc.check_nan_losses(loss)
		misc.check_nan_losses(inf_loss)

		# Perform the backward pass.
		optimizer.zero_grad()
		Loss.backward()

		# Update the parameters.
		optimizer.step()


		# Compute the errors.
		num_topks_correct = metrics.topks_correct(inf_cls, labels, (1, 5))
		top1_err, top5_err = [
			(1.0 - x / batch_size) * 100.0 for x in num_topks_correct
		]

		nce_topks_correct = metrics.topks_correct(logits_ab, nce_labels, (1, 5))
		nce_top1, nce_top5 = [
			(1.0 - x / batch_size) * 100.0 for x in nce_topks_correct
		]


		# Gather all the predictions across all the devices.
		if cfg.NUM_GPUS > 1:
			loss, top1_err, top5_err, nce_top1, nce_top5 = du.all_reduce(
				[loss, top1_err, top5_err, nce_top1, nce_top5]
			)
		# Copy the stats from GPU to CPU (sync point).
		loss, top1_err, top5_err, nce_top1, nce_top5 = (
			loss.item(),
			top1_err.item(),
			top5_err.item(),
			nce_top1.item(),
			nce_top5.item(),
		)

		# Update and log stats.
		train_meter.update_stats(
			top1_err, top5_err, nce_top1, nce_top5, loss, lr, batch_size * cfg.NUM_GPUS
		)
		train_meter.iter_toc()
		
		iter_stats = train_meter.log_iter_stats(cur_epoch, cur_iter)

		if du.is_master_proc() and (cur_iter + 1) % cfg.LOG_PERIOD == 0:
			top1_err, top5_err, nce_top1, nce_top5, loss = iter_stats
			step = cur_epoch * len(train_loader) + cur_iter
			tb_logger.add_scalar('train_loss', loss, step)
			tb_logger.add_scalar('top1_err', top1_err, step)
			tb_logger.add_scalar('top5_err', top5_err, step)
			tb_logger.add_scalar('nce_top1', nce_top1, step)
			tb_logger.add_scalar('nce_top5', nce_top5, step)


		train_meter.iter_tic()

	# Log epoch stats.
	train_meter.log_epoch_stats(cur_epoch)
	train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, classifier, val_meter, cur_epoch, cfg, tb_logger):
	"""
	Evaluate the model on the val set.
	Args:
		val_loader (loader): data loader to provide validation data.
		model (model): model to evaluate the performance.
		val_meter (ValMeter): meter instance to record and calculate the metrics.
		cur_epoch (int): number of the current epoch of training.
		cfg (CfgNode): configs. Details can be found in
			slowfast/config/defaults.py
	"""

	# Evaluation mode enabled. The running stats would not be updated.
	model.eval()
	classifier.eval()
	
	val_meter.iter_tic()

	gt_labels = []
	scores = []
	
	for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
		clip_q, clip_k = inputs
		if isinstance(clip_q, (list,)):
			clip_q = [x.cuda(non_blocking=True) for x in clip_q]
			clip_k = [x.cuda(non_blocking=True) for x in clip_k]
		else:
			clip_q = clip_q.cuda(non_blocking=True)
			clip_k = clip_k.cuda(non_blocking=True)
		del inputs

		labels = labels.cuda()
		gt_labels.append(F.one_hot(labels, 400))

		for key, val in meta.items():
			if isinstance(val, (list,)):
				for i in range(len(val)):
					val[i] = val[i].cuda(non_blocking=True)
			else:
				meta[key] = val.cuda(non_blocking=True)

		with torch.no_grad():
			if cfg.TRAIN.EVAL_FEATURE:
				_, feature = model(clip_q)
			else:
				feature = model(clip_q)
			inf_cls = classifier(feature)

		scores.append(inf_cls)
		# Compute the errors.
		num_topks_correct = metrics.topks_correct(inf_cls, labels, (1, 5))

		# Compute the NCE loss on validation set
		loss_func = losses.get_loss_func('cross_entropy')(reduction="mean")
		loss = loss_func(inf_cls, labels)


		# Combine the errors across the GPUs.
		top1_err, top5_err = [
			(1.0 - x / feature.size(0)) * 100.0 for x in num_topks_correct
		]
		if cfg.NUM_GPUS > 1:
			loss, top1_err, top5_err = du.all_reduce(
						[loss, top1_err, top5_err])

		# Copy the errors from GPU to CPU (sync point).
		loss, top1_err, top5_err = (
					loss.item(), top1_err.item(), top5_err.item())

		val_meter.iter_toc()
		# Update and log stats.
		val_meter.update_stats(
			loss, top1_err, top5_err, feature.size(0) * cfg.NUM_GPUS
		)

		val_meter.log_iter_stats(cur_epoch, cur_iter)
		val_meter.iter_tic()

	gt_labels = torch.cat(gt_labels, dim=0)
	scores = torch.cat(scores, dim=0)

	if cfg.NUM_GPUS > 1:
		gt_labels, scores = du.all_gather([gt_labels, scores])
		
	# Log epoch stats.
	top1_err, top5_err, loss = val_meter.log_epoch_stats(cur_epoch)
	if du.is_master_proc():
		step = cur_epoch * len(val_loader) + cur_iter
		mAP = aps(gt_labels.cpu().numpy(), scores.cpu().numpy())
		logger.info('epoch {} mAP = {}'.format(cur_epoch, mAP))
		tb_logger.add_scalar('val_loss', loss, step)
		tb_logger.add_scalar('val_top1_err', top1_err, step)
		tb_logger.add_scalar('val_top5_err', top5_err, step)
		tb_logger.add_scalar('val_mAP', mAP, step)

	del gt_labels, scores
	val_meter.reset()


def calculate_and_update_precise_bn(loader, model, cfg, num_iters=200):
	"""
	Update the stats in bn layers by calculate the precise stats.
	Args:
		loader (loader): data loader to provide training data.
		model (model): model to update the bn stats.
		num_iters (int): number of iterations to compute and update the bn stats.
	"""

	def _gen_loader():
		for inputs, _, _, _ in loader:
			clip_q, _ = inputs
			if isinstance(clip_q, (list,)):
				clip_q = [x.cuda(non_blocking=True) for x in clip_q]
			else:
				clip_q = clip_q.cuda(non_blocking=True)
			yield clip_q

	# Update the bn stats.
	update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
	"""
	Train a video model for many epochs on train set and evaluate it on val set.
	Args:
		cfg (CfgNode): configs. Details can be found in
			slowfast/config/defaults.py
	"""
	# Set up environment.
	du.init_distributed_training(cfg)
	# Set random seed from configs.
	np.random.seed(cfg.RNG_SEED)
	torch.manual_seed(cfg.RNG_SEED)

	# Setup logging format.
	logging.setup_logging(cfg.OUTPUT_DIR)
	tb_logger = SummaryWriter(cfg.OUTPUT_DIR)

	# Print config.
	logger.info("Train with config:")
	logger.info(pprint.pformat(cfg))

	# Build the video model and print model statistics.
	model = build_model(cfg)
	classifier = build_classifier(cfg)
	gaussian_blur = ct.build_GaussianBlur(cfg)



	if du.is_master_proc() and cfg.LOG_MODEL_INFO:
		misc.log_model_info(model, cfg, is_train=True)

	# Construct the optimizer.
	optimizer = optim.construct_optimizer(model, cfg, classifier)

	# Load a checkpoint to resume training if applicable.
	if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
		logger.info("Load from last checkpoint.")
		last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
		checkpoint_epoch = cu.load_checkpoint(
			last_checkpoint, 
			{'model': model,
			'classifier': classifier, 
			},
			cfg.NUM_GPUS > 1, 
			optimizer
		)
		start_epoch = checkpoint_epoch + 1
	elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
		logger.info("Load from given checkpoint file.")
		checkpoint_epoch = cu.load_checkpoint(
			cfg.TRAIN.CHECKPOINT_FILE_PATH,
			{'model': model},
			cfg.NUM_GPUS > 1,
			optimizer,
			inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
			convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
		)
		start_epoch = checkpoint_epoch + 1
	else:
		start_epoch = 0

	# Create the video train and val loaders.
	train_loader = loader.construct_loader(cfg, "train")
	val_loader = loader.construct_loader(cfg, "val")

	# Create meters.
	if cfg.DETECTION.ENABLE:
		train_meter = AVAMeter(len(train_loader), cfg, mode="train")
		val_meter = AVAMeter(len(val_loader), cfg, mode="val")
	else:
		train_meter = TrainMeter(len(train_loader), cfg)
		val_meter = ValMeter(len(val_loader), cfg)

	# Perform the training loop.
	logger.info("Start epoch: {}".format(start_epoch + 1))

	for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
		# Shuffle the dataset.
		loader.shuffle_dataset(train_loader, cur_epoch)
		# Train for one epoch.
		train_epoch(train_loader, model, classifier, optimizer, train_meter, cur_epoch, cfg, tb_logger, gaussian_blur)

		# Compute precise BN stats.
		if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
			calculate_and_update_precise_bn(
				train_loader, model, cfg, cfg.BN.NUM_BATCHES_PRECISE
			)
		_ = misc.aggregate_split_bn_stats(model)

		# Save a checkpoint.
		if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
			cu.save_checkpoint(
						cfg.OUTPUT_DIR, 
						{'model': model,
						'classifier': classifier, 
						},
						optimizer, 
						cur_epoch, 
						cfg)
		# Evaluate the model on validation set.
		if misc.is_eval_epoch(cfg, cur_epoch):
			eval_epoch(val_loader, model, classifier, val_meter, cur_epoch, cfg, tb_logger)
