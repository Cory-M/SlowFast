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
from slowfast.models import build_model, build_classifier, build_moco_nce
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.misc import *
from sklearn.metrics import average_precision_score as aps

from torchvision.utils import save_image, make_grid


logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, classifier, model_ema, moco_nec, optimizer, train_meter, cur_epoch, cfg, tb_logger, gaussian_blur):
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
	model_ema.eval()

	def set_bn_train(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.train()

	model_ema.apply(set_bn_train)

	train_meter.iter_tic()
	data_size = len(train_loader)

	for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
		# Transfer the data to the current GPU device.
		clip_q, clip_k = inputs
		if isinstance(clip_q, (list,)):
			clip_q = [x.cuda(non_blocking=True) for x in clip_q]
			clip_k = [x.cuda(non_blocking=True) for x in clip_k]
		else:
			clip_q = clip_q.cuda(non_blocking=True)
			clip_k = clip_k.cuda(non_blocking=True)
		del inputs

		if cfg.DATA.CVRL_AUG:
			with torch.no_grad():
				clip_q = [gaussian_blur(x) for x in clip_q]
				clip_k = [gaussian_blur(x) for x in clip_k]

		verb_label, noun_label = labels
		verb_label, noun_label = verb_label.cuda(), noun_label.cuda()

		for key, val in meta.items():
			if isinstance(val, (list,)):
				for i in range(len(val)):
					val[i] = val[i].cuda(non_blocking=True)
			else:
				meta[key] = val.cuda(non_blocking=True)

		# Update the learning rate.
		lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
		optim.set_lr(optimizer, lr)

		
		if cfg.TRAIN.EVAL_FEATURE:
			feat_q, feature = model(clip_q)
		else:
			feat_q = model(clip_q)
			feature = feat_q
	
		with torch.no_grad():
			if cfg.NUM_GPUS > 1 and cfg.BN.SHUFFLE_BN:
				gather_clip_k = du.all_gather(clip_k)
				idx_shuffle = torch.randperm(cfg.TRAIN.BATCH_SIZE).cuda()
				dist.broadcast(idx_shuffle, src=0)
				idx_unshuffle = torch.argsort(idx_shuffle)
				idx_this = idx_shuffle.view(cfg.NUM_GPUS, -1)[dist.get_rank()]
				clip_k = [gather_clip_k[0][idx_this]]
			if cfg.TRAIN.EVAL_FEATURE:
				feat_k, _ = model_ema(clip_k)
			else:
				feat_k = model_ema(clip_k)
			if cfg.NUM_GPUS > 1 and cfg.BN.SHUFFLE_BN:
				gather_feat_k = du.all_gather([feat_k])
				idx_this = idx_unshuffle.view(cfg.NUM_GPUS, -1)[dist.get_rank()]
				feat_k = gather_feat_k[0][idx_this]
		out = moco_nec(feat_q, feat_k)
		nce_labels = torch.zeros([out.shape[0]]).cuda().long()
		loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
		loss = loss_fun(out, nce_labels)

		inf_verb, inf_noun = classifier(feature.detach())

		# Explicitly declare reduction to mean.
		inf_noun_func = losses.get_loss_func('cross_entropy')(reduction="mean")
		inf_verb_func = losses.get_loss_func('cross_entropy')(reduction="mean")

		# Compute the loss.
		inf_noun_loss = inf_noun_func(inf_noun, noun_label)
		inf_verb_loss = inf_verb_func(inf_verb, verb_label)
		Loss = loss + 0.1 * (inf_noun_loss + inf_verb_loss)

		# check Nan Loss.
		misc.check_nan_losses(loss)
		misc.check_nan_losses(inf_verb_loss + inf_noun_loss)

		# Perform the backward pass.
		optimizer.zero_grad()
		Loss.backward()

		# Update the parameters.
		optimizer.step()

		moment_update(model, model_ema, cfg.NCE.ALPHA)
		torch.cuda.synchronize()

		if cfg.DETECTION.ENABLE:
			if cfg.NUM_GPUS > 1:
				loss = du.all_reduce([loss])[0]
			loss = loss.item()

			train_meter.iter_toc()
			# Update and log stats.
			train_meter.update_stats(None, None, None, loss, lr)
		else:
			if cfg.DATA.MULTI_LABEL:
				# Gather all the predictions across all the devices.
				if cfg.NUM_GPUS > 1:
					[loss] = du.all_reduce([loss])
				loss = loss.item()
			else:
				# Compute the errors.
				noun_topks_correct = metrics.topks_correct(inf_noun, noun_label, (1, 5))
				verb_topks_correct = metrics.topks_correct(inf_verb, verb_label, (1, 5))
				noun_top1, noun_top5 = [
					(1.0 - x / out.size(0)) * 100.0 for x in noun_topks_correct
				]
				verb_top1, verb_top5 = [
					(1.0 - x / out.size(0)) * 100.0 for x in verb_topks_correct
				]


				nce_topks_correct = metrics.topks_correct(out, nce_labels, (1, 5))
				nce_top1, nce_top5 = [
					(1.0 - x / out.size(0)) * 100.0 for x in nce_topks_correct
				]

				# Gather all the predictions across all the devices.
				if cfg.NUM_GPUS > 1:
					loss, noun_top1, noun_top5, verb_top1, verb_top5, nce_top1, nce_top5 = du.all_reduce(
						[loss, noun_top1, noun_top5, verb_top1, verb_top5, nce_top1, nce_top5]
					)
				# Copy the stats from GPU to CPU (sync point).
				loss, noun_top1, noun_top5, verb_top1, verb_top5, nce_top1, nce_top5 = (
					loss.item(),
					noun_top1.item(),
					noun_top5.item(),
					verb_top1.item(),
					verb_top5.item(),
					nce_top1.item(),
					nce_top5.item()
				)

			# Update and log stats.
			train_meter.update_stats(
				noun_top1, noun_top5, verb_top1, verb_top5, 
				nce_top1, nce_top5, loss, lr, out.size(0) * cfg.NUM_GPUS
			)
			train_meter.iter_toc()
		
		iter_stats = train_meter.log_iter_stats(cur_epoch, cur_iter)

		if du.is_master_proc() and (cur_iter + 1) % cfg.LOG_PERIOD == 0:
			noun_top1, noun_top5, verb_top1, verb_top5, nce_top1, nce_top5, loss = iter_stats
			step = cur_epoch * len(train_loader) + cur_iter
			tb_logger.add_scalar('train/loss', loss, step)
			tb_logger.add_scalar('train/noun_top1_err', noun_top1, step)
			tb_logger.add_scalar('train/noun_top5_err', noun_top5, step)
			tb_logger.add_scalar('train/verb_top1_err', verb_top1, step)
			tb_logger.add_scalar('train/verb_top5_err', verb_top5, step)
			tb_logger.add_scalar('train/nce_top1_err', nce_top1, step)
			tb_logger.add_scalar('train/nce_top5_err', nce_top5, step)


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

	gt_noun_labels = []
	gt_verb_labels = []
	noun_scores = []
	verb_scores = []
	
	for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
		clip_q, clip_k = inputs
		if isinstance(clip_q, (list,)):
			clip_q = [x.cuda(non_blocking=True) for x in clip_q]
			clip_k = [x.cuda(non_blocking=True) for x in clip_k]
		else:
			clip_q = clip_q.cuda(non_blocking=True)
			clip_k = clip_k.cuda(non_blocking=True)
		del inputs

		verb_label, noun_label = labels
		verb_label, noun_label = verb_label.cuda(), noun_label.cuda()
		gt_noun_labels.append(F.one_hot(noun_label, cfg.MODEL.NUM_NOUN_CLASSES))
		gt_verb_labels.append(F.one_hot(verb_label, cfg.MODEL.NUM_VERB_CLASSES))

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
			inf_verb, inf_noun = classifier(feature)

		verb_scores.append(inf_verb)
		noun_scores.append(inf_noun)
		# Compute the accuracy
		noun_topks_correct = metrics.topks_correct(inf_noun, noun_label, (1, 5))
		verb_topks_correct = metrics.topks_correct(inf_verb, verb_label, (1, 5))

		# Combine the errors across the GPUs.
		noun_top1, noun_top5 = [
			(1.0 - x / feature.size(0)) * 100.0 for x in noun_topks_correct
		]
		verb_top1, verb_top5 = [
			(1.0 - x / feature.size(0)) * 100.0 for x in verb_topks_correct
		]

		if cfg.NUM_GPUS > 1:
			noun_top1, noun_top5, verb_top1, verb_top5 = du.all_reduce(
						[noun_top1, noun_top5, verb_top1, verb_top5])

		# Copy the errors from GPU to CPU (sync point).
		noun_top1, noun_top5, verb_top1, verb_top5 = (
					noun_top1.item(), noun_top5.item(), verb_top1.item(), verb_top5.item())

		val_meter.iter_toc()
		# Update and log stats.
		val_meter.update_stats(
			noun_top1, noun_top5, verb_top1, verb_top5, feature.size(0) * cfg.NUM_GPUS
		)

		val_meter.log_iter_stats(cur_epoch, cur_iter)
		val_meter.iter_tic()

	gt_noun_labels = torch.cat(gt_noun_labels, dim=0)
	gt_verb_labels = torch.cat(gt_verb_labels, dim=0)

	noun_scores = torch.cat(noun_scores, dim=0)
	verb_scores = torch.cat(verb_scores, dim=0)

	if cfg.NUM_GPUS > 1:
		gt_noun_labels, gt_verb_labels, noun_scores, verb_scores = du.all_gather(
					[gt_noun_labels, gt_verb_labels, noun_scores, verb_scores])
		
	# Log epoch stats.
	noun_top1, noun_top5, verb_top1, verb_top5 = val_meter.log_epoch_stats(cur_epoch)
	if du.is_master_proc():
		step = cur_epoch * len(val_loader) + cur_iter
		noun_mAP = aps(gt_noun_labels.cpu().numpy(), noun_scores.cpu().numpy())
		verb_mAP = aps(gt_verb_labels.cpu().numpy(), verb_scores.cpu().numpy())
		logger.info('epoch {} noun mAP = {}'.format(cur_epoch, noun_mAP))
		logger.info('epoch {} verb mAP = {}'.format(cur_epoch, verb_mAP))
		tb_logger.add_scalar('val/noun_top1_err', noun_top1, step)
		tb_logger.add_scalar('val/noun_top5_err', noun_top5, step)
		tb_logger.add_scalar('val/verb_top1_err', verb_top1, step)
		tb_logger.add_scalar('val/verb_top5_err', verb_top5, step)
		tb_logger.add_scalar('val/noun_mAP', noun_mAP, step)
		tb_logger.add_scalar('val/verb_mAP', verb_mAP, step)

	del gt_noun_labels, gt_verb_labels, noun_scores, verb_scores
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
	model_ema = build_model(cfg)
	classifier = build_classifier(cfg)
	moco_nec = build_moco_nce(cfg)
	gaussian_blur = ct.build_GaussianBlur(cfg)
	moment_update(model, model_ema, 0)



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
			'model_ema': model_ema,
			'classifier': classifier, 
			'moco_nec': moco_nec},
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
		train_epoch(train_loader, model, classifier, model_ema, moco_nec, optimizer, train_meter, cur_epoch, cfg, tb_logger, gaussian_blur)

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
						'model_ema': model_ema,
						'classifier': classifier, 
						'moco_nec': moco_nec}, 
						optimizer, 
						cur_epoch, 
						cfg)
		# Evaluate the model on validation set.
		if misc.is_eval_epoch(cfg, cur_epoch):
			eval_epoch(val_loader, model, classifier, val_meter, cur_epoch, cfg, tb_logger)
