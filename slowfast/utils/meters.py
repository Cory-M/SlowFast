#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.ava_eval_helper import (
	evaluate_ava,
	read_csv,
	read_exclusions,
	read_labelmap,
)

from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
	"""
	Get the groundtruth annotations corresponding the "subset" of AVA val set.
	We define the subset to be the frames such that (second % 4 == 0).
	We optionally use subset for faster evaluation during training
	(in order to track training progress).
	Args:
		full_groundtruth(dict): list of groundtruth.
	"""
	ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

	for i in range(3):
		for key in full_groundtruth[i].keys():
			if int(key.split(",")[1]) % 4 == 0:
				ret[i][key] = full_groundtruth[i][key]
	return ret


class AVAMeter(object):
	"""
	Measure the AVA train, val, and test stats.
	"""

	def __init__(self, overall_iters, cfg, mode):
		"""
			overall_iters (int): the overall number of iterations of one epoch.
			cfg (CfgNode): configs.
			mode (str): `train`, `val`, or `test` mode.
		"""
		self.cfg = cfg
		self.lr = None
		self.loss = ScalarMeter(cfg.LOG_PERIOD)
		self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
		self.mode = mode
		self.iter_timer = Timer()
		self.all_preds = []
		self.all_ori_boxes = []
		self.all_metadata = []
		self.overall_iters = overall_iters
		self.excluded_keys = read_exclusions(
			os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
		)
		self.categories, self.class_whitelist = read_labelmap(
			os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
		)
		gt_filename = os.path.join(
			cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
		)
		self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
		self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

		_, self.video_idx_to_name = ava_helper.load_image_lists(
			cfg, mode == "train"
		)

	def log_iter_stats(self, cur_epoch, cur_iter):
		"""
		Log the stats.
		Args:
			cur_epoch (int): the current epoch.
			cur_iter (int): the current iteration.
		"""

		if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
			return

		eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
		eta = str(datetime.timedelta(seconds=int(eta_sec)))
		if self.mode == "train":
			stats = {
				"_type": "{}_iter".format(self.mode),
				"cur_epoch": "{}".format(cur_epoch + 1),
				"cur_iter": "{}".format(cur_iter + 1),
				"eta": eta,
				"time_diff": self.iter_timer.seconds(),
				"mode": self.mode,
				"loss": self.loss.get_win_avg(),
				"lr": self.lr,
			}
		elif self.mode == "val":
			stats = {
				"_type": "{}_iter".format(self.mode),
				"cur_epoch": "{}".format(cur_epoch + 1),
				"cur_iter": "{}".format(cur_iter + 1),
				"eta": eta,
				"time_diff": self.iter_timer.seconds(),
				"mode": self.mode,
			}
		elif self.mode == "test":
			stats = {
				"_type": "{}_iter".format(self.mode),
				"cur_iter": "{}".format(cur_iter + 1),
				"eta": eta,
				"time_diff": self.iter_timer.seconds(),
				"mode": self.mode,
			}
		else:
			raise NotImplementedError("Unknown mode: {}".format(self.mode))

		logging.log_json_stats(stats)

	def iter_tic(self):
		"""
		Start to record time.
		"""
		self.iter_timer.reset()

	def iter_toc(self):
		"""
		Stop to record time.
		"""
		self.iter_timer.pause()

	def reset(self):
		"""
		Reset the Meter.
		"""
		self.loss.reset()

		self.all_preds = []
		self.all_ori_boxes = []
		self.all_metadata = []

	def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
		"""
		Update the current stats.
		Args:
			preds (tensor): prediction embedding.
			ori_boxes (tensor): original boxes (x1, y1, x2, y2).
			metadata (tensor): metadata of the AVA data.
			loss (float): loss value.
			lr (float): learning rate.
		"""
		if self.mode in ["val", "test"]:
			self.all_preds.append(preds)
			self.all_ori_boxes.append(ori_boxes)
			self.all_metadata.append(metadata)
		if loss is not None:
			self.loss.add_value(loss)
		if lr is not None:
			self.lr = lr

	def finalize_metrics(self, log=True):
		"""
		Calculate and log the final AVA metrics.
		"""
		all_preds = torch.cat(self.all_preds, dim=0)
		all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
		all_metadata = torch.cat(self.all_metadata, dim=0)

		if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
			groundtruth = self.full_groundtruth
		else:
			groundtruth = self.mini_groundtruth

		self.full_map = evaluate_ava(
			all_preds,
			all_ori_boxes,
			all_metadata.tolist(),
			self.excluded_keys,
			self.class_whitelist,
			self.categories,
			groundtruth=groundtruth,
			video_idx_to_name=self.video_idx_to_name,
		)
		if log:
			stats = {"mode": self.mode, "map": self.full_map}
			logging.log_json_stats(stats)

	def log_epoch_stats(self, cur_epoch):
		"""
		Log the stats of the current epoch.
		Args:
			cur_epoch (int): the number of current epoch.
		"""
		if self.mode in ["val", "test"]:
			self.finalize_metrics(log=False)
			stats = {
				"_type": "{}_epoch".format(self.mode),
				"cur_epoch": "{}".format(cur_epoch + 1),
				"mode": self.mode,
				"map": self.full_map,
				"gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
				"RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
			}
			logging.log_json_stats(stats)


class TestMeter(object):
	"""
	Perform the multi-view ensemble for testing: each video with an unique index
	will be sampled with multiple clips, and the predictions of the clips will
	be aggregated to produce the final prediction for the video.
	The accuracy is calculated with the given ground truth labels.
	"""

	def __init__(
		self,
		num_videos,
		num_clips,
		num_cls,
		overall_iters,
		multi_label=False,
		ensemble_method="sum",
	):
		"""
		Construct tensors to store the predictions and labels. Expect to get
		num_clips predictions from each video, and calculate the metrics on
		num_videos videos.
		Args:
			num_videos (int): number of videos to test.
			num_clips (int): number of clips sampled from each video for
				aggregating the final prediction for the video.
			num_cls (int): number of classes for each prediction.
			overall_iters (int): overall iterations for testing.
			multi_label (bool): if True, use map as the metric.
			ensemble_method (str): method to perform the ensemble, options
				include "sum", and "max".
		"""

		self.iter_timer = Timer()
		self.num_clips = num_clips
		self.overall_iters = overall_iters
		self.multi_label = multi_label
		self.ensemble_method = ensemble_method
		# Initialize tensors.
		self.video_preds = torch.zeros((num_videos, num_cls))
		if multi_label:
			self.video_preds -= 1e10

		self.video_labels = (
			torch.zeros((num_videos, num_cls))
			if multi_label
			else torch.zeros((num_videos)).long()
		)
		self.clip_count = torch.zeros((num_videos)).long()
		# Reset metric.
		self.reset()

	def reset(self):
		"""
		Reset the metric.
		"""
		self.clip_count.zero_()
		self.video_preds.zero_()
		if self.multi_label:
			self.video_preds -= 1e10
		self.video_labels.zero_()

	def update_stats(self, preds, labels, clip_ids):
		"""
		Collect the predictions from the current batch and perform on-the-flight
		summation as ensemble.
		Args:
			preds (tensor): predictions from the current batch. Dimension is
				N x C where N is the batch size and C is the channel size
				(num_cls).
			labels (tensor): the corresponding labels of the current batch.
				Dimension is N.
			clip_ids (tensor): clip indexes of the current batch, dimension is
				N.
		"""
		for ind in range(preds.shape[0]):
			vid_id = int(clip_ids[ind]) // self.num_clips
			if self.video_labels[vid_id].sum() > 0:
				assert torch.equal(
					self.video_labels[vid_id].type(torch.FloatTensor),
					labels[ind].type(torch.FloatTensor),
				)
			self.video_labels[vid_id] = labels[ind]
			if self.ensemble_method == "sum":
				self.video_preds[vid_id] += preds[ind]
			elif self.ensemble_method == "max":
				self.video_preds[vid_id] = torch.max(
					self.video_preds[vid_id], preds[ind]
				)
			else:
				raise NotImplementedError(
					"Ensemble Method {} is not supported".format(
						self.ensemble_method
					)
				)
			self.clip_count[vid_id] += 1

	def log_iter_stats(self, cur_iter):
		"""
		Log the stats.
		Args:
			cur_iter (int): the current iteration of testing.
		"""
		eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
		eta = str(datetime.timedelta(seconds=int(eta_sec)))
		stats = {
			"split": "test_iter",
			"cur_iter": "{}".format(cur_iter + 1),
			"eta": eta,
			"time_diff": self.iter_timer.seconds(),
		}
		logging.log_json_stats(stats)

	def iter_tic(self):
		self.iter_timer.reset()

	def iter_toc(self):
		self.iter_timer.pause()

	def finalize_metrics(self, ks=(1, 5)):
		"""
		Calculate and log the final ensembled metrics.
		ks (tuple): list of top-k values for topk_accuracies. For example,
			ks = (1, 5) correspods to top-1 and top-5 accuracy.
		"""
		if not all(self.clip_count == self.num_clips):
			logger.warning(
				"clip count {} ~= num clips {}".format(
					", ".join(
						[
							"{}: {}".format(i, k)
							for i, k in enumerate(self.clip_count.tolist())
						]
					),
					self.num_clips,
				)
			)

		stats = {"split": "test_final"}
		if self.multi_label:
			map = get_map(
				self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
			)
			stats["map"] = map
		else:
			num_topks_correct = metrics.topks_correct(
				self.video_preds, self.video_labels, ks
			)
			topks = [
				(x / self.video_preds.size(0)) * 100.0
				for x in num_topks_correct
			]
			assert len({len(ks), len(topks)}) == 1
			for k, topk in zip(ks, topks):
				stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
					topk, prec=2
				)
		logging.log_json_stats(stats)


class ScalarMeter(object):
	"""
	A scalar meter uses a deque to track a series of scaler values with a given
	window size. It supports calculating the median and average values of the
	window, and also supports calculating the global average.
	"""

	def __init__(self, window_size):
		"""
		Args:
			window_size (int): size of the max length of the deque.
		"""
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0

	def reset(self):
		"""
		Reset the deque.
		"""
		self.deque.clear()
		self.total = 0.0
		self.count = 0

	def add_value(self, value):
		"""
		Add a new scalar value to the deque.
		"""
		self.deque.append(value)
		self.count += 1
		self.total += value

	def get_win_median(self):
		"""
		Calculate the current median value of the deque.
		"""
		return np.median(self.deque)

	def get_win_avg(self):
		"""
		Calculate the current average value of the deque.
		"""
		return np.mean(self.deque)

	def get_global_avg(self):
		"""
		Calculate the global mean value.
		"""
		return self.total / self.count


class TrainMeter(object):
	"""
	Measure training stats.
	"""

	def __init__(self, epoch_iters, cfg):
		"""
		Args:
			epoch_iters (int): the overall number of iterations of one epoch.
			cfg (CfgNode): configs.
		"""
		self._cfg = cfg
		self.epoch_iters = epoch_iters
		self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
		self.iter_timer = Timer()
		self.lr = None

		self.metric = {'mb': dict(), 'total': dict()}
		for metric_name in cfg.METRIC.TRAIN:
			# mini batch metrics (used for iter_log)
			self.metric['mb'][metric_name] = ScalarMeter(cfg.LOG_PERIOD)
			# total metrics (used for epoch_log)
			self.metric['total'][metric_name] = 0.0
		
		self.num_samples = 0

	def reset(self):
		"""
		Reset the Meter.
		"""
		self.lr = None
		self.num_samples = 0
		for metric_name in self._cfg.METRIC.TRAIN:
			self.metric['mb'][metric_name].reset()
			self.metric['total'][metric_name] = 0.0

	def iter_tic(self):
		"""
		Start to record time.
		"""
		self.iter_timer.reset()

	def iter_toc(self):
		"""
		Stop to record time.
		"""
		self.iter_timer.pause()

	def update_stats(self, metric_dict, lr, mb_size):
		"""
		Update the current stats.
		Args:
			top1_err (float): top1 error rate.
			top5_err (float): top5 error rate.
			loss (float): loss value.
			lr (float): learning rate.
			mb_size (int): mini batch size.
		"""
		self.lr = lr
		self.num_samples += mb_size

		if not self._cfg.DATA.MULTI_LABEL:
			# Current minibatch stats
			assert (metric_dict.keys() == self.metric['mb'].keys())
			for k, v in metric_dict.items():
				self.metric['mb'][k].add_value(v)
				self.metric['total'][k] += v * mb_size

	def log_iter_stats(self, cur_epoch, cur_iter):
		"""
		log the stats of the current iteration.
		Args:
			cur_epoch (int): the number of current epoch.
			cur_iter (int): the number of current iteration.
		"""
		if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
			return
		eta_sec = self.iter_timer.seconds() * (
			self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
		)
		eta = str(datetime.timedelta(seconds=int(eta_sec)))
		stats = {
			"_type": "train_iter",
			"epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
			"iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
			"time_diff": self.iter_timer.seconds(),
			"eta": eta,
			"lr": self.lr,
			"gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
		}
		if not self._cfg.DATA.MULTI_LABEL:
			for k, v in self.metric['mb'].items():
				stats[k] = v.get_win_avg()
		logging.log_json_stats(stats)
		return {k: v.get_win_avg() for k, v in self.metric['mb'].items()}

	def log_epoch_stats(self, cur_epoch):
		"""
		Log the stats of the current epoch.
		Args:
			cur_epoch (int): the number of current epoch.
		"""
		eta_sec = self.iter_timer.seconds() * (
			self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
		)
		eta = str(datetime.timedelta(seconds=int(eta_sec)))
		stats = {
			"_type": "train_epoch",
			"epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
			"time_diff": self.iter_timer.seconds(),
			"eta": eta,
			"lr": self.lr,
			"gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
			"RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
		}
		if not self._cfg.DATA.MULTI_LABEL:
			for k, v in self.metric['total'].items():
				stats[k] = v / self.num_samples
		logging.log_json_stats(stats)


class ValMeter(object):
	"""
	Measures validation stats.
	"""

	def __init__(self, max_iter, cfg):
		"""
		Args:
			max_iter (int): the max number of iteration of the current epoch.
			cfg (CfgNode): configs.
		"""
		self._cfg = cfg
		self.max_iter = max_iter
		self.iter_timer = Timer()
		# Number of misclassified examples.
		self.num_samples = 0
		self.all_preds = []
		self.all_labels = []
		
		self.metric = {'mb': dict(), 'total': dict(), 'min': dict()}
		for metric_name in cfg.METRIC.EVAL:
			# mini batch metrics (used for iter_log)
			self.metric['mb'][metric_name] = ScalarMeter(cfg.LOG_PERIOD)
			# total metrics (used for epoch_log)
			self.metric['total'][metric_name] = 0.0

		for metric_name in cfg.METRIC.EVAL_MIN:
			# min erros (over the full val set, used for epoch_log)
			self.metric['min'][metric_name] = 100.0

	def reset(self):
		"""
		Reset the Meter.
		"""
		self.iter_timer.reset()
		self.num_samples = 0
		self.all_preds = []
		self.all_labels = []

		for metric_name in self._cfg.METRIC.EVAL:
			self.metric['mb'][metric_name].reset()
			self.metric['total'][metric_name] = 0.0

	def iter_tic(self):
		"""
		Start to record time.
		"""
		self.iter_timer.reset()

	def iter_toc(self):
		"""
		Stop to record time.
		"""
		self.iter_timer.pause()

	def update_stats(self, metric_dict, mb_size):
		"""
		Update the current stats.
		Args:
			top1_err (float): top1 error rate.
			top5_err (float): top5 error rate.
			mb_size (int): mini batch size.
		"""
		self.num_samples += mb_size
		# Current minibatch stats
		assert (metric_dict.keys() == self.metric['mb'].keys())
		for k, v in metric_dict.items():
			self.metric['mb'][k].add_value(v)
			self.metric['total'][k] += v * mb_size

	def update_predictions(self, preds, labels):
		"""
		Update predictions and labels.
		Args:
			preds (tensor): model output predictions.
			labels (tensor): labels.
		"""
		# TODO: merge update_prediction with update_stats.
		self.all_preds.append(preds)
		self.all_labels.append(labels)
	
	def log_iter_stats(self, cur_epoch, cur_iter):
		"""
		log the stats of the current iteration.
		Args:
			cur_epoch (int): the number of current epoch.
			cur_iter (int): the number of current iteration.
		"""
		if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
			return
		eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
		eta = str(datetime.timedelta(seconds=int(eta_sec)))
		stats = {
			"_type": "val_iter",
			"epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
			"iter": "{}/{}".format(cur_iter + 1, self.max_iter),
			"time_diff": self.iter_timer.seconds(),
			"eta": eta,
			"gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
		}
		if not self._cfg.DATA.MULTI_LABEL:
			for k, v in self.metric['mb'].items():
				stats[k] = v.get_win_avg()
		logging.log_json_stats(stats)


	def log_epoch_stats(self, cur_epoch):
		"""
		Log the stats of the current epoch.
		Args:
			cur_epoch (int): the number of current epoch.
		"""
		stats = {
			"_type": "val_epoch",
			"epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
			"time_diff": self.iter_timer.seconds(),
			"gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
			"RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
		}
		if self._cfg.DATA.MULTI_LABEL:
			stats["map"] = get_map(
				torch.cat(self.all_preds).cpu().numpy(),
				torch.cat(self.all_labels).cpu().numpy(),
			)
		else:
			for k, v in self.metric['total'].items():
				stats[k] = v / self.num_samples
			for k, v in self.metric['min'].items():
				self.metric['min'][k] = min(v, stats[k])
				stats['min_' + k] = self.metric['min'][k]

		logging.log_json_stats(stats)
		return {k: v / self.num_samples for k, v in self.metric['total'].items()}

def get_map(preds, labels):
	"""
	Compute mAP for multi-label case.
	Args:
		preds (numpy tensor): num_examples x num_classes.
		labels (numpy tensor): num_examples x num_classes.
	Returns:
		mean_ap (int): final mAP score.
	"""

	logger.info("Getting mAP for {} examples".format(preds.shape[0]))

	preds = preds[:, ~(np.all(labels == 0, axis=0))]
	labels = labels[:, ~(np.all(labels == 0, axis=0))]
	aps = [0]
	try:
		aps = average_precision_score(labels, preds, average=None)
	except ValueError:
		print(
			"Average precision requires a sufficient number of samples \
			in a batch which are missing in this sample."
		)

	mean_ap = np.mean(aps)
	return mean_ap
