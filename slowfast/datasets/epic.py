#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
import pandas as pd


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epic(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.noun_num_classes = 352
        self.verb_num_classes = 125
        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        
        if self.mode == 'test':
            if self.cfg.TEST.SECTION == 'train':
                path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "EPIC_{}_action_labels.csv".format(self.cfg.TEST.SECTION))
            else:
                path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "EPIC_test_{}_timestamps_full.csv".format(self.cfg.TEST.SECTION))
        else:
            path_to_file = os.path.join(
                path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "EPIC_{}_action_labels.csv".format(self.mode))
            )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_folder = []
        self._frame_range = []
        self._verb_labels = []
        self._noun_labels = []
        self._spatial_temporal_idx = []
        self._uids = []
        df = pd.read_csv(path_to_file)
        for i in range(df.shape[0]):
            row = df.iloc[i]
            for idx in range(self._num_clips):
                self._uids.append(row['uid'])
                if self.mode == 'test' and self.cfg.TEST.SECTION == 'train':
                    self._path_to_folder.append(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.TEST.SECTION, row['participant_id'], row['video_id']))
                else:
                    self._path_to_folder.append(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, self.mode, row['participant_id'], row['video_id']))
                self._frame_range.append([row['start_frame'], row['stop_frame']])
                if self.mode == 'test':
                    self._verb_labels.append(-1)
                    self._noun_labels.append(-1)
                else:
                    self._verb_labels.append(row['verb_class'])
                    self._noun_labels.append(row['noun_class'])
                self._spatial_temporal_idx.append(idx)
                self._video_meta[i * self._num_clips + idx] = {}
        assert (
            len(self._path_to_folder) > 0
        ), "Failed to load Epic split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_folder), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        
        # Decode video. Meta info is used to perform selective decoding.
        candidates = []
        num_clips = 1 if self.mode == 'test' else 2

        for i in range(num_clips):
            if self._frame_range[index][0] >= self._frame_range[index][1] + 1 - self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE:
                start_frame_id = self._frame_range[index][0]
                stop_frame_id = self._frame_range[index][1]
            else:
                if self.mode == 'test':
                    start_frame_id = self._frame_range[index][0] + temporal_sample_index * ((self._frame_range[index][1] + 1 - self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE - self._frame_range[index][0]) // self.cfg.TEST.NUM_ENSEMBLE_VIEWS)
                else:
                    start_frame_id = np.random.randint(self._frame_range[index][0], self._frame_range[index][1] + 1 - self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE)
                stop_frame_id = start_frame_id + self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
            
            indices = torch.arange(start_frame_id, stop_frame_id, self.cfg.DATA.SAMPLING_RATE).numpy().astype(int)
            img_paths = [os.path.join(self._path_to_folder[index], 'frame_{}.jpg'.format(str(iidx).zfill(10))) for iidx in indices]
            frames = utils.retry_load_images(img_paths)
            for i in range(self.cfg.DATA.NUM_FRAMES):
                if frames.shape[0] < self.cfg.DATA.NUM_FRAMES:
                    frames = torch.cat((frames, frames[:self.cfg.DATA.NUM_FRAMES - frames.shape[0]]), 0)
                else:
                    break

            assert frames.shape[0] == self.cfg.DATA.NUM_FRAMES
            candidates.append(frames) 
        
        clips = []
        for clip in candidates:
            clip = utils.tensor_normalize(
                        clip, self.cfg.DATA.MEAN, self.cfg.DATA.STD,
                        self.cfg.DATA.NORMALIZATION)

            # T H W C -> C T H W
            clip = clip.permute(3, 0, 1, 2) #TODO
            clip = utils.spatial_sampling(
                clip,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                cvrl_aug=True,
                aug_para=self.cfg.DATA.AUG,
            )
            clip = utils.pack_pathway_output(self.cfg, clip)
            clips.append(clip)

        label = self._verb_labels[index]
        stid = self._spatial_temporal_idx[index]
        uid = self._uids[index]
        return clips, label, index, {}, stid, uid

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_folder)
