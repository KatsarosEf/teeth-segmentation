# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import os
import pickle

import cv2
import torch.utils.data as data
import numpy as np


class MTL_Dataset(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes edge detection, semantic segmentation, surface normals, and depth prediction
    """

    def __init__(self, tasks,
                 root='./DST-dataset/', split='train', seq_len=20, transform=None,
                 meta=True, overfit=False):

        do_seg = 'segment' in tasks
        self.root = root
        self.transform = transform
        self.meta = meta
        self.split = split
        self.threshold = 40
        # Images
        self.images = []
        # Semantic Segmentation
        self.masks_one = []
        self.masks_two = []
        self.masks_three = []
        self.annotator_one = True
        self.annotator_two = True
        self.annotator_three = True

        dataset_dir = os.path.join(root, self.split)
        for video in os.listdir(dataset_dir):
            image_dir, anno_one_dir, anno_two_dir, anno_three_dir = [os.path.join(dataset_dir, video, directory)
                                                         for directory in ['input', 'annotator_one',
                                                                           'annotator_two', 'annotator_three']]

            filenames = sorted(os.listdir(image_dir))
            if seq_len is None:
                seq_len_inner = len(filenames)
            else:
                seq_len_inner = seq_len
            for first_idx in range(0, len(filenames)-seq_len_inner+1, seq_len_inner):

                seq_images, seq_masks_one, seq_masks_two, seq_masks_three = ([] for _ in range(4))

                for file_idx in range(first_idx, first_idx + seq_len_inner):

                    filename = filenames[file_idx]

                    # Images

                    _image = os.path.join(image_dir, filename, "input.png")
                    assert os.path.isfile(_image)
                    seq_images.append(_image)

                    # Segmentation Annotations
                    if self.annotator_one:
                        anno_one = os.path.join(anno_one_dir, filename, "segmentation.png")
                        assert os.path.isfile(anno_one)
                        seq_masks_one.append(anno_one)

                    if self.annotator_two:
                        anno_two = os.path.join(anno_two_dir, filename, "segmentation.png")
                        assert os.path.isfile(anno_two)
                        seq_masks_two.append(anno_two)

                    if self.annotator_three:
                        anno_three = os.path.join(anno_three_dir, filename, "segmentation.png")
                        assert os.path.isfile(anno_three)
                        seq_masks_three.append(anno_three)


                self.images.append(seq_images)
                self.masks_one.append(seq_masks_one)
                self.masks_two.append(seq_masks_two)
                self.masks_three.append(seq_masks_three)


        if seq_len is None:
            return
        # Uncomment to overfit to one sequence
        if overfit:
            n_of = seq_len
            self.images = [self.images[0]][:n_of]

        # Display stats
        print('Number of {} dataset sequences: {:d}'.format(self.split, len(self.images)))
        print('Number of sequence frames: {:d}'.format(seq_len))

    def __getitem__(self, index):

        _img = self._load_img(index)
        sample = {'image': _img}

        if self.annotator_one:
            sample['segment_one'] = self._load_mask(index, self.masks_one)
        if self.annotator_two:
            sample['segment_two'] = self._load_mask(index, self.masks_two)
        if self.annotator_three:
            sample['segment_three'] = self._load_mask(index, self.masks_three)


        if self.meta:
            sample['meta'] = {'paths': self.images[index],
                              'im_size': (_img[0].shape[0], _img[0].shape[1]),
                              'transformations': {}}

        if self.transform is not None:
            sample = self.transform(sample)


        return sample

    def _load_img(self, index):
        return [cv2.imread(path).astype(np.float32) for path in self.images[index]]

    def _load_mask(self, index, folder):
        return [cv2.imread(path).astype(np.float32) for path in folder[index]]

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return 'Dataset (split=' + str(self.split) + ')'
