import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random


class ColorJitter(object):
    def __init__(self):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = [0.2, 0.15, 0.3, 0.1]

    def __call__(self, sample):

        if 'image' in sample.keys():

            sample.update({'image': [Image.fromarray(np.uint8(_image)) for _image in sample['image']]})

            if self.brightness > 0:
                brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                sample.update({'image': [F.adjust_brightness(_image, brightness_factor) for _image in sample['image']]})
                sample['meta']['transformations']['brightness'] = brightness_factor

            if self.contrast > 0:
                contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                sample.update({'image': [F.adjust_contrast(_image, contrast_factor) for _image in sample['image']]})
                sample['meta']['transformations']['contrast'] = contrast_factor

            if self.saturation > 0:
                saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                sample.update({'image': [F.adjust_saturation(_image, saturation_factor) for _image in sample['image']]})
                sample['meta']['transformations']['saturation'] = saturation_factor

            if self.hue > 0:
                hue_factor = np.random.uniform(-self.hue, self.hue)
                sample.update({'image': [F.adjust_hue(_image, hue_factor) for _image in sample['image']]})
                sample['meta']['transformations']['hue'] = hue_factor

            sample.update({'image': [np.asarray(_image).clip(0, 255) for _image in sample['image']]})

        return sample


class RandomColorChannel(object):
    def __call__(self, sample):
        random_order = np.random.permutation(3)
        # only apply to blurry and gt images
        if 'image' in sample.keys() and 'deblur' in sample.keys() and random.random() < 0.5:
            sample.update({'image': [_image[:, :, random_order] for _image in sample['image']]})
            sample['meta']['transformations']['channels_order'] = ''.join(['bgr'[i] for i in random_order])
        else:
            sample['meta']['transformations']['channels_order'] = 'bgr'
        return sample


class RandomGaussianNoise(object):
    def __init__(self):
        gaussian_para = [0, 8]
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, sample):

        shape = sample['image'][0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape).astype(np.float32)
        # only apply to blurry images
        sample.update({'image': [(_image + gaussian_noise).clip(0, 255) for _image in sample['image']]})
        sample['meta']['transformations']['gaussian_noise'] = gaussian_noise
        return sample


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, dims=(640, 480)):
        self.dims = dims

    def __call__(self, sample):
        # Apply to gt, blurry and masks.
        sample.update({'image': [cv2.resize(_image, self.dims, cv2.INTER_AREA) for _image in sample['image']]})
        if 'segment' in sample.keys():
            sample.update({'segment': [cv2.resize(_mask, self.dims, cv2.INTER_AREA) for _mask in sample['segment']]})
        sample['meta']['transformations']['resize'] = self.dims
        return sample


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, sample):
        # Apply to gt, blurry and masks. Additionally, change the homography
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            sample.update({'image': [np.copy(np.fliplr(_image)) for _image in sample['image']]})
            if 'segment' in sample.keys():
                sample.update({'segment': [np.copy(np.fliplr(_mask)) for _mask in sample['segment']]})
            sample['meta']['transformations']['horizontal_flip'] = True
        else:
            sample['meta']['transformations']['horizontal_flip'] = False
        return sample


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, sample):
        # Apply to gt, blurry and masks. Additionally, change the homography
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''

            sample.update({'image': [np.copy(np.flipud(_image)) for _image in sample['image']]})
            if 'segment' in sample.keys():
                sample.update({'segment': [np.copy(np.flipud(_mask)) for _mask in sample['segment']]})
            sample['meta']['transformations']['vertical_flip'] = True
        else:
            sample['meta']['transformations']['vertical_flip'] = False
        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def map_m2l(self, mask):
        return mask

    def __call__(self, sample):
        #TODO write a caller function that maps the colors to classes i.e. [255,0,255] = 1, [255,255,0] = 2, [0,255,255] = 2  and [255,255,255] = background
        sample.update({'image': [torch.from_numpy(_image).permute(2, 0, 1) for _image in sample['image']]})
        if 'segment_one' in sample.keys():
            sample.update({'segment_one': [torch.from_numpy(self.map_m2l(_mask).astype(float)).type(torch.LongTensor) for _mask in sample['segment_one']]})
        if 'segment_two' in sample.keys():
            sample.update({'segment_two': [torch.from_numpy(self.map_m2l(_mask).astype(float)).type(torch.LongTensor) for _mask in sample['segment_two']]})
        if 'segment_three' in sample.keys():
            sample.update({'segment_three': [torch.from_numpy(self.map_m2l(_mask).astype(float)).type(torch.LongTensor) for _mask in sample['segment_three']]})


        return sample


class Normalize(object):
    """Normalize image"""

    def __call__(self, sample):

        sample.update({'image': [_image/255 for _image in sample['image']]})
        return sample


class Standarize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        sample.update({'image': [(_image-self.mean) / self.std for _image in sample['image']]})

        if 'deblur' in sample.keys():
            sample.update({'deblur': [(_deblur-self.mean) / self.std for _deblur in sample['deblur']]})

        return sample
