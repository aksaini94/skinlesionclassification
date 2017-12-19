# -*- coding: utf-8 -*-
"""
Data Loading and Processing
"""

from __future__ import print_function, division
import os
import numpy as np
import torch
import pandas as pd
from skimage import io, transform


import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#from torchvision.transforms.functional import normalize as norm

#from torch._six import string_classes

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def show_landmarks(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001) 


def masking(img, img_segment, level_of_opaqueness):
    #level_of_opaqueness: the more it it, the more opaque the object becomes, values between (0,255)
    img_modified = img
    img_modified[img_segment==0] =level_of_opaqueness
    return img_modified

class SkinLesionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, segment_dir, useSegmentation, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.classification_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.segment_dir = segment_dir
        self.transform= transform
        self.useSegmentation = useSegmentation       

    def __len__(self):
        return len(self.classification_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.classification_frame.ix[idx, 0]+'.jpg')
        segment_name = os.path.join(self.segment_dir, self.classification_frame.ix[idx, 0]+'_segmentation.png')
        image = io.imread(img_name)
        if self.useSegmentation:
            segmented_image = io.imread(segment_name)
            image = masking(image, segmented_image, 255)
        
        sample = {'image': image, 'class1': self.classification_frame.ix[idx, 1],'class2': self.classification_frame.ix[idx, 2]}
             
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

    
def grey_world_func(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)    

class grey_world(object):
    
    def __call__(self, sample):
        image = sample
        image = grey_world_func(image)

       # sample = {'image': torch.from_numpy(image), 'class1': class1,'class2': class2}

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample
        image = image.transpose((2, 0, 1))

        return image

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class Normalize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample
        img = sample

        sample = {'image': img, 'class1': class1,'class2': class2}

        return sample
