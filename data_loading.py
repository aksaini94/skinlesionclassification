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
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
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
        image = sample#['image']
        #class1 = sample['class1']
        #class2 = sample['class2']
        

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

        #sample = {'image': img, 'class1': class1,'class2': class2}

        return img#sample

    
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

        return image#torch.from_numpy(image)
    






class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample#['image']
        #class1 = sample['class1']
        #class2 = sample['class2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

       # sample = {'image': torch.from_numpy(image), 'class1': class1,'class2': class2}

        return image#torch.from_numpy(image)

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
        #assert isinstance(output_size, (int, tuple))
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample#['image']
        #class1 = sample['class1']
        #class2 = sample['class2']
        img = sample

        #img = transforms.functional.normalize(image, self.mean, self.std )
        
        #img = transform.resize(image, (new_h, new_w))

        sample = {'image': img, 'class1': class1,'class2': class2}

        return sample


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = \
            sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')


"""

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transforms.ToTensor())
                  for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes



def imshow(inp, title=None):
    #Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

"""

