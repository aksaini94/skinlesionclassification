
# coding: utf-8

# In[123]:

# Setting up
from __future__ import print_function, division
import os
import numpy as np
import torch
import pandas as pd
from skimage import io, transform

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from training_utils import *
from data_loading import *


#transform1 = transforms.Compose([Rescale((256,256)), RandomCrop(224), ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
transform_list = transforms.Compose([grey_world(),transforms.ToPILImage(),transforms.Scale(250),transforms.RandomHorizontalFlip() ,
                                     transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
# transforms.RandomRotation(90, expand=True) and VerticalFlip() not working for some reason!!
#input_dir= '../datasets/'
input_dir = '/mnt/nfs/work1/lspector/aks/datasets/'

useSegmentaion=True

train = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Training_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Training_Data/',segment_dir=input_dir+'ISIC-2017_Training_Part1_GroundTruth',useSegmentaion =useSegmentaion, transform=transform_list)
validation  = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Validation_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Validation_Data/',segment_dir=input_dir+'ISIC-2017_Validation_Part1_GroundTruth', useSegmentaion =useSegmentaion,transform = transform_list)
test = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Test_v2_Data/',segment_dir=input_dir+'ISIC-2017_Test_v2_Part1_GroundTruth', useSegmentaion =useSegmentaion,transform = transform_list)

train_data = DataLoader(train, batch_size=8,
                        shuffle=True, num_workers=1)
val_data = DataLoader(validation, batch_size=8,
                        shuffle=True, num_workers=1)
test_data = DataLoader(test, batch_size=8,
                        shuffle=True, num_workers=1)

dataset_sizes = {'train':len(train),'val':len(validation),'test':len(test)}
print(dataset_sizes)

dataloaders = {'train':train_data,'val':val_data,'test':test_data}
#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')


# In[119]:

def masking(img, img_segment, level_of_opaqueness):
    #level_of_opaqueness: the more it it, the more opaque the object becomes, values between (0,255)
    img_modified = img
    img_modified[img_segment==0] =level_of_opaqueness
    return img_modified


# In[120]:

from skimage import io, transform
#np.set_printoptions(threshold='nan')
image1 = io.imread('../datasets/ISIC-2017_Training_Data/ISIC_0000000.jpg')
image2 = io.imread('../datasets/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png')
#image_modified = grey_world(image)

#print(image2)

#image2 = np.ones(image2.shape)

#print(image1.shape, image2.shape)
#image_modified = masking(image1, image2,255)
#plt.imshow(image_modified)
#plt.imshow(image)


# In[15]:

#plt.imshow(image1)


# In[16]:

#plt.imshow(image2)


# In[113]:

#for i_batch, sample_batched in enumerate(test_data):
#   print(i_batch, sample_batched['image'].shape[0])
#
    # observe 4th batch and stop.
#    if i_batch == 2:
#        plt.figure()
#        show_landmarks_batch(sample_batched)
#        plt.axis('off')
#        plt.ioff()
#        plt.show()
#        break


# In[124]:

# visualizing some images
# Get a batch of training data
inputs = next(iter(train_data))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs['image'])

#imshow(out, title="First one")


# In[127]:

model_resnet = torchvision.models.resnet18(pretrained=True)
for param in model_resnet.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_resnet.fc.in_features

#print(num_ftrs)
model_resnet.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# In[2]:

# In[128]:

dataset_sizes1 = {'train':len(validation),'val':len(validation),'test':len(test)}
print(dataset_sizes1)

dataloaders1 = {'train':val_data,'val':val_data,'test':test_data}

model_conv = train_model(model_resnet, criterion, optimizer_conv,
                         exp_lr_scheduler, dataloaders,dataset_sizes,num_epochs=2)


# In[ ]:

test_model(model_conv, criterion, dataloaders,dataset_sizes)


# In[18]:

models_dir = '../models/'
#models_dir = input_dir
model_state_list = None
num_epochs = 10
models= [model_resnet, model_inception]
for i in range(len(models)):
    for j in range(1):
        model_state_list = train_model_epochs(model, criterion, optimizer_conv,
                         exp_lr_scheduler, dataloaders,dataset_sizes,num_epochs=num_epochs)
            
        for k,state in enumerate(model_state_list):
            torch.save(model.load_state_dict(state),models_dir+str(i)+str(j)+str(k)+'.pt')


# In[19]:

# ensamle predictions
acc = test_ensamble_model(dataloaders, dataset_sizes, model_dir, (len(models),1, num_epochs))
print(acc)


# In[82]:

# Actual Algorithm
#models_list = (model_conv,model_conv)
#model_data = train_meta_model(models_list, 1, dataloaders, dataset_sizes)


# In[83]:

#test_meta_model(model_data, dataloaders, dataset_sizes)

