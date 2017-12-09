
# coding: utf-8

# In[37]:

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
transform_list = transforms.Compose([grey_world(),transforms.ToPILImage(),transforms.Scale(256), transforms.RandomCrop(224), transforms.ToTensor()])#, transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

input_dir= '../datasets/'
#input_dir = '../../../mnt/nfs/work1/lspector/aks/datasets/'

train = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Training_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Training_Data/', transform=transform_list)
validation  = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Validation_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Validation_Data/', transform = transform_list)
test = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Test_v2_Data/', transform = transform_list)

train_data = DataLoader(train, batch_size=8,
                        shuffle=True, num_workers=1)
val_data = DataLoader(validation, batch_size=8,
                        shuffle=True, num_workers=1)
test_data = DataLoader(test, batch_size=8,
                        shuffle=True, num_workers=1)

dataset_sizes = {'train':len(train),'val':len(validation),'test':len(test)}
print(dataset_sizes)

dataloaders = {'train':train_data,'val':val_data,'test':test_data}
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[32]:

#from skimage import io, transform
#image = io.imread('../datasets/ISIC-2017_Training_Data/ISIC_0000000.jpg')

#image_modified = grey_world(image)


#plt.imshow(image)


# In[23]:


# In[38]:

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features

#print(num_ftrs)
model_conv.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# In[ ]:


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, dataloaders,dataset_sizes,num_epochs=5)


# In[18]:

torch.save(model_conv,'mytraining.pt')


# In[19]:

mymodel = torch.load('mytraining.pt')


# In[20]:

#dataloaders['test'] = dataloaders['test'][0:10]
test_model(mymodel, criterion, dataloaders,dataset_sizes)


# In[82]:

# Actual Algorithm
#models_list = (model_conv,model_conv)
#model_data = train_meta_model(models_list, 1, dataloaders, dataset_sizes)


# In[83]:

#test_meta_model(model_data, dataloaders, dataset_sizes)

