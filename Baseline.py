
# coding: utf-8

# In[136]:

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

useSegmentation=True

train = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Training_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Training_Data/',segment_dir=input_dir+'ISIC-2017_Training_Part1_GroundTruth',useSegmentation = useSegmentation, transform=transform_list)
validation  = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Validation_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Validation_Data/',segment_dir=input_dir+'ISIC-2017_Validation_Part1_GroundTruth', useSegmentation =useSegmentation,transform = transform_list)
test = SkinLesionDataset(csv_file=input_dir+'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
                                    root_dir=input_dir+'ISIC-2017_Test_v2_Data/',segment_dir=input_dir+'ISIC-2017_Test_v2_Part1_GroundTruth', useSegmentation =useSegmentation,transform = transform_list)

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
#image1 = io.imread('../datasets/ISIC-2017_Training_Data/ISIC_0000000.jpg')
#image2 = io.imread('../datasets/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png')
#image_modified = grey_world(image)

#print(image2)

#image2 = np.ones(image2.shape)

#print(image1.shape, image2.shape)
#image_modified = masking(image1, image2,255)
#plt.imshow(image_modified)
#plt.imshow(image)








# In[139]:

model_resnet = torchvision.models.resnet34(pretrained=True)
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


# In[141]:

model_vgg = torchvision.models.vgg16_bn(pretrained=True)
for param in model_vgg.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features

#print(num_ftrs)
model_vgg.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_vgg.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# In[161]:

dataset_sizes1 = {'train':len(validation),'val':len(validation),'test':len(test)}
#print(dataset_sizes1)

dataloaders1 = {'train':val_data,'val':val_data,'test':test_data}

model_conv = train_model(model_resnet, criterion, optimizer_conv,
                         exp_lr_scheduler, dataloaders1,dataset_sizes1,num_epochs=1)


# In[164]:


# In[129]:

#test_model(model_conv, criterion, dataloaders1,dataset_sizes1)


# In[186]:

print("Results using ensamble learning")

#models_dir = '../models/'
models_dir = input_dir
model_state_list = None
num_epochs = 5
model= model_resnet#, model_vgg
model_num = 1 # 1 for resnet
models_list = train_model_epochs(model, model_num,criterion, optimizer_conv,
                                      exp_lr_scheduler, dataloaders,dataset_sizes,num_epochs=num_epochs)
#for k,state in enumerate(models_list):
#    torch.save(state,models_dir+'11'+str(k)+'.pt')


# In[187]:

#print(models_list[0]==models_list[1])

#for k,state in enumerate(model_state_list):
#    torch.save(state,models_dir+'11'+str(k)+'.pt')


# In[188]:

#inputs = next(iter(test_data))

#mymodel  = model_resnet
#mymodel.load_state_dict(torch.load(models_dir+'110.pt'))
#mymodel =  torch.load(models_dir+'110.pt')
#outputs1  = mymodel(Variable(inputs['image']))
#outputs1  = mymodel(Variable(inputs['image']))

#print(outputs1)


#mymodel.load_state_dict(torch.load(models_dir+'111.pt'))
#mymodel =  torch.load(models_dir+'111.pt')
#outputs2  = mymodel(Variable(inputs['image']))
#print(outputs2)

#mymodel.load_state_dict(model_state_list[0])
#mymodel1 =  torch.load(models_dir+'110.pt')
#mymodel2 =  torch.load(models_dir+'111.pt')
#outputs1  = mymodel1(Variable(inputs['image']))
#print(outputs1)
#outputs2  = mymodel2(Variable(inputs['image']))
#print(outputs2)


# In[148]:

# ensamle predictions
acc = test_ensamble_model(model, dataloaders, dataset_sizes, models_dir, (1,1, num_epochs))
print(acc)


# In[82]:

# Actual Algorithm
#models_list = (model_conv,model_conv)
#print("Results using Meta-model")
#model_data = train_meta_model(dataloaders, dataset_sizes, model_dir, (len(models),1, num_epochs))


# In[83]:

#acc = test_meta_model(dataloaders, dataset_sizes, model_dir, (len(models),1, num_epochs))
#print(acc)

