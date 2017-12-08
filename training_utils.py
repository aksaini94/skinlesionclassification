from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn import svm

plt.ion()   # interactive mode


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['class1']

                # wrap them in Variable
                #if use_gpu:
                #    inputs = Variable(inputs.cuda())
                #    labels = Variable(labels.cuda())
                #else:
                inputs, labels = Variable(inputs), Variable(labels)
                labels = labels.long()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #print(outputs), print(labels)
                loss = criterion(outputs, labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return



def test_model(model, criterion,dataloaders,dataset_sizes):
    since = time.time()
    correct = 0.0
    total = 0.0
    loss_final =0.0
    #for data in testloader:
    #    images, labels = data
    #    outputs = net(Variable(images))
    #    _, predicted = torch.max(outputs.data, 1)
    #total += labels.size(0)
    #correct += (predicted == labels).sum()

    phase = 'test'
    # Iterate over data.
    for data in dataloaders['test']:
        # get the inputs
        inputs, labels = data['image'], data['class1']

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        #print(outputs), print(labels)
        loss = criterion(outputs, labels)


        # statistics
        loss_final += loss.data[0]
        correct += torch.sum(preds == labels.data)

    loss_final = loss_final / dataset_sizes[phase]
    acc = correct / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, loss_final, acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model   

def train_meta_model(models_list, num_times, dataloaders, dataset_sizes):
    # model_list = [model1,model2, ...]
    #num_times = number of time each model is to be applied
    
    since = time.time()
    n_models = len(models_list)
    
    feat_list=np.zeros((dataset_sizes['val'],n_models,num_times*2))
    labels_list=np.zeros(dataset_sizes['val'])
    
    # get all the features extracted from the models
    for i,data in enumerate(dataloaders['val']):
        # get the inputs
        inputs, labels = data['image'], data['class1']
        batch_size = data['image'].shape[0]
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        # forward
        for j, model in enumerate(models_list):
            for times in range(num_times):
                outputs = model(inputs)
                feat_list[i*batch_size:(i+1)*batch_size, j,2*times:2*(times+1)]=outputs.data.numpy()
        labels_list[i*batch_size:(i+1)*batch_size]=labels.data.numpy()
 
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        
    clf.fit(feat_list.reshape((dataset_sizes['val'], n_models*num_times*2)), labels_list)  

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    return clf, models_list, num_times


def test_meta_model(model, dataloaders, dataset_sizes):   
    since = time.time()
    
    clf, models_list, num_times = model
    n_models = len(models_list)
    #feat_list=np.zeros((dataset_sizes['test'],n_models,num_times))
    #labels_list=np.zeros(dataset_sizes['test'])
    correct = 0
    phase = 'test'
    for data in dataloaders[phase]:
        inputs, labels = data['image'], data['class1']
        batch_size = data['image'].shape[0]
        
        feat_list=np.zeros((batch_size,n_models,num_times*2))
        #labels_list=np.zeros(dataset_sizes['val'])
        
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        # forward
        for j, model in enumerate(models_list):
            for times in range(num_times):
                outputs = model(inputs)
                feat_list[0:batch_size, j, 2*times:2*(times+1)]=outputs.data.numpy()

        preds = clf.predict(feat_list.reshape((batch_size,n_models*num_times*2)))
        correct += np.sum(preds == labels.data.numpy())

    acc = correct / dataset_sizes[phase]

    print('{}  Acc: {:.4f}'.format(
        phase, acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return acc



