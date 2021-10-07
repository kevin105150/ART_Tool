# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:38:15 2021

@author: csyu
"""

"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
# Pytorch 1.6 and CUDA 10.2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from art.data_generators import PyTorchDataGenerator
from art.utils import load_mnist


import ART_Function as AF
import Attack_Function as Att_F
import Data_Load as DL
import Train_classifier as TC
import Prediction

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Swap axes to PyTorch's NCHW format
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

# Create the Predict model
model, text, select = AF.model_select()

# Select the Attack model
attack_select = int(input('Please select attack model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101):'))
Att_F.model_select(attack_select)

# Select attack function
attack_func, att_range = AF.attack_select()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=1e-4, momentum=0.9)

# load model, optimizer
model, optimizer, isload = AF.Load_Model(text, model, optimizer)

# Basic parameter
torch.backends.cudnn.benchmark = False
device_type = 'cuda:1'
epoch_num = 20
batch_size = 64
max_iter = 20
eps = 0.9
clip_values = (min_pixel_value, max_pixel_value)

# Train Data loader
if select < 3:
    train_set = DL.MNISTDataset(x_train, y_train, DL.data_transform)
else:
    train_set = DL.MNISTDataset(x_train, y_train, DL.data_transform_Resize)
data_loader = DataLoader(dataset=train_set, batch_size=batch_size)
data_Generator = PyTorchDataGenerator(iterator=data_loader, size=int(len(y_train)), batch_size=batch_size)

# Training classifier
print('Start training ...')

classifier = TC.Fit_classifier(
    model = model, 
    clip_values = clip_values,  
    criterion = criterion, 
    optimizer = optimizer, 
    data_Generator = data_Generator, 
    x_train = x_train, 
    y_train = y_train, 
    isload = isload, 
    epoch_num = epoch_num, 
    batch_size = batch_size, 
    device_type = device_type,
    text = text,
    select = select
    )

#Prediction.classifier_predict(classifier, x_test, y_test, text, select)

print('Finish training ...')

# Main
att_classifier = Att_F.attack_classifier(batch_size, epoch_num, clip_values, data_Generator, x_train, y_train, attack_select, device_type)
out_accuracy = Att_F.attack(eps, max_iter, batch_size, attack_func, x_test, y_test, classifier, att_classifier, attack_select, select)

# Save Data
AF.Save_data(attack_func, text, out_accuracy, attack_select, eps)