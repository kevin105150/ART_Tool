# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:46:05 2021

@author: csyu
"""

from art.estimators.classification import PyTorchClassifier
import os

def Fit_classifier(model, clip_values, criterion, optimizer, data_Generator, x_train, y_train, isload, epoch_num, batch_size, device_type, text, select):
    
    
    # Create the ART classifier RGB
    classifier = PyTorchClassifier(
        model = model,
        clip_values = clip_values,
        loss = criterion,
        optimizer = optimizer,
        input_shape = (3, 224, 224),
        nb_classes = 10,
        device_type = device_type
    )
        
    print(classifier.device)
            
    # Train the ART classifier
    if isload == False:
        classifier.fit_generator(data_Generator, nb_epochs=epoch_num)
        classifier.save(filename=text, path=os.path.dirname(os.path.abspath(__file__)))
        
    return classifier