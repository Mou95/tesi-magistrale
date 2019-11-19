from model import vgg19_decaf
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import os
from train import *

def test_dic():
    test_dict = {'patientID' : [], 'fID' : []}
    images_path = os.listdir('dataset_128/test/1')
    for _, image in enumerate(images_path):
        test_dict['patientID'].append(image.split('_')[0])
        test_dict['fID'].append(image.split('_')[1][0])

    return test_dict

def tuning(network, device, num_classes, batch_size, num_epochs, feature_extract, dataloaders_dict):
    result_file = open("best_network","w+")

    # Initialize the model for this run
    model_ft = vgg19_decaf()
    # Send the model to GPU
    model_ft = model_ft.to(device)

    #create dictionary for test set
    test_dict = test_dic()
    # Train and evaluate
    _, train_acc_history, val_acc_history, y_true, y_pred = train_model(model_ft, dataloaders_dict, test_dict, num_epochs, device, result_file)

    return train_acc_history, val_acc_history, y_true, y_pred
    #test_model(model_ft, model_ft.state_dict(), device)

#torch.save(model_ft.state_dict(), "../Weigth/"+net+".pt")
