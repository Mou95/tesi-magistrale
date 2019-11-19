from torchvision import models
import torchvision
import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision import transforms, datasets
import csv
from sklearn import metrics
from  torch.autograd import Variable
import random

def top_3_accuracy(outputs, label):
    corrects = 0
    for x in range(len(outputs)):
        #print(outputs[x])
        ind = torch.topk(outputs[x],k=3).indices
        #print(ind)
        #print(label[x])
        if label[x].tolist() in ind.tolist():
            corrects += 1


    return corrects

def train_model(model, dataloaders_dict_x, dataloaders_dict_y, test_dict, criterion, optimizer, scheduler, num_epochs, device, result_file):
    dict_data = {}
    iter_x = iter(dataloaders_dict_x['train'])
    iter_y = iter(dataloaders_dict_y['train'])
    for i, _ in enumerate(dataloaders_dict_x['train']):
        x, label_x = next(iter_x)
        y, _ = next(iter_y)
        dict_data[i] = {'immagini' : [x,y], 'label' : label_x}

    with open('test.csv', mode='w') as test_csv:
        test_csv = csv.writer(test_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_csv.writerow(['ProxID','fid','ggg'])
                            
        since = time.time()

        phases = ['train','val']

        train_acc_history = []
        val_acc_history = []
        train_acc_history_3 = []
        val_acc_history_3 = []
        epoch_loss_val = []
        

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_acc_3 = 0.0

        index = 0

        '''low_gain = 0
        threshold = 0.03

        top_5 = 0'''

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            y_true = []
            y_pred = []

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    #scheduler.step()
                    model.train()  # Set model to training mode
                elif phase == 'val':
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_3 = 0.0


                if (phase=="train"):
                    # Iterate over data.
                    for i in random.sample(range(0,len(dict_data.keys())), len(dict_data.keys())):
                        x = dict_data[i]['immagini'][0]
                        y = dict_data[i]['immagini'][1]

                        label = dict_data[i]['label']

                        x = Variable(x.to(device))
                        y = Variable(y.to(device))
                        label = label.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss

                            outputs = model(x,y)

                            loss = criterion(outputs, label)

                            _, preds = torch.max(outputs, 1)

                            loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * x.size(0)
                        running_corrects += torch.sum(preds == label.data)
                        running_3 += top_3_accuracy(outputs.clone().detach(), label.data)

                if (phase == "val"):

                    iter_x = iter(dataloaders_dict_x['val'])
                    iter_y = iter(dataloaders_dict_y['val'])
                    for i, _ in enumerate(dataloaders_dict_x['val']):
                        x, label = next(iter_x)
                        y, _ = next(iter_y)

                        x = Variable(x.to(device))
                        y = Variable(y.to(device))
                        labels = Variable(label.to(device))

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss

                            outputs = model(x,y)

                            loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            
                            l = label.cpu().numpy().tolist()
                            p = preds.cpu().numpy().tolist()
                            y_true = y_true + l
                            y_pred = y_pred + p

                        # statistics
                        running_loss += loss.item() * x.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        running_3 += top_3_accuracy(outputs.clone().detach(), labels.data)

                epoch_loss = running_loss / len(dataloaders_dict_x[phase].dataset)
                print("Corrects ",running_corrects, " on ",len(dataloaders_dict_x[phase].dataset))
                epoch_acc = running_corrects.double() / len(dataloaders_dict_x[phase].dataset)
                epoch_acc_3 = running_3 / len(dataloaders_dict_x[phase].dataset)

                if phase == 'val':
                    epoch_loss_val.append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f} Top3Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc_3))
                print('F1: {}'.format(metrics.f1_score(y_true, y_pred, average="weighted")))
                print('Precision: {}'.format(metrics.precision_score(y_true, y_pred, average="weighted")))
                print('Recall: {}'.format(metrics.recall_score(y_true, y_pred, average="weighted")))

                # deep copy the model
                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                    train_acc_history_3.append(epoch_acc_3)
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val' and epoch_acc_3 > best_acc_3:
                    best_acc_3 = epoch_acc_3
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_acc_history_3.append(epoch_acc_3)

                    #Update all models weights for only next epoch
                    '''if update_type == "fully_and_all":
                        if epoch > 0 and val_acc_history[-1] - val_acc_history[-2] < threshold:
                            low_gain += 1
                            if low_gain == 2:
                                print("Change updating")
                                for param_group in optimizer.param_groups:
                                    learn = param_group['lr']
                                optimizer = optim.Adam(set_parameter_requires_grad_dinamically(model, "all_model"), lr=learn/10)
                                update_type = "all_model"
                        else:
                            low_gain = 0'''


                if phase == 'test':
                    test_result = epoch_acc
                    test_result_3 = epoch_acc_3

            print()

    time_elapsed = time.time() - since

    result_file.write("Accuracy training "+str([h.cpu().numpy() for h in train_acc_history])+"\n")
    result_file.write("Accuracy validation "+str([h.cpu().numpy() for h in val_acc_history])+"\n")
    #result_file.write("Accuracy training top-3 "+str(train_acc_history_3)+"\n")
    #result_file.write("Accuracy validation top-3 "+str(val_acc_history_3)+"\n")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #print('Best val Acc top-3: {:4f}'.format(best_acc_3))
    print('Test accuracy: {:4f}'.format(test_result))
    #print('Test accuracy top-3: {:4f}'.format(test_result_3))
    result_file.write('Best val Acc: {:4f}'.format(best_acc) +'\n')
    #result_file.write('Best val Acc top-3: {:4f}'.format(best_acc_3) +'\n')
    #result_file.write('Test accuracy: {:4f}'.format(test_result) +'\n')
    #result_file.write('Test accuracy top-3: {:4f}'.format(test_result_3) +'\n')
    result_file.write("Validation loss: "+str(epoch_loss_val)+"\n")
    result_file.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+'\n')
    result_file.write("Ground truth "+str(y_true)+"\n")
    result_file.write("Prediction "+str(y_pred)+"\n\n")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history, test_result, y_true, y_pred







