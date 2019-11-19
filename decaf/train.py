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
from sklearn import svm


def train_model(model, dataloaders, test_dict, num_epochs, device, result_file):

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
        
        best_acc = 0.0

        
        c_val = [1]

        

        for c in c_val:
            clf = svm.LinearSVC(C=c)

            print('C {}'.format(c))
            print('-' * 10)

            y_true = []
            y_pred = []
            x_o = []
            y_o = []

            # Each epoch has a training and validation phase
            for phase in phases:

                running_corrects = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss

                        outputs = model(inputs)
                        
                        if (phase == 'val'):
                            preds = clf.predict(outputs.cpu().detach().numpy().tolist())
                            l = labels.cpu().detach().numpy().tolist()
                            p = preds.tolist()
                            if (l==p):
                                running_corrects+=1
                            y_true = y_true + l
                            y_pred = y_pred + p
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            l = labels.cpu().detach().numpy().tolist()
                            o = outputs.cpu().detach().numpy().tolist()
                            y_o = y_o + l
                            x_o = x_o + o
                            
                        

                if (phase == 'train'):
                    clf.fit(x_o, y_o) 

                if (phase == 'val'):
                    print("Corrects ",running_corrects, " on ",len(dataloaders[phase].dataset))
                    epoch_acc = running_corrects / len(dataloaders[phase].dataset)
                    print('{} Acc: {:.4f}'.format(phase, epoch_acc))
                    result_file.write('F1: {}'.format(metrics.f1_score(y_true, y_pred, average="micro"))+"\n")
                    result_file.write('Precision: {}'.format(metrics.precision_score(y_true, y_pred, average="micro"))+"\n")
                    result_file.write('Recall: {}'.format(metrics.recall_score(y_true, y_pred, average="micro"))+"\n")

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

            result_file.write("C value = "+str(c)+"\n")

            result_file.write("Accuracy validation "+str(best_acc)+"\n")

            result_file.write("Ground truth "+str(y_true)+"\n")
            result_file.write("Prediction "+str(y_pred)+"\n\n")


            print('Best val Acc: {:4f}'.format(best_acc)+"\n\n")

    # load best model weights
    #model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history, y_true, y_pred

