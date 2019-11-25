from model import initialize_model
from model import train_model
from model import test_model
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import os

def test_dic():
    test_dict = {'patientID' : [], 'fID' : []}
    images_path = os.listdir('dataset/test/1')
    for _, image in enumerate(images_path):
        test_dict['patientID'].append(image.split('_')[0])
        test_dict['fID'].append(image.split('_')[1][0])

    print(test_dict)
    return test_dict

def tuning(network, device, num_classes, batch_size, num_epochs, feature_extract, dataloaders_dict):
    result_file = open("nocrop_with28features.txt","w+")
    result_file.write("12 epochs, 16 batch size, lr 0,0001\n")
    for net in network:
        for learning in [0.0001]:
            # Initialize the model for this run
            model_ft, _ = initialize_model(net, num_classes, feature_extract, use_pretrained=True)
            # Send the model to GPU
            model_ft = model_ft.to(device)
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if name.startswith("classifier"):
                    params_to_update.append(param)
                    param.requires_grad = True
                    print("\t",name)
                else:
                    param.requires_grad = False

            print(net+" "+str(learning))
            result_file.write(net+" con lr = "+str(learning)+"\n")
            # Inizialize optimizer
            optimizer_ft = optim.Adam(params_to_update, lr=learning)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            #create dictionary for test set
            test_dict = test_dic()
            # Train and evaluate
            _, train_acc_history, val_acc_history, _, y_true, y_pred = train_model(model_ft, dataloaders_dict, test_dict, criterion, optimizer_ft,exp_lr_scheduler, num_epochs, device, result_file)

            return train_acc_history, val_acc_history, y_true, y_pred
            #test_model(model_ft, model_ft.state_dict(), device)

        #torch.save(model_ft.state_dict(), "../Weigth/"+net+".pt")
