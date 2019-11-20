from tuning import *
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from confusion_matrix import *

def main():
    print(torch.__version__)
    classes = ["1","2","3","4","5"]

    network = ['vgg']
    #epochs = ['15','20']
    #update = "fully_and_all" #, "fully_connected" "fully_and_all", "all_model"
    #crit = [] # criterion

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "dataset_128_sag/"
    # Number of classes in the dataset
    num_classes = 5
    # Batch size for training (change depending on how much memory you have)
    batch_size = 16
    # Number of epochs to train for
    #num_epochs = 1
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    #feature_extract = True

    # Data augmentation and normalization for training
    # Just normalization for validation
    '''data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.3300, 0.3300, 0.3300], [0.2605, 0.2605, 0.2605])
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
            
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.3300, 0.3300, 0.3300], [0.2605, 0.2605, 0.2605])
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.3300, 0.3300, 0.3300], [0.2605, 0.2605, 0.2605])
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
        ])
    }

    print("Initializing Datasets and Dataloaders...")
    print(os.path.join(data_dir, 'train'))
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {}
    dataloaders_dict['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #from_scratch(network, "all_model", device, num_classes, batch_size, 20, False, dataloaders_dict)
    train_acc_history, val_acc_history, y_true, y_pred = tuning(network, device, num_classes, batch_size, 12, True, dataloaders_dict)

    plot_confusion_matrix(y_true, y_pred, classes)'''
    train_acc_history =  [(0.3460452), (0.53248588), (0.6680791), (0.84463277), (0.8799435), (0.95903955), (0.95056497), (0.81920904), (0.89124294), (0.92090395)]

    val_acc_history = [(0.52631579), (0.47368421), (0.40789474), (0.57894737), (0.68421053), (0.73684211), (0.71052632), (0.61842105), (0.61842105), (0.69736842)]
    plt.title("ResNeXt accuracy from scratch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,10+1),train_acc_history,label="Training")
    plt.plot(range(1,10+1),val_acc_history,label="Validation")
    plt.ylim((0,1.0))
    plt.xticks(np.arange(1, 10+1, 1.0))
    plt.legend()
    plt.show()



if __name__ == '__main__':

    main()
