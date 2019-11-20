from tuning import tuning
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
#from ../decaf/confusion_matrix import *
#from confusion_matrix import *


def main():
    print(torch.__version__)
    classes = ["1","2","3","4","5"]

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir_x = "dataset_128/tra"
    data_dir_y = "dataset_128/sag"
    # Number of classes in the dataset
    num_classes = 5
    # Batch size for training (change depending on how much memory you have)
    batch_size = 2
    # Number of epochs to train for
    num_epochs = 20

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
            
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3511, 0.3511, 0.3511], [0.2399, 0.2399, 0.2399])
        ])
    }

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets_x = {x: datasets.ImageFolder(os.path.join(data_dir_x, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    image_datasets_y = {x: datasets.ImageFolder(os.path.join(data_dir_y, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict_x = {}
    dataloaders_dict_x['train'] = torch.utils.data.DataLoader(image_datasets_x['train'], batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders_dict_x['val'] = torch.utils.data.DataLoader(image_datasets_x['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders_dict_x['test'] = torch.utils.data.DataLoader(image_datasets_x['test'], batch_size=1, shuffle=False, num_workers=0)

    dataloaders_dict_y = {}
    dataloaders_dict_y['train'] = torch.utils.data.DataLoader(image_datasets_y['train'], batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders_dict_y['val'] = torch.utils.data.DataLoader(image_datasets_y['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders_dict_y['test'] = torch.utils.data.DataLoader(image_datasets_y['test'], batch_size=1, shuffle=False, num_workers=0)


    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #from_scratch(network, "all_model", device, num_classes, batch_size, 20, False, dataloaders_dict)
    train_acc_history, val_acc_history, y_true, y_pred = tuning(device, num_classes, batch_size, num_epochs, True, dataloaders_dict_x, dataloaders_dict_y)

    #plot_confusion_matrix(y_true, y_pred, classes)
    '''train_acc_history =  [(0.38304972), (0.49983317), (0.59993327), (0.70937604), (0.79612946), (0.84984985), (0.90323657), (0.91925259), (0.92959626), (0.94661328), (0.95028362), (0.94627961), (0.94627961), (0.94894895), (0.94594595)]
    val_acc_history = [(0.46546547), (0.55255255), (0.57657658), (0.62762763), (0.6996997), (0.72672673), (0.77477477), (0.77777778), (0.79279279), (0.7957958), (0.7957958), (0.81981982), (0.83183183), (0.81981982), (0.82282282)]
    
    plt.title("ResNeXt accuracy from scratch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,15+1),train_acc_history,label="Training")
    plt.plot(range(1,15+1),val_acc_history,label="Validation")
    plt.ylim((0,1.0))
    plt.xticks(np.arange(1, 15+1, 1.0))
    plt.legend()
    plt.show()'''



if __name__ == '__main__':

    main()
