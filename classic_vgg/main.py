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

    '''network = ['vgg']
    #epochs = ['15','20']
    #update = "fully_and_all" #, "fully_connected" "fully_and_all", "all_model"
    #crit = [] # criterion

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "dataset/"
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
    data_transforms = {
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
    
    
    train_acc_history =  [(0.40173507), (0.59993327), (0.8008008), (0.8705372), (0.9009009), (0.91925259), (0.92492492), (0.92692693), (0.93193193), (0.94094094), (0.93993994), (0.93493493)]
    val_acc_history = [(0.51051051), (0.73573574), (0.83483483), (0.89489489), (0.81981982), (0.89189189), (0.88288288), (0.87087087), (0.89189189), (0.88888889), (0.89489489), (0.87987988)]
    
    plt.title("ResNeXt accuracy from scratch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,12+1),train_acc_history,label="Training")
    plt.plot(range(1,12+1),val_acc_history,label="Validation")
    plt.ylim((0,1.0))
    plt.xticks(np.arange(1, 12+1, 1.0))
    plt.legend()
    plt.show()



if __name__ == '__main__':

    main()
