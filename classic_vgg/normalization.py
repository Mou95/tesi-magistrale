import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import os

class MyDataset():
    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

        '''for line in x1:
            img = Image.open("../"+line.split(",,")[0])
            self.data.append(img)
            x+=1
            print(x)

        for line in x2:
            img = Image.open("../"+line.split(",,")[0])
            self.data.append(img)
            x+=1
            print(x)

        for line in x3:
            img = Image.open("../"+line.split(",,")[0])
            self.data.append(img)
            x+=1
            print(x)'''

    def online_mean_and_sd(self):
        """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)
        x = 0

        folder_path = ['dataset_128/train/1/','dataset_128/train/2/','dataset_128/train/3/','dataset_128/train/4/','dataset_128/train/5/',
'dataset_128/val/1/','dataset_128/val/2/','dataset_128/val/3/','dataset_128/val/4/','dataset_128/val/5/']
        for path in folder_path:
            images_path = os.listdir(path)
            for n, image in enumerate(images_path):
                #print(data)
                img = Image.open(os.path.join(path,image))
                data = self.trans(img)
                img.close()
                data = data.unsqueeze(0)
                #print(data.shape)
                b, c, h, w = data.shape
                nb_pixels = b * h * w
                #print(nb_pixels)
                sum_ = torch.sum(data, dim=[0,2,3])
                sum_of_square = torch.sum(data ** 2, dim=[0,2,3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels
                x+=1
                print(x)

        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)



dataset = MyDataset()

mean, std = dataset.online_mean_and_sd()
print(mean,std)

'''transf = torchvision.transforms.ToTensor()
norm = torchvision.transforms.Normalize(mean = [0.3321, 0.3321, 0.3321], std = [0.2618, 0.2618, 0.2618])
toPil = torchvision.transforms.ToPILImage()

img = Image.open("../Images/pablo-picasso/pablo-picasso_woman-with-hat-1962.jpg")
img = toPil(norm(transf(img)))
img.save("../Images/pablo-picasso/pablo-picasso_woman-with-hat-1962.jpg")
img.close()'''
