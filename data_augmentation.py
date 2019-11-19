import pydicom as dicom
import os
import cv2
import numpy as np 
from skimage import color
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import torchvision
import csv

# Specify the .jpg folder path
folder_path = ['multi_input/dataset_128/tra/train/1/','multi_input/dataset_128/tra/train/2/','multi_input/dataset_128/tra/train/3/',
'multi_input/dataset_128/tra/train/4/','multi_input/dataset_128/tra/train/5/','multi_input/dataset_128/sag/train/1/','multi_input/dataset_128/sag/train/2/','multi_input/dataset_128/sag/train/3/',
'multi_input/dataset_128/sag/train/4/','multi_input/dataset_128/sag/train/5/']
#folder_path = ['test_image']
# Specify the output jpg/png folder path

for path in folder_path:
    images_path = os.listdir(path)

    for n, image in enumerate(images_path):
        image_to_transform = sk.io.imread(os.path.join(path, image), plugin='matplotlib')
        image = image.replace('.jpg', '')
        for angle in [-90, 90]:
            to_save = sk.transform.rotate(image_to_transform, angle)
            new_file_path = '%s/%s_aumented_%s.jpg' % (path, image ,str(angle))

            # write image to the disk
            sk.io.imsave(new_file_path, to_save)

        horizontal_flip = image_to_transform[:, ::-1]    
        new_file_path = '%s/%s_hflip.jpg' % (path, image)

        sk.io.imsave(new_file_path, horizontal_flip)

        vertical_flip = image_to_transform[::-1, :]    
        new_file_path = '%s/%s_vflip.jpg' % (path, image)

        sk.io.imsave(new_file_path, vertical_flip)
        img_not = cv2.bitwise_not(image_to_transform)
        new_file_path = '%s/%s_negative.jpg' % (path, image)

        sk.io.imsave(new_file_path, img_not)

        