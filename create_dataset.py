import csv
import os
import pydicom
import shutil
import numpy as np 
from skimage import color
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import torchvision
import cv2

crop = True

def isInstanceNumber(image):
    dic = pydicom.read_file(image[0])
    #if (dic.InstanceNumber in range(image[1][2]-2, image[1][2]+3, 1)):
    if (dic.InstanceNumber == image[1][2]):
        return True
    return False

def sliceNumber(pos):
    xyk = pos.split()
    return int(xyk[0]), int(xyk[1]), int(xyk[2])

def extractImageTrain(locations, path):
    for image in locations:
        print(image)
        ds = pydicom.read_file(image[0])

        img = ds.pixel_array

        # Now, img is pixel_array. it is input of our demo code

        # Convert pixel_array (img) to -> gray image (img_2d_scaled)
        ## Step 1. Convert to float to avoid overflow or underflow losses.
        img_2d = img.astype(float)

        ## Step 2. Rescaling grey scale between 0-255
        img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0

        ## Step 3. Convert to uint
        img_2d_scaled = np.uint8(img_2d_scaled)

        
        #trasform image in rgb
        #image = image.replace('.dcm', '.jpg')
        img_3d = color.gray2rgb(img_2d_scaled)
        if crop:
            x,y = image[1][0], image[1][1] 
            img_3d = img_3d[y-64:y+64, x-64:x+64]
            resized = cv2.resize(img_3d, (224,224), interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(img_3d, (224,224), interpolation = cv2.INTER_AREA)
        path_image = path[int(image[3])-1]+ds.PatientID+"_"+str(ds.InstanceNumber)+"_"+image[2]+".jpg"
        #print(path_image)
        cv2.imwrite(path_image, resized)
        



path='../PROSTATEx/'
path_image = ['multi_input/dataset_128/tra/train/1/','multi_input/dataset_128/tra/train/2/','multi_input/dataset_128/tra/train/3/',
'multi_input/dataset_128/tra/train/4/','multi_input/dataset_128/tra/train/5/']
lstFilesDCM = []  # create an empty list

with open('../PROSTATEx/ProstateX-2-Findings-Train.csv') as find_file:
    with open('../PROSTATEx/ProstateX-2-Images-Train.csv') as image_file:
        find_reader = csv.reader(find_file, delimiter=',')
        image_reader = list(csv.reader(image_file, delimiter=','))
        line_count = 0
        for row in find_reader:
            if line_count == 0:
                line_count += 1
            else:
                for row1 in image_reader:
                    if row1[0] == row[0] and ('tra0' in row1[1] or 'Grappa30' in row1[1]) and row[1] == row1[3]:
                        path_patient = path + row[0]
                        for dirName, subdirList, fileList in os.walk(path_patient):
                            for filename in fileList:
                                if ('t2tsetra' in dirName or 'Grappa30' in dirName):  # check whether the file's DICOM
                                    lstFilesDCM.append((os.path.join(dirName,filename),sliceNumber(row1[6]), row[1], row[4])) 

finalList = []                                                                   
                                    
for i, image in enumerate(lstFilesDCM):
    if isInstanceNumber(image):
        finalList.append(image)

f= open("train_tra.txt","w+")        
for image in finalList:
    f.write(str(image)+'\n')

extractImageTrain(finalList, path_image)