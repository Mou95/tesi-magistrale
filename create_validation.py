import os
import random

path_input = ['multi_input/dataset_128/tra/train/1/','multi_input/dataset_128/tra/train/2/','multi_input/dataset_128/tra/train/3/','multi_input/dataset_128/tra/train/4/','multi_input/dataset_128/tra/train/5/']
path_out = ['multi_input/dataset_128/tra/val/1/','multi_input/dataset_128/tra/val/2/','multi_input/dataset_128/tra/val/3/','multi_input/dataset_128/tra/val/4/','multi_input/dataset_128/tra/val/5/']
path_input_sag = ['multi_input/dataset_128/sag/train/1/','multi_input/dataset_128/sag/train/2/','multi_input/dataset_128/sag/train/3/','multi_input/dataset_128/sag/train/4/','multi_input/dataset_128/sag/train/5/']
path_out_sag = ['multi_input/dataset_128/sag/val/1/','multi_input/dataset_128/sag/val/2/','multi_input/dataset_128/sag/val/3/','multi_input/dataset_128/sag/val/4/','multi_input/dataset_128/sag/val/5/']

for y, path in enumerate(path_input):
    images_path = os.listdir(path)
    images_path.sort()
    images_path_sag = os.listdir(path_input_sag[y])
    images_path_sag.sort()
    print(images_path[0:10])
    print(images_path_sag[0:10])
    list_pos = random.sample(range(0,len(images_path)), int(len(images_path)/10))

    #list_val = random.sample(images_path, int(len(images_path)/10))

    print(list_pos)
    print(sorted(list_pos))

    for pos in list_pos:
        os.rename(os.path.join(path,images_path[pos]), os.path.join(path_out[y], images_path[pos]))
        os.rename(os.path.join(path_input_sag[y],images_path_sag[pos]), os.path.join(path_out_sag[y], images_path_sag[pos]))