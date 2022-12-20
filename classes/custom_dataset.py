import numpy as np 
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader

import glob
import pickle

import torch
import random


class CustomDataset(Dataset):
    def __init__(self, root_dir, dim, channels, transform=None, TotalSamples=100):
        self.root_dir = root_dir
        self.transform = transform
        file_list = glob.glob(self.root_dir + "*")
        print(file_list)
        self.data = []
        self.datashape=(dim,dim,channels)
        self.TotalSamples=TotalSamples

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for file_path in glob.glob(class_path + "/*.pickle"):
                #data.append([file_path, class_name])
                print(file_path)
                x = pickle.load(open(file_path,"rb"))
                x = tf.keras.utils.normalize(x)
                size = x.shape
                print(size)
                for image in range(size[0]):
                    im = [x[image].astype(float)]
                    im = np.array(im)
                    im = im.squeeze()  
                    if im.shape == self.datashape:
                        self.data.append([im, class_name])
        self.data = self.format_data(True)
        #print(self.data)

        self.class_map = {'HCT-116': 0, 'HL60': 1, 'JURKAT': 2, 'LNCAP': 3, 'MCF7': 4, 'PC3': 5, 'THP-1': 6, 'U2OS': 7}
        self.img_dim = (dim, dim)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, class_name = self.data[idx]
        #img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([class_id])
        #return img_tensor, class_id
        return img_tensor.float(), class_id
    
    def class_to_idx(self):
        return print(self.class_map)
    
    #Balance the classes so that they are of equal lengths
    #dataset is a list of [image, label]
    def format_data(self, augment):
        dataset = self.data
        classes = dict([])
        class_index = []
        data = []
        X = []
        y = []
        dataset_new=[]
        reverse_class_map = {0:'HCT-116' , 1:'HL60', 2:'JURKAT', 3:'LNCAP', 4:'MCF7', 5:'PC3', 6:'THP-1', 7:'U2OS'}
        for x in dataset:
            # check if exists in unique_list or not 
            if x[1] not in list(classes.keys()):
                classes[x[1]] = 1
            else:
                classes[x[1]] = classes[x[1]] + 1
            class_index.append(x[1])
            data.append(x[0])
        print(classes.items())

        if augment == True:
            for item in list(classes.keys()):
                indicies = [i for i, x in enumerate(class_index) if x == item] 
                if len(indicies) >= self.TotalSamples:
                    indicies = random.sample(indicies, k = self.TotalSamples)
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                else:
                    aug = []
                    for i in indicies:
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                        #X.append(data[i])
                        #y.append(class_index[i])
                        aug.append(data[i])
                    new_data = self.data_augmentation(aug)
                    for i in range(len(new_data)):
                        #X.append(new_data[i])
                        #y.append(class_index[indicies[0]])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        else:
             for item in list(classes.keys()):
                    indicies = [i for i, x in enumerate(class_index) if x == item]
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        return dataset_new
    ##Rotational data augmentation
    def data_augmentation(self, data):
        new_data = []

        for i in range(self.TotalSamples-len(data)):
            new_image = data[random.randint(1,len(data)-1)]
            for r in range(random.randint(1,3)):
                new_image = np.rot90(new_image)
            new_data.append(new_image)
        return new_data
