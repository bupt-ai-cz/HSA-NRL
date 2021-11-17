import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import numpy as np
import torch
from .utils import noisify

class CHAOYANG(data.Dataset):
    def __init__(self, root, json_name=None, path_list=None, label_list=None, train=True, transform=None):
        imgs = []
        labels = []
        if json_name:
            json_path = os.path.join(root,json_name)
            with open(json_path,'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root,load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
        if (path_list and label_list):
            imgs = path_list
            labels = label_list
        self.transform = transform
        self.train = train  # training set or test set
        self.dataset='chaoyang'
    
        self.nb_classes=4
        if self.train:
            self.train_data, self.train_labels = imgs,labels
            self.train_noisy_labels=[i for i in self.train_labels]
            self.noise_or_not = [True for i in range(self.__len__())]
        else:
            self.test_data, self.test_labels = imgs,labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
    
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)