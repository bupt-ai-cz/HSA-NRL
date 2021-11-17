import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import numpy as np
import torch
from .utils import noisify

class MICCAI(data.Dataset):
    def __init__(self, root="", json_name=None, path_list=None, label_list=None, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, random_state=0, nb_classes = 2):
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
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='miccai'
        self.noise_type=noise_type
        self.nb_classes=nb_classes
        if self.train:
            self.train_data, self.train_labels = imgs,labels
            if noise_type != 'clean':
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state,nb_classes=self.nb_classes)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
        else:
            self.test_data, self.test_labels = imgs,labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            #if self.noise_type is not None:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index][0]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)