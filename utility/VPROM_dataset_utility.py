import glob
import os
from random import shuffle, randint

import matplotlib
import numpy as np
import torch
from numpy import dtype
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import torchvision.models as model
import torch.nn as nn


class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class VPROM_dataset_utility(Dataset):
    def __init__(self, flag, fnames_imgs, fnames_target, img_size, M, shuffle = False, train_folder = 'train/', test_folder = 'test/'):
        self.fnames_imgs = fnames_imgs
        self.fnames_target = fnames_target
        self.img_size = img_size
        self.features_size = 2048
        self.shuffle = shuffle
        self.M = M
        self.resnet101 = model.resnet101(pretrained = True)
        self.resnet101.eval()
        self.modules = list(self.resnet101.children())[:-1]
        self.resnet101Features = nn.Sequential(*self.modules)
        self.is_train = flag
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.switch = [3, 4, 5, 0, 1, 2, 6, 7]

    def __len__(self):
        return len(self.fnames_target)

    def __getitem__(self, index):
        flag = True
        while flag:
            try:
                if self.is_train == 1:

                    current_fnames = self.fnames_imgs[index]

                    feature_path = self.train_folder + str(index) + '_feature_Ours.pt'
                    target_path = self.train_folder + str(index) + '_target_Ours.pt'
                else:

                    current_fnames = self.fnames_imgs[index]

                    feature_path = self.test_folder + str(index) + '_feature_Ours.pt'
                    target_path =  self.test_folder + str(index) + '_target_Ours.pt'

                target = torch.load(target_path, weights_only=True)
                RPM = torch.load(feature_path, weights_only=True)

                flag = False
            except Exception as e:
                print(f'Could not load example {index}')
                index = randint(0, len(self.fnames_target))
                print(f'Try {index} instead')

        if self.shuffle:
            context = RPM[:8, :, :, :, :]
            choices = RPM[8:, :, :, :, :]

            indices = np.arange(8)
            np.random.shuffle(indices)
            new_target = np.where(indices == target.numpy())[0][0]
            new_choices = choices[indices, :, :, :, :]
            switch_2_rows = np.random.rand()
            if switch_2_rows < 0.5:
                context = context[self.switch, :, :, :, :]
            RPM = np.concatenate((context, new_choices))
            target = new_target

        return RPM, target, current_fnames
