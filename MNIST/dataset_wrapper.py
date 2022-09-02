import torch
from torch.utils.data import Dataset
from mnist import MNIST

import random


class DatasetWrapper(Dataset):
    def __init__(self, training = True, folder_path = "dataset"):
        self.x = []
        self.y = []

        self.img_size = 28
        self.channel = 1


        mndata = MNIST(folder_path)

        if training:
            self.x, self.y = mndata.load_training()
        else:
            self.x, self.y = mndata.load_testing()

        self.x = torch.FloatTensor(self.x)

        self.x = self.x.view(-1, self.channel, self.img_size, self.img_size)
        self.y = to_categorical(self.y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def train_validation_split(self, validation_split, shuffle=False):
        training_set = DatasetWrapper()
        validation_set = DatasetWrapper()
        range_from = range(len(self))
        if shuffle:   

            validation_indexes = random.sample(range_from, validation_split)
            training_indexes = [x for x in range_from if x not in validation_indexes]
        else:
            validation_indexes = range_from[:validation_split]
            training_indexes = range_from[validation_split:]

        training_set.x = self.x[training_indexes]
        training_set.y = self.y[training_indexes]

        validation_set.x = self.x[validation_indexes]
        validation_set.y = self.y[validation_indexes]

        return training_set, validation_set

def to_categorical(y):
    categorical_y = []

    
    for i in y:
        temp = [0]*10
        temp[i] = 1
        categorical_y.append(temp)
    
    return torch.FloatTensor(categorical_y)