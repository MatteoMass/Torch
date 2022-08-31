from utils import split_and_one_hot

import random
import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, path=""):
        self.x = []
        self.y = []

        if path != "":
            with open(path) as f:
                lines = f.readlines()
                for l in lines:
                    x, y = split_and_one_hot(l.strip().split()[:-1])
                    self.x.append(x)
                    self.y.append([y])

        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y)
    

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

    def subset_from_indexes(self, indexes):
        subset_dataset = DatasetWrapper()

        subset_dataset.x = self.x[indexes]
        subset_dataset.y = self.y[indexes]

        return subset_dataset


        
        
    
