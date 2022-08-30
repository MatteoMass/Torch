from utils import splitAndOneHot
import torch


class Dataset:
    def __init__(self, path):
        self.x = []
        self.y = []

        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                x, y = splitAndOneHot(l.strip().split()[:-1])
                self.x.append(x)
                self.y.append([y])

        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y)
    
