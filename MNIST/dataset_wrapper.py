import torch
from torch.utils.data import Dataset
from mnist import MNIST
import matplotlib.pyplot as plt



class DatasetWrapper(Dataset):
    def __init__(self, training = "True", folder_path = "dataset"):
        self.x = []
        self.y = []

        self.img_size = 28
        self.channel = 1


        mndata = MNIST(folder_path)

        if training:
            self.x, self.y = mndata.load_training()
        else:
            self.x, self.y = mndata.load_testing()

        self.x = torch.FloatTensor(self.x).view(-1, self.img_size, self.img_size, self.channel)
        self.y = torch.FloatTensor(self.y).view(-1, self.img_size, self.img_size, self.channel)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



if __name__ == "__main__":
    d = DatasetWrapper()
    print(d.x.view(-1, 28, 28, 1).shape)
    plt.imshow(d.x.view(-1, 28, 28, 1)[128])
    plt.show()