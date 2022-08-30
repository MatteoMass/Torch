import torch
import torch.nn as nn
import torch.nn.functional as F
from readDataset import Dataset
from torch.utils.data import DataLoader
INPUT_DIM = 17
OUTPUT_DIM = 1

LR = 0.01
EPOCHS = 10

class Model(nn.Module):

    def __init__(self, hiddenNeuron):
        super().__init__()
        self.firstLayer = nn.Linear(INPUT_DIM, hiddenNeuron)
        self.secondLayer = nn.Linear(hiddenNeuron, OUTPUT_DIM)


    def forward(self, x):
        x = self.firstLayer(x)
        x = F.sigmoid(x)
        x = self.secondLayer(x)
        x = F.sigmoid(x)
        
        return x

if __name__ == '__main__':

    dataset = Dataset("dataset/monks-1.train")

    dl = DataLoader(dataset, batch_size=1)

    model = Model(4)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    losses = []
    for e in range(EPOCHS):
        

            pred_y = model(dataset.x)
            loss = loss(pred_y, dataset.y)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()

            optimizer.step()

    print("ADDESTRATO!")