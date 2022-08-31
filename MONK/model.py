import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_dim, first_hidden_layer, second_hidden_layer, output_dim):
        super().__init__()
        if second_hidden_layer == 0:
            self.architecture = nn.Sequential(
                nn.Linear(input_dim, first_hidden_layer),
                nn.Sigmoid(),
                nn.Linear(first_hidden_layer, output_dim),
                nn.Sigmoid()
            )
        else:
            self.architecture = nn.Sequential(
            nn.Linear(input_dim, first_hidden_layer),
            nn.Sigmoid(),
            nn.Linear(first_hidden_layer, second_hidden_layer),
            nn.Sigmoid(),
            nn.Linear(second_hidden_layer, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):      
        return self.architecture(x)
