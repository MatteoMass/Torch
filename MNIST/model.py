

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_channels, first_conv_kernel, first_out_channels, dense_dim, output_dim):
        super().__init__()
        self.conv_part = nn.Sequential(
             nn.Conv2d(input_channels, first_out_channels, first_conv_kernel ),
             nn.ReLU(),
             nn.MaxPool2d(2,2)
        )


        self.fcn_part = nn.Sequential(
            nn.LazyLinear(dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, output_dim),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        batch = x.shape[0]
        conv_res = self.conv_part(x)
        res = self.fcn_part(conv_res.view(batch, -1))
        return res