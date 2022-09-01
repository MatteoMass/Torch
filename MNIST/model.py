

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_channels, first_conv_kernel, first_out_channels, second_no_channel, second_conv_kernel):
        super().__init__()
        self.conv_part = nn.Sequential(
             nn.Conv2d(input_channels, first_out_channels, first_conv_kernel ),
             nn.ReLU(),
             nn.MaxPool2d(2,2),
             nn.Conv2d(first_out_channels, second_no_channel, second_conv_kernel),
             nn.ReLU(),
             nn.MaxPool2d(2,2)  
        )
        self.fcn_part = nn.Sequential(
            nn.Linear(225),
            nn.Sigmoid(),
            nn.Linear(10),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        conv_res = self.conv_part(x)
        res = self.fcn_part(torch.flatten(conv_res))
        return res



if __name__ == "__main__":
    m = Model(1, (3,3), 3, 9, (3,3))
    t = torch.rand(1, 1, 28, 28)
    print(t)
    print(m(t))