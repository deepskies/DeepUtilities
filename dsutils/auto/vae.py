import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import dsutils as ds
# from dsutils.auto import run_epoch
from dsutils.auto import shape

'''
This network is defined recursively.
|layers| ~ log_2(input_dim)
'''
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, classify=True, layer_type='mlp'):
        super(VAE, self).__init__()

        if layer_type == 'mlp':
            # whats better, this:
            self.enc, self.dec = shape.mlp_vae(shape.log_dims(input_dim, output_dim))
            # or this: (getting rid of shape.mlp_vae)
            # self.dims = shape.log_dims(input_dim, output_dim)
            # self.enc = nn.ModuleList(shape.mlp_layers(self.dims))
            # self.dec = nn.ModuleList(shape.mlp_layers(self.dims[::-1]))
        else:
            raise NotImplementedError("this layer time not supported yet")

    def single_direction(self, x, direction):
        # directions: encoding and decoding
        for layer in direction:
            # if i == self.num_layers - 1 and :
            #     x = F.softmax(layer(x), dim=1)
                # break
            x = torch.tanh(layer(x))
        return x

    def forward(self, x):
        class_prediction = self.single_direction(x, self.enc)
        dec = self.single_direction(class_prediction, self.dec)


    def __repr__(self):
        print(f'encoder: {self.enc}')
        print(f'decoder: {self.dec}')
        return 'network'
