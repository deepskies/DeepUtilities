import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import dsutils as ds
from dsutils.auto import mlp

# from dsutils.auto import train
from dsutils.auto import shape

CONFIG = {
    'type': 'vae',
    'layer_type': 'mlp',
    'factor': 10,
    'classify': True,
    'activation_fxn': 'tanh',
    'lr': 1e-3,
}

class VAE(nn.Module):
    '''
    This network is defined recursively.
    |layers| ~ log_n(input_dim)
    output dim is bottleneck layer size
    The minimum number of layers needs to be 3 i think
    '''
    def __init__(self, input_dim, output_dim, config=None, verbose=True):
        super(VAE, self).__init__()
        self.config = config
        if not self.config:
            self.config = CONFIG

        dec_config = self.config
        dec_config['classify'] = False

        layer_type = self.config.get('layer_type')

        if layer_type == 'mlp':
            factor = self.config['factor']
            self.enc = mlp.MLP(input_dim, output_dim, config=self.config)
            self.dec = mlp.MLP(output_dim, input_dim, config=dec_config)
            if verbose:
                print(f'enc: {self.enc}')
                print(f'dec: {self.dec}')
        else:
            raise NotImplementedError('this layer type not supported yet')


    def forward(self, x):
        # class_prediction = mlp.general_forward(x, self.enc, self.classify, self.activation_fxn)
        class_prediction = self.enc.forward(x)
        dec = self.dec.forward(class_prediction)
        return class_prediction, dec

    def __repr__(self):
        print(f'encoder: {self.enc}')
        print(f'decoder: {self.dec}')
        return 'network'
