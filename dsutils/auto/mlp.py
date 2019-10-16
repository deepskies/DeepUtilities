import torch
import torch.nn as nn
import torch.nn.functional as F

import dsutils as ds
from dsutils.auto import shape
from dsutils.auto import utils
# get one data (X, Y)
# flatten X and recurse down to Y.shape


# add optiion to set hyperparameters and activation fn
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(MLP, self).__init__()
        self.classify = config['classify']
        factor = config['factor']
        self.activation_fxn = utils.activations[config['activation_fxn']]

        # dense layers from tuples
        self.layers = shape.mlp_layers(shape.log_dims(input_dim, output_dim, factor=factor))
        self.num_layers = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.num_layers - 1:
                if self.classify:
                    x = F.softmax(layer(x), dim=1)
                else:
                    x = layer(x)
            else:
                x = self.activation_fxn(layer(x))
        return x
