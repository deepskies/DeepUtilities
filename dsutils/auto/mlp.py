import torch
import torch.nn as nn
import torch.nn.functional as F

import dsutils as ds
from dsutils.auto import shape
from dsutils.auto import utils


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, config=None):
        super(MLP, self).__init__()
        self.config = config
        if not config:
            self.config = {
                    'type': 'mlp',
                    'factor': 10,
                    'classify': True,
                    'activation_fxn': 'tanh',
                    'lr': 1e-3,
                    }
        self.classify = self.config['classify']
        factor = self.config['factor']
        self.activation_fxn = utils.activations[self.config['activation_fxn']]

        layer_dimensions = shape.log_dims(input_dim, output_dim, factor=factor)

        # dense layers from tuples
        self.layers = shape.get_layers(layer_dimensions)
        self.num_layers = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        out = general_forward(x, self.model, self.classify, self.activation_fxn)
        return out

def general_forward(x, model, classify=True, activation_fxn=nn.Tanh()):
    num_layers = len(model)
    for i, layer in enumerate(model):
        if i == num_layers - 1:
            if classify:
                x = F.softmax(layer(x), dim=1)
            else:
                x = layer(x)
        else:
            x = activation_fxn(layer(x))
    return x
