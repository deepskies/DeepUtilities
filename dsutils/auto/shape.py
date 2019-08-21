import math

import torch
import torch.nn as nn

def log_dims(input_dim, output_dim, factor=2):
    '''
    mnist mlp:
    [784, 397, 203, 106, 58, 34, 22, 16, 13, 11, 10]
    '''
    dims = []
    dim = input_dim
    delta = input_dim - output_dim

    while dim > output_dim:
        dims.append(dim)
        dim = (delta // factor) + output_dim
        delta = dim - output_dim

    dims.append(output_dim)
    print(dims)
    return dims

def mlp_layers(layer_dims, verbose=False):
    layers = []
    num_layers = len(layer_dims)
    for i in range(num_layers):
        if i == num_layers - 1:
            break
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
    if verbose:
        print(layers)
    return layers


def get_ksizes(delta):
    ksizes = []

    rough_layer_count = math.log2(delta)
    num_layers = round(rough_layer_count)

    for i in range(num_layers):
        pow = num_layers - i
        ksize = (2 ** (pow - 1)) + 1
        print(ksize)
        ksizes.append(ksize)

    return ksizes


def conv1d_layers(x, ksizes, max_channels=64):
    in_dim = x.dim()
    prev = x

    prev_channels = 1
    channels = 1

    layers = []
    dims = []

    for i, ksize in enumerate(ksizes):
        if channels < max_channels:
            channels = prev_channels * 2
        # if in_dim == 3:
        layer = nn.Conv1d(prev_channels, channels, kernel_size=ksize)
        # if in_dim == 4:
        #     layer = nn.Conv2d(prev_channels, channels, kernel_size=ksize)
        prev_channels = channels
        prev = layer(prev)
        dims.append(prev.shape)
        layers.append(layer)
    numel_wo_batch = dims[-1][1] * dims[-1][2]
    pool_ksize = math.floor(math.log2(numel_wo_batch))
    pool_layer = nn.MaxPool1d(pool_ksize)
    pool_out = pool_layer(prev.view(batch_size, 1, -1)).shape
    linear_input_dim = pool_out[-1]
    layers.append(pool_layer)
    layers.append(nn.Linear(linear_input_dim, self.out_dim))
    layers.append(nn.Softmax(dim=-1))
    return layers
