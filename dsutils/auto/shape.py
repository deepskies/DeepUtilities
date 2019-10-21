import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_dims(input_dim=784, output_dim=10, factor=2, verbose=False):
    '''
    mnist mlp w factor 2:
    [784, 397, 203, 106, 58, 34, 22, 16, 13, 11, 10]
    '''
    if input_dim < output_dim:
        tmp = input_dim
        input_dim = output_dim
        output_dim = tmp
        reverse = True
    else:
        reverse = False

    dims = []
    delta = input_dim - output_dim

    while input_dim > output_dim:
        dims.append(input_dim)
        input_dim = (delta // factor) + output_dim
        delta = input_dim - output_dim

    dims.append(output_dim)

    if verbose:
        print(dims)

    if reverse:
        dims = dims[::-1]
    return dims


def get_layers(layer_dims, layer_type=nn.Linear, verbose=False):
    l = lambda x, y: layer_type(x, y)
    layers = list(map(l, layer_dims, layer_dims[1:]))
    if verbose:
        print(layers)
    return layers


def mlp_vae(layer_dims):
    '''
    takes in list of layer dims, returns encoder and decoder
    possibly redundant code
    '''
    enc = nn.ModuleList(get_layers(layer_dims))
    dec = nn.ModuleList(get_layers(layer_dims[::-1]))
    return enc, dec
