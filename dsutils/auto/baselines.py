import os.path

import h5py

import torch
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
from google.cloud import storage

import dsutils
import dsutils.auto.utils

class Baselines:
    '''
    Baselines should be an easy way to train baseline models on any data as well
        as train

    dataset - * list of tuples [(X1, Y1) , ... , (Xn, Yn)]
              * 'mnist' - a string referencing a dataset
                 if string, dataset downloaded to dsutils/../data (next to setup.py)

              * tf dataset
              * pytorch dataloader


    model - * PyTorch model
            * Tensorflow model
            * string in ['linear', 'conv1d', # todo - 'conv2d', 'node', 'vae', 'gan', 'lstm', 'rnn']

    '''
    def __init__(dataset=None, model=None):
        if isinstance(dataset, str):
            # todo: google cloud h5 retrieve
            self.dataset = dataset
            self.storage_client = storage.Client()
            self.dataloader = dataloader_from_gcs(dataset, self.storage_client)

        elif isinstance(dataset, list):
            # todo: build dataloader
            pass
        elif isinstance(dataset, Dataset):
            self.dataloader = dataset
            pass
        elif isinstance(dataset, tf.data.Dataset)
            self.tf_data = True
            self.dataloader = dataset
        else:
            raise Exception("dataset input type not supported")
