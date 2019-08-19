import os.path

import h5py

import torch
import torch.utils.data # import DataLoader, Dataset

from google.cloud import storage

import dsutils as ds
import dsutils.auto

class Baselines:
    '''
    Baselines should be an easy way to train baseline models on any data as well
        as train

    dataset - * list of tuples [(X1, Y1) , ... , (Xn, Yn)]
              * 'mnist' - a string referencing a dataset
                 if string, dataset downloaded to dsutils/../experiment_name/exp_iter

              * tf dataset
              * pytorch dataloader


    model - * PyTorch model
            * Tensorflow model
            * string in ['linear', 'vae', 'conv1d', # todo - 'conv2d', 'node', 'gan', 'lstm', 'rnn']

    '''
    def __init__(dataset_name='mnist', model=None):
        # todo: logging for diagnostics
        self.logs = []
        self.path = ds.data.experiment_path(dataset_name)

        # for now returns pytorch train_loader, test_loader
        train_loader, test_loader = ds.get_dataset(dataset)




def baselines(dataset):
    base_class = Baselines(dataset)
