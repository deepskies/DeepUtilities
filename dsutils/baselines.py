import os.path

import h5py

import torch
import torch.nn as nn
import torch.utils.data # import DataLoader, Dataset
import torch.optim as optim

from google.cloud import storage

import dsutils as ds
import dsutils.auto.mlp as mlp
import dsutils.auto.vae as vae
import dsutils.auto.run_epoch as run_epoch

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
            * string in ['mlp', 'vae', 'conv1d', # todo - 'conv2d', 'node', 'gan', 'lstm', 'rnn']

    '''
    def __init__(self, dataset_name='mnist', model='mlp', epochs=5, lr=1e-3, classify=True):
        # todo: logging for diagnostics
        self.epochs = epochs
        self.logs = []
        self.path = ds.data.experiment_path(dataset_name)
        self.model_type = model

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = ds.get_dataset(dataset_name)
        self.in_dim, self.out_dim = ds.data.get_dims_from_loader(self.train_loader)

        if model == 'mlp':
            self.model = mlp.MLP(self.in_dim, self.out_dim)
        elif model == 'vae':
            self.model = vae.VAE(self.in_dim, self.out_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

        if classify:
            self.criterion = nn.CrossEntropyLoss()

        self.run()

    def run(self):
        for epoch_id in range(self.epochs):
            run_epoch.train(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch_id, 500)
            run_epoch.test(self.model, self.device, self.test_loader, self.criterion)


def baselines(dataset):
    base_class = Baselines(dataset)
