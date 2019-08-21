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

import dsutils.diagnostics as diag

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
    def __init__(self, dataset_name='mnist', model='mlp', epochs=1, lr=1e-3, classify=True, log_interval=500, factor=5):
        # todo: logging for diagnostics
        self.epochs = epochs
        self.logs = {
                        'actual' : [],
                        'predicted' : [],

                        'train_acc' : [],
                        'test_acc' : []
                    }
        self.path = ds.data.experiment_path(dataset_name)
        self.model_type = model
        self.log_interval = log_interval



        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = ds.get_dataset(dataset_name)
        self.in_dim, self.out_dim = ds.data.get_dims_from_loader(self.train_loader)

        if model == 'mlp':
            self.model = mlp.MLP(self.in_dim, self.out_dim, factor=factor)
        elif model == 'vae':
            self.model = vae.VAE(self.in_dim, self.out_dim)
            print('not working yet, needs custom train')

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

        if classify:
            self.criterion = nn.CrossEntropyLoss()

        self.run()

    def run(self):
        if self.model_type == 'mlp': # or others
            for epoch_id in range(self.epochs):
                run_epoch.train(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch_id, self.log_interval, self.logs)
                run_epoch.test(self.model, self.device, self.test_loader, self.criterion, self.logs)
        elif self.model_type == 'vae':
            run_epoch.vae_train(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch_id, self.log_interval)

        self.diags = diag.Diagnostics(self.path, self.logs['actual'], self.logs['predicted'])

        self.diags.plot_cm()


def baselines(dataset):
    base_class = Baselines(dataset)
