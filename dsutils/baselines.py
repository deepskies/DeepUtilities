import os.path

import h5py

import torch
import torch.utils.data # import DataLoader, Dataset
import torch.optim as optim

from google.cloud import storage

import dsutils as ds

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
    def __init__(dataset_name='mnist', model='mlp', epochs=5, lr=1e-2, classify=True):
        # todo: logging for diagnostics
        self.epochs = epochs
        self.logs = []
        self.path = ds.data.experiment_path(dataset_name)

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = ds.get_dataset(dataset_name)
        self.in_dim, self.out_dim = ds.data.get_dims_from_loader(self.train_loader)
        self.model = ds.auto.mlp.MLP(self.in_dim, self.out_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

        self.run()

    def run(self):
        for i in range(self.epochs):
            ds.auto.run_epoch.train(self.model, self.device, self.train_loader, i)
            ds.auto.run_epoch.test(self.model, self.device, self.test_loader)


def baselines(dataset):
    base_class = Baselines(dataset)
