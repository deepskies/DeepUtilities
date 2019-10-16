import os.path
import json
import time

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
    def __init__(self, path_to_config='./config/mnist.json'):
        # dataset='mnist', model='mlp', epochs=1, lr=1e-3, classify=True, log_interval=500, factor=5
        # todo: logging for diagnostics
        with open(path_to_config) as config_file:
            config = json.load(config_file)

        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.dir = "./experiments/img_results_{}_{}".format(timestamp, config["id"])
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save config file in experiment directory
        # disabled for now
        # with open(self.dir + '/config.json', 'w') as config_file:
        #     json.dump(config, config_file)

        self.model = None
        dataset = config['dataset']
        self.path = ds.data.experiment_path(dataset)

        self.model_configs = config['model_configs']
        self.training_config = config['training_config']
        self.lr = self.training_config['lr']
        self.epochs = self.training_config['epochs']
        self.log_interval = self.training_config['print_freq']
        self.classify = self.training_config['classify']
        self.logs = {
                        'actual' : [],
                        'predicted' : [],

                        'train_acc' : [],
                        'test_acc' : []
                    }

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = ds.get_dataset(dataset)
        self.in_dim, self.out_dim = ds.data.get_dims_from_loader(self.train_loader)

        self.device = config['device'] # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.classify:
            self.criterion = nn.CrossEntropyLoss()

        self.run()

    def run(self):
        for i, model_config in enumerate(self.model_configs):
            model = model_config['type']
            if model == 'automlp':
                factor = model_config['factor']
                self.model = mlp.MLP(self.in_dim, self.out_dim, factor=factor).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

                for epoch_id in range(self.epochs):
                    run_epoch.train(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch_id, self.log_interval, self.logs)
                    run_epoch.test(self.model, self.device, self.test_loader, self.criterion, self.logs)

                torch.save(self.model, self.dir + "model.pt")

        self.diags = diag.Diagnostics(self.path, self.logs['actual'], self.logs['predicted'])

        self.diags.plot_cm()



def baselines(dataset):
    base_class = Baselines(dataset)
