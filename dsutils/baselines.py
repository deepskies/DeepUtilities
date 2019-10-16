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
import dsutils.macros as m
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

        dataset = config['dataset']
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.dir = "./experiments/{}_{}_{}".format(dataset, timestamp, config["id"])

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save config file in experiment directory
        # disabled for now
        # with open(self.dir + '/config.json', 'w') as config_file:
        #     json.dump(config, config_file)

        self.model = None

        self.model_configs = config['model_configs']
        self.training_config = config['training_config']

        self.print_freq = self.training_config['print_freq']
        self.logs = {
                        'actual' : [],
                        'predicted' : [],
                    }

        # for now returns pytorch train_loader, test_loader
        if dataset in m.datasets:
            self.train_loader, self.test_loader = ds.get_dataset(dataset, config)
            self.in_dim, self.out_dim = 784, 10
        else:
            pass

        self.device = config['device'] # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.run()

    def run(self):
        for i, model_config in enumerate(self.model_configs):

            model = model_config['type']
            lr = model_config['lr']
            epochs = model_config['epochs']
            classify = model_config['classify']

            if classify:
                loss_fxn = nn.CrossEntropyLoss()
            else:
                loss_fxn = nn.MSELoss()

            if model == 'automlp':
                factor = model_config['factor']
                self.model = mlp.MLP(self.in_dim, self.out_dim, config=model_config).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            if model == 'vae':
                break

            for epoch_id in range(epochs):

                run_config = {
                    'model': self.model,
                    'device': self.device,
                    'loss_fxn': loss_fxn,
                    'epoch_id': epoch_id,
                    'print_freq': self.print_freq,
                    'logs': self.logs
                }

                run_epoch.train(self.train_loader, self.optimizer, run_config)
                run_epoch.test(self.test_loader, run_config)

        torch.save(self.model, self.dir + "/model.pt")


        self.diags = diag.Diagnostics(self.dir, self.logs['predicted'], self.logs['actual'])

        self.diags.plot_cm()



def baselines(dataset):
    base_class = Baselines(dataset)
