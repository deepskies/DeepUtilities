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

import dsutils.train as train
import dsutils.test as test

import dsutils.diagnostics as diag

class Baselines:
    '''
    path_to_config is required with the following required keys:
    {
        "id": "quick_test",
        "dataset": "mnist",
    }

    roadmap:
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
        self.config = read_json(path_to_config)
        dataset = self.config['dataset']
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.dir = "./experiments/{}_{}_{}/".format(dataset, timestamp, self.config["id"])

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save config file in experiment directory
        # disabled for now
        # with open(self.dir + '/config.json', 'w') as config_file:
        #     json.dump(config, config_file)

        self.model = None
        self.run_configs = []  # to build

        self.model_configs = self.config['model_configs']
        print(f'len model configs: {len(self.model_configs)}')
        self.training_config = self.config['training_config']

        self.print_freq = self.training_config['print_freq']

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = get_dataloaders(dataset, self.config)
        self.in_dim = self.train_loader.dataset[0][0].numel()
        self.out_dim = 10
        self.device = self.config['device'] # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.run()



    def read_config(self):
        # major todo
        pass


    def prep_models(self):
        # builds and configures the model for train and test, builds a
        self.run_configs = []
        for i, model_config in enumerate(self.model_configs):

            model_type = model_config['type']
            lr = model_config['lr']
            epochs = model_config['epochs']
            classify = model_config['classify']
            epochs = model_config['epochs']

            if classify:
                loss_fxn = nn.CrossEntropyLoss()
            else:
                loss_fxn = nn.MSELoss()

            if model_type == 'automlp':
                factor = model_config['factor']
                model = mlp.MLP(self.in_dim, self.out_dim, config=model_config).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

            if model_type == 'vae':
                break

            run_config = {
                'model': model,
                'optimizer': optimizer,
                'device': self.device,
                'loss_fxn': loss_fxn,
                'epochs': epochs,
                'print_freq': self.print_freq,
            }

            self.run_configs.append(run_config)

    def run(self):
        self.prep_models()
        print(self.run_configs)
        run_log = {}
        for i, run_config in enumerate(self.run_configs):
            logs_for_all_epochs = []

            for epoch in range(run_config['epochs']):
                train.single_epoch(self.train_loader, run_config, epoch)
                epoch_log = test.test(self.test_loader, run_config, epoch)
                logs_for_all_epochs.append(epoch_log)

            run_log[i] = logs_for_all_epochs

            torch.save(run_config['model'], self.dir + "model.pt")  # idk if this is updating when calling train

            with open(self.dir + self.model_configs[i]['type'] + '_logs.json', 'w') as json_file:
                json.dump(run_log, json_file)

        # self.diags = diag.Diagnostics()
        n = len(self.run_configs)
        diag.plot_cm(run_log[n-1][0]['predicted'], run_log[n-1][0]['actual'], config=self.config, save_path=self.dir)

def read_json(path):
    with open(path) as config_file:
        d = json.load(config_file)
    return d  # type dict

def get_dataloaders(dataset, config):
    # if its a string assume its a path and read it
    if isinstance(config, str):
        config = read_json(config)

    if dataset in m.datasets:
        train_loader, test_loader = ds.get_dataset(dataset, config)
        return train_loader, test_loader
    else:
        print(f'cant find {dataset}')
        return None
#
# def baselines(dataset):
#     base_class = Baselines(dataset)
