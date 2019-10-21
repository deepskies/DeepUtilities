import argparse
import sys
import os.path
import json
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data # import DataLoader, Dataset
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt

from google.cloud import storage

import dsutils as ds
import dsutils.auto.mlp as mlp
import dsutils.auto.vae as vae

import dsutils.train as train
import dsutils.test as test
import dsutils.data

import dsutils.diagnostics as diag

class Baselines:
    '''
    path_to_config is required with the following required keys:
    {
        "id": "quick_test",
        "dataset": "mnist",
    }
    '''
    def __init__(self, config_path='./config/mnist.json'):
       
        with open(config_path, 'r') as config:
            self.config = json.load(config)

        self.dir = init_experiment(self.config)   
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # ignores config

        self.model = None
        self.run_configs = []
        self.model_configs = self.config['model_configs']
        self.training_config = self.config['training_config']

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = dsutils.data.get_dataset(self.config['dataset'], self.config)

        self.original_elt = self.train_loader.dataset[0][0]
        self.original_shape = self.original_elt.shape
        self.in_dim = self.original_elt.numel()
        self.out_dim = 10
        self.actuals = []
        self.predicted = []
        self.decoded = []
        self.run()

    def run(self):
        self.prep_models()
        print(self.run_configs)

        for i, run_config in enumerate(self.run_configs):
            self.run_model(i, run_config)

        return self.predicted, self.actuals, self.decoded

    def read_config(self):
        # major todo
        pass

    def prep_models(self):
        # builds and configures the model for train and test, builds a
        self.run_configs = []
        for model_config in self.model_configs:

            loss_fxn = nn.CrossEntropyLoss() if model_config['classify'] else nn.MSELoss()
                
            if model_config['type'] == 'automlp':
                model = mlp.MLP(self.in_dim, self.out_dim,
                                config=model_config).to(self.device)
                optimizer = optim.Adam(
                    model.parameters(), lr=model_config['lr'])

            elif model_config['type'] == 'vae':
                model = vae.VAE(self.in_dim, self.out_dim, config=model_config).to(self.device)
                optimizer = optim.Adam(
                    model.parameters(), lr=model_config['lr'])

            run_config = {
                'model': model,
                'optimizer': optimizer,
                'device': self.device,
                'loss_fxn': loss_fxn,
                'epochs': model_config['epochs'],
                'print_freq': self.training_config['print_freq'],
            }

            self.run_configs.append(run_config)


    def run_model(self, i, run_config):
        model_type = self.model_configs[i]['type']
        epochs = run_config['epochs']
        self.actuals = []
        self.decoded = []
        self.predicted = []

        for epoch in range(epochs):
            if model_type == 'vae' or model_type == 'gan':  # if unsupervised
                train.vae_train(self.train_loader, run_config, epoch)
                preds, actuals, decoded = test.vae_test(self.test_loader, run_config, epoch)
                self.predicted.append(preds)  # arbitrary bins
                self.actuals.append(actuals)
                self.decoded.append(decoded)
            else:
                train.single_epoch(self.train_loader, run_config, epoch)
                preds, actuals = test.test(self.test_loader, run_config, epoch)
                self.predicted.append(preds)
                self.actuals.append(actuals)

            diag.plot_cm(preds, actuals, save_path=self.dir +
                         'plots/', show=False, epoch=epoch)
        torch.save(run_config['model'], self.dir + "model.pt")  # idk if this is updating when calling train

def init_experiment(config):
    dataset = config['dataset']
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    dir = "./experiments/{}_{}/".format(
        dataset, timestamp)

    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir + '/plots/')
    return dir

def read_json(path):
    with open(path) as config_file:
        d = json.load(config_file)
    return d 

if __name__ == '__main__':
    b = Baselines('./config/cifar10.json')
