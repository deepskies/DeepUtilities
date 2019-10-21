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

import dsutils.macros as m

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
            os.makedirs(self.dir +'/plots/')


        # Save config file in experiment directory
        # disabled for now
        # with open(self.dir + '/config.json', 'w') as config_file:
        #     json.dump(config, config_file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.run_configs = []  # to build

        self.model_configs = self.config['model_configs']
        print(f'len model configs: {len(self.model_configs)}')
        self.training_config = self.config['training_config']

        self.print_freq = self.training_config['print_freq']

        # for now returns pytorch train_loader, test_loader
        self.train_loader, self.test_loader = dsutils.data.get_dataset(dataset, self.config)

        self.original_elt = self.train_loader.dataset[0][0]
        self.original_shape = self.original_elt.shape
        self.in_dim = self.original_elt.numel()
        self.out_dim = 10
        # self.device = self.config['device'] # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actuals = []
        self.predicted = []
        self.decoded = []
        self.run()

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
                'print_freq': self.print_freq,
            }

            self.run_configs.append(run_config)

    def run(self):
        self.prep_models()
        print(self.run_configs)

        for i, run_config in enumerate(self.run_configs):
            self.run_model(i, run_config)

        # for preds, actual in zip(self.predicted, self.actuals):
        #     diag.plot_cm(preds, actual, save_path=self.dir + 'plots/', show=False, epoch=i)

        return self.predicted, self.actuals, self.decoded

    def run_model(self, i, run_config):
        model_type = self.model_configs[i]['type']
        epochs = run_config['epochs']
        self.actuals = []
        self.decoded = []
        self.predicted = []

        for epoch in range(epochs):
            if model_type == 'vae' or model_type == 'gan':  # if unsupervised
                train.vae_train(self.train_loader, run_config, epoch)
                bins, actuals, decoded = test.vae_test(self.test_loader, run_config, epoch)
                self.predicted.append(bins)
                self.actuals.append(actuals)
                self.decoded.append(decoded)
            else:
                train.single_epoch(self.train_loader, run_config, epoch)
                preds, actuals = test.test(self.test_loader, run_config, epoch)
                self.predicted.append(preds)
                self.actuals.append(actuals)
            diag.plot_cm(preds, actual, save_path=self.dir +
                         'plots/', show=False, epoch=epoch)
        torch.save(run_config['model'], self.dir + "model.pt")  # idk if this is updating when calling train


# todo deprecate
def random_train_images(train_loader, classes, num=10, viewas=None, fn=None):
    dataiter = iter(train_loader)
    imgs = next(dataiter) 
    if viewas:
        images = [torch.tensor(image).view(viewas) for image in imgs]

    if not fn:
        fn = '../experiments/FIX'

    for i in range(10):
        torchvision.utils.save_image(images[i], fn + f'_{i}.jpeg', nrow=8, padding=2,
                                normalize=False, range=None, scale_each=False, pad_value=0)
        

def read_json(path):
    with open(path) as config_file:
        d = json.load(config_file)
    return d  # type dict

#
# def baselines(dataset):
#     base_class = Baselines(dataset)


if __name__ == '__main__':
    b = Baselines('cifar10.json')
