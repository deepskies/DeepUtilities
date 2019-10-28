import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from google.cloud import storage
import dsutils as ds


def experiment_path(dataset_name):
     return ds.__path__[0] + '/../experiments/' + dataset_name + '/'


def get_dataset(dataset_name='mnist', config=None):
    # bs is train_batch_size and tbs is test_batch_size
    if config:
        train_config = config['training_config']
        bs, tbs = train_config.get('batch_size'), train_config.get('test_batch_size')

    if not bs:
        bs = 1
    if not tbs:
        tbs = 1


    if dataset_name == 'mnist':
        train_loader, test_loader = mnist_loaders(bs, tbs)
    elif dataset_name == 'cifar10':
        train_loader, test_loader = cifar_loaders(bs, tbs)
    elif dataset_name == 'FashionMNIST':
        train_loader, test_loader = fashion_mnist_loaders(bs, tbs)
    return train_loader, test_loader


def mnist_loaders(batch_size, test_batch_size):
    transform = transforms.ToTensor()   

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

def cifar_loaders(train_batch_size, test_batch_size):
    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def fashion_mnist_loaders(train_batch_size, test_batch_size):
    transform = transforms.ToTensor()

    trainset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
