import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from google.cloud import storage
import dsutils as ds


def experiment_path(dataset_name):
     return ds.__path__[0] + '/../experiments/' + dataset_name + '/'


def get_dataset(dataset_name='mnist', config=None):

    if config:
        train_config = config['training_config']
        batch_sizes = []
        for item in ['batch_size', 'test_batch_size']:
            try:
                batch_sizes.append(train_config[item])
            except KeyError:
                batch_sizes.append(1)
        train_batch_size, test_batch_size = batch_sizes
    else:
        train_batch_size = 1
        test_batch_size = 1

    if dataset_name == 'mnist':
        train_loader, test_loader = mnist_loaders(train_batch_size, test_batch_size)
    elif dataset_name == 'cifar10':
        train_loader, test_loader = cifar_loaders(train_batch_size, test_batch_size)
    return train_loader, test_loader


def mnist_loaders(batch_size, test_batch_size):

    data_path = experiment_path('minst') + 'data/'

    # download if the path doesn't exist
    dl = not os.path.exists(data_path)

    train_set = datasets.MNIST(data_path, train=True, download=dl, transform=transforms.ToTensor())
    test_set = datasets.MNIST(data_path, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

def cifar_loaders(train_batch_size, test_batch_size):
    data_path = experiment_path('minst') + 'data/'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader
