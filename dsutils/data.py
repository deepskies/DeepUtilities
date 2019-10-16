import os
import torch
from torchvision import datasets, transforms
from google.cloud import storage
import dsutils as ds


def experiment_path(dataset_name):
     return ds.__path__[0] + '/../experiments/' + dataset_name + '/'


def get_dataset(dataset_name='mnist', config=None):

    if config:
        train_config = config['training_config']
        train_batch_size = train_config['batch_size']
        test_batch_size = train_config['test_batch_size']
    else:
        train_batch_size = 1
        test_batch_size = 1

    if dataset_name == 'mnist':
        train_loader, test_loader = mnist_loaders(train_batch_size, test_batch_size)
    elif dataset_name == 'cifar':
        train_loader, test_loader = cifar_loaders()
    return train_loader, test_loader


def mnist_loaders(batch_size, test_batch_size):

    data_path = experiment_path('minst') + 'data/'

    # download if the path doesn't exist
    dl = not os.path.exists(data_path)

    train_set = datasets.MNIST(data_path, train=True, download=dl, transform=transforms.ToTensor())
    test_set = datasets.MNIST(data_path, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader
