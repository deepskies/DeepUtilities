import torch

from google.cloud import storage
import dsutils as ds


def experiment_path(dataset_name):
     return ds.__path__[0] + '/../experiments/' + dataset_name + '/'

def gcs_datasets():
    # todo write actual function
    gcs_datasets = []
    return gcs_datasets

def non_gcs_datasets():
    other_datasets = ['mnist', 'cifar']
    return other_datasets

def all_datasets():
    gcs = gcs_datasets()
    other = non_gcs_datasets()
    return gcs, other

def get_dataset(dataset_name='mnist'):
    # building for mnist rn
    dataloader, experiment_path = fetch(dataset_name)
    gcs_sets, other_sets = datasets_list() # ['mnist']
    all_sets = gcs_sets + other_sets

    if dataset_name in other_sets:
        loaders = non_gcs_fetch(dataset_name)
    elif dataset_name in other_sets:
        # todo
        loaders = gcs_fetch(dataset_name)
    else:
        raise Exception("Dataset not found error")

    return loaders

def non_gcs_fetch(dataset_name):
    sets = non_gcs_datasets()
    # for set in sets:
    #     if dataset_name == set:

    if dataset == 'mnist':
        train_loader, test_loader = mnist_loaders()
    elif dataset == 'cifar':
        train_loader, test_loader = cifar_loaders()
    return train_loader, test_loader

def gcs_fetch(dataset_name):
    data_path = experiment_path(dataset_name) + 'data/'
    exists = os.path.exists(data_path)
    if not exists:
        storage_client = storage.Client()

        bucket_name = dataset_name
        source_blob_name = dataset_name + '.h5'
        destination_file_name = data
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)
        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))

# to deprecate
def fetch(dataset):
    gcs_sets, other_sets = datasets_list() # ['mnist']

    # assuming that type(dataset) == string rn
    if isinstance(dataset, str):
        # this is a hack to get mnist working
        if dataset in other_sets:
           non_gcs_fetch(dataset)
        else:
            # todo: google cloud h5 retrieve
            dataset = dataset
            storage_client = storage.Client()
            dataloader = dataloader_from_gcs(dataset, self.storage_client)

    elif isinstance(dataset, list):
        # todo: build dataloader
        pass
    elif isinstance(dataset, torch.utils.data.Dataset):
        dataloader = dataset
    elif isinstance(dataset, tf.data.Dataset):
        # unsure abt this
        tf_data = True
        dataloader = dataset
    else:
        raise Exception("dataset input type not supported")

    return train_loader, test_loader

# tensorflow-io?
def build_dataloader(dataset):
    # [(X1, Y1), ... , (Xn, Yn)]

    pass


    # todo https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
def mnist_loaders(batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(experiment_path('mnist'), train=True, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),

        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(experiment_path('mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def cifar_loaders():
    # todo
    pass
