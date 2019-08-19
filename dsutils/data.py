from google.cloud import storage
import dsutils as ds


def gcs_datasets():
    # todo write actual function
    gcs_datasets = [] # f()  # todo, returns string list
    return gcs_datasets

def non_gcs_datasets():
    other_datasets = ['mnist', 'cifar']
    return other_datasets

def all_datasets():
    gcs = gcs_datasets()
    other = non_gcs_datasets()
    return gcs, other

def get_dataset(dataset_name):
    # building for mnist rn
    # todo write function to gather list of datasets

    dataloader, experiment_path = fetch(dataset_name)

    if dataset_name in datasets:
    else:
        raise Exception("Dataset not found error")



def fetch(dataset):
    gcs_sets, other_sets = datasets_list() # ['mnist']

    # assuming that type(dataset) == string rn
    if isinstance(dataset, str):
        # this is a hack to get mnist working
        if dataset in other_sets:
            if dataset == 'mnist':
             train_loader, test_loader = mnist_loaders()
            pass
        else:
            # todo: google cloud h5 retrieve
            self.dataset = dataset
            self.storage_client = storage.Client()
            self.dataloader = dataloader_from_gcs(dataset, self.storage_client)

    elif isinstance(dataset, list):
        # todo: build dataloader
        pass
    elif isinstance(dataset, torch.utils.data.Dataset):
        self.dataloader = dataset
    elif isinstance(dataset, tf.data.Dataset)
        self.tf_data = True
        self.dataloader = dataset
    else:
        raise Exception("dataset input type not supported")


# tensorflow-io?
def build_dataloader(dataset):
    # [(X1, Y1), ... , (Xn, Yn)]

    pass

def dataloader_from_gcs(dataset_name):
    data_path = experiment_path + 'data/'
    exists = os.path.exists('datapath')
    if not exists:
        storage_client = storage.Client()

        bucket_name =
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)
        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))

    # todo https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
def mnist_loaders(batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(experiment_path(), train=True, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),

        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
        return train_loader, test_loader

def experiment_path(dataset_name):
     return ds.__path__[0] + '/../experiments/' + dataset_name + '/'
