from google.cloud import storage
import dsutils

# tensorflow-io?

def build_dataloader(dataset):
    pass

def dataloader_from_gcs(dataset):
    path = dsutils.__path__[0]
    data_path = path + '/../data/' + dataset
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


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
