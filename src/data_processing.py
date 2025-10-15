import os
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset_name: str, download_path: str):
    """
    Downloads and extracts a dataset from Kaggle.

    Args:
        dataset_name (str): Name of the Kaggle dataset (e.g. 'fatemehmehrparvar/lung-disease')
        download_path (str): Path to download and extract dataset
    """
    os.makedirs(download_path, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"✅ Dataset downloaded and extracted to {download_path}")

def load_dataset(data_dir, img_height=256, img_width=256, batch_size=32):
    """
    Loads dataset from a given directory and creates TensorFlow datasets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_ds, val_ds, test_ds
