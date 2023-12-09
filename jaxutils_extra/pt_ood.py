import numpy as np
import os
import wget
import tarfile

import torch
from torchvision import datasets, transforms

from jaxutils_extra.pt_image import MoveChannelDim, ToNumpy, get_image_dataset
from jaxutils.data.pt_preprocess import DatafeedImage, NumpyLoader
from pathlib import Path

def load_rotated_dataset(
    dname: str, 
    angle: float, 
    data_dir: str, 
    batch_size: int = 256, 
    num_workers: int = 4, 
    n_data = None, 
    subset_idx: int = -1):
    assert dname in ['MNIST', 'Fashion', 'SVHN', 'CIFAR10', 'CIFAR100', 'EMNIST']

    transform_dict = {
        'MNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ToNumpy(),
            MoveChannelDim(),
        ]),
        'Fashion': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ]),
        'SVHN': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]),
        'CIFAR10': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        'CIFAR100': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
        'EMNIST': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ]),
        'SmallImagenet': None,
        'Imagenet': None,
    }

    dataset_dict = {
        'MNIST': datasets.MNIST,
        'Fashion': datasets.FashionMNIST,
        'SVHN': datasets.SVHN,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'EMNIST': datasets.EMNIST,
        'SmallImagenet': None,
        'Imagenet': None,
    }

    dset_kwargs = {
        'root': data_dir,
        'train': False,
        'download': True,
        'transform': transform_dict[dname]
    }

    if dname == 'SVHN':
        del dset_kwargs['train']
        dset_kwargs['split'] = 'test'

    elif dname == 'EMNIST':
        dset_kwargs['split'] = 'balanced'

    source_dset = dataset_dict[dname](**dset_kwargs)

    # subsample source dataset if desired (either randomly or by subset index)
    if n_data is not None:
        if subset_idx == -1:
            np.random.seed(0)
            perm = np.random.permutation(source_dset.data.shape[0])
            subset = perm[:n_data]
        else:
            subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))
        source_dset.data = source_dset.data[subset]
        source_dset.targets = source_dset.targets[subset]

    source_loader = NumpyLoader(
        source_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return source_loader, source_dset


def load_corrupted_dataset(
    dname: str,
    severity: int,
    corruption_type: int,
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    n_data = None,
    subset_idx: int = -1):
    assert dname in ['CIFAR10']
    data_dir = Path(data_dir)

    transform_dict = {
        'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToNumpy(),
            MoveChannelDim(),
        ]),
        'CIFAR100': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            ToNumpy(),
            MoveChannelDim(),
        ]),
        'Imagenet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ToNumpy(),
            MoveChannelDim(),
        ]),
    }

    if dname == 'CIFAR100':
        if severity == 0:
            # Return the original test set
            _, dataset, _ = get_image_dataset(
                dname, data_dir, flatten_img=False, val_percent=0.,
                random_seed=0, perform_augmentations=False)
        elif severity in [1, 2, 3 ,4, 5]:
            x_file = data_dir / ('CIFAR-100-C/CIFAR100_c%d.npy' % severity)
            np_x = np.load(x_file)
            y_file = data_dir / 'CIFAR-100-C/CIFAR100_c_labels.npy'
            np_y = np.load(y_file).astype(np.int64)
            dataset = DatafeedImage(np_x, np_y, transform_dict[dname])
    
    elif dname == "CIFAR10":
        if severity == 0:
            # Return the original test set
            _, dataset, _ = get_image_dataset(
                dname, data_dir, flatten_img=False, val_percent=0.,
                random_seed=0, perform_augmentations=False)
        elif severity in [1, 2, 3 ,4, 5]:
            path =  data_dir / 'CIFAR-10-C'
            if not os.path.exists(path):
                print("Corrupted Data doesn't exist. Downloading...")
                wget.download("https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1", out=str(data_dir))
                with tarfile.open(str(data_dir / 'CIFAR-10-C.tar'), 'r') as tar:
                    tar.extractall(path)
                print("Corrupted data prepared")

            corrupted_data_files = os.listdir(data_dir / 'CIFAR-10-C/CIFAR-10-C')
            corrupted_data_files.remove('labels.npy')

            if 'README.txt' in corrupted_data_files:
                corrupted_data_files.remove('README.txt')

            corrupted_data_file = corrupted_data_files[corruption_type]
            map = np.lib.format.open_memmap(str(data_dir / 'CIFAR-10-C/CIFAR-10-C')+ "/" + corrupted_data_file, mode='r+')
            subset = map[(severity-1)*10000:(severity)*10000]

            y_file = data_dir / 'CIFAR-10-C/CIFAR-10-C/labels.npy'
            np_y = np.load(y_file).astype(np.int64)
            dataset = DatafeedImage(subset, np_y, transform_dict[dname])
    
    loader = NumpyLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers)

    return loader, dataset