import numpy as np
import os
import wget
import tarfile

import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
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

    transform_dict = {
        'MNIST': transforms.Compose([
            transforms.ToTensor(),
            #ToNumpy(),
            #MoveChannelDim(),
        ]),
        'FMNIST': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }

    dataset_dict = {
        'MNIST': datasets.MNIST,
        'FMNIST': datasets.FashionMNIST,
    }

    dset_kwargs = {
        'root': data_dir,
        'train': False,
        'download': True,
        'transform': transform_dict[dname]
    }
    source_dset = dataset_dict[dname](**dset_kwargs)


    if angle > 0:
        source_dset.data = rotate(source_dset.data, angle)
    
    dset = torch.utils.data.TensorDataset(source_dset.data/255., source_dset.targets)

    source_loader = NumpyLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return source_loader, dset


def load_ood_dataset(
    dname: str, 
    data_dir: str, 
    batch_size: int = 256, 
    num_workers: int = 4, 
    n_data = None, 
    subset_idx: int = -1):

    transform_dict = {
        'MNIST': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'FMNIST': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }

    dataset_dict = {
        'MNIST': datasets.MNIST,
        'FMNIST': datasets.FashionMNIST,
    }

    dset_kwargs = {
        'root': data_dir,
        'train': False,
        'download': True,
        'transform': transform_dict[dname]
    }

    if dname == "MNIST":
        dname_other = "FMNIST"
    else:
        dname_other = "MNIST"

    source_dset = dataset_dict[dname](**dset_kwargs)
    source_dset_data = source_dset.data/255.
    source_dset_targets = torch.zeros_like(source_dset.targets)

    other_dset = dataset_dict[dname_other](**dset_kwargs)
    other_dset_data = other_dset.data/255.
    other_dset_targets = torch.ones_like(other_dset.targets)

    data = torch.concat([source_dset_data, other_dset_data])
    targets = torch.concat([source_dset_targets, other_dset_targets])

    dset = torch.utils.data.TensorDataset(data, targets)

    source_loader = NumpyLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return source_loader, dset

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