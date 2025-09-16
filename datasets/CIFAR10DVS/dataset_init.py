import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
import os
import numpy as np
from os.path import isfile, join
import tonic
from tonic import DiskCachedDataset


def dataset_init(num_workers = 8,
                 batch_size = 128,
                 save_to = "/scratch-node/20235438"):
    
    in_h = 48
    in_w = 48
    in_ch = 2
    out_dim = 10
    
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

    train_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10), ])
    test_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(save_to, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(save_to, 'DVS/DVS_Cifar10'), transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[in_h, in_h], mode='bilinear', align_corners=True),
        transforms.RandomCrop(in_h, padding=in_h // 12),
        transforms.RandomHorizontalFlip(),
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[in_h, in_h], mode='bilinear', align_corners=True), 
    ])

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(save_to, 'DVS/DVS_Cifar10/train_cache_{}'.format(10)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(save_to, 'DVS/DVS_Cifar10/test_cache_{}'.format(10)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = .9
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    # num_train = len(train_dataset)
    # num_per_cls = num_train // 10
    # indices_train, indices_test = [], []
    # portion = .9
    # reduction = 0.2  # Use only 50% of the total dataset

    # for i in range(10):
    #     # Compute the number of samples to keep per class
    #     reduced_per_cls = int(num_per_cls * reduction)
    #     start_idx = i * num_per_cls
    #     reduced_start = start_idx
    #     reduced_end = start_idx + reduced_per_cls

    #     class_indices = list(range(reduced_start, reduced_end))

    #     split = int(reduced_per_cls * portion)
    #     indices_train.extend(class_indices[:split])
    #     indices_test.extend(class_indices[split:])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=True, num_workers=num_workers
    )
    return train_loader, test_loader, out_dim, in_w, in_h, in_ch
