from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import SlicedDataset
from torchvision import transforms as vision_transforms
import os
import torch
import yaml
import numpy as np

def dataset_init(num_workers = 8,
                 batch_size = 128,
                 save_to = "/scratch-node/20235438"):

    # Dataset dimensions
    sensor_size = tonic.datasets.NMNIST.sensor_size 
    time_window = 1000  
    out_dim = 10
    in_dim = 34*34*2

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        lambda x: x.reshape(x.shape[0], -1),
    ])
    
    # Initialize the training and test datasets
    train_dataset = tonic.datasets.NMNIST(save_to=save_to, train=True, first_saccade_only=True, transform=transform)
    test_dataset = tonic.datasets.NMNIST(save_to=save_to, train=False, first_saccade_only=True, transform=transform)     

    # Dataloaders
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=True,
                                drop_last = False
                             )
    
    test_loader  = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=False,
                                drop_last = False 
                             )  


    return train_loader, test_loader, out_dim, in_dim