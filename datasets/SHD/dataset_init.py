from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import SlicedDataset
from torchvision import transforms as vision_transforms
import os
import torch
import yaml

def dataset_init(num_workers = 8,
                 batch_size = 128,
                 save_to = "/scratch-node/20235438"):

    # Dataset dimensions 
    sensor_size = tonic.datasets.SHD.sensor_size
    # number_time_bins = time_steps
    # out_dim = 20
    # in_dim = 700

    # transform = transforms.Compose([
    #     transforms.ToFrame(sensor_size=sensor_size, n_time_bins=number_time_bins),
    #     lambda x: x.reshape(x.shape[0], -1)])
            
    time_window = 10000
    out_dim = 20
    in_dim = 700

    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size,
                            time_window=time_window),
        lambda x: x.reshape(x.shape[0], -1),
                            ])
    # Initialize the training and test datasets
    train_dataset = tonic.datasets.SHD(save_to=save_to, train=True, transform=transform)
    test_dataset = tonic.datasets.SHD(save_to=save_to, train=False, transform=transform)

    # Dataloaders
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=True,
                                drop_last = True
                             )
    
    test_loader  = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=False,
                                drop_last = True 
                             )     

    return train_loader, test_loader, out_dim, in_dim