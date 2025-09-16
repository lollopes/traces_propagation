from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import SlicedDataset
from torchvision import transforms as vision_transforms
import random
import numpy as np
import torch

def dataset_init(num_workers = 8,
                 batch_size = 128,
                 save_to = "/scratch-node/20235438"):

    time_window = 75000
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    out_dim = 11
    in_w = 32
    in_h = 32
    in_ch = 2

    trainset_ori = tonic.datasets.DVSGesture(save_to=save_to, train=True)
    testset_ori = tonic.datasets.DVSGesture(save_to=save_to, train=False)

    # Slicer for data augmentation of Training Dataset
    slicing_time_window = 1575000
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)
    frame_transform = tonic.transforms.Compose([  
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        torch.tensor, vision_transforms.Resize(32)
    ])
    trainset_ori_sl = tonic.SlicedDataset(trainset_ori, slicer=slicer,
                                metadata_path=save_to + '/metadata/online_dvsg_train',
                                transform=frame_transform)
    frame_transform2 = tonic.transforms.Compose([
        torch.tensor,
        vision_transforms.RandomCrop(32, padding=4)
    ])

    train_dataset = tonic.CachedDataset(trainset_ori_sl,
                                cache_path=save_to + '/cache/online_fast_dataloading_train',
                                transform=frame_transform2) 
    
    # Test dataset
    frame_transform_test = tonic.transforms.Compose([ 
            tonic.transforms.ToFrame(sensor_size=sensor_size,
                                    time_window=75000),
            torch.tensor,
            vision_transforms.Resize(32, antialias=True)
        ])

    test_dataset = tonic.CachedDataset(testset_ori,
                            cache_path=save_to + '/cache/online_fast_dataloading_test',
                            transform=frame_transform_test)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=True,
                                drop_last = True,
                                pin_memory = True,
                                worker_init_fn = seed_worker,
                                generator = g
                            )

    test_loader = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(batch_first=True),
                                num_workers = num_workers,
                                shuffle=False,
                                drop_last = True,
                                pin_memory = True,
                                worker_init_fn = seed_worker,
                                generator = g
                            )



    return train_loader, test_loader, out_dim, in_w, in_h, in_ch
