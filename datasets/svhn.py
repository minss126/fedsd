import torch
import torch.utils.data as data
import numpy as np
from torchvision.datasets import SVHN
from PIL import Image

class SVHN_truncated(data.Dataset):
    """
    A custom dataset class for SVHN that allows for truncating the dataset
    to a subset specified by dataidxs.
    """
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # Use 'split' argument for SVHN instead of 'train' boolean
        split = 'train' if self.train else 'test'
        
        # Load the full dataset object from torchvision
        # Transforms are applied in __getitem__, so we don't pass them here.
        svhn_dataobj = SVHN(self.root, split=split, download=self.download)

        # The .data attribute is a numpy array of shape (N, 3, 32, 32)
        data = svhn_dataobj.data
        # The .labels attribute holds the targets
        target = np.array(svhn_dataobj.labels)

        # If dataidxs are provided, slice the dataset
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # The SVHN dataset in torchvision has dimensions (C, H, W).
        # We need to transpose it to (H, W, C) to create a PIL Image.
        img = np.transpose(img, (1, 2, 0))
        
        # Create a PIL Image from the numpy array
        img = Image.fromarray(img, mode='RGB')

        # Apply transforms if they exist
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
    @property
    def num_classes(self):
        return len(np.unique(self.target))