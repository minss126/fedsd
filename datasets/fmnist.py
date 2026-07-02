import torch
import torch.utils.data as data
import numpy as np
import torchvision
from torchvision.datasets import FashionMNIST 
from torchvision import transforms
from PIL import Image

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        fmnist_dataobj = FashionMNIST(self.root, train=self.train, transform=self.transform, 
                                      target_transform=self.target_transform, download=self.download)
        data = fmnist_dataobj.data.numpy() if isinstance(fmnist_dataobj.data, torch.Tensor) else fmnist_dataobj.data
        target = np.array(fmnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img, mode='L') 

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
