import numpy as np
import torch
from torch.utils.data import Dataset


# 2. PyTorch Dataset 클래스 정의
class CIFAR10_1_Dataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data = np.load(data_path)
        self.target = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.target[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.data)
    
    # dataset.num_classes
    @property
    def num_classes(self):
        return len(np.unique(self.target))
