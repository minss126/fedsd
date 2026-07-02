import torch.utils.data as data
import numpy as np
from PIL import Image

class MedMNIST_truncated(data.Dataset):
    """
    A generic custom dataset class for MedMNIST datasets.
    It allows for truncating the dataset to a subset specified by dataidxs
    and works with various MedMNIST classes like OCTMNIST, BloodMNIST, etc.
    
    Args:
        dataset_class: The MedMNIST class to use (e.g., OCTMNIST, BloodMNIST).
        root (str): Root directory of dataset.
        dataidxs (list, optional): A list of indices to subset the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet.
    """
    def __init__(self, dataset_class, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.dataset_class = dataset_class
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # Map the 'train' boolean to the 'split' string used by MedMNIST
        split = 'train' if self.train else 'test'
        
        # Instantiate the provided MedMNIST class
        data_obj = self.dataset_class(split=split, download=self.download, root=self.root)

        # MedMNIST datasets provide data as .imgs and .labels attributes
        data = data_obj.imgs
        # .labels can be (n_samples, 1), so we squeeze it to (n_samples,)
        target = data_obj.labels.squeeze()

        # If dataidxs are provided, slice the dataset
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        # 이미지 배열의 차원 수를 먼저 확인합니다.
        if img.ndim == 2:  # 채널 차원이 없는 흑백 이미지 (예: (28, 28))
            mode = 'L'
        elif img.ndim == 3: # 채널 차원이 있는 이미지
            if img.shape[-1] == 1: # 채널 차원이 있는 흑백 (예: (28, 28, 1))
                img = img.squeeze(axis=-1) # (28, 28)로 만듭니다.
                mode = 'L'
            elif img.shape[-1] == 3: # 컬러 이미지 (예: (28, 28, 3))
                mode = 'RGB'
            else:
                raise ValueError(f"Unsupported image channel count: {img.shape[-1]}")
        else:
            raise ValueError(f"Unsupported image dimension count: {img.ndim}")

        # Create a PIL Image from the numpy array
        img = Image.fromarray(img, mode=mode)

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