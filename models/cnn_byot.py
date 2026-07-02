import torch
import torch.nn as nn
import numpy as np
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from PIL import Image
import torchvision.transforms as transforms
from byot_models import multi_resnet18_kd

IMAGE_SIZE = 32
IMAGES_DIR = os.path.join('..', 'data', 'cifar10', 'data', 'raw', 'img')

transform_train = transforms.Compose([
    # transforms.RandomCrop(IMAGE_SIZE, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr

        self.backbone = multi_resnet18_kd(num_classes=self.num_classes).to(self.device)

        self.size = self.model_size()


    def forward(self, x):
        return self.backbone(x)

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def _load_image(self, img_name):
        if 'test' in img_name:
            name = img_name.split('/')
            img = Image.open(os.path.join(IMAGES_DIR, 'test', name[-1]))
        else:
            img = Image.open(os.path.join(IMAGES_DIR, 'train', img_name))
        if self.training:
            img = transform_train(img)
        else:
            img = transform_test(img)
        img = img.cpu().detach().numpy()
        return img

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size