import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes = 10, ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5,padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 394)
        self.fc2 = nn.Linear(394, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        features = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return features, x