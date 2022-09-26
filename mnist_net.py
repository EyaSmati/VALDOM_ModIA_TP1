import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1,8,5,stride=1)
        self.conv2 = nn.Conv2d(8,16,5,stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)    
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x




if __name__=='__main__':
    
    x = torch.rand(64,1,28,28)
    net = MNISTNet()
    y = net(x)
    assert y.shape == (64,10)
