import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, in_shape=768):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x