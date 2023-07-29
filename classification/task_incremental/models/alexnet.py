import torch.nn as nn
import numpy as np


class AlexNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=3):
        # backbone alignment with HAT
        super().__init__()
        size = 32  # image size
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=size // 8)
        s = compute_conv_output_size(size, size // 8)
        s = s // 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=size // 10)
        s = compute_conv_output_size(s, size // 10)
        s = s // 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048) #self.smoid=2
        self.fc2 = nn.Linear(2048, 2048)
        self.last = nn.Linear(2048, out_dim)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        y = self.logits(h)
        return y


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
