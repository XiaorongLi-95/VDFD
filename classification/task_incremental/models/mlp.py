import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.init as init
# import math


class MLP(nn.Module):

    def __init__(self, out_dim=10, in_dim=784, hidden_dim=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.last = nn.Linear(hidden_dim, out_dim, bias=False)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):

        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.logits(x)
        return x
