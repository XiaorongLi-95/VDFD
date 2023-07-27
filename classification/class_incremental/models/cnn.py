
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, out_dim=10, in_channel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def reset_model(self):
        for module_ in self.modules():
            if hasattr(module_, 'weight') and module_.weight is not None:
                module_.reset_parameters() 



