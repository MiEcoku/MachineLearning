import torch
import torch.nn.functional as F

class Residual(torch.nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, stride = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = stride)
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = stride)
        else :
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)

