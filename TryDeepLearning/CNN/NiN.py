import torch

class NiN(torch.nn.Module):
    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding = 0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size = 1),
            torch.nn.ReLU()
        )
    
    def __init__(self, in_channels = 1):
        super(NiN, self).__init__()
        self.net = torch.nn.Sequential(
            self.nin_block(in_channels, 96, kernel_size = 11, stride = 4, padding = 0),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            self.nin_block(96, 256, kernel_size = 5, stride = 1, padding = 2),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            self.nin_block(256, 384, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.Dropout(p = 0.5),
            self.nin_block(384, 10, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )
    
    def forward(self, x):
        return self.net(x)

x = torch.randn(1, 1, 224, 224)

mode = NiN()

for layer in mode.net:
    x = layer(x)
    print(layer.__class__.__name__, ' output shape:\t', x.shape)