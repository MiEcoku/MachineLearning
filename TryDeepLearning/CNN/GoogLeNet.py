import torch
import torch.nn.functional as F

class Inception(torch.nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        
        self.p1_1 = torch.nn.Conv2d(in_channels, c1, kernel_size = 1)

        self.p2_1 = torch.nn.Conv2d(in_channels, c2[0], kernel_size = 1)
        self.p2_2 = torch.nn.Conv2d(c2[0], c2[1], kernel_size = 3, padding = 1)

        self.p3_1 = torch.nn.Conv2d(in_channels, c3[0], kernel_size = 1)
        self.p3_2 = torch.nn.Conv2d(c3[0], c3[1], kernel_size = 5, padding = 2)

        self.p4_1 = torch.nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.p4_2 = torch.nn.Conv2d(in_channels, c4, kernel_size = 1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2( F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2( F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2( self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim = 1)

class GoogLeNet(torch.nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        b1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        b2 = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        b3 = torch.nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        b4 = torch.nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        b5 = torch.nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )
        self.net = torch.nn.Sequential(b1, b2, b3, b4, b5, torch.nn.Linear(1024, 10))

    def forward(self, x):
        return self.net(x)

x = torch.randn(1, 1, 96, 96)

mode = GoogLeNet()

for layer in mode.net:
    x = layer(x)
    print(layer.__class__.__name__, ' output shape:\t', x.shape)