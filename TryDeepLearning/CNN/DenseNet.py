import torch

def conv_block(input_channels, num_channels):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(input_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1)
    )

class DenseBlock(torch.nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels * i + input_channels, num_channels)
            )
        self.net = torch.nn.Sequential(*layer)
    
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim = 1)
        return x

def transition_block(input_channels, num_channels):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(input_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(input_channels, num_channels, kernel_size = 1),
        torch.nn.AvgPool2d(kernel_size = 2, stride = 2)
    )

class DenseNet(torch.nn.Module):
    def __init__(self, num_channels = 64, growth_rate = 32, num_convs_in_dense_blocks = [4, 4, 4, 4]):
        super(DenseNet, self).__init__()
        self.b1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
