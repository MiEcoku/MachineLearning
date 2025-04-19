import torch

class VGG(torch.nn.Module):

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
            )
            layers.append(torch.nn.ReLU())
            in_channels = out_channels
        layers.append(torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        return torch.nn.Sequential(*layers)
    
    def __init__(self, conv_arch, in_channels = 1):
        super(VGG, self).__init__()
        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        
        self.net = torch.nn.Sequential(
            *conv_blks,
            torch.nn.Flatten(),
            torch.nn.Linear(out_channels * 7 * 7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        return self.net(x)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

mode = VGG(conv_arch)

x = torch.randn(1, 1, 224, 224)
for blk in mode.net:
    x = blk(x)
    print(blk.__class__.__name__, ' output shape:\t', x.shape)