import torch

class AlexNet(torch.nn.Module):
    def __init__(self, out_features = 10):
        super(AlexNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, stride = 4, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 6400, out_features = 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(in_features = 4096, out_features = 4096),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(in_features = 4096, out_features = out_features)
        
        )
    
    def forward(self, x):
        return self.net(x)

