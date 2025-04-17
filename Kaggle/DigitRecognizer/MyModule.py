import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.liner = torch.nn.Linear(in_features = 16 * 14 * 14, out_features = 10, bias = False)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.liner(x)
        return x

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 16 * 5 * 5, out_features = 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features = 120, out_features = 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features = 84, out_features = 10)
        )
    
    def forward(self, x):
        return self.net(x)
