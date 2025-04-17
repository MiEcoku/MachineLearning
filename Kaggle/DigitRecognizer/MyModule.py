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
