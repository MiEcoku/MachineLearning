import torch

class MyModel(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super(MyModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = 1)
        )
        
    
    def forward(self, x):
        return self.net(x)
