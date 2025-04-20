import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features = 10, out_features = 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features = 64, out_features = 2)
        )

    def forward(self, x):
        return self.net(x)