import torch

def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else :
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            mean = x.mean(dim = 0)
            var = ((x - mean) ** 2).mean(dim = 0)
        else :
            mean = x.mean(dim = (0, 2, 3), keepdim = True)
            var = ((x - mean) ** 2).mean(dim = (0, 2, 3), keepdim = True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * x_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else :
            shape = (1, num_features, 1, 1)
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        
        Y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beat, self.moving_mean, self.moving_var, eps = 1e-5, momentum = 0.9
        )
        return Y
