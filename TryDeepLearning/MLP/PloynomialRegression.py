import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def synthetic_data():
    max_degree, n_train, n_test = 20, 100, 100
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    features = np.random.normal(size = (n_train + n_test, 1))
    np.random.shuffle(features)
    ploy_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        ploy_features[:, i] /= math.gamma(i + 1)
    labels = np.dot(ploy_features, true_w)
    labels += np.random.normal(scale = 0.1, size = labels.shape)
    return [torch.tensor(x, dtype = torch.float32) for x in [true_w, features, ploy_features, labels]]

class MyModule(torch.nn.Module):
    def __init__(self, in_fea):
        super(MyModule, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_fea, out_features = 1, bias = False)
        )
    
    def forward(self, x):
        return self.net(x)

def main(args) -> None:
    true_w, features, ploy_features, labels = synthetic_data()
    labels = labels.reshape(-1, 1)
    # print(true_w, '\n', features[:2], '\n', ploy_features[:2], '\n', labels[:2])

    # print(features.shape, '\n', ploy_features.shape, '\n', labels.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = MyModule(args.in_f).to(device)
    data = (ploy_features, labels)

    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data), 
        shuffle = True, batch_size = args.batch_size
    )
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(mode.parameters(), lr = args.lr)

    epochs = args.epochs

    for epoch in range(epochs):
        for input, val in data_iter:
            optimizer.zero_grad()
            output = mode.forward(input[:, :args.in_f])
            l = loss(output, val)
            l.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print('epoch: {}, loss: {}'.format(epoch + 1, loss(mode.net(ploy_features[:, :args.in_f]), labels)))

    print(mode.net[0].weight.data.numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--in_f', type = int, default = 4)


    opt = parser.parse_args()

    main(opt)
