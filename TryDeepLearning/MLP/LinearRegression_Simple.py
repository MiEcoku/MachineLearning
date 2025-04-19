import torch
import matplotlib.pyplot as plt

def create_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

def show_data(features, labels) -> None:
    plt.scatter(features.detach().numpy(), labels.detach().numpy())
    plt.show()
    
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModule, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features = 2, out_features = 1, bias = True)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

def main():
    weights_true = torch.tensor([2, -3.4])
    bias_true = torch.tensor([4.2])
    num_examples = 1000

    features, labels = create_data(weights_true, bias_true, num_examples)

    # show_data(features[:, 1], labels)

    dataset = (features, labels)
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*dataset),
        shuffle = True,
        batch_size = 10
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = MyModule().to(device)
    optimizer = torch.optim.SGD(mode.parameters(), lr = 0.03)
    epochs = 3

    loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            output = mode.forward(X)
            l = loss(output, y)
            l.backward()
            optimizer.step()
        l = loss(mode.forward(features), labels)
        print('epoch {}, loss{}'.format(epoch, l))
    
    # print(*mode.parameters())

if __name__ == '__main__':
    main()
