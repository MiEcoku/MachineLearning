import torch
import argparse
from Animator import animator
import numpy as np

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return (x, y.reshape((-1, 1)))

class MyModule(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super(MyModule, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = 1, bias = True)
        )
    
    def forward(self, x):
        return self.net(x)

def main(args):
    true_w, true_b = torch.ones((args.input_channel, 1)) * 0.01, 0.05
    n_train, n_test = 20, 100

    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        shuffle = True, 
        batch_size = args.batch_size
    )

    # print(train_data[0].type)

    test_data = synthetic_data(true_w, true_b, n_test)
    test_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*test_data),
        batch_size = args.batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = MyModule(args.input_channel)
    mode.to(device)

    loss = torch.nn.MSELoss(reduction = 'none')
    L2 = lambda x: torch.sum(x.pow(2)) / 2

    for param in mode.net[0].parameters():
        param.data.normal_()
    
    optimizer = torch.optim.SGD(
        [{"params": mode.net[0].weight, 'weight_decay': args.lambd},
         {"params": mode.net[0].bias}], lr = args.lr)

    # print(mode.net[0].weight.data.norm(2))

    axes = animator(xlabel = 'epoch', ylabel = 'loss', yscale = 'log', legend = ['train loss', 'test loss'])

    for epoch in range(args.epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            output = mode.forward(x)
            l = loss(output, y) + (args.lambd * L2(mode.net[0].weight.data)).sum()
            l.mean().backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            train_loss = loss(mode.forward(train_data[0]), train_data[1]).mean().item()
            test_loss = loss(mode.forward(test_data[0]), test_data[1]).mean().item()
            # print('epochs : {}, train_loss : {}, test_loss : {}'.format(
            #         epoch + 1,
            #         train_loss,
            #         test_loss
            #     ))
            axes.add(epoch + 1, (train_loss, test_loss))

    axes.show()
    print("w's L2 norm is {}".format(mode.net[0].weight.data.norm(2).item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr',            type = float, default = 0.003)
    parser.add_argument('--batch_size',    type = int,   default = 5)
    parser.add_argument('--input_channel', type = int,   default = 200)
    parser.add_argument('--epochs',        type = int,   default = 100)
    parser.add_argument('--lambd',         type = float, default = 0.)


    opt = parser.parse_args()

    main(opt)
