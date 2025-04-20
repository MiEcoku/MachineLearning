import torch
import argparse
import pandas
from my_dataset import getData
from MyModule import MLP

def main(args):
    train, test = getData()
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train),
        shuffle = True,
        batch_size = args.batch_size
    )

    mode = MLP()

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(mode.parameters(), lr = args.lr, momentum = 0.9)
    # optimizer = torch.optim.Adam(mode.parameters(), lr = args.lr)

    for epoch in range(args.epochs):
        running_loss = 0
        total = 0
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(mode(X), y)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            total += X.shape[0]
        print(f'epoch: {epoch + 1}, running loss: {running_loss / total}')
        total, running_loss = 0, 0
    
    mode.eval()
    with torch.no_grad():
        y = mode(test)
        _, output = torch.max(y, dim = 1)
        output = output.detach().numpy().astype(int)
        submision = pandas.DataFrame(
            {'PassengerId' : [i for i in range(train[0].shape[0] + 1, train[0].shape[0] + test.shape[0] + 1)],
             'Survived' : output}
        )
        submision.to_csv('submission.csv', index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',     type = int,     default = 100)
    parser.add_argument('--batch_size', type = int,     default = 64)
    parser.add_argument('--lr',         type = float,   default = 0.001)
    
    opt = parser.parse_args()

    main(opt)
