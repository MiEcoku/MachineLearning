import torch
import argparse
from my_dataset import getData
from MyModule import CNN
import pandas as pd

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = getData()

    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_dataset),
        shuffle = True, 
        batch_size = args.batch_size
    )
    # drop_last = False

    mode = CNN().to(device)
    optimizer = torch.optim.SGD(mode.parameters(), lr = args.lr)
    # optimizer = torch.optim.Adam(mode.parameters(), lr = args.lr)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0
        for i, data in enumerate(train_iter):
            X, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            l = loss(mode.forward(X), y)
            l.backward()
            optimizer.step()
            
            running_loss += l.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1} loss : {running_loss / 100:.3f}]')
                running_loss = 0

    mode.eval()
    with torch.no_grad():
        X = test_dataset.to(device)
        _, id = torch.max(mode.forward(X), 1)
        id = id.to('cpu')
        id = id.detach().numpy()
        submision = pd.DataFrame(
            {'ImageId' : [i for i in range(1, 1 + id.shape[0])], 'Label' : id}
        )
        submision.to_csv('submission.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr',         type = float, default = 0.001)
    parser.add_argument('--epochs',     type = int,   default = 100)
    parser.add_argument('--batch_size', type = int,   default = 64)

    opt = parser.parse_args()

    main(opt)
