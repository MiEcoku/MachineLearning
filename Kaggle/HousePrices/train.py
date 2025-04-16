import torch
import argparse
from My_model import MyModel
from my_dataset import get_data
from Animator import animator
import pandas as pd


def train(mode, 
          train_features, train_labels, test_features, test_labels,
          num_ecpochs, learning_rate, weight_decay, batch_size
          ):
    train_ls, test_ls = [], []
    train_data = (train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        shuffle = True,
        batch_size = batch_size
    )

    optimizer = torch.optim.Adam(mode.parameters(), lr = learning_rate, weight_decay = weight_decay)
    loss = torch.nn.MSELoss()

    log_mse = lambda x, y: torch.sqrt(
        loss(
            torch.log(torch.clamp(x, 1, float('inf'))),
            torch.log(y)
        )
    )

    for epoch in range(num_ecpochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            output = mode.forward(X)
            l = loss(output, y)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_ls.append(log_mse(mode.forward(train_features), train_labels))
            if test_labels is not None:
                test_ls.append(log_mse(mode.forward(test_features), test_labels))
    
    return train_ls, test_ls

def get_K_fold_data(k, i, x, y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat([x_train, x_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return x_train, y_train, x_valid, y_valid

def k_fold(K, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    axes = animator(xlabel = 'epoch', ylabel = 'rmse', legend = ['train', 'valid'], yscale = 'log')

    train_l_sum, valid_l_sum = 0, 0
    for i in range(K):
        data = get_K_fold_data(K, i, x_train, y_train)
        mode = MyModel(x_train.shape[1]).to(device)
        train_ls, valid_ls = train(mode, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     axes.add(list(range(1, num_epochs + 1)), [train_ls, valid_ls])
        #     axes.show()
        print(f'折{i + 1}, train log rmse{float(train_ls[-1]):f}, '
              f'valid log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / K, valid_l_sum / K

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_features, train_labels, testdata = get_data(train_file = './data/train.csv', test_file = './data/test.csv')
    
    train_l, valid_l = k_fold(args.K, 
                              train_features, train_labels, 
                              args.epochs, args.lr, args.weight_decay, args.batch_size
                              )

    mode = MyModel(train_features.shape[1]).to(device)

    train_ls, _ = train(mode, 
                        train_features, train_labels, None, None, 
                        args.epochs, args.lr, args.weight_decay, args.batch_size
                        )
    
    preds = mode.forward(testdata).detach().numpy()
    # print(preds.shape)
    test_data = pd.DataFrame({'Id' : [i for i in range(1461, preds.shape[0] + 1461)], 'SalePrice' : preds.reshape(1, -1)[0]})
    # print(test_data)
    test_data.to_csv('submission.csv', index = False)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr',           type = float,   default = 5)
    parser.add_argument('--batch_size',   type = int,     default = 64)
    parser.add_argument('--epochs',       type = int,     default = 100)
    parser.add_argument('--K',            type = int,     default = 5)
    parser.add_argument('--weight_decay', type = float,   default = 0)

    opt = parser.parse_args()

    main(opt)
