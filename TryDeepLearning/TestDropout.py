import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
from Animator import animator

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

droprate1, droprate2 = 0.2, 0.5
class MyModule(torch.nn.Module):
    def __init__(self, in_channel, out_channel, hidden1, hidden2, is_train):
        super(MyModule, self).__init__()
        # self.net = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(in_features = in_channel, out_features = hidden1),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(droprate1),
        #     torch.nn.Linear(in_features = hidden1, out_features = hidden2),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(droprate2),
        #     torch.nn.Linear(in_features = hidden2, out_features = out_channel)
        # )
        self.flatten = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(in_features = in_channel, out_features = hidden1)
        self.lin2 = torch.nn.Linear(in_features = hidden1, out_features = hidden2)
        self.lin3 = torch.nn.Linear(in_features = hidden2, out_features = out_channel)
        self.relu = torch.nn.ReLU()
        self.is_train = is_train
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        if self.is_train:
            x = dropout_layer(x, droprate1)
        x = self.relu(self.lin2(x))
        if self.is_train:
            x = dropout_layer(x, droprate2)
        return self.lin3(x)

# def show_img(imgs, titles = None):
#     fig, axes = plt.subplots(2, 5)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             ax.imshow(img.numpy())
#         else:
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     plt.show()


def main(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_train = True
    if args.test:
        is_train = False
    
    transforms = torchvision.transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train = True, transform = transforms, download = True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = "../data", train = False, transform = transforms, download = True
    )
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle = True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = args.batch_size, shuffle = False)
    classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    
    # show_img([mnist_train[i][0][0] for i in range(10)], 
    #          [classes[mnist_train[i][1]] for i in range(10)])

    mode = MyModule(
        in_channel = args.in_channel, out_channel = args.out_channel,
        hidden1 = args.hidden1, hidden2 = args.hidden2,
        is_train = is_train
        ).to(device)

    optimizer = torch.optim.SGD(mode.parameters(), lr = args.lr)
    loss = torch.nn.CrossEntropyLoss()

    axes = animator(xlabel = 'epoch', ylabel = 'loss', legend = ['train loss', 'train acc', 'test acc'])

    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        test_acc = 0
        total = 0

        mode.is_train = True
        for i, data in enumerate(train_iter, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = mode.forward(inputs)

            _, predicte = torch.max(output, 1)
            train_acc += (predicte == labels).sum().item()
            total += labels.size(0)
            
            l = loss(output, labels)
            l.backward()
            optimizer.step()
            
            train_loss += l.item()
        train_loss = train_loss / total
        train_acc = 1. * train_acc / total
        total = 0

        mode.is_train = False
        with torch.no_grad():
            for data in test_iter:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = mode.forward(inputs)
                _, predicte = torch.max(outputs, 1)
                test_acc += (predicte == labels).sum().item()
                total += labels.size(0)
        test_acc = 1. * test_acc / total

        axes.add(epoch, (train_loss * 200, train_acc, test_acc))

    axes.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',  type = int,   default = 256)
    parser.add_argument('--lr',          type = float, default = 0.5)
    parser.add_argument('--epochs',      type = int,   default = 10)
    parser.add_argument('--in_channel',  type = int,   default = 784)
    parser.add_argument('--out_channel', type = int,   default = 10)
    parser.add_argument('--hidden1',     type = int,   default = 256)
    parser.add_argument('--hidden2',     type = int,   default = 256)
    
    parser.add_argument('--test',    action = 'store_true', default = False)

    opt = parser.parse_args()

    main(opt)
