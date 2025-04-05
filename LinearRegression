import torch
import matplotlib.pyplot as plt
import random

def synthetic_data(weight, bias, num_examples):
    x = torch.normal(0, 1, (num_examples, len(weight)))
    y = torch.matmul(x, weight) + bias
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_weight = torch.tensor([2, -3.4])
true_bias = 4.2
features, labels = synthetic_data(true_weight, true_bias, 1000)

### 查看数据趋势
# plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)]
        ) 
        yield features[batch_indices], labels[batch_indices]



### Loss Function
MSE = lambda x, y: (x - y) ** 2 / 2

### 回归模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b;

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

lr = 0.03
epochs = 3
net = linreg
batch_size = 10

# for x, y in data_iter(batch_size, features, labels):
#     print(x, '\n', y)
#     break

for epoch in range(epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = MSE(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = MSE(net(features, w, b), labels)
        print('epoch %d, loss %lf' % (epoch + 1, float(train_l.mean())))

print('w 的误差:{}'.format(true_weight - w.reshape(true_weight.shape)))
print('b 的误差:{}'.format(true_bias - b))
