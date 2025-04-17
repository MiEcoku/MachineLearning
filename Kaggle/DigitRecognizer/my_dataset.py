import torch
import pandas as pd
import matplotlib.pyplot as plt

def show_imgs(imgs, labels) -> None:
    _, axes = plt.subplots(1, 5)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(labels[i])
    
    plt.show()

def show_info(trainData, labels, testData) -> None:
    print('-' * 100)
    print(f'train dataset has {trainData.shape[0]} rows and {trainData.shape[1]} columns')
    print(f'test dataset has {testData.shape[0]} rows and {testData.shape[1]} columns')

def getData(train_file = './data/train.csv', test_file = './data/test.csv'):
    trainData = pd.read_csv(train_file)
    testData = pd.read_csv(test_file)
    labels = trainData['label']
    trainData.drop(['label'], axis = 1, inplace = True)

    # imgs = trainData.to_numpy()[0:5, :].reshape(5, 28, 28)
    # label = labels.to_numpy()[0:5].reshape(5, 1)
    # show_imgs(imgs, label)

    # show_info(trainData, labels, testData)

    train_dataset = trainData.to_numpy().reshape(trainData.shape[0], 1, 28, 28)
    test_dataset = testData.to_numpy().reshape(testData.shape[0], 1, 28, 28)
    labels = labels.to_numpy()
    # [batch_size, channel, high, width]

    train_dataset = torch.tensor(train_dataset, dtype = torch.float32)
    train_labels = torch.zeros(size = [labels.shape[0], 10])
    for i in range(labels.shape[0]):
        train_labels[i, labels[i]] = 1.
    test_dataset = torch.tensor(test_dataset, dtype = torch.float32)

    # print(train_dataset.shape)

    return (train_dataset, train_labels), test_dataset


def main():
    train_dataset, test_dataset = getData()

    

if __name__ == '__main__':
    main()
