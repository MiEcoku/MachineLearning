import pandas
import matplotlib
import torch

def getData():
    train_path = './train.csv'
    test_path = './test.csv'

    traindata = pandas.read_csv(train_path)
    testdata = pandas.read_csv(test_path)
    labels = traindata['Survived'].values

    traindata.drop(['Survived'], axis = 1, inplace = True)
    features = pandas.concat((traindata, testdata)).reset_index(drop = True)

    features.drop(['PassengerId', 'Cabin', 'Ticket', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)

    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x : (x - x.mean()) / (x.std())
    )
    features[numeric_features] = features[numeric_features].fillna(0)

    features = pandas.get_dummies(features, dummy_na = True).values.astype(float)

    # print(features.to_numpy())

    train_features = torch.tensor(features[ : traindata.shape[0]], dtype = torch.float32)
    test_features = torch.tensor(features[traindata.shape[0] : ], dtype = torch.float32)
    train_labels = torch.zeros(size = [labels.shape[0], 2])
    train_labels[[i for i in range(labels.shape[0])], labels] = 1

    # print(train_features.shape)
    
    return (train_features, train_labels), test_features
    # matplotlib.pyplot.figure(figsize = (6, 4))



if __name__ == '__main__':
    getData()