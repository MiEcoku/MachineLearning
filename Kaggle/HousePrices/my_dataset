import os
import pandas as pd
import torch
import numpy as np
 

def get_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    labels = train['SalePrice']
    
    train.drop(['SalePrice', 'Id'], axis = 1, inplace = True)
    test.drop(['Id'], axis = 1, inplace = True)

    combine = pd.concat((train, test)).reset_index(drop = True)

    # print('-' * 100)
    # print('Train dataset :', train.shape[1], 'columns and ', train.shape[0], 'rows')
    # print('Test dataset :', test.shape[1], 'columns and ', test.shape[0], 'rows')
    # print('-' * 100)
    # print("train's None: ", train.isnull().sum()[train.isnull().sum() > 0].shape[0])
    # print("test's None: ", test.isnull().sum()[test.isnull().sum() > 0].shape[0])
    # print("combine's None: ", combine.isnull().sum()[combine.isnull().sum() > 0].shape[0])
    # print('-' * 100)
    
    numeric_features = combine.dtypes[combine.dtypes != 'object'].index
    combine[numeric_features] = combine[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    combine[numeric_features] = combine[numeric_features].fillna(0)

    combine = pd.get_dummies(combine, dummy_na = True)

    # print(combine.dtypes[combine.dtypes == 'object'])
    # print(combine[ : train.shape[0]].isnull().sum()[combine[ : train.shape[0]].isnull().sum() > 0])
    # print(combine[ : train.shape[0]].dtypes[combine[ : train.shape[0]].dtypes == 'object'])

    train_features = torch.tensor(combine[ : train.shape[0]].values.astype(float), dtype = torch.float32)
    train_labels = torch.tensor(labels.values.reshape(-1, 1), dtype = torch.float32)
    test_features = torch.tensor(combine[train.shape[0] : ].values.astype(float), dtype = torch.float32)

    return train_features, train_labels, test_features


def main():
    train_file = './data/train.csv'
    test_file = './data/test.csv'

    trainset, labels, testset = get_data(train_file, test_file)
    # print(trainset)
    # print(labels)
    # print(testset)

    


if __name__ == '__main__':
    main()
