from torch import from_numpy,zeros
from os import listdir
from pandas import read_csv
from json import load
from lib.env import *
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader,TensorDataset

def get_labeled_nursing_ids():
    return sorted([int(file.split('_')[0]) for file in listdir(f'../data/nursing_andrew')])
def load_feature_label_pair(index=0,data_dir=DATA_DIR,label_dir=LABEL_DIR):
    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    if index in skip_idx:
        raise(Exception("file not labeled"))
    X = load_features(index,data_dir)
    y = load_labels(index,len(X),label_dir)
    return X,y
def load_features(index=0,data_dir=DATA_DIR):
    return from_numpy(read_csv(f'{data_dir}/{index}/raw_data.csv',header=None)[[2,3,4]].to_numpy())
def load_labels(index,length,label_dir=LABEL_DIR):
    with open(f'{label_dir}/{index}_data.json','r') as f:
        data = load(f)
    y = zeros(length)
    for puff in data['puffs']:
        y[puff['start']:puff['end']] = 1
    y = y.unsqueeze(1).float()
    return y
def window_nursing(X,y_true,window_size=101):
    x = X[:,0].unsqueeze(1)
    y = X[:,1].unsqueeze(1)
    z = X[:,2].unsqueeze(1)
    xs = [x[:-(window_size-1)]]
    ys = [y[:-(window_size-1)]]
    zs = [z[:-(window_size-1)]]
    for i in range(1,window_size-1):
        xs.append(x[i:i-(window_size-1)])
        ys.append(y[i:i-(window_size-1)])
        zs.append(z[i:i-(window_size-1)])
    xs.append(x[(window_size-1):])
    ys.append(y[(window_size-1):])
    zs.append(z[(window_size-1):])
    xs = torch.cat(xs,axis=1).float()
    ys = torch.cat(ys,axis=1).float()
    zs = torch.cat(zs,axis=1).float()
    X = torch.cat([xs,ys,zs],axis=1)
    y_true = y_true[window_size//2:-(window_size//2)]
    return X,y_true
def load_nursing_list(idxs,data_dir,label_dir):
    X = torch.Tensor()
    y = torch.Tensor()

    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    for idx in idxs:
        if(idx in skip_idx):
            continue
        Xi,yi = load_nursing_by_index(idx,data_dir=data_dir,label_dir=label_dir)
        X = torch.cat([X,Xi])
        y = torch.cat([y,yi])
    return X,y
def load_and_window_nursing_list(idxs,data_dir=f'../data/nursing',label_dir=f'../data/nursing_andrew',window_size=101):
    X = torch.Tensor()
    y = torch.Tensor()

    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    for idx in idxs:
        if(idx in skip_idx):
            continue
        Xi,yi = load_nursing_by_index(idx,data_dir=data_dir,label_dir=label_dir)
        Xi,yi = window_nursing(Xi,yi,window_size=window_size)
        X = torch.cat([X,Xi])
        y = torch.cat([y,yi])
    return X,y
def load_and_window_nursing_by_index(idx,window_size=201):
    return window_nursing(*load_nursing_by_index(idx,dir='../data/nursing.chrisogonas/'),window_size=window_size)
def load_and_window_nursing_list_for_convolution(idxs,data_dir=f'../data/nursing',label_dir=f'../data/nursing_andrew',window_size=101):
    X = torch.Tensor()
    y = torch.Tensor()

    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    for idx in idxs:
        if(idx in skip_idx):
            continue
        Xi,yi = load_nursing_by_index(idx,data_dir=data_dir,label_dir=label_dir)
        Xi,yi = window_nursing_for_convolution(Xi,yi,window_size=window_size)
        X = torch.cat([X,Xi])
        y = torch.cat([y,yi])
    return X,y
def window_nursing_for_convolution(X,y,window_size=101):
    X = X.unsqueeze(2)
    xs = [X[:-(window_size-1)]]
    for i in range(1,window_size-1):
        xs.append(X[i:i-(window_size-1)])
    xs.append(X[(window_size-1):])
    X = torch.cat(xs,axis=2)
    y = y[window_size//2:-(window_size//2)]
    return X,y
def load_data(window_size=101,n=71):
    train_idx = range(n)
    X,y = load_and_window_nursing_list(train_idx,window_size=window_size)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=0)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=.25,stratify=y_train,random_state=0)
    trainloader = DataLoader(TensorDataset(X_train,y_train),batch_size=64,shuffle=True)
    devloader = DataLoader(TensorDataset(X_dev,y_dev),batch_size=64,shuffle=True)
    testloader = DataLoader(TensorDataset(X_test,y_test),batch_size=64,shuffle=True)
    return trainloader,devloader,testloader
def load_data_convolution(window_size=101,n=71):
    train_idx = range(n)
    X,y = load_and_window_nursing_list_for_convolution(train_idx,window_size=window_size)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=0)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=.25,stratify=y_train,random_state=0)
    trainloader = DataLoader(TensorDataset(X_train,y_train),batch_size=64,shuffle=True)
    devloader = DataLoader(TensorDataset(X_dev,y_dev),batch_size=64,shuffle=True)
    testloader = DataLoader(TensorDataset(X_test,y_test),batch_size=64,shuffle=True)
    return trainloader,devloader,testloader
def load_data_cv(foldi=0,window_size=101):
    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    all_idx = list(range(71))
    for idx in skip_idx:
        all_idx.remove(idx)
    random.seed(0)
    random.shuffle(all_idx)
    k_folds = 5
    fold_size = int(len(all_idx)/k_folds)
    test_idx = all_idx[foldi*fold_size:(foldi+1)*fold_size]
    for idx in test_idx:
        all_idx.remove(idx)
    train_idx = all_idx
    X,y = load_and_window_nursing_list(train_idx,window_size=window_size)
    X_train,X_dev,y_train,y_dev = train_test_split(X,y,test_size=.05,stratify=y)
    trainloader = DataLoader(TensorDataset(X_train,y_train),batch_size=64,shuffle=True)
    devloader = DataLoader(TensorDataset(X_dev,y_dev),batch_size=64,shuffle=True)
    return trainloader,devloader,test_idx
def load_data_convolution_cv(foldi=0,window_size=101):
    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    all_idx = list(range(71))
    for idx in skip_idx:
        all_idx.remove(idx)
    random.seed(0)
    random.shuffle(all_idx)
    k_folds = 5
    fold_size = int(len(all_idx)/k_folds)
    test_idx = all_idx[foldi*fold_size:(foldi+1)*fold_size]
    for idx in test_idx:
        all_idx.remove(idx)
    train_idx = all_idx
    X,y = load_and_window_nursing_list_for_convolution(train_idx,window_size=window_size)
    X_train,X_dev,y_train,y_dev = train_test_split(X,y,test_size=.05,stratify=y)
    trainloader = DataLoader(TensorDataset(X_train,y_train),batch_size=64,shuffle=True)
    devloader = DataLoader(TensorDataset(X_dev,y_dev),batch_size=64,shuffle=True)
    return trainloader,devloader,test_idx
def load_cv_train_idx(foldi=0,window_size=101):
    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    all_idx = list(range(71))
    for idx in skip_idx:
        all_idx.remove(idx)
    random.seed(0)
    random.shuffle(all_idx)
    k_folds = 5
    fold_size = int(len(all_idx)/k_folds)
    test_idx = all_idx[foldi*fold_size:(foldi+1)*fold_size]
    for idx in test_idx:
        all_idx.remove(idx)
    train_idx = all_idx
    return train_idx
def load_cv_test_idx(foldi=0,window_size=101):
    skip_idx = [19,24,26,32,34,38,40,45,52,55,70]
    all_idx = list(range(71))
    for idx in skip_idx:
        all_idx.remove(idx)
    random.seed(0)
    random.shuffle(all_idx)
    k_folds = 5
    fold_size = int(len(all_idx)/k_folds)
    test_idx = all_idx[foldi*fold_size:(foldi+1)*fold_size]
    for idx in test_idx:
        all_idx.remove(idx)
    train_idx = all_idx
    return test_idx
def load_nursing_by_index(index,data_dir='../data/nursing/',label_dir='../data/nursing_andrew'):
    i = index
    df = pd.read_csv(f'{data_dir}/{i}/raw_data.csv')
    with open(f'{label_dir}/{i}.json','r') as f:
        data = load(f)
    y_true = np.zeros(len(df))
    for puff in data['puffs']:
        y_true[puff['start']:puff['end']] = 1
    X = torch.from_numpy(df[['x','y','z']].to_numpy())
    y = torch.from_numpy(y_true)
    X = X.float()[::5] # downsample to 20 Hz
    y = y.unsqueeze(1).float()[::5]
    return X,y