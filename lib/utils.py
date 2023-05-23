import pandas as pd
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
import seaborn as sns
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random

def load_nursing_by_index(index,data_dir='../data/nursingv1/',label_dir='../data/nursingv1_andrew'):
    i = index
    df = pd.read_csv(f'{data_dir}/{i}/raw_data.csv',header=None)
    with open(f'{label_dir}/{i}_data.json','r') as f:
        data = json.load(f)
    y_true = np.zeros(len(df))
    for puff in data['puffs']:
        y_true[puff['start']:puff['end']] = 1
    X = torch.from_numpy(df[[2,3,4]].to_numpy())
    y = torch.from_numpy(y_true)
    X = X.float()[::5] # downsample to 20 Hz
    y = y.unsqueeze(1).float()[::5]
    return X,y
def load_thrasher_by_index(index,dir):
    """
    timestamp : 
        https://developer.android.com/reference/android/hardware/SensorEvent#timestamp
        'The time in nanoseconds at which the event happened. For a given sensor, each new sensor event should be monotonically increasing using the same time base as SystemClock.elapsedRealtimeNanos().'

        https://developer.android.com/reference/android/os/SystemClock#elapsedRealtimeNanos()
        Returns nanoseconds since boot, including time spent in sleep.

    real time :
        epoch time in milliseconds
    """
    import os
    files = os.listdir(dir)
    df = pd.read_csv(f'{dir}/{files[index]}/raw/{files[index]}.0.csv',skiprows=1)
    with open(f'{dir}/{files[index]}/log.csv',"r") as f:
        lines = f.readlines()
    times,events = [],[]
    for line in lines:
        line = line.replace("\n","").split(":")
        times.append(int(line[0]))
        events.append(line[1].strip())
    # convert time to seconds
    df.timestamp = df.timestamp * 1e-9
    df['real time'] = df['real time']/1000
    print(f'Length from shape: {df.shape[0]/20}')
    print(f'Length from timestamp: {(df.timestamp[len(df)-1]-df.timestamp[0])}')
    print(f'Length from epoch time : {(df["real time"][len(df)-1]-df["real time"][0])}')
    # df['rawlabel'] = df['rawlabel']*10
    # df['label'] = df['label']*10
    df['real time'] = (df['real time']*1000)
    df['log'] = np.zeros(len(df))
    X = torch.from_numpy(df[['acc_x','acc_y','acc_z']].to_numpy())
    x = X[:,0].unsqueeze(1)
    y = X[:,1].unsqueeze(1)
    z = X[:,2].unsqueeze(1)
    xs = [x[:-99]]
    ys = [y[:-99]]
    zs = [z[:-99]]
    for i in range(1,99):
        xs.append(x[i:i-99])
        ys.append(y[i:i-99])
        zs.append(z[i:i-99])
    xs.append(x[99:])
    ys.append(y[99:])
    zs.append(z[99:])
    xs = torch.cat(xs,axis=1).float()
    ys = torch.cat(ys,axis=1).float()
    zs = torch.cat(zs,axis=1).float()

    X = torch.cat([xs,ys,zs],axis=1)
    df = df.reset_index()
    df['index'] = df['index']/20
    return df,(times,events),X
def cms(y_true,y_pred,dir='.',filename=f'cm.jpg',loss=0):
    fig,axes = plt.subplots(1,3,sharey=True,figsize=(10,5))
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,ax=axes[0],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,ax=axes[1],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,ax=axes[2],cbar=False,fmt='.2f')
    axes[0].set_title('Recall')
    axes[1].set_title('Precision')
    axes[2].set_title('Count')
    plt.suptitle(f'macro-recall : {balanced_accuracy_score(y_true=y_true,y_pred=y_pred)} loss : {loss}')
    plt.savefig(f'{dir}/{filename}',dpi=200,bbox_inches='tight')
def run_old_state_machine_on_thresholded_predictions(predictions):
    state = 0
    states = []
    puff_locations = []
    currentInterPuffIntervalLength = 0
    currentPuffLength = 0
    for i,smokingOutput in enumerate(predictions):
        states.append(state)
        if (state == 0 and smokingOutput == 0.0):
            # no action
            state = 0
        elif (state == 0 and smokingOutput == 1.0):
            # starting validating puff length
            state = 1
            currentPuffLength += 1
        elif (state == 1 and smokingOutput == 1.0):
            # continuing not yet valid length puff
            currentPuffLength += 1
            if (currentPuffLength > 14) :
                # valid puff length!
                state = 2
        elif (state == 1 and smokingOutput == 0.0):
            # never was a puff, begin validating end
            state = 3
            currentInterPuffIntervalLength += 1
        elif (state == 2 and smokingOutput == 1.0):
            # continuing already valid puff
            currentPuffLength += 1
        elif (state == 2 and smokingOutput == 0.0):
            # ending already valid puff length
            state = 4 # begin validating inter puff interval
            currentInterPuffIntervalLength += 1
        elif (state == 3 and smokingOutput == 0.0): 
            currentInterPuffIntervalLength += 1
            if (currentInterPuffIntervalLength > 49):
                # valid interpuff
                state = 0
                currentPuffLength = 0
                currentInterPuffIntervalLength = 0
        elif (state == 3 and smokingOutput == 1.0):
            # was validating interpuff for puff that wasn't valid
            currentPuffLength += 1
            currentInterPuffIntervalLength = 0
            if (currentPuffLength > 14) :
                # valid puff length!
                state = 2
            state = 1
        elif (state == 4 and smokingOutput == 0.0) :
            currentInterPuffIntervalLength += 1
            if (currentInterPuffIntervalLength > 49):
                # valid interpuff for valid puff
                state = 0
                currentPuffLength = 0
                currentInterPuffIntervalLength = 0
                puff_locations.append(i)
        elif (state == 4 and smokingOutput == 1.0):
            # back into puff for already valid puff
            currentInterPuffIntervalLength = 0
            currentPuffLength += 1
            state = 2

    states = states[1:] + [0]
    return states,puff_locations
def run_new_state_machine_on_thresholded_predictions(predictions):
    state = 0
    states = []
    puff_locations = []
    currentInterPuffIntervalLength = 0
    currentPuffLength = 0
    for i,smokingOutput in enumerate(predictions):
        states.append(state)
        if (state == 0 and smokingOutput == 0.0):
            # no action
            state = 0
        elif (state == 0 and smokingOutput == 1.0):
            # starting validating puff length
            state = 1
            currentPuffLength += 1
        elif (state == 1 and smokingOutput == 1.0):
            # continuing not yet valid length puff
            currentPuffLength += 1
            if (currentPuffLength > 14) :
                # valid puff length!
                state = 2
        elif (state == 1 and smokingOutput == 0.0):
            # never was a puff, begin validating end
            state = 3
            currentInterPuffIntervalLength += 1
        elif (state == 2 and smokingOutput == 1.0):
            # continuing already valid puff
            currentPuffLength += 1
        elif (state == 2 and smokingOutput == 0.0):
            # ending already valid puff length
            state = 4 # begin validating inter puff interval
            currentInterPuffIntervalLength += 1
        elif (state == 3 and smokingOutput == 0.0): 
            currentInterPuffIntervalLength += 1
            if (currentInterPuffIntervalLength > 49):
                # valid interpuff
                state = 0
                currentPuffLength = 0
                currentInterPuffIntervalLength = 0
        elif (state == 3 and smokingOutput == 1.0):
            # was validating interpuff for puff that wasn't valid
            currentPuffLength += 1
            currentInterPuffIntervalLength = 0
            if (currentPuffLength > 14) :
                # valid puff length!
                state = 2
            else:
                state = 1
        elif (state == 4 and smokingOutput == 0.0) :
            currentInterPuffIntervalLength += 1
            if (currentInterPuffIntervalLength > 49):
                # valid interpuff for valid puff
                state = 0
                currentPuffLength = 0
                currentInterPuffIntervalLength = 0
                puff_locations.append(i)
        elif (state == 4 and smokingOutput == 1.0):
            # back into puff for already valid puff
            currentInterPuffIntervalLength = 0
            currentPuffLength += 1
            state = 2

    states = states[1:] + [0]
    return states,puff_locations
def forward_casey(X):
    from pandas import read_csv
    from tqdm import tqdm
    ranges = read_csv('../data/casey_network/range',header=None).to_numpy()
    iW = read_csv('../data/casey_network/input',header=None).to_numpy()
    hW = read_csv('../data/casey_network/hidden',header=None).to_numpy()
    output = []
    def oldMinMaxNorm(X):
        """
        attempted version on github (incorrect)
        """
        return ((X-X.min())/(X.max()-X.min())).tolist()
    def correctedMinMaxNorm(X):
        """
        corrected
        """
        return ((2*((X-ranges[:,0])/(ranges[:,1]-ranges[:,0])))-1).tolist()
    def tanSigmoid(X):
        output = []
        for x in X:
            output.append((2/(1+np.exp(-2*x)))-1)
        return output
    def logSigmoid(x):
        return (1/(1+np.exp(-1*x)))
    def oldForward(X):
        a = [1] + oldMinMaxNorm(X)
        b = [1] + tanSigmoid(iW @ a)
        c = hW @ b
        d = logSigmoid(c[0])
        return d
    def correctedForward(X):
        a = [1] + correctedMinMaxNorm(X)
        b = [1] + tanSigmoid(iW @ a)
        c = hW @ b
        d = logSigmoid(c[0])
        return d
    for x in tqdm(X):
        output.append(oldForward(x))
    return output + [0]*99
def forward_casey_corrected(X):
    from pandas import read_csv
    from tqdm import tqdm
    ranges = read_csv('../data/casey_network/range',header=None).to_numpy()
    iW = read_csv('../data/casey_network/input',header=None).to_numpy()
    hW = read_csv('../data/casey_network/hidden',header=None).to_numpy()
    output = []
    def oldMinMaxNorm(X):
        """
        attempted version on github (incorrect)
        """
        return ((X-X.min())/(X.max()-X.min())).tolist()
    def correctedMinMaxNorm(X):
        """
        corrected
        """
        return ((2*((X-ranges[:,0])/(ranges[:,1]-ranges[:,0])))-1).tolist()
    def tanSigmoid(X):
        output = []
        for x in X:
            output.append((2/(1+np.exp(-2*x)))-1)
        return output
    def logSigmoid(x):
        return (1/(1+np.exp(-1*x)))
    def oldForward(X):
        a = [1] + oldMinMaxNorm(X)
        b = [1] + tanSigmoid(iW @ a)
        c = hW @ b
        d = logSigmoid(c[0])
        return d
    def correctedForward(X):
        a = [1] + correctedMinMaxNorm(X)
        b = [1] + tanSigmoid(iW @ a)
        c = hW @ b
        d = logSigmoid(c[0])
        return d
    for x in tqdm(X):
        output.append(correctedForward(x))
    return output + [0]*99
def test_evaluation(dataloader,model,criterion,dir='.',filename=f'cm.jpg',plot=True,device='cuda'):
    from tqdm import tqdm
    y_true = torch.Tensor()
    y_pred = torch.Tensor()
    model_was_training = False
    if(model.training):
        # note that this changes the state of the model outside the scope of this function
        model_was_training = True
        model.eval()

    loss_dev_total = 0
    for (X,y) in tqdm(dataloader):
        X,y = X.to(device),y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        loss_dev_total += loss.item()
        y_true = torch.cat([y_true,y.detach().cpu()])
        y_pred = torch.cat([y_pred,torch.sigmoid(logits).detach().cpu()])

    if(plot):
        cms(y_true=y_true,y_pred=y_pred.round(),loss=(loss_dev_total/len(dataloader)),dir=dir,filename=filename)

    if(model_was_training):
        model.train()

    return loss_dev_total/len(dataloader),y_true,y_pred
def window_nursing_single_channel(X,y,window_size=101):
    X = X[:,0].unsqueeze(1)
    xs = [X[:-(window_size-1)]]
    for i in range(1,window_size-1):
        xs.append(X[i:i-(window_size-1)])
    xs.append(X[(window_size-1):])
    X = torch.cat(xs,axis=1)
    y = y[window_size//2:-(window_size//2)]
    return X,y
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
def load_and_window_nursing_list(idxs,data_dir=f'../data/nursingv1',label_dir=f'../data/nursingv1_andrew',window_size=101):
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
def load_and_window_nursing_list_for_convolution(idxs,data_dir=f'../data/nursingv1',label_dir=f'../data/nursingv1_andrew',window_size=101):
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
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,stratify=y)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=.25,stratify=y_train)
    trainloader = DataLoader(TensorDataset(X_train,y_train),batch_size=64,shuffle=True)
    devloader = DataLoader(TensorDataset(X_dev,y_dev),batch_size=64,shuffle=True)
    testloader = DataLoader(TensorDataset(X_test,y_test),batch_size=64,shuffle=True)
    return trainloader,devloader,testloader
def load_data_convolution(window_size=101,n=71):
    train_idx = range(n)
    X,y = load_and_window_nursing_list_for_convolution(train_idx,window_size=window_size)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,stratify=y)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=.25,stratify=y_train)
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