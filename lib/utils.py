import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,recall_score,precision_score
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix
from seaborn import heatmap
from tqdm import tqdm
from lib.env import *

def window_epoched_signal(X,windowsize,n_channels,zero_padding=True):
    """
    only works for odd windows, puts label at center
    """
    if(zero_padding):
        X = torch.cat([torch.zeros(windowsize//2,n_channels),X,torch.zeros(windowsize//2,n_channels)])
    cat = [X[:-(windowsize-1)]]
    for i in range(1,(windowsize-1)):
        cat.append(X[i:i-(windowsize-1)])
    cat.append(X[(windowsize-1):])
    cat = [xi.unsqueeze(2) for xi in cat]
    X = torch.cat(cat,axis=2).float()
    return X
def cm_grid(y_true,y_pred,save_path='cm.jpg'):
    fig,axes = plt.subplots(2,2,figsize=(5,5))
    heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,fmt='.2f',cbar=False,ax=axes[0][0])
    heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,fmt='.2f',cbar=False,ax=axes[0][1])
    heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='all'),annot=True,fmt='.2f',cbar=False,ax=axes[1][0])
    heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,fmt='.0f',cbar=False,ax=axes[1][1])
    axes[0][0].set_title('Recall')
    axes[0][1].set_title('Precision')
    axes[1][0].set_title('Proportion')
    axes[1][1].set_title('Count')
    axes[0][0].set_xticks([])
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    axes[1][1].set_yticks([])
    # axes[0][0].set_yticklabels(['P','S','W'])
    # axes[1][0].set_yticklabels(['P','S','W'])
    # axes[1][0].set_xticklabels(['P','S','W'])
    # axes[1][1].set_xticklabels(['P','S','W'])
    plt.savefig(save_path,dpi=200,bbox_inches='tight')
def metrics(y_true,y_pred):
    return {
        'precision':precision_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'recall':recall_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'f1':f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    }
def evaluate(dataloader,model,criterion,DEVICE=DEVICE):
    model.eval()
    with torch.no_grad():
        y_true = torch.Tensor()
        y_pred = torch.Tensor()
        y_logits = torch.Tensor()
        loss_total = 0
        for (Xi,yi) in tqdm(dataloader):
            y_true = torch.cat([y_true,yi.round()])
            
            logits = model(Xi)
            loss = criterion(logits,yi)
            loss_total += loss.item()
            
            y_logits = torch.cat([y_logits,torch.sigmoid(logits).detach().cpu()])
            y_pred = torch.cat([y_pred,torch.sigmoid(logits).round().detach().cpu()])
    model.train()
    return loss_total/len(dataloader),metrics(y_true,y_pred),y_true,y_pred,y_logits
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