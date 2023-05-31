"""
author: Andrew Smith
date: Mar 21
description:
"""
import json
import argparse
from datetime import datetime
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from lib.utils import *
from lib.models import *

# argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-p", "--project", type=str, default='project',help="Project directory name")
parser.add_argument("-b", "--batch", type=int, default=64,help="Batch Size")
parser.add_argument("-l", "--lr", type=float, default=3e-4,help="Learning Rate")
parser.add_argument("-o", "--dropout", type=float, default=.2,help="Dropout")
parser.add_argument("-i", "--hidden", type=int, default=32,help="Hidden Layer Neurons")
parser.add_argument("-u", "--directory", type=str, default='.',help="Data Directory",required=False)
args = parser.parse_args()

data_dir = args.directory
current_date = str(datetime.now()).replace(' ','_')
project_dir = args.project
early_stopping = True
patience = 20
window_size = 101

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
config = {
    'BATCH_SIZE':args.batch,
    'EPOCHS':args.epochs,
    'RESUME':args.resume,
    'START_TIME':current_date,
    'LEARNING_RATE':args.lr,
    'DATA_DIR':data_dir,
}

if not os.path.isdir(project_dir):
    os.system(f'mkdir {project_dir}')
if not os.path.isdir(f'{project_dir}/{current_date}'):
    os.system(f'mkdir {project_dir}/{current_date}')

trainloader,devloader,testloader = load_data_convolution()
model = FCN()
params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)
if(config['RESUME']):
    print("Resuming previous training")
    if os.path.exists(f'{project_dir}/model.pt'):
        model.load_state_dict(torch.load(f=f'{project_dir}/model.pt'))
    else:
        print("Model file does not exist.")
        print("Exiting because resume flag was given and model does not exist. Either remove resume flag or move model to directory.")
        exit(0)
    with open(f'{project_dir}/config.json','r') as f:
        previous_config = json.load(f)
    config['START_EPOCH'] = previous_config['END_EPOCH'] + 1
    config['best_dev_loss'] = previous_config['best_dev_loss']
else:
    config['START_EPOCH'] = 0

config['END_EPOCH'] = config['START_EPOCH'] + config['EPOCHS'] - 1

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

loss_tr = []
loss_dev = []
patiencei = 0
best_model_index = 0
pbar = tqdm(range(config['EPOCHS']))
for epoch in pbar:
    # train loop
    model.train()
    loss_tr_total = 0
    for (X_tr,y_tr) in trainloader:
        X_tr,y_tr = X_tr.to(device),y_tr.to(device)
        logits = model(X_tr)
        loss = criterion(logits,y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr_total += loss.item()
    loss_tr.append(loss_tr_total/len(trainloader))

    # dev loop
    model.eval()
    loss_dev_total = 0
    for (X_dv,y_dv) in devloader:
        X_dv,y_dv = X_dv.to(device),y_dv.to(device)
        logits = model(X_dv)
        loss = criterion(logits,y_dv)
        loss_dev_total += loss.item()
    loss_dev.append(loss_dev_total/len(devloader))
    if(early_stopping):
        if(epoch == 0):
            # first epoch
            config['best_dev_loss'] = loss_dev[-1]
        else:
            if(loss_dev[-1] < config['best_dev_loss']):
                # new best loss
                config['best_dev_loss'] = loss_dev[-1]
                patiencei = 0
                best_model_index = epoch
                torch.save(model.state_dict(), f=f'{project_dir}/best_model.pt')
                test_evaluation(devloader,model,criterion,dir=f'{project_dir}',filename='cm_best.jpg',device=device)
            else:
                patiencei += 1
                if(patiencei == 20):
                    print("early stopping")
                    break
    best = config['best_dev_loss']
    pbar.set_description(f'\033[94m Train Loss: {loss_tr[-1]:.4f}\033[93m Dev Loss: {loss_dev[-1]:.4f}\033[92m Best Loss: {best:.4f} \033[91m Stopping: {patiencei}\033[0m')


    # plot recent loss
    plt.plot(loss_tr[-30:])
    plt.plot(loss_dev[-30:])
    plt.savefig(f'{project_dir}/{current_date}/loss_last_30.jpg')
    plt.close()

    # plot all loss
    plt.plot(loss_tr)
    plt.plot(loss_dev)
    plt.savefig(f'{project_dir}/{current_date}/loss_all_epochs.jpg')
    plt.close()

    # save on checkpoint
    torch.save(model.state_dict(), f=f'{project_dir}/{current_date}/{epoch}.pt')

test_evaluation(devloader,model,criterion,dir=f'{project_dir}/{current_date}',filename='cm.jpg',device=device)

torch.save(model.state_dict(), f=f'{project_dir}/{current_date}/model.pt')
torch.save(model.state_dict(), f=f'{project_dir}/model.pt')

# save config
with open(f'{project_dir}/config.json', 'w') as f:
     f.write(json.dumps(config))
with open(f'{project_dir}/{current_date}/config.json', 'w') as f:
     f.write(json.dumps(config))