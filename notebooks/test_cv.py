import torch
from torch import nn
from tqdm import tqdm

from lib.utils import *
from lib.models import *
window_size = 101

# model = MLP(window_size=window_size)
model = FCN()
model.load_state_dict(torch.load(f=f'fcn_64/best_model.pt'))
trainloader,devloader,testloader = load_data_convolution()
device = 'cuda'
model.to(device)

criterion = nn.BCEWithLogitsLoss()

test_evaluation(trainloader,model,criterion,dir=f'.',filename='train.jpg',device=device)
test_evaluation(devloader,model,criterion,dir=f'.',filename='dev.jpg',device=device)
test_evaluation(testloader,model,criterion,dir=f'.',filename='test.jpg',device=device)