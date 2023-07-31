from torch import nn
from torch.nn.functional import relu
class MLP(nn.Module):
    def __init__(self,window_size) -> None:
        super().__init__()
        self.d1 = nn.Dropout1d(p=.1)
        self.fc1 = nn.Linear(window_size*3,500)

        self.d2 = nn.Dropout1d(p=.2)
        self.fc2 = nn.Linear(500,500)

        self.d3 = nn.Dropout1d(p=.2)
        self.fc3 = nn.Linear(500,500)

        self.d4 = nn.Dropout1d(p=.3)
        self.fc4 = nn.Linear(500,1)

    def forward(self,x):
        x = self.d1(x)
        x = self.fc1(x)
        x = relu(x)

        x = self.d2(x)
        x = self.fc2(x)
        x = relu(x)

        x = self.d3(x)
        x = self.fc3(x)
        x = relu(x)

        x = self.d4(x)
        x = self.fc4(x)

        return x

class FCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=3,out_channels=128,kernel_size=8,stride=1)
        self.bn1 = nn.BatchNorm1d(num_features=128)

        self.c2 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=5,stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=256)

        self.c3 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.gap = nn.AvgPool1d(kernel_size=88)
        self.fc1 = nn.Linear(in_features=128,out_features=1)
    def forward(self,x):
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = relu(x)

        x = self.gap(x)
        x = self.fc1(x.squeeze(2))
        return x
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=8,padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)

        self.c2 = nn.Conv1d(in_channels=out_channels,out_channels=out_channels,kernel_size=5,padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.c3 = nn.Conv1d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding='same')
        self.bn3 = nn.BatchNorm1d(num_features=out_channels)

        self.c4 = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.bn4 = nn.BatchNorm1d(num_features=out_channels)
    def forward(self,x):
        residual = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)

        residual = self.c4(residual)
        residual = self.bn4(residual)

        x = x + residual
        x = relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = ResBlock(in_channels=3,out_channels=64)
        self.block2 = ResBlock(in_channels=64,out_channels=128)
        self.gap = nn.AvgPool1d(kernel_size=101)
        self.fc1 = nn.Linear(in_features=128,out_features=1)
    def forward(self,x,classification=True):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        if(classification):
            x = self.fc1(x.squeeze(2))
        return x