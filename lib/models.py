from torch import nn
from torch.nn.functional import relu
class MLP(nn.Module):
    """
    Musa initial epoch9 model for poster
    """
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits  
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=8,stride=1,padding='same',bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.gap = nn.AvgPool1d(kernel_size=501)

        self.fc1 = nn.Linear(16,4)
        self.do1 = nn.Dropout(p=.5)
        self.fc2 = nn.Linear(4,1)
    
    def forward(self, x):
        x = x.view(-1,1,501)
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.gap(x)
        
        x = self.fc1(x.squeeze(2))
        x = relu(x)
        # x = self.do1(x)
        x = self.fc2(x)

        return x  