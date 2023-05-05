from os.path import join
from torch.utils.data import Dataset
from torch import load
class Dataset2p0(Dataset):
    def __init__(self,dir,labels):
        self.labels = load(f'{labels}')
        self.dir = dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = join(self.dir, str(idx)+".pt")
        X = load(path)
        y = self.labels[idx]

        return (X,y)