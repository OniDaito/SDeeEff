"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

loader.net - data loader for our SDF program.

"""

import torch
import math
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, data):
        super(TrainDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        x = self.data[ind][1:] / 255.0
        y = self.data[ind][0]
        return x, y

class TestDataset(TrainDataset):
    def __getitem__(self, ind):
        x = self.data[ind] / 255.0
        return x

class TestSDF():
    ''' A simple test SDF that returns a distance from a simple scene.'''

    def __init__(self):
        self.sphere = (0, 0, 0, 0.5)  # x, y, z and radius

    def get_distance(self, coord):
        x =  self.sphere[0] - coord[0]
        y =  self.sphere[1] - coord[1]
        z =  self.sphere[2] - coord[2]

        d = math.sqrt(x*x + y*y + z*z) - self.sphere[3]

        return torch.tensor([d], dtype=torch.float32)


'''
train_data = pd.read_csv('data/train.csv').values
test_data  = pd.read_csv('data/test.csv' ).values # (28000, 784)

print(train_data.shape) # (42000, 785)
print(test_data.shape)  # (28000, 784)



train_set = TrainDataset(train_data)
test_set  = TestDataset(test_data)

batch_size = 512
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
'''