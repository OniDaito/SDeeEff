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
import random
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    def __init__(self, data):
        super(SDFDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        x = self.data[ind][:3]
        y = self.data[ind][3]
        return x, y

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

    def get_volume(self, coords):
        volume = torch.zeros([len(coords)])
        
        for step, coord in enumerate(coords):
            d = self.get_distance(coord)[0]
            volume[step] = d

        return volume

    def random_sample(self, num_samples):
        samples = []

        for i in range(num_samples):
            x = random.random() * 2.0 - 1.0
            y = random.random() * 2.0 - 1.0
            z = random.random() * 2.0 - 1.0
            d = self.get_distance((x, y, z))
            samples.append((x, y, z, d))

        return samples