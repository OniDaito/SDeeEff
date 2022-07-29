"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

model.net - our sine mlp representation of an SDF
"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class SDF(nn.Module):
    """ Input is an X, Y, Z co-ordinate. Output is a distance. """
    def __init__(self):
        super(SDF, self).__init__()
        hfeatures = 256
        use_bias = True # Apparently, we really do need a bias! :D 
        self.layer0 = nn.Linear(3, hfeatures, bias=use_bias)
        self.layer1 = nn.Linear(hfeatures, hfeatures, bias=use_bias)
        self.layer2 = nn.Linear(hfeatures, hfeatures, bias=use_bias)
        self.layer3 = nn.Linear(hfeatures, hfeatures, bias=use_bias)
        self.layer4 = nn.Linear(hfeatures, hfeatures, bias=use_bias)
        self.layer5 = nn.Linear(hfeatures, 1, bias=use_bias)
    
    def forward(self, x):
        y = torch.sin(self.layer0(x))
        y = torch.sin(self.layer1(y))
        y = torch.sin(self.layer2(y))
        y = torch.sin(self.layer3(y))
        y = torch.sin(self.layer4(y))
        y = self.layer5(y)
        return y

    def get_distance(self, coord, device):
        assert(len(coord) == 3)
        coord = torch.tensor([coord[0], coord[1], coord[2]], dtype=torch.float32, device=device)
        res = self.forward(coord)
        return float(res[0])
