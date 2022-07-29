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
        self.layer0 = nn.Linear(3, 5)
        self.layer1 = nn.Linear(5, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 1)
    
    def forward(self, x):
        y = torch.sin(self.layer0(x))
        y = torch.sin(self.layer1(y))
        y = torch.sin(self.layer2(y))
        y = self.layer3(y)

        return y