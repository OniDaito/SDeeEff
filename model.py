"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

sdf.net - the core of the SDF program

"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class SDF(nn.Module):
    def __init__(self):
        super(SDF, self).__init__()
        self.layer0 = nn.Linear(10, 10)
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)
    
    def forward(self, x):
        y = torch.sin(self.layer0(x))
        y = torch.sin(self.layer1(y))
        y = torch.sin(self.layer2(y))
        y = torch.sin(self.layer3(y))

        return y