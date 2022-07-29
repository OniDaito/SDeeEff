"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

train.py - train our MLP
"""

from model import SDF
from loader import TestSDF
import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import random
from tqdm import tqdm
from tkinter import Tk
from viz import SDFView

def criterion(pred, target):
    loss = nn.L1Loss()
    #loss = nn.MSELoss()
    return loss(pred, target)  

def train(args, model, optimiser, device):
    model.train()
    loader = TestSDF()
    losses = []

    coords = []

    for i in range(args.samples):
        x = random.random() * 2.0 - 1.0
        y = random.random() * 2.0 - 1.0
        z = random.random() * 2.0 - 1.0
        # x = random.gauss(0.0, 0.2)
        # y = random.gauss(0.0, 0.2)
        # z = random.gauss(0.0, 0.2)
        coords.append((x, y, z))

    for epoch in range(args.epochs):

      
    
        #for batch_num, input_data in enumerate(train_loader):
        #target = loader.get_volume(coords).to(device=device)

        # Create a volume we shall fill
        #pred = torch.zeros([len(coords)], dtype=torch.float32, device=device)

        # Loop over the SDF space in discrete steps 
        for step, coord in enumerate(coords):
            optimiser.zero_grad()
            x = torch.tensor(coord, dtype=torch.float32, device=device)
            target = loader.get_distance(coord).to(device=device)

            output = model(x)
            #pred[step] = output

            # print(pred, target)
            loss = criterion(output, target)
            loss.backward()
            losses.append(loss.item())
      
            optimiser.step()

            #if batch_num % 40 == 0:
            #batch_num = 0
            #print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Train")

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--rez",
        type=int,
        default=128,
        help="How many voxels along each dimension from -1 to 1 (default: 64)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="How many samples of the volume to take for each train datum (default: 500)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print("Using device", device)

    # Initialise our models
    model = SDF().to(device)
    optimiser = torch.optim.Adam(model.parameters())
    print(model)
    # now train
    train(args, model, optimiser, device)
    print("Training Done")
    model = model.to('cpu')
    model.eval()

    root = Tk()
    ex = SDFView(root, model, rez=args.rez)
    root.geometry(str(args.rez) + "x" + str(args.rez))
    root.mainloop()