from model import SDF
from loader import TestSDF
import torch
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import os
import sys
import pandas as pd


def criterion(pred, target):
    pass

def train(args, model, optimiser, device):
    model.train()
    loader = TestSDF()

    # Divide up the space by rez -1, to -1
    coords = []
    step = 2.0 / args.rez
    num_steps = 2.0 / step

    for i in range(num_steps):
        x = i * step
        
        for j in range(num_steps):
            y = j * step
            
            for k in range(num_steps):
                z = k * step
                coords.push_back((x, y, z))


    for epoch in range(args.epochs):
        losses = []
    
        #for batch_num, input_data in enumerate(train_loader):
        
        # Loop over the SDF space in discrete steps 
        for coord in coords:

            optimiser.zero_grad()
            x = torch.tensor(coord, dtype=torch.float32, device=device)
            target = loader.get_distance(coord)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimiser.step()

            #if batch_num % 40 == 0:
            batch_num = 0
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Train")

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.004,
        help="learning rate (default: 0.004)",
    )
    parser.add_argument(
        "--rez",
        type=int,
        default=128,
        help="How many voxels along each dimension from -1 to 1 (default: 128)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print("Using device", device)

    # Initialise our models