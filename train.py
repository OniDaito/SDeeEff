from model import SDF

import torch
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import os
import sys


def criterion(pred, target):
    pass

def train(args, model, optimiser, device):

    model.train()

    for epoch in range(args.epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimiser.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimiser.step()

            if batch_num % 40 == 0:
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
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print("Using device", device)