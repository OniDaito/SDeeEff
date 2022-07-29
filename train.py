"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

train.py - train our MLP

https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
"""

from model import SDF
from loader import TestSDF
import torch
import torch.nn as nn
import argparse
from tkinter import Tk
from tqdm import tqdm
from tkinter import Tk
from viz import SDFView
from loader import SDFDataset
from torch.utils.data import DataLoader

def criterion(pred, target):
    loss = nn.L1Loss()
    #loss = nn.MSELoss()
    return loss(pred, target)  


def create_data(args):
    field = TestSDF()

    train_data = torch.tensor(field.random_sample(args.samples))
    test_data  = torch.tensor(field.random_sample(100))

    print(train_data.shape)
    print(test_data.shape)

    train_set = SDFDataset(train_data)
    test_set  = SDFDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def train(args, model, train_loader, optimiser, viewer, device):
    model.train()
    losses = []

    for epoch in range(args.epochs):

        for batch_num, input_data in enumerate(train_loader):
            optimiser.zero_grad()
            x, y = input_data
            x = x.to(device)
            y = y.to(device).squeeze()
            output = model(x).squeeze()
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimiser.step()

            #if batch_num % 40 == 0:
            #batch_num = 0
            #print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))

        model.eval()
        viewer.raycast(model, device)
        viewer.update()
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF Train")

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
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

    # Create the starting window
    root = Tk()
    viewer = SDFView(root, rez=args.rez)
    root.geometry(str(args.rez) + "x" + str(args.rez))

    # Create the data
    train_loader, test_loader = create_data(args)

    # Initialise our models
    model = SDF().to(device)
    optimiser = torch.optim.Adam(model.parameters())
    print(model)

    # now train
    train(args, model, train_loader, optimiser, viewer, device)
    root.mainloop()
