"""A program based on Blackle Mori's work on the SDF fields.
I've tidied it up a bit and split things out to make it easier
to read. It runs as a stand-alone python script and not an
ipynb for ease of use.

https://github.com/marian42/mesh_to_sdf

"""

import torch
import os
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from math import sqrt
import numpy as np
from mesh_to_sdf import get_surface_point_cloud, sample_sdf_near_surface
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere, scale_to_unit_sphere
import trimesh
import re
from model import Siren
import argparse


# Overriding for non-unit sphere
def sample_uniform_points_in_sphere(amount, radius):
    sphere_points = np.random.uniform(-radius, radius, size=(amount * 2 + 20, 3))
    sphere_points = sphere_points[np.linalg.norm(sphere_points, axis=1) < radius] # radius was 1

    points_available = sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = sphere_points
        result[points_available:, :] = sample_uniform_points_in_sphere(amount - points_available, radius)
        return result
    else:
        return sphere_points[:amount, :]


class SDFFitting(Dataset):
    def __init__(self, filename, num_samples, unit=False):
        super().__init__()
        mesh = trimesh.load(filename)

        if unit:
            mesh = scale_to_unit_sphere(mesh) # Our meshes might be bigger? Of course, we might loose detail at this scale thanks to floating point res.
        
        # mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0
        surface_point_cloud = get_surface_point_cloud(
            mesh, surface_point_method="sample"
        )

        #self.coords, self.samples = sample_sdf_near_surface(mesh, number_of_points=num_samples // 2)

        self.coords, self.samples = surface_point_cloud.sample_sdf_near_surface(
            num_samples // 2, use_scans=False, sign_method="normal"
        )
        
        unit_sphere_points = sample_uniform_points_in_sphere(num_samples, 10.0)
        samples = surface_point_cloud.get_sdf_in_batches(
            unit_sphere_points, use_depth_buffer=False
        )
        self.coords = np.concatenate([self.coords, unit_sphere_points]).astype(
            np.float32
        )
        self.samples = np.concatenate([self.samples, samples]).astype(np.float32)
        self.samples = torch.from_numpy(self.samples)[:, None]
        self.coords = torch.from_numpy(self.coords)
        print(self.coords.shape, self.samples.shape)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.samples


def dump_data(dat):
    dat = dat.cpu().detach().numpy()
    return dat


def print_vec4(ws):
    # Can set .2. or .3 if we like?
    vec = "vec4(" + ",".join(["{0:.3f}".format(w) for w in ws]) + ")"
    vec = re.sub(r"\b0\.", ".", vec)
    return vec


def print_mat4(ws):
    # Can set .2. or .3 if we like?
    mat = (
        "mat4("
        + ",".join(["{0:.3f}".format(w) for w in np.transpose(ws).flatten()])
        + ")"
    )
    mat = re.sub(r"\b0\.", ".", mat)
    return mat


def serialize_to_glsl(siren, varname):
    """Given the model, serialise out to text, ready for cutting and pasting into
    our sdf.glsl file."""
    # first layer
    omega = siren.omega
    chunks = int(siren.hidden_features / 4)
    lin = siren.net[0] if siren.first_linear else siren.net[0].linear
    in_w = dump_data(lin.weight)
    in_bias = dump_data(lin.bias)
    om = 1 if siren.first_linear else omega

    for row in range(chunks):
        if siren.first_linear:
            line = "vec4 %s0_%d=(" % (varname, row)
        else:
            line = "vec4 %s0_%d=sin(" % (varname, row)

        for ft in range(siren.in_features):
            feature = x_vec = in_w[row * 4 : (row + 1) * 4, ft] * om
            line += ("p.%s*" % ["y", "z", "x"][ft]) + print_vec4(feature) + "+"

        bias = in_bias[row * 4 : (row + 1) * 4] * om
        line += print_vec4(bias) + ");"
        print(line)

    # hidden layers
    for layer in range(siren.hidden_layers):
        layer_w = dump_data(siren.net[layer + 1].linear.weight)
        layer_bias = dump_data(siren.net[layer + 1].linear.bias)

        for row in range(chunks):
            line = ("vec4 %s%d_%d" % (varname, layer + 1, row)) + "=sin("

            for col in range(chunks):
                mat = layer_w[row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4] * omega
                line += (
                    print_mat4(mat) + ("*%s%d_%d" % (varname, layer, col)) + "+\n    "
                )

            bias = layer_bias[row * 4 : (row + 1) * 4] * omega
            line += print_vec4(bias) + ")/%0.1f+%s%d_%d;" % (
                sqrt(layer + 1),
                varname,
                layer,
                row,
            )
            print(line)

    # output layer
    out_w = dump_data(siren.net[-1].weight)
    out_bias = dump_data(siren.net[-1].bias)

    for outf in range(siren.out_features):
        line = "return "

        for row in range(chunks):
            vec = out_w[outf, row * 4 : (row + 1) * 4]
            line += (
                ("dot(%s%d_%d," % (varname, siren.hidden_layers, row))
                + print_vec4(vec)
                + ")+\n    "
            )

        print(line + "{:0.3f}".format(out_bias[outf]) + ";")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_siren(
    dataloader,
    hidden_features,
    hidden_layers,
    omega,
    total_steps=20000,
    learning_rate=1e-4,
    load_dict=""
):
    """Train a new siren model on the new model we've loaded."""
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    img_curr = Siren(
        in_features=3,
        out_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        outermost_linear=True,
        omega=omega,
        first_linear=False,
    )

    if os.path.exists(load_dict):
        img_curr.load_state_dict(torch.load(load_dict))

    img_curr.cuda()
    # optim = torch.optim.Adagrad(params=img_curr.parameters())
    # optim = torch.optim.Adam(lr=1e-3, params=img_curr.parameters())
    optim = torch.optim.Adam(
        lr=learning_rate, params=img_curr.parameters(), weight_decay=0.01 # was 0.01
    )
    perm = torch.randperm(model_input.size(1))

    update = int(total_steps / 50)
    batch_size = 256 * 256

    scheduler = ExponentialLR(optim, gamma=0.9)

    for step in range(total_steps):
        #if step == 500:
        #    optim.param_groups[0]["weight_decay"] = 0.0

        idx = step % int(model_input.size(1) / batch_size)
        model_in = model_input[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        truth = ground_truth[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        model_output, coords = img_curr(model_in)

        loss = (model_output - truth) ** 2
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step % update) == update - 1:
            lr = get_lr(optim)
            perm = torch.randperm(model_input.size(1))
            print("Step %d, Current loss %0.6f, lr %0.6f" % (step, loss, lr))
            #scheduler.step()

    return img_curr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF SIREN training")
    parser.add_argument("--obj", default="bunny2.obj")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--samples", type=int, default=256 * 256 * 4)
    parser.add_argument("--load", default="")
    args = parser.parse_args()

    sdf = SDFFitting(args.obj, args.samples)
    sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)

    # Default is 16, 4, 15
    sdf_siren = train_siren(
        sdfloader, 16, 4, 15, total_steps=args.steps, learning_rate=args.lr, load_dict=args.load
    )

    torch.save(sdf_siren.state_dict(), "latest_model.pt")
    serialize_to_glsl(sdf_siren, "f")
