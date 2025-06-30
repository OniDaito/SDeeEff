"""A program based on Blackle Mori's work on the SDF fields.
I've tidied it up a bit and split things out to make it easier
to read. It runs as a stand-alone python script and not an
ipynb for ease of use.

https://github.com/marian42/mesh_to_sdf

"""

import torch
import os
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from math import sqrt
import numpy as np
import re
from model import Siren
import argparse
from mesh import SDFFitting


def dump_data(dat):
    dat = dat.cpu().detach().numpy()
    return dat


def print_vec4(ws):
    # Can set .2. or .3 if we like?
    vec = "vec4(" + ",".join(["{:.2f}".format(w) for w in ws]) + ")"
    vec = re.sub(r"\b0\.", ".", vec)
    return vec


def print_mat4(ws):
    # Can set .2. or .3 if we like?
    mat = (
        "mat4("
        + ",".join(["{:.2f}".format(w) for w in np.transpose(ws).flatten()])
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

    output_string = ""

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
        output_string += line

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
            output_string += line

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

        output_string += line + "{:0.3f}".format(out_bias[outf]) + ";"

    return output_string


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_siren(
    dataloader,
    hidden_features,
    hidden_layers,
    omega,
    total_steps=200000,
    learning_rate=5e-4,
    load_dict="",
    sched=False,
    interval=100,
    batch_size=65536,
    weight_decay=0.01,
    decay_interval=5000
):
    """Train a new siren model on the new model we've loaded."""
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    model = Siren(
        in_features=3,
        out_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        outermost_linear=True,
        omega=omega,
        first_linear=False,
    )

    if os.path.exists(load_dict):
        model.load_state_dict(torch.load(load_dict))

    model = model.cuda()
    #optim = torch.optim.Adagrad(lr=learning_rate, params=img_curr.parameters())
    # optim = torch.optim.Adam(lr=1e-3, params=img_curr.parameters())
    optim = torch.optim.Adam(
        lr=learning_rate, params=model.parameters(), weight_decay=weight_decay 
    )

    perm = torch.randperm(model_input.size(1))
    scheduler = ExponentialLR(optim, gamma=0.999)
    best_loss = 1.0

    for step in range(total_steps):
        if step == decay_interval:
            optim.param_groups[0]["weight_decay"] = 0.0

        idx = step % int(model_input.size(1) / batch_size)
        model_in = model_input[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        truth = ground_truth[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        model_output, coords = model(model_in)

        loss = (model_output - truth) ** 2
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step % interval) == interval - 1:

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "current_model.pt")
                output_string = serialize_to_glsl(model, "f")

                # Now load the template and replace <--FRAGMENT--> with our string
                final_shader = ""

                with open("template.glsl", "r") as f:
                    template = f.read()
                    final_shader = template.replace("<--FRAGMENT-->", output_string)
                
                with open("current_fragment.glsl", "w") as f:
                    f.write(final_shader)

            lr = get_lr(optim)
            perm = torch.randperm(model_input.size(1))
            print("Step %d, Current loss %0.6f, lr %0.6f" % (step, loss, lr))

            if sched:
                scheduler.step()

    # Save the last model as well as the best, just in case
    torch.save(model.state_dict(), "final_model.pt")
    output_string = serialize_to_glsl(model, "f")

    # Now load the template and replace <--FRAGMENT--> with our string
    final_shader = ""

    with open("template.glsl", "r") as f:
        template = f.read()
        final_shader = template.replace("<--FRAGMENT-->", output_string)
    
    with open("final_fragment.glsl", "w") as f:
        f.write(final_shader)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF SIREN training")
    parser.add_argument("--obj", default="bunny2.obj")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--decay-interval", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--load", default="")
    parser.add_argument("--omega", type=int, default=30)
    parser.add_argument("--unit", dest="unit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=65535)
    parser.add_argument("--sched", action="store_true")
    args = parser.parse_args()

    sdf = SDFFitting(args.obj, args.samples, unit=True)
    sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)

    # Default is 16, 2, 15
    sdf_siren = train_siren(
        sdfloader,
        args.features,
        args.layers,
        args.omega,
        total_steps=args.steps,
        learning_rate=args.lr,
        load_dict=args.load,
        sched=args.sched,
        interval=args.interval,
        batch_size=args.batch_size,
        decay_interval=args.decay_interval
    )
