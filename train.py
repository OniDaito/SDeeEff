""" A program based on Blackle Mori's work on the SDF fields."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from math import sqrt
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere

import trimesh
import pyrender
import re


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid




class SDFFitting(Dataset):
    def __init__(self, filename, samples):
        super().__init__()
        mesh = trimesh.load(filename)
        #mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0
        surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method='sample')

        self.coords, self.samples = surface_point_cloud.sample_sdf_near_surface(samples//2, use_scans=False, sign_method='normal')
        unit_sphere_points = sample_uniform_points_in_unit_sphere(samples//2)
        samples = surface_point_cloud.get_sdf_in_batches(unit_sphere_points, use_depth_buffer=False)
        self.coords = np.concatenate([self.coords, unit_sphere_points]).astype(np.float32)
        self.samples = np.concatenate([self.samples, samples]).astype(np.float32)
        
        #colors = np.zeros(self.coords.shape)
        #colors[self.samples < 0, 2] = 1
        #colors[self.samples > 0, 0] = 1
        #cloud = pyrender.Mesh.from_points(self.coords, colors=colors)
        #scene = pyrender.Scene()
        #scene.add(cloud)
        #viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

        self.samples = torch.from_numpy(self.samples)[:,None]
        self.coords = torch.from_numpy(self.coords)
        print(self.coords.shape, self.samples.shape)
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.samples
    
sdf = SDFFitting("bunny2.obj", 256*256*4)
sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)


def dump_data(dat):
  dat = dat.cpu().detach().numpy()
  return dat

def print_vec4(ws):
  vec = "vec4(" + ",".join(["{0:.2f}".format(w) for w in ws]) + ")"
  vec = re.sub(r"\b0\.", ".", vec)
  return vec

def print_mat4(ws):
  mat = "mat4(" + ",".join(["{0:.2f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
  mat = re.sub(r"\b0\.", ".", mat)
  return mat

def serialize_to_shadertoy(siren, varname):
  #first layer
  omega = siren.omega
  chunks = int(siren.hidden_features/4)
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
        feature = x_vec = in_w[row*4:(row+1)*4,ft]*om
        line += ("p.%s*" % ["y","z","x"][ft]) + print_vec4(feature) + "+"
    bias = in_bias[row*4:(row+1)*4]*om
    line += print_vec4(bias) + ");"
    print(line)

  #hidden layers
  for layer in range(siren.hidden_layers):
    layer_w = dump_data(siren.net[layer+1].linear.weight)
    layer_bias = dump_data(siren.net[layer+1].linear.bias)
    for row in range(chunks):
      line = ("vec4 %s%d_%d" % (varname, layer+1, row)) + "=sin("
      for col in range(chunks):
        mat = layer_w[row*4:(row+1)*4,col*4:(col+1)*4]*omega
        line += print_mat4(mat) + ("*%s%d_%d"%(varname, layer, col)) + "+\n    "
      bias = layer_bias[row*4:(row+1)*4]*omega
      line += print_vec4(bias)+")/%0.1f+%s%d_%d;"%(sqrt(layer+1), varname, layer, row)
      print(line)

  #output layer
  out_w = dump_data(siren.net[-1].weight)
  out_bias = dump_data(siren.net[-1].bias)
  for outf in range(siren.out_features):
    line = "return "
    for row in range(chunks):
      vec = out_w[outf,row*4:(row+1)*4]
      line += ("dot(%s%d_%d,"%(varname, siren.hidden_layers, row)) + print_vec4(vec) + ")+\n    "
    print(line + "{:0.3f}".format(out_bias[outf])+";")


def train_siren(dataloader, hidden_features, hidden_layers, omega):
  model_input, ground_truth = next(iter(dataloader))
  model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

  img_curr = Siren(in_features=3, out_features=1, hidden_features=hidden_features, 
                   hidden_layers=hidden_layers, outermost_linear=True, omega=omega, first_linear=False)
  img_curr.cuda()
  #optim = torch.optim.Adagrad(params=img_curr.parameters())
  #optim = torch.optim.Adam(lr=1e-3, params=img_curr.parameters())
  optim = torch.optim.Adam(lr=1e-4, params=img_curr.parameters(), weight_decay=.01)
  perm = torch.randperm(model_input.size(1))

  total_steps = 20000
  update = int(total_steps/50)
  batch_size = 256*256
  for step in range(total_steps):
    if step == 500:
        optim.param_groups[0]['weight_decay'] = 0.
    idx = step % int(model_input.size(1)/batch_size)
    model_in = model_input[:,perm[batch_size*idx:batch_size*(idx+1)],:]
    truth = ground_truth[:,perm[batch_size*idx:batch_size*(idx+1)],:]
    model_output, coords = img_curr(model_in)

    loss = (model_output - truth)**2
    loss = loss.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
           
    if (step % update) == update-1:
      perm = torch.randperm(model_input.size(1))
      print("Step %d, Current loss %0.6f" % (step, loss))

  return img_curr

sdf_siren = train_siren(sdfloader, 16, 2, 15)
serialize_to_shadertoy(sdf_siren, "f")
