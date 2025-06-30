import torch
import numpy as np
from mesh import prep_mesh
import pyrender

num_samples = 25000

samples, coords = prep_mesh('cleaned2.obj', unit=True, num_samples=num_samples)


colours = np.zeros(coords.shape)
#colours[samples < 0, 2] = 1
#colours[samples > 0, 0] = 1
cloud = pyrender.Mesh.from_points(coords, colors=colours)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)