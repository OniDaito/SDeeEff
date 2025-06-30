import torch
import numpy as np
from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere, scale_to_unit_sphere
import trimesh
from torch.utils.data import Dataset


class SDFFitting(Dataset):
    def __init__(self, filename, samples, unit=True):
        super().__init__()
        mesh = trimesh.load(filename)
        mesh = scale_to_unit_sphere(mesh)
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

def prep_mesh(filename, unit=True, num_samples=256000):
    mesh = trimesh.load(filename)
    radius = 10.0

    if unit:
        mesh = scale_to_unit_sphere(mesh) # Our meshes might be bigger? Of course, we might loose detail at this scale thanks to floating point res.
        radius = 1.0

    #surface_point_cloud = get_surface_point_cloud(
    #    mesh, surface_point_method="scan",  bounding_radius=radius, scan_count=100, scan_resolution=400
    #)

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method='sample')

    coords, samples = surface_point_cloud.sample_sdf_near_surface(
        num_samples // 2, use_scans=False, sign_method="normal"
    )
    
    sphere_points = None

    if unit:
        sphere_points = sample_uniform_points_in_unit_sphere(num_samples//2)
    else:
        sphere_points = sample_uniform_points_in_sphere(num_samples, radius)
    
    samples = surface_point_cloud.get_sdf_in_batches(
        sphere_points, use_depth_buffer=False
    )
    coords = np.concatenate([coords, sphere_points]).astype(
        np.float32
    )
    samples = np.concatenate([samples, samples]).astype(np.float32)
    samples = torch.from_numpy(samples)[:, None]
    coords = torch.from_numpy(coords)

    return (samples, coords)