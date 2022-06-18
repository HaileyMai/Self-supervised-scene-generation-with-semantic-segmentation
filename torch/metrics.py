"""
Assumes the output of data_util.save_predictions() and computes metrics to quantify performance.

Computed metrics based on SPSG paper:
    Based on rendered views and original color images:
        Structure Similarity Image Metric (SSIM)
        Frechet Inception Distance (FID)
        Feature-l1: l1 distance between feature embeddings under a specific network
    Geometric Reconstruction:
        Intersection over Union (IoU)
        Recall
        Chamfer Distance
For semantics:
    TODO
"""

import argparse
import os
import random
from unicodedata import bidirectional
from PIL import Image
import numpy as np

import plyfile
import torch

from SSIM_PIL import compare_ssim
from chamferdist import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the files")
parser.add_argument("--voxel_size", type=float, default=2.0, help="voxel_size used for voxelization in cm")
parser.add_argument("--num_samples", type=int, default=30000, help="number of points sampled from mesh for chamfer distance")

args = parser.parse_args()
print(args)


def sample_points(ply_file): #TODO is this the correct way to sample?
    vertices = ply_file.elements[0]
    faces = ply_file.elements[1]

    samples = []
    for i in range(args.num_samples):
        face_id = random.randint(0, faces.count - 1)
        face_vertices, _, _, _ = faces.data[face_id]
        face_vertices = [vertices[i] for i in face_vertices]

        corners = []
        for i in range(3):
            corner = [face_vertices[i]["x"], face_vertices[i]["y"], face_vertices[i]["z"]]
            corners += [corner]
        
        factors = np.random.dirichlet(np.ones(3))

        z = corners[0][2] * factors[0] + corners[1][2] * factors[1] + corners[2][2] * factors[2]
        y = corners[0][1] * factors[0] + corners[1][1] * factors[1] + corners[2][1] * factors[2]
        x = corners[0][0] * factors[0] + corners[1][0] * factors[1] + corners[2][0] * factors[2]

        samples += [[x, y, z]]
    
    return samples



files = set(["_".join(f.split("_")[:-1]) for f in os.listdir(args.path)])
chamfer = ChamferDistance()

ssim_sum = 0
chamfer_sum = 0
for file in files:
    target_color_image = np.array(Image.open(file + "_target.png"))
    output_color_image = np.array(Image.open(file + "_pred.png"))

    ssim = compare_ssim(target_color_image, output_color_image) #TODO value for tile_size ? 
    ssim_sum += ssim

    #TODO voxelize mesh (with interior?) -> for IoU and Recall
    target_mesh = plyfile.PlyData.read(file + "_target-mesh.ply")
    pred_mesh = plyfile.PlyData.read(file + "_pred-mesh.ply")
    
    #Chamfer Distance
    target_points = torch.tensor(sample_points(target_mesh)).reshape(1, args.num_samples, 3)
    pred_points = torch.tensor(sample_points(pred_mesh)).reshape(1, args.num_samples, 3)
    
    chamfer_dist = chamfer(target_points, pred_points, bidirectional=True).item()
    chamfer_sum += chamfer_dist

print(f"SSIM: {ssim_sum / len(files)}")
print(f"Chamfer Distance: {chamfer_sum / len(files)}")