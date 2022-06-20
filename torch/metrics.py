"""
Assumes the output of data_util.save_predictions() and computes metrics to quantify performance.

Computed metrics based on SPSG paper:
    Based on rendered views and original color images:
        Structure Similarity Image Metric (SSIM) -> this script
        Frechet Inception Distance (FID) -> use: https://github.com/bioinf-jku/TTUR
        Feature-l1: 
    Geometric Reconstruction:
        Intersection over Union (IoU) -> TODO
        Recall -> TODO
        Chamfer Distance -> this script
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

import sample_util

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the files")
parser.add_argument("--voxel_size", type=float, default=2.0, help="voxel_size used for voxelization in cm")
parser.add_argument("--num_samples", type=int, default=30000, help="number of points sampled from mesh for chamfer distance")
parser.add_argument("--inception_v3_path", type=str, default="inception_v3.pth", help="path where the Inception v3 model is saved or should be saved")

args = parser.parse_args()
print(args)


files = set(["_".join(f.split("_")[:-1]) for f in os.listdir(args.path)])
chamfer = ChamferDistance()

#Inception v3 Network used for Feature-l1
if os.path.exists(args.inception_v3_path):
    inception_v3 = torch.load(args.inception_v3_path)
else:
    inception_v3 = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
    torch.save(inception_v3, args.inception_v3_path)
inception_v3.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
inception_v3.to(device)

ssim_sum = 0
chamfer_sum = 0
feature_sum = 0
for file in files:
    target_color_image = np.array(Image.open(file + "_target.png"))
    output_color_image = np.array(Image.open(file + "_pred.png"))

    ssim = compare_ssim(target_color_image, output_color_image) #TODO value for tile_size ? 
    ssim_sum += ssim

    #TODO voxelize mesh (with interior?) -> for IoU and Recall
    
    target_mesh = plyfile.PlyData.read(file + "_target-mesh.ply")
    pred_mesh = plyfile.PlyData.read(file + "_pred-mesh.ply")
    
    #Chamfer Distance
    target_points, _ = sample_util.sample_from_region_ply(target_mesh, args.num_samples, force_total_n=True)
    pred_points, _ = sample_util.sample_from_region_ply(pred_mesh, args.num_samples, force_total_n=True)
    
    target_points = torch.tensor(target_points).reshape(1, args.num_samples, 3).to(device)
    pred_points = torch.tensor(pred_points).reshape(1, args.num_samples, 3).to(device)

    chamfer_dist = chamfer(target_points, pred_points, bidirectional=True).item()
    chamfer_sum += chamfer_dist

    #Feature-l1
    with torch.no_grad():
        t_target = torch.tensor(target_color_image).to(device)
        t_out = torch.tensor(output_color_image).to(device)
        feat_target = inception_v3(t_target)
        feat_out = inception_v3(t_out)

        dist = torch.sum(feat_target - feat_out).item()
        feature_sum += dist



print(f"SSIM: {ssim_sum / len(files)}")
print(f"Chamfer Distance: {chamfer_sum / len(files)}")
print(f"Feature-l1 dist: {feature_sum / len(files)}")