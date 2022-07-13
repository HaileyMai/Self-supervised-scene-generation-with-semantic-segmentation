import struct
import time
from http.client import UNAUTHORIZED

import os
import glob

import collections

import matplotlib
from plyfile import PlyData, PlyElement
from os import path, listdir
import json
import random
import data_util
import torch
import loss
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import zipfile
import utils.marching_cubes.marching_cubes as mc
import sample_util
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
First run download-mp.py -o [output_dir] --type region_segmentations
Category mapping should be taken from https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv
"""


def plot_colortable(colors, title, emptycols=0):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i in range(n):  # skip void
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, eigen13_label[i], fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=colors[i], edgecolor='0.7')
        )

    return fig


def add_semantics_to_chunk_sdf(sdf_file_name, points, cat, index, vis_path=None):
    sdf, world2grid, known, colors, _ = data_util.load_sdf(
        sdf_file_name, load_sparse=False, load_known=True, load_color=True)
    dimz, dimy, dimx = sdf.shape[0], sdf.shape[1], sdf.shape[2]

    x = np.ones((points.shape[0], 4))
    x[:, :3] = points[:, :3]
    x = np.matmul(world2grid, np.transpose(x))
    x = np.transpose(x)[:, :3]

    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    x_floor = np.floor(x)
    x_round = np.rint(x)

    def find_valid_points(x_, concatenate=None):
        lower_bound = np.all(x_ >= 0, axis=1)
        upper_bound = np.all(x_ < [dimx, dimy, dimz], axis=1)
        inbounds = np.logical_and(lower_bound, upper_bound)
        if concatenate is None:
            points_ = np.column_stack((x_, index[cat]))
        else:
            points_ = np.column_stack((x_, np.repeat(index[cat], concatenate)))
        points_ = points_[inbounds].astype(int)  # keep only values that are in the given grid
        return points_

    points = (cube[None, :, :] + x_floor[:, None, :]).reshape(-1, 3)
    points = find_valid_points(points, 8)  # denser: box around points
    points = points[points[:, 3] < 14]
    _, unique = np.unique(points[:, :3], axis=0, return_index=True)
    points = points[unique]
    points_round = find_valid_points(x_round)  # most accurate points

    dense_semantics = 14 * np.ones([dimz, dimy, dimx], dtype=np.uint8)
    dense_semantics[points[:, 2], points[:, 1], points[:, 0]] = points[:, 3]
    dense_semantics[points_round[:, 2], points_round[:, 1], points_round[:, 0]] = points_round[:, 3]

    # visualization for debug
    if vis_path is not None:
        target_for_sdf, target_for_colors = loss.compute_targets(sdf[None, None, ...], args.truncation, True,
                                                                 known[None, None, ...], colors[None, ...])
        dense_sem_color = mapping_color[dense_semantics]
        dense_sem_color = dense_sem_color.astype(np.uint8)
        dense_sem_color = torch.from_numpy(dense_sem_color).byte()
        [sp1, _, sp3] = os.path.splitext(os.path.basename(sdf_file_name))[0].split('__')
        mc.marching_cubes(torch.from_numpy(target_for_sdf[0, 0]), dense_sem_color, isovalue=0, truncation=3,
                          thresh=10, output_filename=os.path.join(vis_path, sp1 + '__sem__' + sp3 + '.ply'))
        target_for_colors = torch.from_numpy(target_for_colors).byte()
        mc.marching_cubes(torch.from_numpy(target_for_sdf[0, 0]), target_for_colors[0], isovalue=0, truncation=3,
                          thresh=10, output_filename=os.path.join(vis_path, sp1 + '__color__' + sp3 + '.ply'))
    return dense_semantics


def extend_sdf_file(segmentation, sdf_file, output_dir, output_vis_dir, region_sampled_points, region_sampled_cat,
                    index):
    # print(f"Now extending {sdf_file}.")
    room, _, sdf_number = os.path.splitext(os.path.basename(sdf_file))[0].split('__')
    region = room.split('room')[-1]

    sdf, world2grid, _, _, _ = data_util.load_sdf(sdf_file, load_sparse=False, load_known=False, load_color=False)
    limits = np.concatenate((np.array([[0, 0, 0, 1]]), np.array([[sdf.shape[2], sdf.shape[1], sdf.shape[0], 1]])))
    grid2world = np.linalg.inv(world2grid)  # transformation already considered voxel size

    limits = np.matmul(grid2world, np.transpose(limits))
    limits = np.transpose(limits)[:, :3]
    valid = np.logical_and(region_sampled_points >= limits[0] - 0.3, region_sampled_points <= limits[1] + 0.3)
    valid = np.all(valid, axis=1)
    dense_sem = add_semantics_to_chunk_sdf(sdf_file, region_sampled_points[valid], region_sampled_cat[valid],
                                           index, vis_path=output_vis_dir)

    out_path = path.join(output_dir, segmentation + "_room" + str(region) + "__sem__" + str(sdf_number) + ".sdf")
    with open(sdf_file, "rb") as i:
        with open(out_path, "wb") as o:
            o.write(i.read())  # copy everything
            o.write(struct.pack('Q', dense_sem.shape[0] * dense_sem.shape[1] * dense_sem.shape[2]))
            o.write(dense_sem.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seg_path", type=str, required=True,
                        help="output directory of the region segmentation download")
    parser.add_argument("--mapping", type=str, required=True,
                        help="table that contains the mapping of raw_categories to ids")
    parser.add_argument("--sdf_path", type=str, required=True, help="directory of the .sdf files")
    parser.add_argument("--output_dir", type=str, default=".", help="where to write the extended .sdf files")
    parser.add_argument("--output_vis_dir", type=str, default=None, help="where to write the color and sem meshes")
    parser.add_argument("--truncation", type=float, default=3, help="truncation in voxels")
    parser.add_argument("--samples_per_face", type=int, default=4,
                        help="how many points are sampled from every face in average")
    parser.add_argument("--max_scenes", type=int, default=None, help="set maximum number of scenes processed")
    parser.add_argument("--max_vis", type=int, default=1, help="set maximum number of scenes to visualize")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    mapping_table = pd.read_csv(args.mapping, sep="\t")[["count", "eigen13id", "eigen13class", "mpcat40index"]]
    raw_index = np.array(mapping_table["eigen13id"])
    raw_index = np.concatenate((np.array([0]), raw_index), axis=-1)
    count = np.array(mapping_table[["count", "eigen13id"]].groupby("eigen13id").sum())
    count[7] -= mapping_table["count"][mapping_table["mpcat40index"] == 41].sum()
    weight = 1 / count
    weight /= weight.sum()

    _, unique = np.unique(mapping_table["eigen13id"], axis=0, return_index=True)
    label = list(zip(mapping_table["eigen13id"][unique], mapping_table["eigen13class"][unique]))
    label = sorted(label, key=lambda x: x[0])
    eigen13_label = np.array(label)[:, 1]
    eigen13_label[-1] = "unlabeled"

    mapping_color = matplotlib.cm.tab20(range(20))[:15, :3] * 255
    mapping_color[0] = np.array([255, 255, 255])  # void: white
    mapping_color[-1] = np.array([0, 0, 0])  # unlabeled as default: black
    color_list = tuple(map(tuple, mapping_color / 255))
    category_img = plot_colortable(color_list[:], "Category List")
    category_img.savefig("Category_list.png")
    np.savez("category", mapping_color=mapping_color.astype(np.uint8), class_name=eigen13_label[:-1], weight=weight.reshape(-1))

    seg_dir = path.join(args.seg_path, "v1", "scans")
    if not os.path.exists(args.output_vis_dir):
        os.makedirs(args.output_vis_dir)

    num_scenes = 0
    for segmentation in listdir(seg_dir):
        if args.max_scenes is not None and args.max_scenes <= num_scenes:
            print("Max number of scenes reached, done.")
            exit()
        
        if os.path.exists(os.path.join(args.output_dir, segmentation + "_room0__sem__0.sdf")):
            print(f"{segmentation} already exists, skipping.")
            continue
        
        # Check before parsing ply's to avoid unnecessary processing.
        sdf_paths = glob.glob(os.path.join(args.sdf_path, segmentation + '*cmp*'))
        if len(sdf_paths) == 0:
            print(f"Found no sdf files for {segmentation}, skipping.")
            continue

        unzip_path = path.join(seg_dir, segmentation)
        print("=========================")
        if path.exists(path.join(unzip_path, segmentation)):
            print(f"{segmentation}: already unzipped, extracting semantics...")
        else:
            zip_path = path.join(seg_dir, segmentation, "region_segmentations.zip")
            if not path.exists(zip_path):
                print(f"{segmentation} does not contain a region_segmentations.zip")
                continue
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            print(f"{segmentation}: extracted region_segmentations")

        # sample points from all regions/rooms in the segmentation
        # can avoid inconsistency between .ply and .sdf files (region and chunk)
        ply_dir = path.join(unzip_path, segmentation, "region_segmentations")
        region = 0
        start = time.time()
        print(f"Sampling points ...")
        region_sampled_points, region_sampled_cat = None, None
        while path.exists(path.join(ply_dir, "region" + str(region) + ".ply")):
            # print(f"-region {region}")

            ply_path = path.join(ply_dir, "region" + str(region) + ".ply")
            sampled_points, sampled_cat = sample_util.sample_from_region_ply(ply_path, num=args.samples_per_face)
            if len(sampled_points) == 0:
                region += 1
                continue

            if region_sampled_points is None:
                region_sampled_points = sampled_points
                region_sampled_cat = sampled_cat
            else:
                region_sampled_points = np.concatenate((region_sampled_points, sampled_points))
                region_sampled_cat = np.concatenate((region_sampled_cat, sampled_cat))

            region += 1

        took = time.time() - start
        print(f"Processed {region} regions, sampled {region_sampled_points.shape[0]} points, took {took:.3f} s.")

        if args.output_vis_dir is not None and num_scenes < args.max_vis:
            output_vis_dir = args.output_vis_dir
        else:
            output_vis_dir = None

        # add valid points to corresponding sdf file(s)
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_dict = {
                executor.submit(extend_sdf_file, segmentation, sdf_file, args.output_dir, output_vis_dir,
                                region_sampled_points, region_sampled_cat, raw_index): sdf_file for sdf_file in sdf_paths}

            for future in as_completed(future_dict):
                sdf = future_dict[future]
                try:
                    future.result()
                except Exception as e:
                    print((sdf, e))

        took = time.time() - start
        print(f"Processed {len(sdf_paths)} sdf files, took {took:.3f} s.")

        num_scenes += 1

    print("Done.")
