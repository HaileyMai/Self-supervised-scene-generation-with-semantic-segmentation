from http.client import UNAUTHORIZED
from plyfile import PlyData, PlyElement
from os import path, listdir
import json
import random
import data_util
import numpy as np
import pandas as pd
import argparse
import zipfile

"""
First run download-mp.py -o [output_dir] --type region_segmentations
Category mapping should be taken from https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv

Ex.: extend_sdfs.py --seg_path ../../region_segmentations --mapping ../../Matterport/metadata/category_mapping.tsv --sdf_path ../../data --output_dir ../../data_extended
"""

#TODO
#add option to delete unzipped files at the end
#change the data_util.load_sdf extension to a not so hacky one
#test for the occupancy and correctness of semantics on the grid 
#-> it is possible to add arbitrarily many points for every face
#--force parameter for overwriting output files

parser = argparse.ArgumentParser()

parser.add_argument("--seg_path", type=str, required=True, help="output directory of the region segmentation download")
parser.add_argument("--mapping", type=str, required=True, help="table that contains the mapping of raw_categories to ids")
parser.add_argument("--category_taxonomy", type=str, default="mpcat40index", help="what taxonomy to use, should be a column of the mapping table")
parser.add_argument("--raw_category", type=str, default="raw_category", help="column of mapping that contains the raw category names")
parser.add_argument("--sdf_path", type=str, required=True, help="directory of the .sdf files")
parser.add_argument("--output_dir", type=str, default=".", help="where to write the extended .sdf files")

args = parser.parse_args()
print(args)

mapping_table = pd.read_csv(args.mapping, sep="\t")
assert args.category_taxonomy in mapping_table.columns and args.raw_category in mapping_table.columns
#mapping is a dict from string to id
mapping = dict(zip(mapping_table[args.raw_category], mapping_table[args.category_taxonomy]))

seg_dir = path.join(args.seg_path, "v1", "scans")

for segmentation in listdir(seg_dir):
    unzip_path = path.join(seg_dir, segmentation)
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

    ply_dir = path.join(unzip_path, segmentation, "region_segmentations")
    region = 0
    while path.exists(path.join(ply_dir, "region" + str(region) + ".ply")):
        print(f"-region {region}")
        semseg_path = path.join(ply_dir, "region" + str(region) + ".semseg.json")
        with open(semseg_path) as f:
            semseg = json.load(f)["segGroups"]
        
        ply_path = path.join(ply_dir, "region" + str(region) + ".ply")
        data = PlyData.read(ply_path)
        vertices = data.elements[0]
        faces = data.elements[1]

        point_sems = []
        for i in range(faces.count):
            face_vertices, _, object_id, category_id = faces.data[i]
            face_vertices = [vertices[i] for i in face_vertices]
            
            #calculate center of face
            z = (face_vertices[0]["z"] + face_vertices[1]["z"] + face_vertices[2]["z"]) / 3
            y = (face_vertices[0]["y"] + face_vertices[1]["y"] + face_vertices[2]["y"]) / 3
            x = (face_vertices[0]["x"] + face_vertices[1]["x"] + face_vertices[2]["x"]) / 3
            
            raw_category = semseg[object_id]["label"]
            if raw_category in mapping:
                sem = mapping[raw_category]
            else:
                sem = 0 #default

            #add center
            center_sem = [z, y, x, sem]
            point_sems += [center_sem]
            
            #add corners
            for i in range(3):
                point_sem = [face_vertices[i]["z"], face_vertices[i]["y"], face_vertices[i]["x"], sem]
                point_sems += [point_sem]
            
        point_sems = np.array(point_sems)

        sdf_base = path.join(args.sdf_path, segmentation + "_room" + str(region) + "__cmp__")
        sdf_number = 0

        while path.exists(sdf_base + str(sdf_number) + ".sdf"):
            #print(f"--sdf {sdf_number}")
            sdf_path = sdf_base + str(sdf_number) + ".sdf"
            [locs, sdf], [dimz, dimy, dimx], world2grid, known, colors = data_util.load_sdf(sdf_path, True, False, True)
            
            points = point_sems.copy()

            x = np.ones((points.shape[0], 4))
            x[:, :3] = points[:, :3]
            x = np.matmul(world2grid, np.transpose(x))
            x = np.transpose(x)
            x = np.divide(x[:, :3], x[:, 3, None])
            x = np.rint(x)

            lower_bound = np.all(x >= 0, axis=1)
            upper_bound = np.all(x < [dimz, dimy, dimx], axis=1)
            inbounds = np.logical_and(lower_bound, upper_bound)
            points = np.column_stack((x, points[:, 3]))
            points = points[inbounds] #keep only values that are in the given grid
            
            _, unique = np.unique(points[:, :3], axis=0, return_index=True) #TODO how to choose a label for a grid coordinate
            points = points[unique]

            points = points.astype(int)
            
            #convert to dense to keep format the same as colors
            dense_sem = np.zeros([dimz, dimy, dimx], dtype=np.uint8)
            dense_sem[points[:, 0], points[:, 1], points[:, 2]] = points[:, 3]

            out_path = path.join(args.output_dir, segmentation + "_room" + str(region) + "__sem__" + str(sdf_number) + ".sdf")
            
            with open(sdf_path, "rb") as i:
                with open(out_path, "wb") as o:
                    o.write(i.read()) #copy everything
                    o.write(dense_sem.tobytes())
            
            sdf_number += 1

        region += 1

    print("Done.")
