import torch
import numpy as np

import data_util
import scene_dataloader
import model as model_util
import loss as loss_util
import style
from utils.raycast_rgbd.raycast_rgbd import RaycastRGBD

# raycaster_rgbd = RaycastRGBD(args.batch_size, args.input_dim, args.style_width, args.style_height,
#                              depth_min=0.1 / args.voxelsize, depth_max=raycast_depth_max / args.voxelsize,
#                              thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment,
#                              max_num_locs_per_sample=max_num_locs_per_sample)
batch_size = 2
raycaster_rgbd = RaycastRGBD(batch_size, (128, 64, 64), 320, 256,
                             depth_min=0.1 / 0.02, depth_max=6.0 / 0.02,
                             thresh_sample_dist=50.5*0.9, ray_increment=0.9,
                             max_num_locs_per_sample=640000)

train_files, val_files, _OVERFIT = data_util.get_train_files("../data/sdf/", "../filelists/inc.txt", None, 0)
train_dataset = scene_dataloader.SceneDataset(train_files, (128, 64, 64), 3, True, True, (0.5, 1.5), 0, 'lab',
                                              "../data/data-frames/", "../data/images/", image_dims=(320, 256),
                                              load_depth=True, subsamp2d_factor=1, randomize_frames=1, num_overfit=0)
print('train_dataset', len(train_dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               collate_fn=scene_dataloader.collate_voxels)

sdf, world2grid, known, color, semantic = data_util.load_sdf("../data/sdf/1LXtFkjw3qL_room0__sem__0.sdf",
                                                             load_sparse=False, load_known=True, load_color=True,
                                                             color_file=None, load_semantic=True)
target_for_sdf = data_util.preprocess_sdf_pt(sdf, 3)

locs = torch.nonzero(torch.abs(target_for_sdf[:, 0]) < 3)
locs = torch.cat([locs[:, 1:], locs[:, :1]], 1).contiguous()
vals = target_for_sdf[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]].contiguous()
colors = color[locs[:, -1], locs[:, 0], locs[:, 1], locs[:, 2], :].float() / 255.0

depth_image, color_image, pose, intrinsic = data_util.load_frame("../data/images/1LXtFkjw3qL/depth/", color_files[b][f], camera_files[b][f],
                                                                 depth_image_dims, color_image_dims, color_normalization, load_depth, load_color)
poses = sample['images_pose'].cuda()
intrinsics = sample['images_intrinsic'].cuda()
view_matrix = style.compute_view_matrix(sample['world2grid'].cuda(), poses)

target_normals = loss_util.compute_normals_sparse(locs, vals, target_for_sdf.shape[2:],
                                                  transform=torch.inverse(view_matrix))
raycast_color, _, raycast_normal, raycast_semantic = raycaster_rgbd(locs, vals, colors.contiguous(), target_normals,
                                                                    view_matrix, intrinsics)