from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import shutil
import random
import torch
import torch.nn.functional as F
import numpy as np
import gc

import data_util
import scene_dataloader
import model as model_util
import loss as loss_util
from utils.raycast_rgbd.raycast_rgbd import RaycastRGBD
import style


COLOR_SPACES = ['rgb', 'lab']

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', required=True, help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output2', help='folder to output predictions')
# model params
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=0, help='#points / voxel dim.')
parser.add_argument('--nf_gen', type=int, default=20, help='controls #channels of generator')
parser.add_argument('--no_pass_geo_feats', dest='pass_geo_feats', action='store_false')
parser.add_argument('--input_mask', type=int, default=1, help='input mask')
# test params
parser.add_argument('--max_input_height', type=int, default=128,
                    help='truncate input to this height (in voxels), 0 to disable')
parser.add_argument('--num_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--max_to_process', type=int, default=150, help='max num to process')
parser.add_argument('--vis_only', dest='vis_only', action='store_true')
parser.add_argument('--weight_color_loss', type=float, default=1.0, help='weight color loss vs rest (0 to disable).')
parser.add_argument('--weight_semantic_loss', type=float, default=1.0,
                    help='weight semantic loss vs rest (0 to disable).')
parser.add_argument('--color_thresh', type=float, default=15.0, help='mask colors with all values < color_thresh')
parser.add_argument('--color_truncation', type=float, default=0, help='truncation in voxels for color')
parser.add_argument('--augment_rgb_scaling', dest='augment_rgb_scaling', action='store_true')
parser.add_argument('--augment_scale_min', type=float, default=0.5, help='for color augmentation')
parser.add_argument('--augment_scale_max', type=float, default=1.5, help='for color augmentation')
parser.add_argument('--color_space', type=str, default='lab', help='[rgb, lab]')
parser.add_argument('--cpu', dest='cpu', action='store_true')
# 2d proj part
parser.add_argument('--voxelsize', type=float, default=0.02, help='voxel size in meters.')
parser.add_argument('--style_width', type=int, default=480, help='width of input for 2d style')
parser.add_argument('--style_height', type=int, default=384, help='height of input for 2d style')


parser.set_defaults(vis_only=False, augment_rgb_scaling=False, cpu=False, pass_geo_feats=True)
args = parser.parse_args()
if args.input_dim == 0:  # set default values
    args.input_dim = (128, 260, 328)
args.input_nf = 4
UP_AXIS = 0  # z is 0th
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# create model
nf_in_color = 3 if args.input_mask == 0 else 4
model = model_util.Generator(nf_in_geo=1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats,
                             truncation=args.truncation, max_data_size=args.input_dim)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)

# raycaster
ray_increment = 0.3 * args.truncation
thresh_sample_dist = 50.5 * ray_increment
max_num_locs_per_sample = 1200000  # too high: run out of memory, too low: can only raycast lower locs
raycast_depth_max = 6.0

# specify camera intrinsic and extrinsic (R looking down, t will be added later)
intrinsics = torch.tensor([[269.1120, 269.2970, args.style_width//2, args.style_height//2]])
camera_pose = torch.eye(4)
camera_pose[:3, 0] = torch.tensor([1, 0, 0])
camera_pose[:3, 1] = torch.tensor([0, -1, 0])
camera_pose[:3, 2] = torch.tensor([0, 0, -1])
camera_pose = camera_pose[None, :]

if not args.cpu:
    model = model.cuda()
    intrinsics = intrinsics.cuda()
    camera_pose = camera_pose.cuda()


def test(dataloader, output_vis, num_to_vis):
    model.eval()

    hierarchy_factor = 4
    num_proc = 0
    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs = sample['input']
            mask = sample['mask']

            sdfs = sample['sdf']
            colors = sample['colors']
            semantics = sample['semantics']
            sdfs = data_util.preprocess_sdf_pt(sdfs.cuda(), args.truncation)

            max_input_dim = np.array(inputs.shape[2:])
            # truncate height in order to view from above
            if args.max_input_height > 0 and max_input_dim[UP_AXIS] > args.max_input_height:
                max_input_dim[UP_AXIS] = args.max_input_height
                mask_input = inputs[0][:, UP_AXIS] < args.max_input_height
                inputs = inputs[:, :, :args.max_input_height]
                if mask is not None:
                    mask = mask[:, :, :args.max_input_height]
            max_input_dim = ((max_input_dim + hierarchy_factor - 1) // hierarchy_factor) * hierarchy_factor
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (
                num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1], max_input_dim[2]))
            # pad input and target to max_input_dim
            padded = torch.zeros(1, inputs.shape[1], max_input_dim[0], max_input_dim[1], max_input_dim[2])
            padded[:, 0].fill_(-args.truncation)
            padded[:, :, :min(args.max_input_height, inputs.shape[2]), :inputs.shape[3], :inputs.shape[4]] = \
                inputs[:, :, :args.max_input_height, :, :]
            inputs = padded
            padded_mask = torch.zeros(1, 1, max_input_dim[0], max_input_dim[1], max_input_dim[2])
            padded_mask[:, :, :min(args.max_input_height, mask.shape[2]), :mask.shape[3], :mask.shape[4]] = \
                mask[:, :, :args.max_input_height, :, :]
            mask = padded_mask

            model.update_sizes(max_input_dim)
            output_occ = None
            try:
                if not args.cpu:
                    inputs = inputs.cuda()
                    mask = mask.cuda()
                output_occ, output_sdf, output_color, output_semantic = model(
                    inputs, mask, pred_sdf=[True, True], pred_color=args.weight_color_loss > 0,
                    pred_semantic=args.weight_semantic_loss > 0)

                # if output_occ is not None:
                #     occ = torch.nn.Sigmoid()(output_occ.detach()) > 0.5
                #     locs = torch.nonzero((torch.abs(output_sdf.detach()[:, 0]) < args.truncation) & occ[:, 0]).cpu()
                # else:
                locs = torch.nonzero(torch.abs(output_sdf[:, 0]) < args.truncation).cpu()
                locs = torch.cat([locs[:, 1:], locs[:, :1]], 1)
                output_sdf = [locs, output_sdf[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]]
                if args.weight_color_loss == 0:
                    output_color = None
                else:
                    output_color = output_color[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]
                    output_color = (output_color + 1) * 0.5
                if args.weight_semantic_loss == 0:
                    output_semantic = None
                else:
                    output_semantic = output_semantic[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]
            except:
                print('exception')
                gc.collect()
                continue

            num_proc += 1
            if num_vis < num_to_vis:
                try:
                    raycaster_rgbd = RaycastRGBD(1, max_input_dim, width=args.style_width, height=args.style_height,
                                                 depth_min=0.1 / args.voxelsize,
                                                 depth_max=raycast_depth_max / args.voxelsize,
                                                 thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment,
                                                 max_num_locs_per_sample=max_num_locs_per_sample)
                    # input raycast
                    grid2world = np.linalg.inv(sample['world2grid'])
                    cam_pos = np.matmul(grid2world, np.array([max_input_dim[2]//2, max_input_dim[1]//2, max_input_dim[0]*2, 1]))
                    cam_pos = torch.from_numpy(cam_pos).cuda()
                    camera_pose[:, :3, 3] = cam_pos[:, :3]

                    view_matrix = style.compute_view_matrix(sample['world2grid'].cuda(), camera_pose)
                    input_locs = torch.nonzero(torch.abs(inputs[:, 0]) < args.truncation)
                    input_locs = torch.cat([input_locs[:, 1:], input_locs[:, :1]], 1)
                    input_vals = inputs[input_locs[:, -1], :, input_locs[:, 0], input_locs[:, 1], input_locs[:, 2]]
                    input_normals = loss_util.compute_normals(inputs[:, :1], input_locs,
                                                              transform=torch.inverse(view_matrix))
                    raycast_color, _, raycast_normal, _ = raycaster_rgbd(
                        input_locs.cuda(), input_vals[:, :1].contiguous(), input_vals[:, 1:].contiguous(), input_normals,
                        None, view_matrix, intrinsics)
                    if args.weight_color_loss > 0:
                        invalid = raycast_color == -float('inf')
                        input2d = raycast_color.clone() * 2 - 1
                        input2d[invalid] = 0
                    normals = raycast_normal.clone()
                    invalid = raycast_normal == -float('inf')
                    normals[invalid] = 0
                    if args.weight_color_loss > 0:
                        input2d = torch.cat([input2d, normals], 3)
                    else:
                        input2d = normals
                    input2d = input2d.permute(0, 3, 1, 2).contiguous()

                    target_for_sdfs = torch.zeros(1, 1, max_input_dim[0], max_input_dim[1], max_input_dim[2])
                    target_for_sdfs[:, 0].fill_(-args.truncation)
                    target_for_sdfs[:, :, :min(args.max_input_height, sdfs.shape[2]), :sdfs.shape[3], :sdfs.shape[4]] = \
                        sdfs[:, :, :args.max_input_height, :, :]
                    target_for_sdfs = target_for_sdfs.cuda()
                    target_for_colors = torch.zeros(1, max_input_dim[0], max_input_dim[1], max_input_dim[2], 3)
                    target_for_colors[:, :min(args.max_input_height, colors.shape[1]), :colors.shape[2], :colors.shape[3],
                    :] = \
                        colors[:, :args.max_input_height, :, :, :]
                    target_for_colors = target_for_colors.cuda()
                    target_for_semantics = 41 * torch.ones(1, 1, max_input_dim[0], max_input_dim[1], max_input_dim[2])
                    if semantics is not None:
                        target_for_semantics[:, :, :min(args.max_input_height, semantics.shape[2]), :semantics.shape[3],
                        :semantics.shape[4]] = semantics[:, :, :args.max_input_height, :, :]
                    target_for_semantics = target_for_semantics.cuda()

                    # target raycast
                    locs = torch.nonzero(torch.abs(target_for_sdfs[:, 0]) < args.truncation)
                    locs = torch.cat([locs[:, 1:], locs[:, :1]], 1).contiguous()
                    vals = target_for_sdfs[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]].contiguous()
                    colors = target_for_colors[locs[:, -1], locs[:, 0], locs[:, 1], locs[:, 2], :].float() / 255.0
                    target_normals = loss_util.compute_normals_sparse(locs, vals, target_for_sdfs.shape[2:],
                                                                      transform=torch.inverse(view_matrix))
                    target_semantics = target_for_semantics[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]
                    target_semantics_onehot = F.one_hot(target_semantics[:, 0].long())[..., :-1].float().contiguous()
                    raycast_color, _, raycast_normal, raycast_semantic = raycaster_rgbd(
                        locs, vals, colors.contiguous(), target_normals, target_semantics_onehot, view_matrix, intrinsics)
                    if args.weight_color_loss > 0:
                        invalid = raycast_color == -float('inf')
                        target2d = raycast_color.clone() * 2 - 1
                        target2d[invalid] = 0
                    invalid = raycast_normal == -float('inf')
                    normals = raycast_normal.clone()
                    normals[invalid] = 0
                    if args.weight_color_loss > 0:
                        target2d = torch.cat([target2d, normals], 3)
                    else:
                        target2d = normals
                    target2d = target2d.permute(0, 3, 1, 2).contiguous()
                    target2d_label = None
                    if semantics is not None:
                        cat = torch.cat((raycast_semantic.clone(), torch.ones(raycast_semantic.shape[:-1] + (1,)).cuda()), dim=-1)
                        _, target2d_label = torch.max(cat, dim=-1, keepdim=True)
                        target2d_label = target2d_label.to(torch.uint8)

                    # prediction raycast
                    if args.weight_color_loss > 0:
                        color = output_color
                    else:
                        color = torch.zeros(output_sdf[0].shape[0], 3).cuda()
                    if args.weight_semantic_loss > 0:
                        semantic = output_semantic.clone()
                    else:
                        semantic = 41 * torch.ones(output_sdf[0].shape[:-1] + (41,)).cuda()
                    output_normals = loss_util.compute_normals_sparse(output_sdf[0].cuda(), output_sdf[1],
                                                                      target_for_sdfs.shape[2:],
                                                                      transform=torch.inverse(view_matrix))
                    raycast_color, raycast_depth, raycast_normal, raycast_semantic = raycaster_rgbd(
                        output_sdf[0].cuda(), output_sdf[1], color, output_normals, semantic, view_matrix, intrinsics)
                    if args.weight_color_loss > 0:
                        raycast = torch.cat([raycast_color, raycast_normal], 3)
                    else:
                        raycast = raycast_normal
                    synth = raycast.permute(0, 3, 1, 2).contiguous()
                    invalid = synth == -float('inf')
                    synth[:, :3] = synth[:, :3] * 2 - 1  # normalize
                    synth[invalid] = 0
                    pred2d_label = None
                    if args.weight_semantic_loss > 0:
                        cat = torch.cat((raycast_semantic.detach(), torch.ones(raycast_semantic.shape[:-1] + (1,)).cuda()),
                                        dim=-1)
                        _, pred2d_label = torch.max(cat, dim=-1, keepdim=True)
                        pred2d_label = pred2d_label.to(torch.uint8)

                    vis_pred_sdf = [None]
                    vis_pred_color = [None]
                    vis_pred_semantic = [None]
                    if len(output_sdf[0]) > 0:
                        vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
                        if output_color is not None:  # convert colors to vec3uc
                            output_color = torch.clamp(output_color.detach() * 255, 0, 255)
                            vis_pred_color[0] = output_color.cpu().numpy()
                        if output_semantic is not None:
                            vis_pred_semantic[0] = output_semantic.detach().cpu().numpy()
                    vis_pred_images_color = None
                    vis_tgt_images_color = None
                    vis_input_images_color = None
                    vis_pred_depth = None
                    vis_target_depth = None
                    vis_pred_images_semantic = None
                    vis_target_images_semantic = None

                    if input2d is not None:
                        vis_input_images_color = input2d.cpu().numpy()
                        if args.weight_color_loss > 0:
                            vis_input_images_color[:, :3] = (vis_input_images_color[:, :3] + 1) * 0.5
                        vis_input_images_color = np.transpose(vis_input_images_color, [0, 2, 3, 1])
                    if target2d is not None:
                        if output_color is not None:
                            synth[:, :3] = (synth[:, :3] + 1) * 0.5
                            target2d[:, :3] = (target2d[:, :3] + 1) * 0.5
                        vis_pred_images_color = synth.detach().cpu().numpy()
                        vis_pred_images_color = np.transpose(vis_pred_images_color, [0, 2, 3, 1])
                        vis_tgt_images_color = target2d.cpu().numpy()
                        vis_tgt_images_color = np.transpose(vis_tgt_images_color, [0, 2, 3, 1])

                    if pred2d_label is not None:
                        vis_pred_images_semantic = pred2d_label.cpu().numpy()
                    if target2d_label is not None:
                        vis_target_images_semantic = target2d_label.cpu().numpy()
                    if semantics is not None:
                        target_for_semantics = target_for_semantics.cpu().numpy()
                    else:
                        target_for_semantics = None

                    data_util.save_predictions(output_vis, np.arange(1), sample['name'], inputs.cpu().numpy(),
                                               target_for_sdfs.cpu().numpy(), target_for_colors.cpu().numpy(),
                                               target_for_semantics, None, vis_tgt_images_color,
                                               vis_target_images_semantic, vis_pred_sdf, vis_pred_color, vis_pred_semantic,
                                               None, vis_pred_images_color, vis_pred_images_semantic, sample['world2grid'],
                                               args.truncation, np.load("category_color.npz")['mapping_color'],
                                               args.color_space, input_images=vis_input_images_color,
                                               pred_depth=vis_pred_depth, target_depth=vis_target_depth)
                except:
                    print('exception vis')
                    gc.collect()
                    continue
                num_vis += 1
            gc.collect()
    sys.stdout.write('\n')


def main():
    # data files
    test_files, _, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '', 0)
    if len(test_files) > args.max_to_process:
        test_files = test_files[:args.max_to_process]
    else:
        args.max_to_process = len(test_files)
    random.seed(42)
    random.shuffle(test_files)
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, True,
                                                 args.augment_rgb_scaling,
                                                 (args.augment_scale_min, args.augment_scale_max),
                                                 args.color_truncation, args.color_space,
                                                 # load_semantic=args.weight_semantic_loss > 0, TODO
                                                 target_path=args.target_data_path,
                                                 max_input_height=args.max_input_height)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                                  collate_fn=scene_dataloader.collate_voxels)
    print('test_dataset', len(test_dataset))

    if os.path.exists(args.output):
        if args.vis_only:
            print('warning: output dir %s exists, will overwrite any existing files')
        else:
            input('warning: output dir %s exists, press key to delete and continue' % args.output)
            shutil.rmtree(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    output_vis_path = os.path.join(args.output, 'vis')
    if not os.path.exists(output_vis_path):
        os.makedirs(output_vis_path)

    # start testing
    print('starting testing...')
    test(test_dataloader, output_vis_path, args.num_to_vis)


if __name__ == '__main__':
    main()
