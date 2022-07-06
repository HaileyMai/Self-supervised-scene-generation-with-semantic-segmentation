from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import shutil
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
import model as model_util
import loss as loss_util

COLOR_SPACES = ['rgb', 'lab']

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', required=True, help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
# model params
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=0, help='#points / voxel dim.')
parser.add_argument('--nf_gen', type=int, default=20, help='controls #channels of generator')
parser.add_argument('--no_pass_geo_feats', dest='pass_geo_feats', action='store_false')
parser.add_argument('--input_mask', type=int, default=1, help='input mask')
# test params
parser.add_argument('--num_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--max_to_process', type=int, default=150, help='max num to process')
parser.add_argument('--vis_only', dest='vis_only', action='store_true')
parser.add_argument('--weight_color_loss', type=float, default=1.0, help='weight color loss vs rest (0 to disable).')
parser.add_argument('--weight_semantic_loss', type=float, default=0.1,
                    help='weight semantic loss vs rest (0 to disable).')
parser.add_argument('--color_thresh', type=float, default=15.0, help='mask colors with all values < color_thresh')
parser.add_argument('--color_truncation', type=float, default=0, help='truncation in voxels for color')
parser.add_argument('--augment_rgb_scaling', dest='augment_rgb_scaling', action='store_true')
parser.add_argument('--augment_scale_min', type=float, default=0.5, help='for color augmentation')
parser.add_argument('--augment_scale_max', type=float, default=1.5, help='for color augmentation')
parser.add_argument('--color_space', type=str, default='lab', help='[rgb, lab]')
parser.add_argument('--stride', type=int, default=32, help='stride for chunking (0 - chunk size)')

parser.set_defaults(vis_only=False, augment_rgb_scaling=False, pass_geo_feats=True)
args = parser.parse_args()
assert (args.color_space in COLOR_SPACES)
if args.input_dim == 0:  # set default values
    args.input_dim = (128, 64, 64)
args.input_nf = 1 + 3
UP_AXIS = 0
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# create model
nf_in_color = 3 if args.input_mask == 0 else 4
model = model_util.Generator(nf_in_geo=1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats,
                             truncation=args.truncation, max_data_size=args.input_dim).cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)

class_name = np.load("category.npz")['class_name']
mapping_color = np.load("category.npz")['mapping_color']


def compute_intersection_union(chunk_target_sdf, output_sdf, known, chunk_target_semantic=None,
                               output_semantic=None, class_index=None):
    inside_target = torch.zeros(chunk_target_sdf.shape, dtype=torch.bool)
    inside_target[torch.abs(chunk_target_sdf) < args.truncation] = True

    inside_output = torch.zeros(output_sdf.shape, dtype=torch.bool)
    inside_output[torch.abs(output_sdf) < args.truncation] = True

    if chunk_target_semantic is not None and output_semantic is not None and class_index is not None:
        assert chunk_target_semantic.shape == output_semantic.shape
        mask = torch.logical_and(chunk_target_semantic != 14, known)  # ignore unlabeled or unknown voxels
        inside_target[chunk_target_semantic != class_index] = False
        inside_output[output_semantic != class_index] = False
    else:
        mask = known

    union = torch.logical_or(inside_target[mask], inside_output[mask])
    intersection = torch.logical_and(inside_target[mask], inside_output[mask])
    union = torch.sum(union)
    intersection = torch.sum(intersection)
    return intersection.item(), union.item()


def test(dataloader, output_vis, num_to_vis):
    model.eval()

    chunk_dim = args.input_dim
    args.max_input_height = chunk_dim[0]
    if args.stride == 0:
        args.stride = chunk_dim[1]
    pad = 2

    num_proc = 0
    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        intersection_total = 0
        union_total = 0
        intersection_classes_total = np.zeros(model.n_classes)
        union_classes_total = np.zeros(model.n_classes)
        sample_total = 0
        for t, sample in enumerate(dataloader):
            inputs = sample['input']
            sdfs = sample['sdf']
            mask = sample['mask']
            known = sample['known']
            colors = sample['colors']
            semantics = sample['semantics']

            max_input_dim = np.array(sdfs.shape[2:])
            if args.max_input_height > 0 and max_input_dim[UP_AXIS] > args.max_input_height:
                max_input_dim[UP_AXIS] = args.max_input_height
                inputs = inputs[:, :, :args.max_input_height]
                if mask is not None:
                    mask = mask[:, :, :args.max_input_height]
                if sdfs is not None:
                    sdfs = sdfs[:, :, :args.max_input_height]
                if known is not None:
                    known = known[:, :, :args.max_input_height]
                if colors is not None:
                    colors = colors[:, :args.max_input_height]
                if semantics is not None:
                    semantics = semantics[:, :, :args.max_input_height]
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (
                num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1], max_input_dim[2]))

            output_colors = torch.zeros(colors.shape)
            output_sdfs = torch.zeros(sdfs.shape)
            output_norms = torch.zeros(sdfs.shape)
            output_occs = torch.zeros(sdfs.shape, dtype=torch.uint8)
            output_semantics = (torch.zeros((sdfs.shape[:1] + (14,) + sdfs.shape[2:])))  # unlabeled class

            # chunk up the scene
            chunk_input = torch.ones(1, args.input_nf, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_mask = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_known = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_target_sdf = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_target_colors = torch.zeros(1, chunk_dim[0], chunk_dim[1], chunk_dim[2], 3, dtype=torch.uint8).cuda()
            chunk_target_semantic = 14 * torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2],
                                                    dtype=torch.uint8).cuda()

            intersection_sum = 0
            union_sum = 0
            intersection_classes_sum = np.zeros(model.n_classes)
            union_classes_sum = np.zeros(model.n_classes)
            for y in range(0, max_input_dim[1], args.stride):
                for x in range(0, max_input_dim[2], args.stride):
                    chunk_input_mask = torch.abs(
                        inputs[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]) < args.truncation
                    if torch.sum(chunk_input_mask).item() == 0:
                        continue
                    sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d) (%d, %d)    ' % (
                        num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1],
                        max_input_dim[2], y, x))

                    fill_dim = [min(sdfs.shape[2], chunk_dim[0]), min(sdfs.shape[3] - y, chunk_dim[1]),
                                min(sdfs.shape[4] - x, chunk_dim[2])]
                    chunk_target_sdf.fill_(float('inf'))
                    chunk_target_colors.fill_(0)
                    chunk_input[:, 0].fill_(-args.truncation)
                    chunk_input[:, 1:].fill_(0)
                    chunk_mask.fill_(0)
                    chunk_input[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]] = inputs[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]
                    chunk_mask[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]] = mask[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]
                    chunk_known[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]] = known[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]
                    chunk_known = chunk_known <= 1
                    chunk_target_sdf[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]] = sdfs[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]
                    chunk_target_colors[:, :fill_dim[0], :fill_dim[1], :fill_dim[2], :] = colors[:, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]
                    if semantics is not None:
                        chunk_target_semantic[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]] = semantics[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]]

                    output_occ = None
                    output_occ, output_sdf, output_color, output_semantic = model(
                        chunk_input, chunk_mask, pred_sdf=[True, True], pred_color=args.weight_color_loss > 0,
                        pred_semantic=args.weight_semantic_loss > 0)

                    if semantics is not None:
                        output_label = torch.argmax(output_semantic, dim=1, keepdim=True)
                        for cl in range(model.n_classes):
                            i, u = compute_intersection_union(chunk_target_sdf, output_sdf, chunk_known,
                                                              chunk_target_semantic, output_label, class_index=cl)
                            intersection_classes_sum[cl] += i
                            union_classes_sum[cl] += u
                    i, u = compute_intersection_union(chunk_target_sdf, output_sdf, chunk_known)
                    intersection_sum += i
                    union_sum += u

                    if output_occ is not None:
                        occ = torch.nn.Sigmoid()(output_occ.detach()) > 0.5
                        locs = torch.nonzero((torch.abs(output_sdf.detach()[:, 0]) < args.truncation) & occ[:, 0]).cpu()
                    else:
                        locs = torch.nonzero(torch.abs(output_sdf[:, 0]) < args.truncation).cpu()
                    locs = torch.cat([locs[:, 1:], locs[:, :1]], 1)
                    output_sdf = [locs, output_sdf[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]].detach().cpu()]
                    if args.weight_color_loss == 0:
                        output_color = None
                    else:
                        output_color = [locs, output_color[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]]
                    if args.weight_semantic_loss == 0:
                        output_semantic = None
                    else:
                        output_semantic = output_semantic[locs[:, -1], :, locs[:, 0], locs[:, 1], locs[:, 2]]

                    output_locs = output_sdf[0] + torch.LongTensor([0, y, x, 0])
                    if args.stride < chunk_dim[1]:
                        min_dim = [0, y, x]
                        max_dim = [0 + chunk_dim[0], y + chunk_dim[1], x + chunk_dim[2]]
                        if y > 0:
                            min_dim[1] += pad
                        if x > 0:
                            min_dim[2] += pad
                        if y + chunk_dim[1] < max_input_dim[1]:
                            max_dim[1] -= pad
                        if x + chunk_dim[2] < max_input_dim[2]:
                            max_dim[2] -= pad
                        for k in range(3):
                            max_dim[k] = min(max_dim[k], sdfs.shape[k + 2])
                        outmask = (output_locs[:, 0] >= min_dim[0]) & (output_locs[:, 1] >= min_dim[1]) & (
                                output_locs[:, 2] >= min_dim[2]) & (output_locs[:, 0] < max_dim[0]) & (
                                          output_locs[:, 1] < max_dim[1]) & (output_locs[:, 2] < max_dim[2])
                    else:
                        outmask = (output_locs[:, 0] < output_sdfs.shape[2]) & (
                                output_locs[:, 1] < output_sdfs.shape[3]) & (
                                          output_locs[:, 2] < output_sdfs.shape[4])
                    output_locs = output_locs[outmask]
                    output_sdf = [output_sdf[0][outmask], output_sdf[1][outmask]]
                    if output_color is not None:
                        output_color = [output_color[0][outmask], output_color[1][outmask]]
                        output_color = (output_color[1] + 1) * 0.5

                        output_colors[0, output_locs[:, 0], output_locs[:, 1], output_locs[:, 2], :] += output_color.detach().cpu()
                    if output_occ is not None:
                        output_occs[:, :, :chunk_dim[0], y:y + chunk_dim[1], x:x + chunk_dim[2]] = occ[:, :, :fill_dim[0], :fill_dim[1], :fill_dim[2]]
                    if output_semantic is not None:
                        output_semantic = output_semantic[outmask]
                        output_semantics[0, :, output_locs[:, 0], output_locs[:, 1],
                        output_locs[:, 2]] += output_semantic.permute(1, 0).detach().cpu()
                    output_sdfs[0, 0, output_locs[:, 0], output_locs[:, 1], output_locs[:, 2]] += output_sdf[1][:, 0].detach().cpu()
                    output_norms[0, 0, output_locs[:, 0], output_locs[:, 1], output_locs[:, 2]] += 1

            if semantics is not None:
                iou_classes = intersection_classes_sum / union_classes_sum
                print(f"\nMean IoU of {model.n_classes} classes: ")
                for i in range(model.n_classes):
                    print(f"{class_name[i]}: {iou_classes[i]:.3f}")
                intersection_classes_total += intersection_classes_sum
                union_classes_total += union_classes_sum
            print(f"**Mean IoU: {intersection_sum / union_sum:.3f}")
            intersection_total += intersection_sum
            union_total += union_sum
            sample_total += 1

            # normalize
            mask = output_norms > 0
            output_norms = output_norms[mask]
            output_sdfs[mask] = output_sdfs[mask] / output_norms
            output_sdfs[~mask] = -float('inf')
            mask = mask.view(1, mask.shape[2], mask.shape[3], mask.shape[4])
            output_colors[mask, :] = output_colors[mask, :] / output_norms.view(-1, 1)
            output_colors = torch.clamp(output_colors * 255, 0, 255)

            sdfs = torch.clamp(sdfs, -args.truncation, args.truncation)
            output_sdfs = torch.clamp(output_sdfs, -args.truncation, args.truncation)

            if num_vis < num_to_vis:
                inputs = inputs.cpu().numpy()
                locs = torch.nonzero(torch.abs(output_sdfs[0, 0]) < args.truncation)
                vis_pred_sdf = [None]
                vis_pred_color = [None]
                vis_pred_semantic = [None]
                sdf_vals = output_sdfs[0, 0, locs[:, 0], locs[:, 1], locs[:, 2]].view(-1)
                vis_pred_sdf[0] = [locs.cpu().numpy(), sdf_vals.cpu().numpy()]
                if args.weight_color_loss > 0:
                    vals = output_colors[0, locs[:, 0], locs[:, 1], locs[:, 2]]
                    vis_pred_color[0] = vals.cpu().numpy()
                if output_semantic is not None:
                    vals_sem = output_semantics[0, :, locs[:, 0], locs[:, 1], locs[:, 2]]
                    vis_pred_semantic[0] = vals_sem.permute(1, 0).cpu().numpy()
                if output_occs is not None:
                    pred_occ = output_occs.cpu().numpy().astype(np.float32)
                if semantics is not None:
                    semantics = semantics.numpy()
                data_util.save_predictions(output_vis, np.arange(1), sample['name'], inputs, sdfs.numpy(),
                                           colors.numpy(), semantics, None, None, None, vis_pred_sdf, vis_pred_color,
                                           vis_pred_semantic, None, None, None, sample['world2grid'], args.truncation,
                                           mapping_color, args.color_space)
                num_vis += 1
            num_proc += 1
            gc.collect()

        print("\n=========== Summary =============")
        print(f"Evaluate {sample_total} regions: ")
        if semantics is not None:
            iou_classes = intersection_classes_total / union_classes_total
            print(f"Mean IoU of {model.n_classes} classes: ")
            for i in range(model.n_classes):
                print(f"{class_name[i]}: {iou_classes[i]:.3f}")
        print(f"**Mean IoU total: {intersection_total / union_total:.3f}")

        with open(os.path.join(args.output, "IoU.txt"), "w") as f:  # TODO two columns
            if semantics is not None:
                np.savetxt(f, class_name, '%s', delimiter=' ')
                np.savetxt(f, iou_classes, '%.3f', delimiter=' ')
            f.write("\nTotal:\n")
            f.write(str(intersection_total / union_total))

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
                                                 target_path=args.target_data_path,
                                                 load_semantic=args.weight_semantic_loss > 0)
    print('test_dataset', len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                  collate_fn=scene_dataloader.collate_voxels)

    if os.path.exists(args.output):
        if args.vis_only:
            print('warning: output dir %s exists, will overwrite any existing files')
        # else:
        # input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
        # shutil.rmtree(args.output)
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
