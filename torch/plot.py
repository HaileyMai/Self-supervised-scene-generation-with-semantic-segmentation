import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', required=True, help='path to log data')
parser.add_argument('--val', default=True, help='plot val log data')
parser.add_argument('--name', default='', help='name of the experiment')
parser.add_argument('--output', default='./plot', help='folder to output')
args = parser.parse_args()

# TODO read from args.txt
f = open(os.path.join(args.log_path, "args.txt"), "r")
params = f.read()[10:-2].split("voxelsize=0.02, ")[1]
values = [e.split('=')[1] for e in params.split(', ')]
keys = [e.split('=')[0] for e in params.split(', ')]
params = dict(zip(keys, values))
weight_depth_loss = float(params['weight_depth_loss'])
weight_disc_loss = float(params['weight_disc_loss'])
weight_discgen_loss = float(params['weight_discgen_loss'])
weight_occ_loss = float(params['weight_occ_loss'])
weight_sdf_loss = float(params['weight_sdf_loss'])
weight_color_loss = float(params['weight_color_loss'])
weight_semantic_loss = float(params['weight_semantic_loss'])

if not os.path.exists(args.output):
    os.makedirs(args.output)

log = pd.read_csv(os.path.join(args.log_path, "log.csv"))

iteration = np.array(log["iter"])
train_iou_occ = np.array(log["train_iou(occ)"])

train_loss_total = np.array(log["train_loss(total)"])
train_loss_occ = np.array(log["train_loss(occ)"])
train_loss_sdf = np.array(log["train_loss(sdf)"])
train_loss_sdf[train_loss_sdf < 0] = 0
train_loss_depth = np.array(log["train_loss(depth)"])
train_loss_depth[train_loss_depth < 0] = 0
if weight_color_loss > 0:
    train_loss_color = np.array(log["train_loss(color)"])
    train_loss_color[train_loss_color < 0] = 0
else:
    train_loss_color = 0
if weight_semantic_loss > 0:
    train_loss_semantic = np.array(log["train_loss(semantic)"])
    train_loss_semantic[train_loss_semantic < 0] = 0
else:
    train_loss_semantic = 0

# train_loss_disc = np.array(log["train_loss(disc)"])
# train_loss_disc_real = np.array(log["train_loss(disc-real)"])
# train_loss_disc_fake = np.array(log["train_loss(disc-fake)"])
# train_loss_gen = np.array(log["train_loss(gen)"])
# train_loss_style = np.array(log["train_loss(style)"])
# train_loss_content = np.array(log["train_loss(content)"])

plt.figure(1)
# plt.plot(iteration, train_loss_total, label='total')
plt.plot(iteration, train_loss_occ * weight_occ_loss + train_loss_sdf * weight_sdf_loss
         + train_loss_depth * weight_depth_loss + train_loss_color * weight_color_loss
         + train_loss_semantic * weight_semantic_loss, label='total')
plt.plot(iteration, train_loss_occ * weight_occ_loss + train_loss_sdf * weight_sdf_loss, label='occ+sdf')
# plt.plot(iteration, train_loss_occ * weight_occ_loss, label='occ')
# plt.plot(iteration, train_loss_sdf * weight_sdf_loss, label='sdf')
plt.plot(iteration, train_loss_depth * weight_depth_loss, label='depth2d')
if weight_color_loss > 0:
    plt.plot(iteration, train_loss_color * weight_color_loss, label='color2d')
if weight_semantic_loss > 0:
    plt.plot(iteration, train_loss_semantic * weight_semantic_loss, label='semantic')
plt.title(args.name + ' train loss')
plt.legend()
plt.axis([0, None, 0, None])
plt.xlabel('iter')
plt.ylabel('losses')
plt.savefig(os.path.join(args.output, args.name + '_train_losses'))
plt.show()

plt.figure()
plt.plot(iteration, train_iou_occ)
plt.axis([0, None, 0, 1])
plt.yticks(np.arange(0, 1.1, .1))
plt.xlabel('iter')
plt.ylabel('iou')
plt.show()

if args.val:
    log_val = pd.read_csv(os.path.join(args.log_path, "log_val.csv"))
    epoch = np.array(log_val["epoch"])
    iteration = np.array(log_val["iter"])
    # train_iou_occ = np.array(log_val["train_iou(occ)"])

    train_loss_total = np.array(log_val["train_loss(total)"])
    train_loss_occ = np.array(log_val["train_loss(occ)"])
    train_loss_sdf = np.array(log_val["train_loss(sdf)"])
    train_loss_sdf[train_loss_sdf < 0] = 0
    train_loss_depth = np.array(log_val["train_loss(depth)"])
    train_loss_depth[train_loss_depth < 0] = 0

    # train_loss_disc = np.array(log_val["train_loss(disc)"])
    # train_loss_disc_real = np.array(log_val["train_loss(disc-real)"])
    # train_loss_disc_fake = np.array(log_val["train_loss(disc-fake)"])
    # train_loss_gen = np.array(log_val["train_loss(gen)"])
    # train_loss_style = np.array(log_val["train_loss(style)"])
    # train_loss_content = np.array(log_val["train_loss(content)"])

    val_loss_total = np.array(log_val["val_loss(total)"])
    val_loss_occ = np.array(log_val["val_loss(occ)"])
    val_loss_sdf = np.array(log_val["val_loss(sdf)"])
    val_loss_sdf[val_loss_sdf < 0] = 0
    val_loss_depth = np.array(log_val["val_loss(depth)"])
    val_loss_depth[val_loss_depth < 0] = 0

    if weight_color_loss > 0:
        train_loss_color = np.array(log_val["train_loss(color)"])
        train_loss_color[train_loss_color < 0] = 0
        val_loss_color = np.array(log_val["val_loss(color)"])
        val_loss_color[val_loss_color < 0] = 0
    else:
        train_loss_color = 0
        val_loss_color = 0
    if weight_semantic_loss > 0:
        train_loss_semantic = np.array(log_val["train_loss(semantic)"])
        train_loss_semantic[train_loss_semantic < 0] = 0
        val_loss_semantic = np.array(log_val["val_loss(semantic)"])
        val_loss_semantic[val_loss_semantic < 0] = 0
    else:
        train_loss_semantic = 0
        val_loss_semantic = 0

    # val_loss_disc = np.array(log_val["val_loss(disc)"])
    # val_loss_disc_real = np.array(log_val["val_loss(disc-real)"])
    # val_loss_disc_fake = np.array(log_val["val_loss(disc-fake)"])
    # val_loss_gen = np.array(log_val["val_loss(gen)"])
    # val_loss_style = np.array(log_val["val_loss(style)"])
    # val_loss_content = np.array(log_val["val_loss(content)"])

    plt.figure(2)
    # plt.plot(epoch, train_loss_total, label='train_total')
    # plt.plot(epoch, val_loss_total, '--x', label='val_total')
    plt.plot(epoch, train_loss_occ * weight_occ_loss + train_loss_sdf * weight_sdf_loss
             + train_loss_depth * weight_depth_loss + train_loss_color * weight_color_loss
             + train_loss_semantic * weight_semantic_loss)
    plt.plot(epoch, val_loss_occ * weight_occ_loss + val_loss_sdf * weight_sdf_loss
             + val_loss_depth * weight_depth_loss + val_loss_color * weight_color_loss
             + val_loss_semantic * weight_semantic_loss, '--x')
    # plt.plot(epoch, train_loss_occ * weight_occ_loss + train_loss_sdf * weight_sdf_loss, label='train_occ+sdf')
    # plt.plot(epoch, val_loss_occ * weight_occ_loss + val_loss_sdf * weight_sdf_loss, '--x', label='val_occ+sdf')
    plt.plot(train_loss_depth * weight_depth_loss, label='train_depth2d')
    plt.plot(val_loss_depth * weight_depth_loss, '--x', label='val_depth2d')
    if weight_color_loss > 0:
        plt.plot(epoch, train_loss_color * weight_color_loss, label='train_color2d')
        plt.plot(epoch, val_loss_color * weight_color_loss, '--x', label='val_color2d')
    if weight_semantic_loss > 0:
        plt.plot(epoch, train_loss_semantic * weight_semantic_loss, label='train_semantic')
        plt.plot(epoch, val_loss_semantic * weight_semantic_loss, '--x', label='val_semantic')
    plt.title(args.name + ' val loss')
    plt.legend()
    plt.axis([0, None, 0, None])
    plt.xticks(np.arange(min(epoch), max(epoch)+1, 1))
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.savefig(os.path.join(args.output, args.name + '_val_losses'))
    plt.show()
