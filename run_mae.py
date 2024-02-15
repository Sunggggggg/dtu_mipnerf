import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from einops import rearrange

import torch.nn as nn
from config import mae_args_parser
# DDP
import torch.multiprocessing as mp
from set_multi_gpus import set_ddp
from torch.nn.parallel import DistributedDataParallel as DDP
# dataset
from load_dtu import load_nerf_dtu_data
from MAE import make_input, IMAGE_MAE, PATCH_MAE, image_plot, to8b, PRO_MAE, augmenting_images

row = 5

def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # log 
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # save args   
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # save weight, fig
    os.makedirs(os.path.join(basedir, expname, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'figures'), exist_ok=True)

    # load dataset
    train_imgs, train_c2w, train_p2c, scan_list = load_nerf_dtu_data(args.datadir, args.mae_input, factor=args.scale, random_idx=args.random_idx)

    # Save masked view point poses
    scan_list = [f'scan{i}' for i in [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]]
    print("[SAVE] MAE trainig poses", train_c2w.shape)
    os.makedirs(os.path.join(basedir, expname, 'poses'), exist_ok=True)
    for scan in scan_list :
        np.save(os.path.join(basedir, expname, 'poses', scan), train_c2w)

    print("Data load shape")
    print(f"image shape {train_imgs.shape}")
    print(f"poses shape {train_c2w.shape}")

    H, W = int(train_imgs.shape[2]), int(train_imgs.shape[3])
    num_scan = len(scan_list)

    # Plot images
    fig_path = os.path.join(basedir, expname, 'figures')
    for idx in range(num_scan):
        dir_path = os.path.join(fig_path, scan_list[idx])
        os.makedirs(dir_path, exist_ok=True)

    train_imgs = make_input(train_imgs, fig_path, scan_list, n=row)     # [B, 3, N, H, W]

    # Model build
    if args.emb_type == "IMAGE" :
        mae = IMAGE_MAE(args, H, W).to(rank)
    else :
        mae = PATCH_MAE(args, H, W).to(rank)
    #mae = PRO_MAE(args, H, W).to(rank)

    optimizer = torch.optim.Adam(params=mae.parameters(), lr=args.lrate)
    
    # Move gpu
    train_imgs = torch.Tensor(train_imgs).to(rank)  # [B, 3, N, H, W]
    train_c2w = torch.Tensor(train_c2w).to(rank)    # [B, N, 3, 4]
    print("Training input shape", train_imgs.shape, train_c2w.shape)

    # if use multi gpus
    mae.train()
    mae = DDP(mae, device_ids=[rank], find_unused_parameters=True)
    print("Data parallel model with Multi gpus!")
    
    # Train
    with open(os.path.join(basedir, expname, 'model_param.txt'), 'w') as f :
        for name, para in mae.named_parameters():
            f.write(f"{name} : {para.requires_grad}\n")

    start = 0
    epochs = args.epochs

    print("Train begin")
    start = start + 1
    for i in trange(start, epochs):
        # Train 
        imgs, poses, object_shuffle_idx = augmenting_images(train_imgs, train_c2w, num_scan)
        
        loss, pred, mask = mae(imgs, poses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log/saving
        if i % args.i_figure == 0 :
            # pred : [B, N, (HxWx3)] 
            pred_img = rearrange(pred, 'B N (H W c) -> B c N H W', H=H, W=W, c=3)
            print(f"Reconstruct image {pred_img.shape}")

            for idx in range(num_scan) :
                fig_name = f'pred_{i}.png'
                png_path = os.path.join(basedir, expname, 'figures', scan_list[object_shuffle_idx[idx]], fig_name)
                image_plot(pred_img[idx], row=row, save_fig=png_path)
 
        if i % args.i_weight == 0 : 
            model_name = f'mae_weight.tar'
            print("[SAVE] Model weights", model_name)
            torch.save({
            'model_state_dict' : mae.module.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(basedir, expname, 'weights', model_name))
            
        if i % args.i_print == 0 :
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.mean().item()}")

if __name__ == '__main__' :
    parser = mae_args_parser()
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)