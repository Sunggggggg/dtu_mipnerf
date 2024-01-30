import os
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# 
from config import config_parser
from set_multi_gpus import set_ddp, myDDP
from metric import get_metric
# 
from scheduler import MipLRDecay
from loss import MipNeRFLoss
from model import MipNeRF
#
from load_dtu import load_dtu_data, get_rays_dtu, get_radii, shift_origins, dtu_sampling_pose_interp
from nerf_render import *


def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    # Load dataset
    near, far = 0.5, 3.5
    # images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test \ # blender
    images, c2w, p2c, render_poses, i_train, i_test = load_dtu_data(data_dir=args.datadir, factor=args.scale)
    H, W = int(images.shape[1]), int(images.shape[2])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # logging dir
    logdir = os.path.join(basedir, expname, 'eval.txt')

    # Build model
    model = MipNeRF(
        use_viewdirs=args.use_viewdirs,
        randomized=args.randomized,
        ray_shape=args.ray_shape,
        white_bkgd=args.white_bkgd,
        num_levels=args.num_levels,
        N_samples=args.N_samples,
        hidden=args.hidden,
        density_noise=args.density_noise,
        density_bias=args.density_bias,
        rgb_padding=args.rgb_padding,
        resample_padding=args.resample_padding,
        min_deg=args.min_deg,
        max_deg=args.max_deg,
        viewdirs_min_deg=args.viewdirs_min_deg,
        viewdirs_max_deg=args.viewdirs_max_deg,
        device=torch.device(rank),
    )
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    scheduler = MipLRDecay(optimizer, lr_init=args.lr_init, lr_final=args.lr_final, 
                           max_steps=args.max_iters, lr_delay_steps=args.lr_delay_steps, 
                           lr_delay_mult=args.lr_delay_mult)
    # optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    # Training hyperparams
    N_rand = args.N_rand
    max_iters = args.max_iters + 1
    start = 0 + 1
    nerf_weight_path = 'nerf_weights.tar'

    # Load pretrained model
    if args.nerf_weight != None :
        print("Load MipNeRF model weight :", args.nerf_weight)
        ckpt = torch.load(args.nerf_weight) # 

        model.load_state_dict(ckpt['network_fn_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        nerf_weight_path = 'nerf_tune_weights.tar'

    # Set multi gpus
    model = DDP(model, device_ids=[rank])

    # Loss func (Mip-NeRF)
    loss_func = MipNeRFLoss(args.coarse_weight_decay)
    
    # Randomly sampling function
    sampling_pose_function = lambda N : dtu_sampling_pose_interp(N, poses=torch.Tensor(render_poses))

    #################################
     # Move training data to GPU
    model.train()
    c2w = torch.Tensor(c2w).to(rank)
    p2c = torch.Tensor(p2c).to(rank)        # [3, 3]
    render_poses = torch.Tensor(render_poses).to(rank)

    for i in trange(start, max_iters):
        # 1. Random select image
        img_i = np.random.choice(i_train)
        pose = c2w[img_i, :3,:4]
        K = p2c[img_i]
        target = images[img_i]

        target = torch.Tensor(target).to(rank)
        pose = torch.Tensor(pose).to(rank)          # [3, 4]
        # 2. Generate rays
        rays_o, rays_d = get_rays_dtu(H, W, K, pose)
        radii = get_radii(rays_d)
        rays_o = shift_origins(rays_o, rays_d, 0.0)
    
        # 3. Random select rays
        if i < args.precrop_iters:
            dH = int(H//2 * args.precrop_frac)
            dW = int(W//2 * args.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1)
            if i == start:
                print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()        # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)
        radii = radii[select_coords[:, 0], select_coords[:, 1]]     # (N_rand, 1)
        lossmult = torch.ones_like(radii)                             # (N_rand, 1) 
        batch_rays = torch.stack([rays_o, rays_d], 0)                 # (2, N_rand, 3)
        target = target[select_coords[:, 0], select_coords[:, 1]]     # (N_rand, 3)
        
        # 4. Rendering 
        comp_rgbs, _, _ = render_mipnerf(H, W, K, chunk=args.chunk, mipnerf=model, 
                                         rays=batch_rays, radii=radii, near=near, far=far, use_viewdirs=args.use_viewdirs)
        
        # 5. loss and update
        loss, (mse_loss_c, mse_loss_f), (train_psnr_c, train_psnr_f) = loss_func(comp_rgbs, target, lossmult.to(rank))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Rest is logging
        if i%args.i_weights==0 and i > 0:
            path = os.path.join(basedir, expname, nerf_weight_path)
            torch.save({
                'network_fn_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict()
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', c2w[i_test].shape)
            with torch.no_grad():
                rgbs = render_path(c2w[i_test], H, W, p2c, args.chunk, model, 
                                    near=near, far=far, use_viewdirs=args.use_viewdirs, 
                                    savedir=testsavedir)
                eval_psnr, eval_ssim, eval_lpips = get_metric(rgbs[:, -1], images[i_test], None, torch.device(rank))    # Use fine model
            if rank == 0 :
                with open(logdir, 'a') as file :
                    file.write(f"{i:06d}-iter PSNR : {eval_psnr:.3f}, SSIM : {eval_ssim:.3f}, LPIPS : {eval_lpips:.3f}\n")
            print('Saved test set')

        if i%args.i_print==0 and rank == 0 :
            tqdm.write(f"[TRAIN]    Iter: {i:06d} Total Loss: {loss.item():.6f} PSNR: {train_psnr_f.item():.4f}")

if __name__ == '__main__' :
    parser = config_parser()
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))
