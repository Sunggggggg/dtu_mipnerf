import os
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from .visualize import image_plot

def make_input(imgs, fig_path, object_list, n=5, save_fig=True):
    imgs = imgs.transpose(0, -1, 1, 2, 3)

    if save_fig :
        for idx, _object in enumerate(object_list) :
            png_path = os.path.join(fig_path, _object, 'input.png')
            image_plot(imgs[idx], row=n, save_fig=png_path)
            
    return imgs

def mae_input_format(imgs, poses, mae_input, emb_type='IMAGE'):
    """ NeRF input format with MAE input format (F = nerf_input / N = mae_input)
    args
    imgs  (torch) [F, H, W, 3]
    poses (torch) [F, 4, 4] 

    return
        Cam_pos_encoding = True
        imgs        [B, 3, N, H, W]
        poses       [B, N, 4, 4]

        Cam_pos_encoding = False
        imgs        [B, 3, Hxn, Wxn]
        poses       [B, N, 4, 4] 
    """
    imgs = imgs.permute(3, 0, 1, 2).unsqueeze(0)    # [1, 3, N, H, W]
    poses = poses.unsqueeze(0)                      # [1, N, 4, 4]

    return imgs, poses

def augmenting_images(train_images, train_pose, num_scan):
    """
    images  : [O, 3, N, H, W]
    poses   : [O, N, 4, 4]
    """
    # Shuffle
    object_shuffle_idx = torch.rand((num_scan)).argsort()
    shuffle_imgs, shuffle_poses = train_images[object_shuffle_idx], train_pose[object_shuffle_idx]

    return shuffle_imgs, shuffle_poses, object_shuffle_idx




