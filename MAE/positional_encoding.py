import torch
import numpy as np
import cv2

def get_2d_sincos_pos_embed(embed_dim, num_h, num_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(num_h, dtype=np.float32)
    grid_w = np.arange(num_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, num_h, num_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed    

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0, embed_dim
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def matrix2angle(poses):
    """ Extrinsic matrix to theta, phi
    poses   [B, N, 4, 4]
    """
    xyz = poses[..., :3, -1]                    # [B, N, 3]
    x, y, z = torch.split(xyz, 1, dim=-1)

    theta = torch.arctan2(y, x).squeeze(-1)                 # [B, N]
    phi = torch.arctan2((x**2+y**2)**0.5, z).squeeze(-1)    # [B, N]
    
    return theta, phi

def view_sinusoid_encoding(thetas, phis, d_hid, cls_token=False):
    """ Positional encoding 
    thetas       [B, N]
    phis         [B, N]

    return
        [B, N, d_hid]
    {(sin, cos), (sin, cos)}, (sin, cos) ...
    """
    B, N = thetas.shape
    assert thetas.shape == phis.shape, ""

    batch_list = list()
    for b in range(B):
        t = get_1d_sincos_pos_embed_from_grid(d_hid//2, thetas[b].float().detach().cpu().numpy())
        p = get_1d_sincos_pos_embed_from_grid(d_hid//2, phis[b].float().detach().cpu().numpy())

        emb = np.concatenate([t, p], axis=1)
        batch_list.append(emb)
    pos_embed = np.array(batch_list)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([B, 1, d_hid]), pos_embed], axis=1)

    return torch.from_numpy(pos_embed).type(thetas.type())

def get_embed_dim(dim, embed_element=3):
    """
    embed_element : x, y, z 
    """
    embed_dim = dim

    if embed_dim % embed_element*2 != 0 :   #1024%6 = 4
        pad = embed_element*2 - embed_dim % (embed_element*2) 
        embed_dim += pad + embed_element*2
    else :
        pad = 0
    return embed_dim, pad

def get_3d_sincos_pos_embed(poses, embed_dim, cls_token=False):
    """
    xyz : [B, N, 4, 4]
    """
    B = poses.shape[0]
    # x,y,z
    xyz = poses[..., :3, -1]
    x, y, z = torch.split(xyz, 1, dim=-1)   # [B, N, 1]

    if embed_dim == 1024 :
        pad_embed_dim = embed_dim+8
    elif embed_dim == 2048 :
        pad_embed_dim = embed_dim+4

    # 
    emb_list = list()
    for b in range(B):
        root_poses = np.stack([cv2.Rodrigues(pose[:3,:3].float().detach().cpu().numpy())[0] for pose in poses[b]])  # [N, 3, 1]
        enc_r1 = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, root_poses[:, 0])
        enc_r2 = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, root_poses[:, 0])
        enc_r3 = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, root_poses[:, 0])

        enc_x = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, x[b].float().detach().cpu().numpy())    # [N, embed_dim//3]
        enc_y = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, y[b].float().detach().cpu().numpy())
        enc_z = get_1d_sincos_pos_embed_from_grid(pad_embed_dim//6, z[b].float().detach().cpu().numpy())

        emb_list.append(np.concatenate([enc_r1, enc_r2, enc_r3, enc_x, enc_y, enc_z], 1)[..., :embed_dim])
    
    pos_embed = np.array(emb_list)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([xyz.shape[0], 1, embed_dim]), pos_embed], axis=1)
    return torch.from_numpy(pos_embed).type(poses.type())

