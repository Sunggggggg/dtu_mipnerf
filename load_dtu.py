import os
import numpy as np
import cv2
from PIL import Image
import torch

data_dir = 'scan8'
light_str = 'max'
factor = 4

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
    sh = img.shape
    max_fn = lambda x: max(x, patch_size)
    out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
    img = cv2.resize(img, out_shape, mode)
    return img

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def recenter_poses(poses):
    """Recenter poses around the origin."""
    cam2world = poses_avg(poses)
    poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
    return unpad_poses(poses)

def rescale_poses(poses):
    """Rescales camera poses according to maximum x/y/z value."""
    s = np.max(np.abs(poses[:, :3, -1]))
    out = np.copy(poses)
    out[:, :3, -1] /= s
    return out

def focus_pt_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
    """Calculates a forward facing spiral path for rendering for DTU."""

    # Get radii for spiral path using 60th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
    render_poses = np.stack(render_poses, axis=0)

    return render_poses

def load_dtu_data(data_dir="data/Rectified/images", 
                    dtu_mask_path="data/Rectified/mask", 
                    dtu_scan="scan8", 
                    factor=4,
                    dtu_splite_type='pixelnerf',
                    dtuhold=8):
    """
    data_dir : Data load path

    """
    # Load renderings 
    n_images = len(os.listdir(data_dir)) // 8

    images, p2c, c2w = [], [], []
    for i in range(1, n_images+1) :
        # Load image
        fname = os.path.join(data_dir, f'rect_{i:03d}_{light_str}.png')
        image = np.array(Image.open(fname), dtype=np.float32) / 255.

        # Load projection matrix from file.
        fname = os.path.join(data_dir, f'../../Calibration/cal18/pos_{i:03d}.txt')
        projection = np.loadtxt(fname, dtype=np.float32)
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        c2w.append(pose)
        
        if factor > 0 :
            image = downsample(image, factor)
            camera_mat = np.diag([1./factor, 1./factor, 1.]).astype(np.float32) @ camera_mat
            p2c.append(np.linalg.inv(camera_mat))
        images.append(image)

    images = np.stack(images)   # [N, H, W, 3]
    p2c = np.stack(p2c)         # [N, 3, 3]
    c2w = np.stack(c2w)         # [N, 3, 4]

    print(images.shape, p2c.shape, c2w.shape)

    # Center and scale poses.
    c2w = rescale_poses(recenter_poses(c2w))

    if dtu_splite_type == 'pixelnerf' :
        i_train = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        i_exclude = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        i_test = [i for i in np.arange(49) if i not in i_train + i_exclude]
    else :
        i_all = np.arange(images.shape[0])
        i_train = i_all[i_all % dtuhold != 0]
        i_test = i_all[i_all % dtuhold == 0]
    print("Train", i_train, "Test", i_test)
    # Make Novel view
    render_poses = generate_spiral_path_dtu(c2w, 40)        # [N, 3, 4]

    return images, c2w, p2c, render_poses, i_train, i_exclude, i_test
    # Load mask
    if dtu_mask_path :
        masks = []
        idr_scans = ['scan40', 'scan55', 'scan63', 'scan110', 'scan114']

        if dtu_scan in idr_scans :
            maskf_fn = lambda x: os.path.join(dtu_mask_path, dtu_scan, 'mask', f'{x:03d}.png')
        else :
            maskf_fn = lambda x: os.path.join(dtu_mask_path, dtu_scan, f'{x:03d}.png')

        for i in i_test :
            fname = maskf_fn(i)
            image = np.array(Image.open(fname), dtype=np.float32)[:, :, :3] / 255.
            image = (image == 1).astype(np.float32)
            if factor > 0 :
                image = downsample(image, factor, 8, mode=cv2.INTER_NEAREST)
            masks.append(image)
        masks = np.stack(masks)

    return images, c2w, p2c

def load_nerf_dtu_data(basedir, factor=4):
    nerf_dtu_dir = os.path.join(basedir, 'Rectified')
    scan_list = ['scan40', 'scan55', 'scan63', 'scan110', 'scan114']
    
    # Same input image size
    train_imgs, train_c2w, train_p2c = [], [], []
    for scan in scan_list :
        scandir = os.path.join(nerf_dtu_dir, scan)
        images, c2w, p2c, render_poses, i_train, i_exclude, i_test = load_dtu_data(data_dir=scandir, factor=factor)

        print(scan, 'Loaded dtu', images.shape, render_poses.shape)
        print(i_train, i_exclude, i_test)
        
        # train
        train_imgs.append(images[i_exclude])          # [N, H, W, 3]
        train_c2w.append(c2w[i_exclude])              # [N, 3, 4]
        train_p2c.append(p2c[i_exclude])              # 

    train_imgs = np.stack(train_imgs, 0)        # [O, N, H, W, 3]    
    train_c2w = np.stack(train_c2w, 0)          # [O, N, 3, 4]
    train_p2c = np.stack(train_p2c, 0)          # [O, N, 3, 3]

    return train_imgs, train_c2w, train_p2c, scan_list

###################################################################################################
# Sampling Diet NeRF type
from scipy.spatial.transform import Rotation

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def interp(pose1, pose2, s):
    assert pose1.shape == (3, 4)
    assert pose2.shape == (3, 4)

    # Camera translation 
    C = (1 - s) * pose1[:, -1] + s * pose2[:, -1]
    assert C.shape == (3,)

    # Rotation from camera frame to world frame
    R1 = Rotation.from_matrix(pose1[:, :3])
    R2 = Rotation.from_matrix(pose2[:, :3])
    R = slerp(R1.as_quat(), R2.as_quat(), s)
    R = Rotation.from_quat(R)
    R = R.as_matrix()
    assert R.shape == (3, 3)
    transform = np.concatenate([R, C[:, None]], axis=-1)
    return torch.tensor(transform, dtype=pose1.dtype)

def interp3(pose1, pose2, pose3, s12, s3):
    return interp(interp(pose1, pose2, s12).cpu(), pose3, s3)

def dtu_sampling_pose_interp(N, poses):
    """
    return [N, 3, 4]
    """
    sample_poses = []
    
    for _ in range(N) :
        rend_i = np.random.choice(poses.shape[0], size=3, replace=False)
        pose1, pose2, pose3 = poses[rend_i, :3, :4].cpu()
        s12, s3 = np.random.uniform([0.0, 1.0], size=2)
        s12, s3 = [np.clip(s, 0.3, 1.0) for s in [s12, s3]]
        sample_poses.append(interp3(pose1, pose2, pose3, s12, s3))
    return torch.stack(sample_poses, 0)