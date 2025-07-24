import numpy as np
import torch


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

def compute_projection(img_meta, stride=1):
    projection = []
    intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
    intrinsic[:2] /= ratio
    extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
    for extrinsic in extrinsics:
        projection.append(intrinsic @ extrinsic[:3])
    return torch.stack(projection)