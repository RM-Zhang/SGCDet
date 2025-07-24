import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class DenseHead(nn.Module):
    def __init__(
        self,
        *args,
        voxel_size=None,
        n_voxels=None,
        embed_dims,
        cross_transformer,
        **kwargs
    ):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.n_voxels = torch.tensor(n_voxels)
        self.embed_dims = embed_dims

        self.cross_transformer = build_transformer(cross_transformer)

        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.n_voxels[0]), torch.arange(self.n_voxels[1]),torch.arange(self.n_voxels[2]), 
            indexing='ij') # [v_h, v_w, v_z]
        idx = torch.arange(self.n_voxels[0] * self.n_voxels[1] * self.n_voxels[2]) # [v_h*v_w*v_z]
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1) # [v_h*v_w*v_z, 4] 4->h,w,z,idx

        points = torch.stack(torch.meshgrid([
            torch.arange(self.n_voxels[0]),
            torch.arange(self.n_voxels[1]),
            torch.arange(self.n_voxels[2])
        ]))
        new_origin =  - self.n_voxels / 2. * self.voxel_size
        points = points * self.voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
        points = points.view(3,-1).permute(1,0) # [v_h*v_w*v_z, 3] standard coordinates
        
        return vox_coords, points
    
    def forward(self, mlvl_feats, img_meta=None, proposal=None, **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device        

        assert bs == 1
        
        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()
        volume_queries = torch.zeros([(self.n_voxels[0])*(self.n_voxels[1])*(self.n_voxels[2]), self.embed_dims]).to(device)
        if proposal is None:
            proposal = torch.ones([(self.n_voxels[0])*(self.n_voxels[1])*(self.n_voxels[2])]).to(device)
        unmasked_idx = torch.nonzero(proposal > 0).view(-1) # [L] unmaksed voxel indices
        
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            volume_queries,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            bev_pos=None,
            prev_bev=None,
            img_meta=img_meta,
            **kwargs
        ).squeeze(0) # [bs, L, c] -> [L, c]

        volume_out = torch.zeros([(self.n_voxels[0])*(self.n_voxels[1])*(self.n_voxels[2]), self.embed_dims]).to(device)
        volume_out[vox_coords[unmasked_idx, 3], :] = seed_feats
        volume_out = volume_out.reshape(self.n_voxels[0], self.n_voxels[1], self.n_voxels[2], self.embed_dims)
        volume_out = volume_out.permute(3, 0, 1, 2).unsqueeze(0) # [bs, c, v_h, v_w, v_z]
        return volume_out