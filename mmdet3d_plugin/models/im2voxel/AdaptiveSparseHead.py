import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_head


def topk_wo_grad(occ_preds_flatten, topk=10):
    topk_values, topk_indices = torch.topk(occ_preds_flatten, k=topk, dim=1)
    mask_hard = torch.zeros_like(occ_preds_flatten)
    mask_hard.scatter_(1, topk_indices, 1.0)
    return mask_hard


@HEADS.register_module()
class AdaptiveSparseHead(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 topk_list=None,
                 voxel_size_list=None,
                 n_voxels_list=None,
                 base_head_configs=None,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.topk_list = topk_list if topk_list is not None else []
        self.voxel_size_list = voxel_size_list if voxel_size_list is not None else []
        self.n_voxels_list = n_voxels_list if n_voxels_list is not None else []

        self.base_heads = nn.ModuleList()
        base_head_configs = base_head_configs if base_head_configs is not None else []
        for config in base_head_configs:
            self.base_heads.append(build_head(config))

        self.occ_pred_heads = nn.ModuleList()
        for _ in range(len(self.base_heads) - 1):
            self.occ_pred_heads.append(nn.Sequential(nn.Linear(embed_dims, 1), nn.Sigmoid()))
        
        self.loss = nn.BCELoss()
        
    def forward(self, mlvl_feats, img_meta, mlvl_dpt_dists):
        bs = mlvl_feats[0].shape[0]
        assert bs == 1

        volumes = [None] * len(self.base_heads)
        occ_preds_list = []
        indices_list = [None] * len(self.base_heads)

        finest_downsample_factor = 4
        for i in range(len(self.base_heads)):
            downsample_factor_current = finest_downsample_factor * (2**(len(self.base_heads) - 1 - i))
            height = img_meta['img_shape'][0] // downsample_factor_current
            width = img_meta['img_shape'][1] // downsample_factor_current

            feat_idx = len(self.base_heads) - 1 - i
            current_feat = mlvl_feats[feat_idx][:, :, :, :height, :width]
            current_dpt_dist = mlvl_dpt_dists[feat_idx][:, :, :, :height, :width]

            if i == 0:
                volumes[i] = self.base_heads[i]([current_feat], img_meta, mlvl_dpt_dists=[current_dpt_dist])
            else:
                upsampled_volume = F.interpolate(
                    volumes[i-1], 
                    scale_factor=2, 
                    mode='trilinear', 
                    align_corners=False
                )
                
                occ_preds_current = self.occ_pred_heads[i-1](upsampled_volume.permute(0,2,3,4,1)).reshape(bs, -1)
                occ_preds_list.append(occ_preds_current)
                if (i - 1) < len(self.topk_list):
                    indices_current = topk_wo_grad(occ_preds_current, topk=self.topk_list[i-1]).squeeze(0)
                    indices_list[i] = indices_current
                
                volumes[i] = upsampled_volume + self.base_heads[i](
                    [current_feat], 
                    img_meta, 
                    proposal=indices_list[i], 
                    mlvl_dpt_dists=[current_dpt_dist]
                )

        volume_out = volumes[-1] 
        if occ_preds_list == []:
            occ_preds = None
            bs, _, v_h, v_w, v_z = volume_out.shape
            valid = torch.ones([bs, 1, v_h, v_w, v_z]).to(volume_out.device)
        else:
            occ_preds = torch.cat(occ_preds_list[::-1], dim=1) 
            valid = self.get_valid(indices_list[len(self.base_heads)-1]).unsqueeze(0).unsqueeze(0).detach()
        
        return volume_out, valid, occ_preds
    
    def get_valid(self, indices_0):
        high_res = indices_0.view(self.n_voxels_list[-1][0], self.n_voxels_list[-1][1], self.n_voxels_list[-1][2])
        valid = high_res.bool().long()
        return valid
    
    def occ_loss(self, occ_pred, sem_occ_gt, geo_occ_gt):
        bs, N = occ_pred.shape
        loss_occ = self.loss(occ_pred, geo_occ_gt[:, 0:N].float()).mean() * 0.5
        return {'loss_occ': loss_occ}