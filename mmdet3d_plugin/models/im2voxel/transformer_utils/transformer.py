# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from .deformable_cross_attention import MSDeformableAttention3D


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    """
    def __init__(self,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            # if isinstance(m, MSDeformableAttention3D) or isinstance(m, DeformSelfAttention):
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            ref_3d,
            vox_coords,
            unmasked_idx,
            bev_pos=None,
            prev_bev=None,
            img_meta=None,
            **kwargs):
        """
        obtain voxel features.
        """

        bs = mlvl_feats[0].size(0)
        # To do, implement a function which supports bs > 1
        assert bs == 1
        
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # [v_h*v_w*v_z, bs, c]
        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx, 3], :, :]
        
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, bs, c]
            unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx, 3], :, :]

        unmasked_ref_3d = ref_3d[vox_coords[unmasked_idx, 3], :]
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device) # [1,1,N_unmasked,3] unmasked voxel 3D coordinates
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=unmasked_bev_queries.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            unmasked_bev_queries,
            feat_flatten,
            feat_flatten,
            ref_3d=unmasked_ref_3d,
            bev_pos=unmasked_bev_bev_pos if bev_pos is not None else None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_meta=img_meta,
            prev_bev=None,
            **kwargs
        )
        return bev_embed


@TRANSFORMER.register_module()
class PerceptionTransformer_DFA3D(PerceptionTransformer):
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            ref_3d,
            vox_coords,
            unmasked_idx,
            bev_pos=None,
            prev_bev=None,
            img_meta=None,
            mlvl_dpt_dists=None,
            **kwargs):
        """
        obtain voxel features. needs mlvl_dpt_dists for 3D attention
        """

        bs = mlvl_feats[0].size(0)
        # To do, implement a function which supports bs > 1
        assert bs == 1
        
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # [v_h*v_w*v_z, bs, c]
        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx, 3], :, :] # [N, bs, c]
        
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, bs, c]
            unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx, 3], :, :]

        unmasked_ref_3d = ref_3d[vox_coords[unmasked_idx, 3], :] # [N, 3]
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device) # [1, 1, N, 3]
        
        feat_flatten = []
        spatial_shapes = []
        dpt_dist_flatten = []
        for lvl, (feat, dpt_dist) in enumerate(zip(mlvl_feats, mlvl_dpt_dists)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # [num_cam, bs, hw, c]
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2) # [num_cam, bs, hw, dpt]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=unmasked_bev_queries.device)
        
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        dpt_dist_flatten = dpt_dist_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, dpt)

        bev_embed = self.encoder(
            unmasked_bev_queries, # [N, bs, C]
            feat_flatten,
            feat_flatten,
            value_dpt_dist=dpt_dist_flatten,
            ref_3d=unmasked_ref_3d, # [1, 1, N, 3]
            bev_pos=unmasked_bev_bev_pos if bev_pos is not None else None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_meta=img_meta,
            prev_bev=None,
            **kwargs
        )   
        return bev_embed # (bs, num_query, embed_dims)