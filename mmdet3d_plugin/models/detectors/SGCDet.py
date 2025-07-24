import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from .utils import *


@DETECTORS.register_module()
class SGCDet(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 depth_head,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 voxel_size,
                 voxel_head=None,
                 head_2d=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_gt_dpt=False,
                 depth_loss=False,
                 occ_loss=False,
                 lighting_augmentation=False):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.depth_head = build_head(depth_head)
        self.neck_3d = build_neck(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.voxel_size = voxel_size
        self.voxel_head = build_head(voxel_head) if voxel_head is not None else None
        self.head_2d = build_head(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_gt_dpt = use_gt_dpt
        self.depth_loss = depth_loss
        self.occ_loss = occ_loss
        self.lighting_augmentation = lighting_augmentation
        print(f"model size: {self.compute_model_size():.3f}MB")
            
    def compute_model_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"param_size: {param_size/1024**2}, buffer_size: {buffer_size/1024**2}")
        return size_all_mb

    def build_volume(self, batch):
        img = batch['img']
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img) # [[B*N, 256, 60, 80], [B*N, 512, 30, 40], [B*N, 1024, 15, 20], [B*N, 2048, 8, 10]] [B*N, C, H, W]
        features_2d = self.head_2d.forward(x[-1], batch['img_metas']) if self.head_2d is not None else None
        x = list(self.neck(x))    # [[B*N, 256, 60, 80], [B*N, 256, 30, 40], [B*N, 256 , 15, 20], [B*N, 256 , 8, 10]]
        for i in range(len(x)):
            x[i] = x[i].reshape([batch_size, -1] + list(x[i].shape[1:])) # [B, N, C, H, W]

        stride = int(4)
        
        if self.use_gt_dpt:
            dpt_dist = self.depth_head.get_downsampled_gt_depth(batch['depth_maps']) # [B*N*h*w, d]
            [b, n, _, h, w] = x[0].shape
            dpt_dist = dpt_dist.view(b,n,h,w,-1).permute(0,1,4,2,3) # [B, N, D, H, W]
        else:
            img = img.reshape([batch_size, -1] + list(img.shape[1:]))
            if self.depth_loss == True:
                dpt_dist = self.depth_head(xs=x[0].detach(), imgs=img, img_metas=batch['img_metas'], stride=stride) # [B, N, D, H, W]
            else:
                dpt_dist = self.depth_head(x[0], imgs=img, img_metas=batch['img_metas'], stride=stride) # [B, N, D, H, W]
        mlvl_dpt_dists = [dpt_dist]
        mlvl_dpt_dists.append(F.interpolate(dpt_dist, scale_factor=(1,0.5,0.5), mode='nearest'))
        mlvl_dpt_dists.append(F.interpolate(dpt_dist, scale_factor=(1,0.25,0.25), mode='nearest'))
        
        volume, valid, occ = self.voxel_head(x, batch['img_metas'][0], mlvl_dpt_dists)
        
        if valid == None:
            _, _, v_h, v_w, v_z = volume.shape
            valid = torch.ones([batch_size, 1, v_h, v_w, v_z]).to(volume.device)
        
        return volume, valid, features_2d, dpt_dist, occ

    def extract_feat(self, volumes):
        return self.neck_3d(volumes)

    def forward_train(self, batch):
        avg_vols, valids, features_2d, dpt_dist, occ = self.build_volume(batch)
        x = self.extract_feat(avg_vols)
        
        losses = {}
        losses_det, sem_occ, geo_occ = self.bbox_head.forward_train(x, valids.float(), batch['img_metas'], batch['gt_bboxes_3d'], batch['gt_labels_3d'])
        losses.update(losses_det)
        if self.head_2d is not None:
            losses.update(
                self.head_2d.loss(*features_2d, batch['img_metas']))
        if self.depth_loss == True:
            losses.update(self.depth_head.loss(batch['depth_maps'], dpt_dist))
        if self.occ_loss == True:
            losses.update(self.voxel_head.occ_loss(occ, sem_occ, geo_occ))
            
        return losses
        
    def forward_test(self, batch):
        # not supporting aug_test for now
        return self.simple_test(batch)

    def simple_test(self, batch):
        avg_vols, valids, features_2d, dpt_dist, occ = self.build_volume(batch)
    
        x = self.extract_feat(avg_vols)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, valids.float(), batch['img_metas'])
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass
    