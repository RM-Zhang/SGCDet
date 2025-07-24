import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
from .extractor_matching import ResNetFPN


def knn(x, ref, k, maskself=False):
    '''
    x: (b,c,num_src)
    ref: (b, c, num_ref)
    k: top-k neigbour for each src
    assume x and ref are the same here!!
    '''
    
    inner = -2 * torch.matmul(x.transpose(2, 1), ref) #(B,num_src,num_ref)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) #(B,1,num_src)
    yy = torch.sum(ref ** 2, dim=1, keepdim=True) #(B,1,num_ref)
    
    pairwise_distance = -yy - inner - xx.transpose(2, 1) #(B, num_src, num_ref)
    
    # mask out self
    if maskself:
        assert x.shape == ref.shape
        mask = torch.arange(xx.shape[2]) # (num_src)
        pairwise_distance[:, mask, mask] = -100000
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,k)
    return idx


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, maskself=False):
    '''
    Args:
        tar_pose: target pose [num_tgt, 4, 4]. c2w
        ref_poses: reference poses [num_src, 4, 4]. C2W. we pick neighbour views from this pool
        num_select: the number of nearest views to select
    Returns: the selected indices # (num_tgt, k)
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)

    tar_cam_locs = tar_pose[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N) only translation
    ref_cam_locs = ref_poses[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N)
    neigh_ids = knn(tar_cam_locs, ref_cam_locs, k=num_select, maskself=maskself)[0] # (num_src, k)

    return neigh_ids


def get_closest_frame_ids(num_cams, num_select):
    assert num_select % 2 == 0
    
    main_indices = torch.arange(num_cams).unsqueeze(1)  # [N, 1]
    past_offsets = torch.arange(-num_select//2, 0).unsqueeze(0)  # [1, K//2]
    future_offsets = torch.arange(1, num_select//2 + 1).unsqueeze(0)  # [1, K//2]
    offsets = torch.cat([past_offsets, future_offsets], dim=1)  # [1, K]
    
    closest_frames = main_indices + offsets  # [N, K]
    closest_frames[0:num_select//2, :] = closest_frames[0:num_select//2, :] + num_select//2 + 1
    closest_frames[num_cams-num_select//2:num_cams, :] = closest_frames[num_cams-num_select//2:num_cams, :] - num_select//2 -1
    return closest_frames


def collect_proj(w2c, intr, neighbor_ids):
    '''
    w2c: (num_src, 4, 4) cuda
    intr: (4, 4). cuda or (num_src, 4, 4)
    neighbour_ids: (num_src, k)
    return:
        [(b, 4, 4), (b, 4, 4) ...] len=k. b=num_src. each (4, 4) is K[R|t], world2img
    '''
    if len(intr.shape) == 2:
        intr = intr.unsqueeze(0).repeat(w2c.shape[0], 1, 1) # (4, 4) -> (num_src, 4, 4)
    proj = torch.matmul(intr, w2c) # (num_src, 4, 4)
    num_src = neighbor_ids.shape[0]
    num_nei = neighbor_ids.shape[1]
    nei_projs = proj[neighbor_ids.view(-1)].view(num_src, num_nei, *proj.shape[1:]) # (num_src, k, 4, 4)
    nei_projs = torch.unbind(nei_projs, dim=1) 
    return proj, nei_projs


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4] k[R|t]. [R|t] is w2c
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        if len(depth_values.shape) == 2:
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                                1)  # [B, 3, Ndepth, H*W]
        else: # depth_values.shape: (b, n_depth, h, w)
            depth_values = depth_values.view(batch, num_depth, -1).unsqueeze(1) # (b,1, n_depth, h*w)
            # print(depth_values.shape, rot_xyz.shape)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values  # [B, 3, Ndepth, H*W]
            
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class ConvBnReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SimpleUnet2D(nn.Module):
    def __init__(self, in_channel):
        super(SimpleUnet2D, self).__init__()

        self.conv1 = ConvBnReLU2D(in_channel, 2*in_channel, stride=2)
        self.conv2 = ConvBnReLU2D(2*in_channel, 2*in_channel)
        self.conv3 = ConvBnReLU2D(2*in_channel, 4*in_channel, stride=2)
        self.conv4 = ConvBnReLU2D(4*in_channel, 4*in_channel)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(4*in_channel, 2*in_channel, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*in_channel),
            nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(2*in_channel, in_channel, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0)) # (b, 2*d, h/2, w/2)
        x = self.conv4(self.conv3(conv2)) # (b, 4*d, h/4, w/4)
        x = conv2 + self.conv9(x) # (b, 2*d, h/2, w/2)
        x = conv0 + self.conv11(x) # (b, d, h, w)
        return x
    
   
@HEADS.register_module()
class DepthNet_Fusion(nn.Module):
    def __init__(self, neighbor_img_num, downsample_factor, dbound, mono_channels=256, loss_weight=0.5, max_tol=0, init_weight='ImageNet'):
        super().__init__()
        self.fp16_enabled = False
        self.max_tol = max_tol
        self.downsample_factor = downsample_factor
        self.loss_weight = loss_weight
        
        self.neighbor_img_num = neighbor_img_num
        self.dbound = dbound
        self.depth_channels = round((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.depth_values = np.arange(self.dbound[0], self.dbound[1], self.dbound[2], dtype=np.float32) + self.dbound[2]/2 # depth interval center
        
        self.fnet_mvs = ResNetFPN(input_dim=3, output_dim=128, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=init_weight)
        self.correlation_regulation = SimpleUnet2D(in_channel=self.depth_channels)
        
        self.fnet_mono = ConvBnReLU2D(in_channels=mono_channels, out_channels=128)
        self.mono_regulation = SimpleUnet2D(in_channel=128)
        
        self.fusion_regulation = SimpleUnet2D(in_channel=self.depth_channels + 128)
        self.depth_reg = nn.Conv2d(self.depth_channels + 128, self.depth_channels, kernel_size=3, stride=1, padding=1)


    @auto_fp16(apply_to=('xs'))
    def forward(self, xs, imgs, img_metas, stride):
        B, num_src, C, H, W = xs.shape
        depth_preds = torch.empty((B, num_src, self.depth_channels, H, W), device=xs.device)
        
        batch_index = 0
        for x, img, img_meta in zip(xs, imgs, img_metas):
            f_mvs = self.fnet_mvs(img)
            channel_num = f_mvs.shape[1]
            
            src_w2c = torch.tensor(np.array(img_meta['lidar2img']['extrinsic']), device=x.device) # stack to tensor (num_src, 4, 4)
            src_intrinsic = torch.tensor(np.array(img_meta['lidar2img']['intrinsic']), device=x.device) # (4, 4). correspond to original img resolution. or (num_src, 4, 4)
            
            # whether intrinsic is list: intrin_list_flag
            intrin_list_flag = isinstance(img_meta['lidar2img']['intrinsic'], list)
            ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
            src_feat_intrinsic = src_intrinsic.clone()
            if not intrin_list_flag: # (4,4)
                src_feat_intrinsic[:2] /= ratio # (4,4). correspond to src_feat intrisinc
            else: # (num_src, 4, 4)
                src_feat_intrinsic[:,:2] /= ratio
            
            # 1. get neighbour frames
            k=min(self.neighbor_img_num, num_src-1) # incase num_src is very few....
            ## choose position nearest frames
            # src_c2w = src_w2c.inverse() # c2w (num_src, 4, 4) 
            # neighbor_ids =  get_nearest_pose_ids(src_c2w, src_c2w, k, maskself=True) # (num_src, k)
            ## choose time closest frames
            neighbor_ids = get_closest_frame_ids(num_src, k)
            
            # 2. plane sweep
            nei_features = f_mvs[neighbor_ids.view(-1)].view(num_src, k, *f_mvs.shape[1:]) # (num_src, k, c, h, w)
            nei_features = torch.unbind(nei_features, dim=1) # [(bs, c, h, w), (bs, c, h, w), ...] len=k
            # nei_img = torch.unbind(img[neighbor_ids.view(-1)].view(num_src, k, *img.shape[1:]), dim=1)
            ref_proj, nei_projs = collect_proj(src_w2c, src_feat_intrinsic, neighbor_ids) # (b, 4, 4). [(bs, 4, 4), (bs, 4, 4), ...]. intrinsic should change !!!
            depth_values = torch.tensor(self.depth_values, device=x.device).unsqueeze(0).repeat(num_src, 1) # (bs, n_depth)
            
            # we follow definition in mvsnet. neigbour is src in mvsnet.
            # 3. cost volume
            correlation = torch.zeros((num_src, self.depth_channels, H, W), device=x.device) # (num_src, num_depth, h, w)
            for nei_fea, nei_proj in zip(nei_features, nei_projs): # len=k
                # warpped features
                warped_features = homo_warping(nei_fea, nei_proj, ref_proj, depth_values) # (num_src, c, num_depth, h, w)
                
                if self.training:
                    correlation = correlation + ((warped_features*f_mvs.unsqueeze(2)).sum(dim=1))/torch.sqrt(torch.tensor(channel_num).float()) # (num_src, num_depth, h, w)
                else:
                    correlation += ((warped_features*f_mvs.unsqueeze(2)).sum(dim=1))/torch.sqrt(torch.tensor(channel_num).float())
            correlation = correlation / k
            
            # 4. cost volume regularization
            cost_reg = self.correlation_regulation(correlation) # (num_src, num_depth, h, w)
            
            f_mono = self.fnet_mono(x)
            mono_reg = self.mono_regulation(f_mono)
            
            prob_volume = self.fusion_regulation(torch.cat([cost_reg, mono_reg],dim=1))
            prob_volume = self.depth_reg(prob_volume)
            prob_volume = F.softmax(prob_volume, dim=1) # (num_src, d, h, w). depth probability
            
            depth_preds[batch_index] = prob_volume
            batch_index += 1
        return depth_preds
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        
        if self.downsample_factor % 1 == 0:
            B, N, H, W = gt_depths.shape
            gt_depths = gt_depths.view(
                B * N,
                H // self.downsample_factor,
                self.downsample_factor,
                W // self.downsample_factor,
                self.downsample_factor,
                1,
            )  # B*N H/ds ds W/ds ds 1
            gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  # B*N H/ds W/ds 1 ds ds
            gt_depths = gt_depths.view(
                -1, self.downsample_factor * self.downsample_factor)  # B*N*H/ds*W/ds ds*ds
            gt_depths_tmp = torch.where(gt_depths == 0.0,
                                        1e5 * torch.ones_like(gt_depths),
                                        gt_depths)
            gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # B*N*H/ds*W/ds 1(min=1e5)
            gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                    W // self.downsample_factor)  # B*N H/ds W/ds
        else:
            gt_depths = F.interpolate(gt_depths, scale_factor=1 / self.downsample_factor, mode='nearest')
            B, N, H, W = gt_depths.shape
            gt_depths = gt_depths.view(B*N, H, W)
            
        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:] # 0: [min,min+delta]; D-1:[max-delta,max]
        gt_depths_errtol = self.error_tol(gt_depths)
        return gt_depths_errtol.float()
    
    def error_tol(self, gt_depths_onehot_):
        if self.max_tol < 1:
            return gt_depths_onehot_
        error_tol = [-self.max_tol, self.max_tol+1]
        padding = gt_depths_onehot_.new_zeros(gt_depths_onehot_.shape[0], 1)
        gt_depths_onehot = gt_depths_onehot_.clone()
        for error in range(error_tol[0], error_tol[1]):
            if error < 0 :  # move left
                gt_depths_onehot = gt_depths_onehot + torch.cat([gt_depths_onehot[..., 1:], padding], dim=-1)
            elif error > 0:  # move right
                gt_depths_onehot = gt_depths_onehot + torch.cat([padding, gt_depths_onehot[..., :-1]], dim=-1)
        gt_depths_onehot = (gt_depths_onehot / (gt_depths_onehot + 1e-5))
        return gt_depths_onehot
    
    @force_fp32(apply_to=('depth_preds'))
    def loss(self, depth_labels, depth_preds):
        # depth_labels [B,N,H,W]
        # depth_preds [B,N,C,H,W]
        if depth_labels.dim() == 3:
            depth_labels = depth_labels.unsqueeze(0)
        depth_labels = self.get_downsampled_gt_depth(depth_labels) # [B * N * H/ds * W/ds, C]
        
        depth_preds = depth_preds.permute(0,1,3,4,2).contiguous().view(-1, self.depth_channels) # [batch_size*num_cam*H*W, C]
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        depth_preds = torch.clamp(depth_preds, min=1e-7, max=1-1e-7)
        
        depth_loss = (F.binary_cross_entropy(
            depth_preds[fg_mask],
            depth_labels[fg_mask],
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum()))
            
        return {"loss_dpt": self.loss_weight * depth_loss}