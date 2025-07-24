import numpy as np
import torch
import cv2 as cv
import mmcv
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_transformer_layer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(
        self, 
        *args,
        return_intermediate=False,
        dbound=None, 
        **kwargs):
        super(VoxFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.dbound = dbound
        self.fp16_enabled = False

    @staticmethod
    def _compute_projection(img_meta, stride):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def point_sampling(self, reference_points, img_meta=None):
        # input: reference_points [bs, D, HWZ, 2] D -> num_points_in_pillar

        assert reference_points.shape[0]==1 # only support bs=1

        eps = 1e-5
        ogfH = img_meta['img_shape'][0] # 239
        ogfW = img_meta['img_shape'][1] # 320

        # [bs, 1, HWZ, 3] / [bs, D, HWZ, 3]
        origin = torch.tensor(img_meta['lidar2img']['origin']).to(reference_points.device)
        reference_points =  reference_points + origin
        
        # [bs, D, HWZ, 3] -> [D, bs, HWZ, 3]
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        projection = self._compute_projection(img_meta, stride=1).to(reference_points.device) # [num_cam, 3, 4]  stride=1 because reference_points_cam/ogfW 
        num_cam = projection.shape[0]
        
        # [D, B, num_cam, num_query, 3]
        reference_points = reference_points.view(D, B, 1, num_query, 3).repeat(1, 1, num_cam, 1, 1)

        # [D, B, num_cam, num_query, 4]
        reference_points = torch.cat((reference_points, torch.ones(*reference_points.shape[:-1], 1).type_as(reference_points)), dim=-1)
        projection = projection.unsqueeze(0).unsqueeze(0) # [1, 1, num_cam, 3, 4]
        reference_points_cam = torch.matmul(projection, reference_points.permute(0,1,2,4,3)).permute(0,1,2,4,3) # [D, B, num_cam, num_query, 3]
        points_d = reference_points_cam[..., 2:3]
        
        reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(points_d, torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # [D, B, num_cam, num_query, 3]
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        reference_points_cam[..., 2] = (reference_points_cam[..., 2] - self.dbound[0]) / (self.dbound[1]-self.dbound[0])
        
        volume_mask = (points_d > eps)
        # [D, B, num_cam, num_query, 1]
        volume_mask = (volume_mask & (reference_points_cam[..., 0:1] > eps)
                & (reference_points_cam[..., 0:1] < (1.0 - eps))
                & (reference_points_cam[..., 1:2] > eps)
                & (reference_points_cam[..., 1:2] < (1.0 - eps))
                )

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # [num_cam, B, num_query, D, 3]
        volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # [D, B, num_cam, num_query, 1] -> [num_cam, B, num_query, D]
        ## different from VoxFormerEncoder_DFA3D
        reference_points_cam = reference_points_cam[:,:,:,:,0:2] # [num_cam, B, num_query, D, 2]
        return reference_points_cam, volume_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                ref_3d=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                img_meta=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerEncoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape [num_query, bs, embed_dims]
            key & value (Tensor): Input multi-cameta features with shape [num_cam, num_key, bs, embed_dims]
        Returns:
            Tensor: Results with shape [bs, num_query, embed_dims]
        """

        output = bev_query
        intermediate = []
        
        # [num_cam, bs, num_query, D, 2(hw)/3(hwd)], normalized [0,1] [num_cam, bs, num_query, D]
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_meta=img_meta)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_3d=ref_3d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                img_meta = img_meta,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output # (bs, num_query, embed_dims)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxFormerEncoder_DFA3D(VoxFormerEncoder):
    def __init__(
        self,
        *args,
        return_intermediate=False, 
        dbound=None,
        **kwargs):
        super(VoxFormerEncoder_DFA3D, self).__init__(*args, return_intermediate=return_intermediate, dbound=dbound, **kwargs)
    
    @staticmethod
    def _compute_projection(img_meta, stride):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def point_sampling(self, reference_points, img_meta=None):
        assert reference_points.shape[0]==1 # only support bs=1

        eps = 1e-5
        ogfH = img_meta['img_shape'][0] # 239
        ogfW = img_meta['img_shape'][1] # 320

        # [bs, 1, HWZ, 3] / [bs, D, HWZ, 3]
        origin = torch.tensor(img_meta['lidar2img']['origin']).to(reference_points.device)
        reference_points =  reference_points + origin
        
        # [bs, D, HWZ, 3] -> [D, bs, HWZ, 3]
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        projection = self._compute_projection(img_meta, stride=1).to(reference_points.device) # [num_cam, 3, 4]  stride=1 because reference_points_cam/ogfW
        num_cam = projection.shape[0]
        
        # [D, B, num_cam, num_query, 3]
        reference_points = reference_points.view(D, B, 1, num_query, 3).repeat(1, 1, num_cam, 1, 1)

        # [D, B, num_cam, num_query, 4]
        reference_points = torch.cat((reference_points, torch.ones(*reference_points.shape[:-1], 1).type_as(reference_points)), dim=-1)
        projection = projection.unsqueeze(0).unsqueeze(0) # [1, 1, num_cam, 3, 4]
        reference_points_cam = torch.matmul(projection, reference_points.permute(0,1,2,4,3)).permute(0,1,2,4,3) # [D, B, num_cam, num_query, 3]
        points_d = reference_points_cam[..., 2:3]
        
        reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(points_d, torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # [D, B, num_cam, num_query, 3]
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        reference_points_cam[..., 2] = (reference_points_cam[..., 2] - self.dbound[0]) / (self.dbound[1]-self.dbound[0])
        
        volume_mask = (points_d > eps)
        # [D, B, num_cam, num_query, 1]
        volume_mask = (volume_mask & (reference_points_cam[..., 0:1] > eps)
                & (reference_points_cam[..., 0:1] < (1.0 - eps))
                & (reference_points_cam[..., 1:2] > eps)
                & (reference_points_cam[..., 1:2] < (1.0 - eps))
                )

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # [num_cam, B, num_query, D, 3]
        volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # [D, B, num_cam, num_query, 1] -> [num_cam, B, num_query, D]
        return reference_points_cam, volume_mask


@TRANSFORMER_LAYER.register_module()
class VoxFormerLayer(MyCustomBaseTransformerLayer):
    """Implements encoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(VoxFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            **kwargs)
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerEncoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape [bs, num_queries, embed_dims]
            key (Tensor): The key tensor with shape [num_cam, num_keys, bs, embed_dims]
            value (Tensor): The value tensor with same shape as `key`.
        Returns:
            Tensor: forwarded results with shape [bs, num_queries, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


