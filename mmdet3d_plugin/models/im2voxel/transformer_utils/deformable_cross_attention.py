from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from .multi_scale_3ddeformable_attn_function import MultiScale3DDeformableAttnFunction_fp16, MultiScale3DDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def Grid_Sample_2D_Feature(
            query,
            value=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None):

    num_heads, num_levels, num_points = 1, 1, 1
    bs, num_query, _ = query.shape
    bs, num_value, _ = value.shape
    assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    
    value = value.view(bs, num_value, num_heads, -1) # [bs, num_key, num_heads, c_head]
    
    # attention weights all ones
    attention_weights = torch.ones([bs, num_query, num_heads, num_levels, num_points], device=value.device)
    
    if reference_points.shape[-1] == 2:

        bs, num_query, num_Z_anchors, xy = reference_points.shape # num_Z_anchors=1
        reference_points = reference_points[:, :, None, None, None, :, :] # [bs, num_query, 1, 1, 1, num_Z_anchors, 2]
        
        sampling_locations_ref = reference_points.repeat(1,1,num_heads,num_levels,num_points,1,1)
        num_all_points = num_points * num_Z_anchors

        sampling_locations_ref = sampling_locations_ref.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy) # init sampling locations
        
    elif reference_points.shape[-1] == 4:
        assert False
    else:
        raise ValueError(
            f'Last dim of reference_points must be'
            f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
    if torch.cuda.is_available() and value.is_cuda:
        if value.dtype == torch.float16:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        else:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations_ref,
            attention_weights, 128)
    return output


def Grid_Sample_3D_Feature(
            query,
            value=None,
            value_dpt_dist=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None):

    num_heads, num_levels, num_points = 1, 1, 1
    bs, num_query, _ = query.shape
    bs, num_value, _ = value.shape
    assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    
    value = value.view(bs, num_value, num_heads, -1) # [bs, num_key, num_heads, c_head]
    _, _, dim_depth = value_dpt_dist.shape
    value_dpt_dist = value_dpt_dist.view(bs, num_value, 1, dim_depth).repeat(1,1,num_heads, 1) # [bs, num_key, num_heads, dim_depth]
    
    # attention weights all ones
    attention_weights = torch.ones([bs, num_query, num_heads, num_levels, num_points], device=value.device)
    
    spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * dim_depth
    spatial_shapes_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1).contiguous() # [num_levels, 3] e.g.[[H,W,D]]

    if reference_points.shape[-1] == 3:

        bs, num_query, num_Z_anchors, xy = reference_points.shape # num_Z_anchors=1
        reference_points = reference_points[:, :, None, None, None, :, :] # [bs, num_query, 1, 1, 1, num_Z_anchors, 3]
        
        sampling_locations_ref = reference_points.repeat(1,1,num_heads,num_levels,num_points,1,1)
        num_all_points = num_points * num_Z_anchors

        sampling_locations_ref = sampling_locations_ref.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy) # init sampling locations
        
    elif reference_points.shape[-1] == 4:
        assert False
    else:
        raise ValueError(
            f'Last dim of reference_points must be'
            f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
    if torch.cuda.is_available() and value.is_cuda:
        if value.dtype == torch.float16:
            MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp16
        else:
            MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp32
        output, depth_score = MultiScaleDeformableAttnFunction.apply(
            value, value_dpt_dist, spatial_shapes_3D, level_start_index, sampling_locations_ref,
            attention_weights, 128)
    return output


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class MSDeformableAttention3D_DFA3D(MSDeformableAttention3D):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8, im2col_step=64, dropout=0.1, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__(embed_dims, num_heads, num_levels, num_points, im2col_step, dropout, batch_first, norm_cfg, init_cfg)
        self.sampling_offsets_depth = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 1)
        self.init_smpl_off_weights()
        
    def init_smpl_off_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets_depth, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([(thetas.cos() + thetas.sin()) / 2], -1)
        grid_init = grid_init.view(self.num_heads, 1, 1, 1).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets_depth.bias.data = grid_init.view(-1)
    
    def forward(self,
                query,
                key=None,
                value=None,
                value_dpt_dist=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            value_dpt_dist(Tensor): The depth distribution of each image feature (value), with shape
                `(bs, num_key,  dpt)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            reference_points (Tensor):  The normalized reference
                points cames with shape (bs, num_query, D, 3),
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1) # [bs, num_key, num_heads, c_head]
        _, _, dim_depth = value_dpt_dist.shape
        value_dpt_dist = value_dpt_dist.view(bs, num_value, 1, dim_depth).repeat(1,1,self.num_heads, 1) # [bs, num_key, num_heads, dim_depth]
        sampling_offsets_uv = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        sampling_offsets_depth = self.sampling_offsets_depth(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 1)
        sampling_offsets = torch.cat([sampling_offsets_uv, sampling_offsets_depth], dim = -1) # offsets without normalized [0,1]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        spatial_shapes_3D = self.get_spatial_shape_3D(spatial_shapes, dim_depth) # [num_levels, 3] e.g.[[H,W,D]]
        if reference_points.shape[-1] == 3:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes_3D[..., 1], spatial_shapes_3D[..., 0], spatial_shapes_3D[..., 2]], -1) # [num_levels, 3] e.g.[[W,H,D]]

            bs, num_query, num_Z_anchors, xy = reference_points.shape # voxel num_Z_anchors=1
            reference_points = reference_points[:, :, None, None, None, :, :] # [bs, num_query, 1, 1, 1, num_Z_anchors, 3]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :] # [1, 1, 1, num_levels, 1, 3]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            # sampling_locations_ref = reference_points.repeat(1,1,num_heads,num_levels,num_points,1,1)
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy) # deformable sampling locations
            # sampling_locations_ref = sampling_locations_ref.view(
            #     bs, num_query, num_heads, num_levels, num_all_points, xy) # projection sampling locations
        
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        '''
        # The original usage of 2D Deformable Attention.
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        '''
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp32
            output, depth_score = MultiScaleDeformableAttnFunction.apply(
                value, value_dpt_dist, spatial_shapes_3D, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        
        # weight_update is useful when self.use_empty == True.
        weight_update = (depth_score.mean(dim=-1) * attention_weights).flatten(-2).sum(dim=-1, keepdim=True)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output, weight_update
    
    def get_spatial_shape_3D(self, spatial_shape, depth_dim):
        spatial_shape_depth = spatial_shape.new_ones(*spatial_shape.shape[:-1], 1) * depth_dim
        spatial_shape_3D = torch.cat([spatial_shape, spatial_shape_depth], dim=-1)
        return spatial_shape_3D.contiguous()


@ATTENTION.register_module()
class DeformCrossAttention(BaseModule):
    """An attention module used in VoxFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 deformable_attn=True,
                 inter_view_aggregation='attn',
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(DeformCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.deformable_attn = deformable_attn
        self.inter_view_aggregation = inter_view_aggregation
        if inter_view_aggregation == 'attn':
            self.attention_pooling = nn.MultiheadAttention(embed_dim=embed_dims, num_heads=8, batch_first=False)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, channels = query.size()

        D = reference_points_cam.size(3)
        num_cams = reference_points_cam.size(0)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        reference_points_rebatch = reference_points_rebatch.view(bs*num_cams, max_len, D, 2)
        
        queries_per_image = Grid_Sample_2D_Feature(queries_rebatch.view(bs*num_cams, max_len, self.embed_dims),
                                                value=value,
                                                reference_points=reference_points_rebatch,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index)
        if self.deformable_attn ==True:
            queries = self.deformable_attention(query=queries_per_image, key=key, value=value,
                                                reference_points=reference_points_rebatch,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index, **kwargs)
            queries = queries + queries_per_image
        else:
            queries = queries_per_image
        
        queries = queries.view(bs, num_cams, max_len, self.embed_dims)

        ## average pooling 
        # slots = torch.zeros_like(query)
        # for j in range(bs):
        #     for i, index_query_per_img in enumerate(indexes):
        #         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # count = bev_mask.sum(-1) > 0    # (num_cams, bs, num_query)
        # count = count.permute(1, 2, 0).sum(-1)    # (bs, num_query)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]
        # slots = self.output_proj(slots)
        # return self.dropout(slots) + inp_residual
        
        slots = torch.zeros([num_cams, bs, num_query, channels], device=queries.device) # [num_cams, bs, num_query, channels]
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[i, j, index_query_per_img] = queries[j, i, :len(index_query_per_img)]
        count = bev_mask.sum(-1) > 0    # (num_cams, bs, num_query)
        count = count.permute(1, 2, 0).sum(-1)    # (bs, num_query)
        
        valid_index = count.nonzero()[:,1] # (L)
        valid_num = count[:,valid_index] # (bs, L)
        valid_slots = slots[:,:,valid_index,:] # (num_cams, bs, L, channels)
        valid_mask = bev_mask[:,:,valid_index,:] # (num_cams, bs, L, 1)
        slots_mean = (valid_slots * valid_mask).sum(dim=0) / valid_num[..., None] # [bs, L, channels]
        slots_mean = self.output_proj(slots_mean)  # [bs, L, channels]

        if self.inter_view_aggregation == 'attn':
            valid_slots = valid_slots.squeeze(1) # [num_cams, L, channels]
            slots_mean = slots_mean.squeeze(0).unsqueeze(0) # [1, L, channels]
            key_padding = ~valid_mask.squeeze(3).squeeze(1).transpose(1,0) # [L, num_cams]
            slots_mean, _ = self.attention_pooling(slots_mean, valid_slots, valid_slots, key_padding)
        
        output = torch.zeros([bs, num_query, channels], device=queries.device) # [bs, num_query, channels]
        output[:,valid_index,:] = slots_mean
        return self.dropout(output) + inp_residual
    
    
@ATTENTION.register_module()
class DeformCrossAttention_DFA3D(DeformCrossAttention):
    def __init__(self,
                 embed_dims=256,
                 deformable_attn=True,
                 inter_view_aggregation='attn',
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=None,
                 **kwargs):
        super().__init__(embed_dims, deformable_attn, inter_view_aggregation, dropout, init_cfg, batch_first, deformable_attention,  **kwargs)
        
    @force_fp32(apply_to=('query', 'key', 'value', 'value_dpt_dist', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                value_dpt_dist=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape (bs, num_query, embed_dims)
            key (Tensor): The key tensor with shape (num_cams, num_key, bs, embed_dims)
            value (Tensor): The value tensor with shape (num_cams, num_key, bs, embed_dims)
            value_dpt_dist(Tensor): The depth distribution of each image feature (value), with shape (num_cams, num_key, bs, dbound)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default: None.
            reference_points (Tensor):  The normalized reference points with shape (1, 1, num_query, embed_dims)
                all elements is range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
            reference_points_cam (Tensor):  The normalized reference points cames with shape (num_cams, 1, num_query, 1, 3)
            bev_mask (Tensor):  whether a point is visiable in any view, with shape (num_cams, 1, num_query, 1)
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, channels = query.size()

        D = reference_points_cam.size(3)
        num_cams = reference_points_cam.size(0)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask): # (num_cams, 1, num_query, 1)
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1) # [num_index] 该图像能索引到的voxel的编号
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 3])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam): # (num_cams, 1, num_query, 1, 3)
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        
        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims)
        value_dpt_dist = value_dpt_dist.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, value_dpt_dist.shape[-1])
        reference_points_rebatch = reference_points_rebatch.view(bs*num_cams, max_len, D, 3)
        
        queries_per_image = Grid_Sample_3D_Feature(queries_rebatch.view(bs*num_cams, max_len, self.embed_dims),
                                                value=value,
                                                value_dpt_dist=value_dpt_dist,
                                                reference_points=reference_points_rebatch,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index)
        if self.deformable_attn ==True:
            queries, _ = self.deformable_attention(query=queries_per_image, key=key, value=value,
                                                value_dpt_dist=value_dpt_dist,
                                                reference_points=reference_points_rebatch,
                                                spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index, **kwargs)
        else:
            queries = queries_per_image
        
        queries = queries.view(bs, num_cams, max_len, self.embed_dims)

        ## average pooling 
        # slots = torch.zeros_like(query)
        # for j in range(bs):
        #     for i, index_query_per_img in enumerate(indexes):
        #         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # count = bev_mask.sum(-1) > 0    # (num_cams, bs, num_query)
        # count = count.permute(1, 2, 0).sum(-1)    # (bs, num_query)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]
        # slots = self.output_proj(slots)
        # return self.dropout(slots) + inp_residual
        
        slots = torch.zeros([num_cams, bs, num_query, channels], device=queries.device) # [num_cams, bs, num_query, channels]
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[i, j, index_query_per_img] = queries[j, i, :len(index_query_per_img)]
        count = bev_mask.sum(-1) > 0    # (num_cams, bs, num_query)
        count = count.permute(1, 2, 0).sum(-1)    # (bs, num_query)
        
        valid_index = count.nonzero()[:,1] # (L)
        valid_num = count[:,valid_index] # (bs, L)
        valid_slots = slots[:,:,valid_index,:] # (num_cams, bs, L, channels)
        valid_mask = bev_mask[:,:,valid_index,:] # (num_cams, bs, L, 1)
        slots_mean = (valid_slots * valid_mask).sum(dim=0) / valid_num[..., None] # [bs, L, channels]
        slots_mean = self.output_proj(slots_mean)  # [bs, L, channels]

        if self.inter_view_aggregation == 'attn':
            valid_slots = valid_slots.squeeze(1) # [num_cams, L, channels]
            slots_mean = slots_mean.squeeze(0).unsqueeze(0) # [1, L, channels]
            key_padding = ~valid_mask.squeeze(3).squeeze(1).transpose(1,0) # [L, num_cams]
            slots_mean, _ = self.attention_pooling(slots_mean, valid_slots, valid_slots, key_padding)
        
        output = torch.zeros([bs, num_query, channels], device=queries.device) # [bs, num_query, channels]
        output[:,valid_index,:] = slots_mean
        return self.dropout(output) + inp_residual

