# depth range
downsample_factor = 8 # scannet
dbound = [0.2, 5, 0.4]

# adaptive sparse head
voxel_size_list = [(.64, .64, .8),
                   (.32, .32, .4),
                   (.16, .16, .2)]
n_voxels_list = [(10, 10, 4),
                 (20, 20, 8),
                 (40, 40, 16)]
topk_list = [800, 6400]

# channel number
embed_dims=256

cross_transformer = dict(
    type='PerceptionTransformer_DFA3D',
    embed_dims=embed_dims,
    encoder=dict(
        type='VoxFormerEncoder_DFA3D',
        num_layers=1,
        return_intermediate=False,
        dbound=dbound,
        transformerlayers=dict(
            type='VoxFormerLayer',
            attn_cfgs=[
                dict(
                    type='DeformCrossAttention_DFA3D',
                    deformable_attention=dict(
                        type='MSDeformableAttention3D_DFA3D',
                        embed_dims=embed_dims,
                        num_heads=8,
                        num_points=4,
                        num_levels=1,
                        im2col_step=128),
                    embed_dims=embed_dims,
                    inter_view_aggregation='attn',
                    dropout=0,
                )
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=embed_dims*2,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('cross_attn', 'norm', 'ffn', 'norm'))))

base_head_configs = [
    dict(type='DenseHead',
         voxel_size=voxel_size_list[0],
         n_voxels=n_voxels_list[0],
         embed_dims=embed_dims,
         cross_transformer=cross_transformer),
    dict(type='DenseHead',
         voxel_size=voxel_size_list[1],
         n_voxels=n_voxels_list[1],
         embed_dims=embed_dims,
         cross_transformer=cross_transformer),
    dict(type='DenseHead',
         voxel_size=voxel_size_list[2],
         n_voxels=n_voxels_list[2],
         embed_dims=embed_dims,
         cross_transformer=cross_transformer)
]

model = dict(
    type='SGCDet',
    depth_loss=False,
    occ_loss=True,
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=embed_dims,
        num_outs=4),
    depth_head = dict(
        type='DepthNet_Fusion',
        neighbor_img_num=2,
        downsample_factor=downsample_factor,
        dbound=dbound,
        loss_weight=0.5,
        max_tol=0),
    voxel_head = dict(
        type='AdaptiveSparseHead',
        embed_dims=embed_dims,
        topk_list=topk_list,
        voxel_size_list=voxel_size_list,
        n_voxels_list=n_voxels_list,
        base_head_configs=base_head_configs,
    ),
    neck_3d=dict(
        type='FastIndoorImVoxelNeck',
        in_channels=embed_dims,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ScanNetImVoxelHeadV2',
        loss_bbox=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
        n_classes=18,
        n_channels=128,
        n_reg_outs=6,
        n_scales=3,
        limit=27,
        centerness_topk=18),
    voxel_size=voxel_size_list[-1],
    n_voxels=n_voxels_list[-1])
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    iou_thr=.25,
    score_thr=.01)


dataset_type = 'ScanNetMultiViewDataset'
data_root = '/home/zrm/code/Det3D/data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=40,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(240, 320))
        ]),
    dict(type='LoadDepthMap', depth_shift=1000.), # ScanNet's depth is in millimeter
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='PackDet3DInputs', keys=
         ['img', 'depth_maps', 'depth_masks', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=100,
        sample_method="linear",
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(240, 320))
        ]),
    dict(type='LoadDepthMap', depth_shift=1000.), # ScanNet's depth is in millimeter
    dict(type='PackDet3DInputs', keys=['img', 'depth_maps', 'depth_masks']),
]
data = dict(
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)
train_dataloader_config = dict(
    batch_size=1,
    num_workers=4)
val_dataloader_config = dict(
    batch_size=1,
    num_workers=4)
test_dataloader_config = dict(
    batch_size=1,
    num_workers=4)


"""Training params."""
learning_rate=0.0002
training_steps=(1201*36) # 12 epoch, 6 repeat, 2 card

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.0001
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    final_div_factor=1e4,
    interval="step",
    frequency=1
)