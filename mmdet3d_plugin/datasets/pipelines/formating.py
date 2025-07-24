# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PackDet3DInputs(object):
    def __init__(self, keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug')):
        super(PackDet3DInputs, self).__init__()
        self.keys = keys
        self.meta_keys = meta_keys
        
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
                     
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        
        if 'depth_maps' in results:
            if isinstance(results['depth_maps'], list):
                # process multiple imgs in single frame
                imgs = [img for img in results['depth_maps']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['depth_maps'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['depth_maps'])
                results['depth_maps'] = DC(to_tensor(img), stack=True)
        
        if 'depth_masks' in results:
            if isinstance(results['depth_masks'], list):
                # process multiple imgs in single frame
                imgs = [img for img in results['depth_masks']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['depth_masks'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['depth_masks'])
                results['depth_masks'] = DC(to_tensor(img), stack=True)
                
        if 'img_render' in results:
            if isinstance(results['img_render'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img_render']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img_render'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img_render'] = DC(to_tensor(img), stack=True)
        
        if 'depth_maps_render' in results:
            if isinstance(results['depth_maps_render'], list):
                # process multiple imgs in single frame
                imgs = [img for img in results['depth_maps_render']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['depth_maps_render'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['depth_maps_render'])
                results['depth_maps_render'] = DC(to_tensor(img), stack=True)
        
        if 'depth_masks_render' in results:
            if isinstance(results['depth_masks_render'], list):
                # process multiple imgs in single frame
                imgs = [img for img in results['depth_masks_render']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['depth_masks_render'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['depth_masks_render'])
                results['depth_masks_render'] = DC(to_tensor(img), stack=True)
        
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, meta_keys={self.meta_keys})'
        return repr_str