import numpy as np
import os.path as osp
from collections import defaultdict
from mmdet.datasets import DATASETS
import scipy.spatial as scipy_spatial

from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox import DepthInstance3DBoxes
from .dataset_wrappers import MultiViewMixin


def compute_poses_center(poses):
    poses = np.stack(poses)
    # (num_poses, 3)
    pose_origins = poses[:, :3, 3]
    pose_center = pose_origins.mean(axis=0)
    return pose_center


@DATASETS.register_module()
class ARKitDataset(MultiViewMixin, Custom3DDataset):
    CLASSES = ("cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", 
               "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool",
               "chair", "table", "tv_monitor", "sofa")

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = defaultdict(list)
        for i in range(len(info["img_paths"])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])
            dep_filename = osp.join(self.data_root, info['depth_paths'][i])
            input_dict["img_prefix"].append(None)
            input_dict["img_info"].append(dict(filename=img_filename))
            extrinsic = np.linalg.inv(info["poses"][i])
            input_dict["lidar2img"].append(extrinsic.astype(np.float32))
            input_dict["depth_paths"].append(dep_filename)
        input_dict = dict(input_dict)

        origin = compute_poses_center(info["poses"])
        # origin = np.array([.0, .0, .5])
        
        intrinsic = np.eye(4).astype(np.float32)
        intrinsic[0:3,0:3] = info["intrinsic"]
        input_dict["lidar2img"] = dict(
            extrinsic=input_dict["lidar2img"],
            intrinsic=intrinsic,
            origin=origin.astype(np.float32),
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and len(annos["gt_bboxes_3d"]) == 0:
                return None
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if info["annos"]["gt_num"] != 0:
            gt_bboxes_3d = info["annos"]["gt_boxes_upright_depth"].astype(
                np.float32)  # k, 7
            gt_labels_3d = info["annos"]["class"].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        return anns_results