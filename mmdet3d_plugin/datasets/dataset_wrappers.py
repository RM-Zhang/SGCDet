import os
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from os import path as osp
from mmdet3d.datasets.builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)


class MultiViewMixin:
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8).tolist()

    @staticmethod
    def draw_corners(img, corners, color, projection):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

    def show(self, results, out_dir):
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            ann_info = self.get_ann_info(i)
            gt_corners =  ann_info['gt_bboxes_3d'].corners
            gt_labels =  ann_info['gt_labels_3d']

            ## arkit
            scene_name = self.data_infos[i]['img_paths'][0].split('/')[-2]
            ## scannet
            # data_info = self.data_infos[i]
            # pts_path = data_info['pts_path']
            # scene_name = pathlib.Path(pts_path).stem
            # axis_align_matrix = data_info['annos']['axis_align_matrix'].astype(np.float32)
            # mesh_vertices = np.fromfile(osp.join(self.data_root, pts_path), dtype=np.float32).reshape(-1, 6)
            # pts = np.ones((mesh_vertices.shape[0], 4))
            # pts[:, 0:3] = mesh_vertices[:, 0:3]
            # pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
            # aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]], axis=1)
            
            if len(result['boxes_3d']) != 0:
                corners = result['boxes_3d'].corners.numpy()
                scores = result['scores_3d'].numpy()
                labels = result['labels_3d'].numpy()

                out_scene_dir = pathlib.Path(out_dir, scene_name)
                out_scene_dir.mkdir(exist_ok=True, parents=True)
                # _write_obj(aligned_mesh_vertices, osp.join(out_scene_dir, f'points.obj'))
                np.save(out_scene_dir / "bbox_pred.npy", corners)
                np.save(out_scene_dir / "score_pred.npy", scores)
                np.save(out_scene_dir / "label_pred.npy", labels)
                np.save(out_scene_dir / "bbox_gt.npy", gt_corners)
                np.save(out_scene_dir / "label_gt.npy", gt_labels)
                
    def show_2d(self, results, out_dir):
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            scene_name = pathlib.Path(pts_path).stem
            out_scene_dir = pathlib.Path(out_dir, scene_name)
            out_scene_dir.mkdir(exist_ok=True, parents=True)
                
            info = self.get_data_info(i)
            for j in range(len(info['img_info'])):
                img = skimage.io.imread(info['img_info'][j]['filename'])
                extrinsic = info['lidar2img']['extrinsic'][j]
                intrinsic = info['lidar2img']['intrinsic'][:3, :3]
                projection = intrinsic @ extrinsic[:3]
                if not len(result['scores_3d']):
                    continue
                corners = result['boxes_3d'].corners.numpy()
                scores = result['scores_3d'].numpy()
                labels = result['labels_3d'].numpy()
                for corner, score, label in zip(corners, scores, labels):
                    self.draw_corners(img, corner, self.colors[label], projection)
                out_file_name = os.path.split(info['img_info'][j]['filename'])[-1][:-4]
                skimage.io.imsave(os.path.join(out_scene_dir, f'{out_file_name}.png'), img)


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()