import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results
    

@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images, sample_method="random"):
        self.transforms = Compose(transforms)
        self.n_images = n_images

        assert sample_method in ["random", "linear", "uniform_random"], f"Not support {sample_method}"
        self.smpl_method = sample_method

    def __call__(self, results):
        imgs = []
        extrinsics = []
        depth_paths = []
        ids = np.arange(len(results['img_info']))
        if self.smpl_method == "random":
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
        elif self.smpl_method == "uniform_random":
            base_ids = np.linspace(0, len(ids)-1, self.n_images, dtype=int)
            offset_range = 2  
            offsets = np.random.randint(-offset_range, offset_range + 1, size=self.n_images)
            offsets = np.zeros_like(base_ids)
            if self.n_images > 2:
                mid_size = self.n_images - 2
                random_offsets = np.random.randint(-offset_range, offset_range + 1, size=mid_size)
                offsets[1:-1] = random_offsets
            jittered_ids = base_ids + offsets
            jittered_ids = np.clip(jittered_ids, 0, len(ids) - 1)
            jittered_ids = np.sort(jittered_ids)
            ids = jittered_ids
        else:
            ids = np.linspace(0, len(ids)-1, self.n_images, dtype=int)

        for i in sorted(ids.tolist()):
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
            depth_paths.append(results['depth_paths'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        results['lidar2img']['extrinsic'] = extrinsics
        results['depth_paths'] = depth_paths
        return results
