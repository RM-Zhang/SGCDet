import mmcv
import numpy as np
import imageio.v2 as imageio
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadDepthMap:
    def __init__(self, depth_shift):
        self.depth_shift = depth_shift

    def __call__(self, results):
        depth_maps = []
        depth_masks = []
        depth_paths = results['depth_paths']
        for path in depth_paths:
            # # uint16
            dep_map = np.array(imageio.imread(path))
            mask = dep_map != 0
            
            # Convert to meter
            dep_map = dep_map.astype(np.float32) / self.depth_shift
            depth_maps.append(dep_map)
            depth_masks.append(mask)
        results['depth_maps'] = np.stack(depth_maps, axis=0)
        results['depth_masks'] = np.stack(depth_masks, axis=0)
        
        if 'depth_paths_render' in results.keys():
            depth_maps = []
            depth_masks = []
            depth_paths = results['depth_paths_render']
            for path in depth_paths:
                # # uint16
                dep_map = np.array(imageio.imread(path))
                mask = dep_map != 0
                # Convert to meter
                dep_map = dep_map.astype(np.float32) / self.depth_shift
                depth_maps.append(dep_map)
                depth_masks.append(mask)
            results['depth_maps_render'] = np.stack(depth_maps, axis=0)
            results['depth_masks_render'] = np.stack(depth_masks, axis=0)
            
        return results