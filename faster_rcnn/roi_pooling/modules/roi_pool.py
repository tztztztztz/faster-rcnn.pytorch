import torch
from torch.nn.modules.module import Module
if torch.cuda.is_available():
    from ..functions.roi_pool import RoIPoolFunction
else:
    # only for cpu debug
    class RoIPoolFunction(object):
        def __init__(self, a, b, c):
            pass
        def __call__(self, a, b):
            pass

class _RoIPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)

