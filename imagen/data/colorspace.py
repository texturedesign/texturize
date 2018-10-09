# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch

from ..utils import torch_min, torch_max, torch_mean


class MinMeanMaxBuilder:

    def __init__(self):
        pass

    def build(self, data, normalize=False):
        means = torch_mean(data, dims=(0, 1))
        mins = torch_min(data, dims=(0, 1))
        maxs = torch_max(data, dims=(0, 1))

        if normalize:
            dmin, dmax = (mins - means), (maxs - means)
            dmin[dmin == 0.0] = 1.0
            dmax[dmax == 0.0] = 1.0
        else:
            dmin, dmax = -1.0, +1.0

        data_min = torch.clamp((data - means) / dmin, min=0.0)
        data_max = torch.clamp((data - means) / dmax, min=0.0)
        data_mean = torch.clamp(torch.abs(1.0 - data_max - data_min), min=0.0)
        return torch.cat([data_min, data_max, data_mean], dim=2)
