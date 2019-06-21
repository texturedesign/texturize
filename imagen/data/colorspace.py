# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch

from ..utils import torch_min, torch_max, torch_mean, torch_std


class MinMeanMaxBuilder:
    def __init__(self, normalize=False):
        self.normalize = normalize

    def build(self, data):
        data = data - torch_mean(data, dims=(2, 3))
        means = 0.0

        mins = torch_min(data, dims=(2, 3)) * 0.5 - 0.5 * torch_std(data, dims=(2, 3))
        maxs = torch_max(data, dims=(2, 3)) * 0.5 + 0.5 * torch_std(data, dims=(2, 3))

        if self.normalize:
            dmin, dmax = (mins - means), (maxs - means)
            dmin[dmin == 0.0] = 1.0
            dmax[dmax == 0.0] = 1.0
        else:
            dmin, dmax = -1.0, +1.0

        data_min = torch.clamp((data - means) / dmin, min=0.0)
        data_max = torch.clamp((data - means) / dmax, min=0.0)
        data_mean = torch.clamp(torch.abs(1.0 - data_max - data_min), min=0.0)
        return torch.cat([data_min, data_max, data_mean], dim=1)
