# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import itertools

import torch
from ..utils import torch_gather_2d, torch_pad_replicate


class PatchBuilder:

    def __init__(self, patch_size=3, weights=None):
        self.min = -((patch_size - 1) // 2)
        self.max = patch_size + self.min - 1
        self.patch_size = patch_size

        if weights is None:
            weights = torch.ones(size=(patch_size**2,))
        else:
            weights = torch.tensor(weights, dtype=torch.float)

        self.weights = weights / weights.sum()

    def extract(self, array):
        padded = torch_pad_replicate(array, (abs(self.min), self.max, abs(self.min), self.max))
        h, w = padded.shape[0] - self.patch_size + 1, padded.shape[1] - self.patch_size + 1
        output = []
        for y, x in itertools.product(self.coords, repeat=2):
            p = padded[y:h+y,x:w+x]
            output.append(p)
        return torch.cat(output, dim=2)

    def reconstruct(self, patches):
        layer_count = len(self.coords) ** 2
        layers = patches.view(patches.shape[:2] + (layer_count, -1))

        oh = patches.shape[0] + abs(self.min) + self.max
        ow = patches.shape[1] + abs(self.min) + self.max
        output = patches.new_zeros((oh, ow) + (patches.shape[-1] // layer_count,))
        weights = torch.zeros((oh, ow, 1), dtype=torch.float, device=patches.device)

        ph, pw = patches.shape[:2]
        for i, (y, x) in enumerate(itertools.product(self.coords, repeat=2)):
            output[y:ph+y,x:pw+x,:] += layers[:,:,i,:] * self.weights[i]
            weights[y:ph+y,x:pw+x] += self.weights[i]

        weights[weights == 0.0] = 1.0
        return (output / weights)[abs(self.min):oh-self.max,abs(self.min):ow-self.max]

    @property
    def coords(self):
        return range(self.patch_size)
