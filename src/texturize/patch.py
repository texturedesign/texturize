# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import itertools

import torch
import torch.nn.functional as F


class PatchBuilder:
    def __init__(self, patch_size=3, weights=None):
        self.min = -((patch_size - 1) // 2)
        self.max = patch_size + self.min - 1
        self.patch_size = patch_size

        if weights is None:
            weights = torch.ones(size=(patch_size ** 2,))
        else:
            weights = torch.tensor(weights, dtype=torch.float32)

        self.weights = weights / weights.sum()

    def extract(self, array):
        padded = F.pad(
            array,
            pad=(abs(self.min), self.max, abs(self.min), self.max),
            mode="constant",
            value=0.0
        )
        h, w = (
            padded.shape[2] - self.patch_size + 1,
            padded.shape[3] - self.patch_size + 1,
        )
        output = []
        for y, x in itertools.product(self.coords, repeat=2):
            p = padded[:, :, y : h + y, x : w + x]
            output.append(p)
        return torch.cat(output, dim=1)

    @property
    def coords(self):
        return range(self.patch_size)
