# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import itertools

import torch
from ..utils import torch_gather_2d, torch_pad_replicate


class PatchMatcher:
    """Implementation of patchmatch that uses normalized cross-correlation
    of patches to determine the similarity of patches.
    """

    def __init__(self, content, style, indices='random'):
        assert len(content.shape) == 4 and len(style.shape) == 4
        assert content.shape[0] == 1 and style.shape[0] == 1

        # Normalize the arrays that were provided by the user.
        self.content = self.normalize_patches(content)
        self.style = self.normalize_patches(style)

        # Initialize the coordinates for the starting state.
        if indices == 'zero':
            self.indices = self.create_indices_empty().zero_()
        elif indices == 'random':
            self.indices = self.create_indices_random()
        elif indices == 'linear':
            self.indices = self.create_indices_linear()
        else:
            assert isinstance(indices, torch.Tensor), "Long tensor of shape (y,x,2) is expected."
            self.indices = indices.long()

        # Compute the scores for the first chosen coordinates.
        b, c, h, w = self.content.shape
        self.scores = torch.zeros((b, 1, h, w), dtype=torch.float, device=self.content.device)
        self.improve_patches(self.indices)

    def create_indices_empty(self):
        return torch.empty((1,2) + self.content.shape[-2:], dtype=torch.long, device=self.content.device)

    def create_indices_random(self):
        indices = self.create_indices_empty()
        torch.randint(low=0, high=self.style.shape[2], size=self.content.shape[-2:], out=indices[:, 0, :, :])
        torch.randint(low=0, high=self.style.shape[3], size=self.content.shape[-2:], out=indices[:, 1, :, :])
        return indices

    def create_indices_linear(self):
        indices = self.create_indices_empty()
        b, _, h, w = indices.shape
        indices[:, 0, :, :] = torch.arange(h, dtype=torch.float).mul(self.style.shape[2] / h).view((b, -1, 1)).long()
        indices[:, 1, :, :] = torch.arange(w, dtype=torch.float).mul(self.style.shape[3] / w).view((b, 1, -1)).long()
        return indices

    def normalize_patches(self, patches):
        """Setup patches for normalized cross-correlation. The same patch times itself
        should result in 1.0.
        """
        patches_norm = torch.sqrt(torch.sum(patches ** 2.0, dim=1, keepdim=True))
        return patches / patches_norm

    def improve_patches(self, candidate_indices):
        """Compute the similarity score for target patches and possible improvements
        using normalized cross-correlation.  Where the score is better, update the
        indices and score array to reflect the new chosen coordinates.
        """
        candidate_repro = torch_gather_2d(self.style, candidate_indices)
        candidate_scores = torch.sum(self.content * candidate_repro, dim=1, keepdim=True)

        better = candidate_scores > self.scores

        self.indices[:,:,:,:] = torch.where(better, candidate_indices, self.indices)
        self.scores[:,:,:,:] = torch.where(better, candidate_scores, self.scores)

        print('.', end='', flush=True)

    def search_patches_random(self, radius=8, times=4):
        """Generate random coordinates within a radius for each pixel, then compare the 
        patches to see if the current selection can be improved.
        """
        for i in range(times):
            if i % 2 == 0:
                offset = torch.randint(low=-radius, high=radius+1, size=self.indices.shape,
                                       dtype=torch.long, device=self.indices.device)
                candidates = (self.indices + offset)
            else:
                candidates = self.create_indices_random()

            candidates[:,0,:,:].clamp_(min=0, max=self.style.shape[2] - 1)
            candidates[:,1,:,:].clamp_(min=0, max=self.style.shape[3] - 1)
            self.improve_patches(candidates)

    def search_patches_propagate(self, steps=[1, 2, 4, 8]):
        """Generate nearby coordinates for each pixel to see if offseting the neighboring
        pixel would provide better results.
        """
        padding = max(steps)
        cells = itertools.chain(*[[(-s, 0), (+s, 0), (0, -s), (0, +s)] for s in steps])
        for y, x in cells:
            # Create a lookup map with offset coordinates from each coordinate.
            lookup = self.indices.new_empty(self.indices.shape)
            lookup[:,0,:,:] = torch.arange(0, lookup.shape[2], dtype=torch.long).view((1, -1, 1))
            lookup[:,1,:,:] = torch.arange(0, lookup.shape[3], dtype=torch.long).view((1, 1, -1))
            lookup[:,0,:,:] += y + padding
            lookup[:,1,:,:] += x + padding

            # Compute new padded buffer with the current best coordinates.
            indices = torch_pad_replicate(self.indices.float(), (padding, padding, padding, padding)).long()
            indices[:,0,:,:] -= y
            indices[:,1,:,:] -= x

            # Lookup the neighbor coordinates and clamp if the calculation overflows.
            candidates = torch_gather_2d(indices, lookup)
            candidates[:,0,:,:].clamp_(min=0, max=self.style.shape[2] - 1)
            candidates[:,1,:,:].clamp_(min=0, max=self.style.shape[3] - 1)

            self.improve_patches(candidates)
