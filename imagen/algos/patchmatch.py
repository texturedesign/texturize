# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import itertools

import torch
from ..utils import torch_gather_2d, torch_pad_replicate


class PatchMatcher:
    """Implementation of patchmatch that uses normalized cross-correlation
    of patches to determine the similarity of patches.
    """

    def __init__(self, content, style, indices='random'):
        # Compute the norm of the vector for each pixel.
        content_norm = torch.sqrt(torch.sum(content ** 2.0, dim=2, keepdim=True))
        style_norm = torch.sqrt(torch.sum(style ** 2.0, dim=2, keepdim=True))
        # Normalize the arrays that were provided by the user.
        self.content = content / content_norm
        self.style = style / style_norm

        # Initialize the coordinates for the starting state.
        if indices == 'zero':
            self.indices = self.create_indices_zero()
        elif indices == 'random':
            self.indices = self.create_indices_random()
        elif indices == 'linear':
            self.indices = self.create_indices_linear()
        else:
            assert isinstance(indices, torch.Tensor), "Long tensor of shape (y,x,2) is expected."
            self.indices = indices.long()

        # Compute the scores for the first chosen coordinates.
        self.scores = torch.zeros(self.content.shape[:2] + (1,), dtype=torch.float, device=self.content.device)
        self.improve_patches(self.indices)

    def create_indices_zero(self):
        return torch.zeros(self.content.shape[:2] + (2,), dtype=torch.long, device=self.content.device)

    def create_indices_random(self):
        indices = self.create_indices_zero()
        torch.randint(low=0, high=self.style.shape[0], size=self.content.shape[:2], out=indices[:, :, 0])
        torch.randint(low=0, high=self.style.shape[1], size=self.content.shape[:2], out=indices[:, :, 1])
        return indices

    def create_indices_linear(self):
        indices = self.create_indices_zero()
        indices[:, :, 0] = torch.arange(self.content.shape[0], dtype=torch.float).mul(self.style.shape[0] / self.content.shape[0]).view((-1, 1)).long()
        indices[:, :, 1] = torch.arange(self.content.shape[1], dtype=torch.float).mul(self.style.shape[1] / self.content.shape[1]).view((1, -1)).long()
        return indices

    def improve_patches(self, candidate_indices):
        """Compute the similarity score for target patches and possible improvements
        using normalized cross-correlation.  Where the score is better, update the
        indices and score array to reflect the new chosen coordinates.
        """
        candidate_repro = torch_gather_2d(self.style, candidate_indices)
        candidate_scores = torch.sum(self.content * candidate_repro, dim=2, keepdim=True)

        better = candidate_scores > self.scores

        self.indices[:,:] = torch.where(better, candidate_indices, self.indices)
        self.scores[:,:] = torch.where(better, candidate_scores, self.scores)

    def search_patches_random(self, radius=8, times=4):
        """Generate random coordinates within a radius for each pixel, then compare the 
        patches to see if the current selection can be improved.
        """
        for _ in range(times):
            offset = torch.randint(low=-radius, high=radius+1, size=self.indices.shape,
                                   dtype=torch.long, device=self.indices.device)
            candidates = (self.indices + offset)
            candidates[:,:,0].clamp_(min=0, max=self.style.shape[0] - 1)
            candidates[:,:,1].clamp_(min=0, max=self.style.shape[1] - 1)
            self.improve_patches(candidates)
    
    def search_patches_propagate(self, steps=[1, 2]):
        """Generate nearby coordinates for each pixel to see if offseting the neighboring
        pixel would provide better results.
        """
        padding = max(steps)
        cells = itertools.chain(*[[(-s, 0), (+s, 0), (0, -s), (0, +s)] for s in steps])
        for y, x in cells:
            # Create a lookup map with offset coordinates from each coordinate.
            lookup = self.indices.new_empty(self.indices.shape)
            lookup[:,:,0] = torch.arange(0, lookup.shape[0], dtype=torch.long).view((-1, 1))
            lookup[:,:,1] = torch.arange(0, lookup.shape[1], dtype=torch.long).view((1, -1))
            lookup[:,:,0] += y + padding
            lookup[:,:,1] += x + padding

            # Compute new padded buffer with the current best coordinates.
            indices = torch_pad_replicate(self.indices.float(), (padding, padding, padding, padding)).long()
            indices[:,:,0] -= y
            indices[:,:,1] -= x

            # Lookup the neighbor coordinates and clamp if the calculation overflows.
            candidates = torch_gather_2d(indices, lookup)
            candidates[:,:,0].clamp_(min=0, max=self.style.shape[0] - 1)
            candidates[:,:,1].clamp_(min=0, max=self.style.shape[1] - 1)

            self.improve_patches(candidates)
