# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import itertools

import torch


def torch_flatten_1d(a):
    return a.contiguous().view(a.nelement())

def torch_flatten_2d(a):
    return a.contiguous().view((-1, a.shape[-1]))

def torch_gather_2d(array, indices):
    """Extract the content of an array using the 2D coordinates provided.
    """
    idx = indices[:, :, 0] * array.shape[1] + indices[:, :, 1]
    x = torch.index_select(torch_flatten_2d(array), 0, torch_flatten_1d(idx))
    return x.view(indices.shape[:2] + array.shape[-1:])

def torch_pad_replicate(array, padding):
    array = array.permute(2, 0, 1)[None]
    array = torch.nn.functional.pad(array, padding, mode='replicate')
    return array[0].permute(1, 2, 0)


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
