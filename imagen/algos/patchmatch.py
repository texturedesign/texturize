# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch


def torch_flatten_1d(a):
    return a.contiguous().view(a.nelement())

def torch_flatten_2d(a):
    return a.contiguous().view((-1, a.size()[-1]))

def torch_gather_2d(array, indices):
    idx = indices[:, :, 0] * array.size(1) + indices[:, :, 1]
    x = torch.index_select(torch_flatten_2d(array), 0, torch_flatten_1d(idx))
    return x.view(indices.shape[:2] + array.shape[-1:]).detach()


class PatchMatcher:

    def __init__(self, content, style, indices=None):
        content_norm = torch.sqrt(torch.sum(content ** 2.0, dim=2, keepdim=True))
        style_norm = torch.sqrt(torch.sum(style ** 2.0, dim=2, keepdim=True))

        self.content = content / content_norm
        self.style = style / style_norm

        self.indices = indices or self.create_indices_random()
        self.scores = torch.zeros(self.content.shape[:2] + (1,), dtype=torch.float, device=self.content.device)

        self.evaluate_patches(self.indices)

    def create_indices_random(self):
        indices = torch.zeros(self.content.shape[:2] + (2,), dtype=torch.long, device=self.content.device)
        torch.randint(low=0, high=self.style.shape[0], size=self.content.shape[:2], out=indices[:, :, 0])
        torch.randint(low=0, high=self.style.shape[1], size=self.content.shape[:2], out=indices[:, :, 1])
        return indices

    def evaluate_patches(self, candidate_indices):
        candidate_repro = torch_gather_2d(self.style, candidate_indices)
        candidate_scores = torch.sum(self.content * candidate_repro, dim=2, keepdim=True)

        better = candidate_scores > self.scores

        self.indices[:,:] = torch.where(better, candidate_indices, self.indices)
        self.scores[:,:] = torch.where(better, candidate_scores, self.scores)

    def search_patches_random(self):
        pass
    
    def search_patches_propagate(self):
        pass


def transform(content, style, iterations=4):
    matcher = PatchMatcher(content, style, indices=None)

    for i in range(iterations):
        matcher.search_patches_propagate()
        matcher.search_patches_random()

    return content, matcher.indices
