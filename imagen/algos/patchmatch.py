# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch


class PatchMatcher:

    def __init__(self, content, style, indices=None):
        self.content = content
        self.style = style
        self.indices = indices or self.create_indices_random()

    def create_indices_random(self):
        indices = torch.zeros(self.content.shape[:2] + (2,), dtype=torch.long, device=self.content.device)
        torch.randint(low=0, high=self.style.shape[0], size=self.content.shape[:2], out=indices[:, :, 0])
        torch.randint(low=0, high=self.style.shape[1], size=self.content.shape[:2], out=indices[:, :, 1])
        return indices

    def search_patches_random(self):
        pass
    
    def search_patches_propagate(self):
        pass


def transform(content, style, iterations=4):
    matcher = PatchMatcher(content, style, indices=None)

    for i in range(iterations):
        matcher.search_patches_propagate()
        matcher.search_patches_search()

    return content, matcher.indices
