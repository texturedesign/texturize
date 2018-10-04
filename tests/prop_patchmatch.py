# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.algos import patchmatch

from hypothesis import given, settings, strategies as H


def make_square_tensor(size, channels):
    return torch.rand((channels, size, size), dtype=torch.float)

def Tensor(min_size=1) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_tensor,
        size=H.integers(min_value=min_size, max_value=32),
        channels=H.integers(min_value=min_size, max_value=8))


@given(content=Tensor(), style=Tensor())
def test_indices_range(content, style):
    """Determine that random indices are in range.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.indices[:,:,0].min() >= 0
    assert pm.indices[:,:,0].max() < style.shape[0]

    assert pm.indices[:,:,1].min() >= 0
    assert pm.indices[:,:,1].max() < style.shape[1]
    

@given(content=Tensor(4), style=Tensor(4))
def test_indices_random(content, style):
    """Determine that random indices are indeed random.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.indices.min() != pm.indices.max()


@given(array=Tensor())
def test_identity(array):
    """Apply patch-match transformation with the same array.
    """
    result, indices = patchmatch.transform(array, array)
