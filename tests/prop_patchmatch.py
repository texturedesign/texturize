# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.algos import patchmatch

from hypothesis import given, settings, strategies as H


def make_square_tensor(size, channels):
    return torch.rand((size, size, channels), dtype=torch.float)

def Tensor(min_size=1, channels=None) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_tensor,
        size=H.integers(min_value=min_size, max_value=32),
        channels=H.integers(min_value=channels or 1, max_value=channels or 8))


@given(content=Tensor(channels=4), style=Tensor(channels=4))
def test_indices_range(content, style):
    """Determine that random indices are in range.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.indices[:,:,0].min() >= 0
    assert pm.indices[:,:,0].max() < style.shape[0]

    assert pm.indices[:,:,1].min() >= 0
    assert pm.indices[:,:,1].max() < style.shape[1]


@given(content=Tensor(channels=3), style=Tensor(channels=3))
def test_scores_range(content, style):
    """Determine that random indices are indeed random.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.scores.min() >= 0.0
    assert pm.scores.max() <= 1.0


@given(content=Tensor(4, channels=3), style=Tensor(4, channels=3))
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
