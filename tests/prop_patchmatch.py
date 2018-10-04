# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.algos import patchmatch

import pytest
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
    """Determine that the scores of random patches are in correct range.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.scores.min() >= 0.0
    assert pm.scores.max() <= 1.0


@given(content=Tensor(4, channels=3), style=Tensor(4, channels=3))
def test_indices_random(content, style):
    """Determine that random indices are indeed random in larger grids.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.indices.min() != pm.indices.max()


@given(array=Tensor(min_size=2))
def test_indices_linear(array):
    """Indices of the indentity transformation should be linear.
    """
    pm = patchmatch.PatchMatcher(array, array, indices='linear')
    assert (pm.indices[:,:,0] == torch.arange(start=0, end=array.shape[0]).view(-1, 1)).all()
    assert (pm.indices[:,:,1] == torch.arange(start=0, end=array.shape[1]).view(1, -1)).all()


@given(array=Tensor(min_size=2))
def test_scores_identity(array):
    """The score of the identity operation with linear indices should be one.
    """
    pm = patchmatch.PatchMatcher(array, array, indices='linear')
    assert pytest.approx(1.0) == pm.scores.min()


@given(content=Tensor(4, channels=2), style=Tensor(4, channels=2))
def test_scores_zero(content, style):
    """Scores must be zero if inputs vary on different dimensions.
    """
    content[:,:,0], style[:,:,1] = 0.0, 0.0
    pm = patchmatch.PatchMatcher(content, style)
    assert pytest.approx(0.0) == pm.scores.max()


@given(content=Tensor(4, channels=2), style=Tensor(4, channels=2))
def test_scores_one(content, style):
    """Scores must be one if inputs only vary on one dimension.
    """
    content[:,:,0], style[:,:,0] = 0.0, 0.0
    pm = patchmatch.PatchMatcher(content, style)
    assert pytest.approx(1.0) == pm.scores.min()
