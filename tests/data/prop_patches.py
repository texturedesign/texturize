# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.data import patches

from hypothesis import given, event, strategies as H


def make_square_tensor(size, channels):
    return torch.rand((size, size, channels), dtype=torch.float)

def Tensor(range=(1,32), channels=None) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_tensor,
        size=H.integers(min_value=range[0], max_value=range[-1]),
        channels=H.integers(min_value=channels or 1, max_value=channels or 8))

Coord = H.tuples(H.integers(), H.integers())
CoordList = H.lists(Coord, min_size=1, max_size=32)


@given(patch_size=H.integers(1,11))
def test_extract_min_max(patch_size):
    pb = patches.PatchBuilder(patch_size)
    assert abs(pb.min) <= pb.max
    assert (pb.max - pb.min) + 1 == patch_size
    assert len(pb.coords) == patch_size


@given(array=Tensor(), patch_size=H.integers(1,5))
def test_extract_size(array, patch_size):
    pb = patches.PatchBuilder(patch_size)
    result = pb.extract(array)
    assert result.shape[2] == array.shape[2] * (patch_size ** 2)
    assert result.shape[:2] == array.shape[:2]


@given(array=Tensor(range=(2,8)), coords=CoordList)
def test_extract_patches_even(array, coords):
    pb = patches.PatchBuilder(patch_size=2)
    result = pb.extract(array)

    for y, x in coords:
        y = y % (array.shape[0] - 1)
        x = x % (array.shape[1] - 1)
        column = torch.cat([array[y+0,x+0], array[y+0,x+1], array[y+1,x+0], array[y+1,x+1]], dim=0)
        assert (result[y,x] == column).all()

    column_br = torch.cat([array[-1,-1], array[-1,-1], array[-1,-1], array[-1,-1]], dim=0)
    assert (result[-1,-1] == column_br).all()


@given(array=Tensor(range=(3,8)), coords=CoordList)
def test_extract_patches_odd(array, coords):
    pb = patches.PatchBuilder(patch_size=3)
    result = pb.extract(array)
    
    for y, x in coords:
        y = y % (array.shape[0] - 2)
        x = x % (array.shape[1] - 2)
        column = torch.cat([array[y+0,x+0], array[y+0,x+1], array[y+0,x+2],
                            array[y+1,x+0], array[y+1,x+1], array[y+1,x+2],
                            array[y+2,x+0], array[y+2,x+1], array[y+2,x+2]], dim=0)
        assert (result[y+1,x+1] == column).all()

    column_br = torch.cat([array[-2,-2], array[-2,-1], array[-2,-1],
                           array[-1,-2], array[-1,-1], array[-1,-1],
                           array[-1,-2], array[-1,-1], array[-1,-1]], dim=0)
    assert (result[-1,-1] == column_br).all()


@given(array=Tensor())
def test_reconstruct_patches_identity(array):
    pb = patches.PatchBuilder(patch_size=1)
    intermed = pb.extract(array)
    repro = pb.reconstruct(intermed)

    assert array.shape == repro.shape
    assert (array == repro).all()


@given(array=Tensor(range=[2], channels=1))
def test_reconstruct_patches_even(array):
    pb = patches.PatchBuilder(patch_size=2, weights=(1.0, 0.0, 0.0, 0.0))
    intermed = pb.extract(array)
    repro = pb.reconstruct(intermed)

    assert array.shape == repro.shape
    assert (array == repro).all()


@given(array=Tensor(range=[3]))
def test_reconstruct_patches_odd(array):
    pb = patches.PatchBuilder(patch_size=3, weights=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    intermed = pb.extract(array)
    repro = pb.reconstruct(intermed)

    assert array.shape == repro.shape
    assert (array == repro).all()
