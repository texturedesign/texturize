# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.utils import torch_interp

import pytest
from hypothesis import given, event, strategies as H


def make_tensor(size):
    return torch.rand((size,), dtype=torch.float)


def Tensor(range=(1, 32)) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_tensor, size=H.integers(min_value=range[0], max_value=range[-1])
    )


@given(array=Tensor())
def test_interp_linear_unit(array):
    result = torch_interp(array, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert (array == result).all()


@given(array=Tensor())
def test_interp_linear_extrapolate(array):
    result = torch_interp(array, [-1.0, 2.0], [-1.0, 2.0])
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)


@given(array=Tensor())
def test_interp_nonlinear_intervals(array):
    result = torch_interp(array, [0.0, 0.1, 0.8, 1.0], [0.0, 0.1, 0.8, 1.0])
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)


@given(array=Tensor())
def test_interp_inverse(array):
    result = 1.0 - torch_interp(array, [0.0, 0.75, 1.0], [1.0, 0.25, 0.0])
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)


@given(array=Tensor())
def test_interp_offset(array):
    result = torch_interp(array, [0.0, 1.0], [2.0, 3.0]) - 2.0
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)


@given(array=Tensor())
def test_interp_multiply(array):
    result = (torch_interp(array, [0.0, 1.0], [2.0, 4.0]) - 2.0) * 0.5
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)


@given(array=Tensor())
def test_interp_clamp(array):
    result = torch_interp(array, [0.2, 0.8], [0.2, 0.8])
    print(array.clamp(0.2, 0.8), result)
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array.clamp(0.2, 0.8) - result)


@given(array=Tensor())
def test_interp_negative(array):
    result = 0.5 + torch_interp(array, [0.0, 0.5, 1.0], [-2.0, 0.0, 2.0]) / 4.0
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(array - result)
