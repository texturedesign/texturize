# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.data import histogram

import pytest
from hypothesis import given, event, strategies as H


def make_tensor(size, mean, std):
    return torch.empty(1,1,size,size).normal_(mean=mean, std=std)\
                .clamp_(mean - std * 0.5, mean + std * 0.5)

def Tensor(range=(64,128)) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_tensor,
        size=H.integers(min_value=range[0], max_value=range[-1]),
        mean=H.floats(min_value=-10.0, max_value=+10.0),
        std=H.floats(min_value=0.1, max_value=2.0))


@given(source=Tensor())
def test_extract_deterministic(source):
    h1 = histogram.extract_histograms(source)
    h2 = histogram.extract_histograms(source)
    assert (h1[0] == h2[0]).all()

@given(source=Tensor())
def test_extract_balanced(source):
    h = histogram.extract_histograms(source)
    assert pytest.approx(0.0, abs=1e-6) == torch.mean(h[0] - 1.0 / h[0].shape[2])

@given(source=Tensor())
def test_extract_normalized(source):
    h = histogram.extract_histograms(source)
    assert pytest.approx(1.0, abs=1e-6) == torch.sum(h[0])

@given(source=Tensor())
def test_match_identity(source):
    h = histogram.extract_histograms(source)
    output = histogram.match_histograms(source, h)
    assert pytest.approx(0.0, abs=1e-6) == torch.max(output - source)

@given(source=Tensor(), offset=H.floats(min_value=-10.0, max_value=+10.0))
def test_match_offset(source, offset):
    h = histogram.extract_histograms(source)
    output = histogram.match_histograms(source + offset, h)
    assert pytest.approx(0.0, abs=1e-4) == torch.max(output - source)
