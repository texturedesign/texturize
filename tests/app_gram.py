# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from texturize.api import process_octaves


def test_gram_single(image, size=(96, 88)):
    for r in process_octaves([image(size)], octaves=2, size=size, mode="gram"):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
        assert abs(r.iteration) < 200
        assert r.loss < 5e-3


def test_gram_variations(image, size=(72, 64)):
    for r in process_octaves(
        [image(size)], variations=2, octaves=2, size=size, mode="gram"
    ):
        assert len(r.images) == 2
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
        assert abs(r.iteration) < 200
        assert r.loss < 5e-3
