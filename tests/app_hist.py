# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from texturize.api import process_octaves


def test_hist_single(image, size=(32, 48)):
    for r in process_octaves([image(size)], octaves=2, size=size, mode="hist"):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
        # assert r.loss < 1e-1
