# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from texturize.api import process_octaves


def test_patch_single(image, size=(64, 48)):
    for r in process_octaves(
        [image(size)], octaves=2, size=size, mode="patch", threshold=1e-3
    ):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
        # assert r.loss < 5.0
