# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import pytest
import PIL.Image

from texturize.api import process_octaves


def test_patch_single(image, size=(64, 48)):
    for result in process_octaves(
        image(size), octaves=2, size=size, mode="patch", threshold=1e-3
    ):
        assert len(result.images) == 1
        assert all(isinstance(img, PIL.Image.Image) for img in result.images)
        assert all(img.size == size for img in result.images)
        assert result.loss < 5.0
