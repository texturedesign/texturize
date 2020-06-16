# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import pytest
import PIL.Image

from texturize.api import process_octaves


def test_patch_single(image, size=(64, 48)):
    for _, loss, images in process_octaves(
        image(size), octaves=2, size=size, mode="patch", threshold=1e-3
    ):
        assert len(images) == 1
        assert all(isinstance(img, PIL.Image.Image) for img in images)
        assert all(img.size == size for img in images)
        assert loss < 5.0
