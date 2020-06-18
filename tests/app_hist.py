# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import pytest
import PIL.Image

from texturize.api import process_octaves


def test_hist_single(image, size=(32, 48)):
    for _, loss, images in process_octaves(
        image(size), octaves=2, size=size, mode="hist"
    ):
        assert len(images) == 1
        assert all(isinstance(img, PIL.Image.Image) for img in images)
        assert all(img.size == size for img in images)
        assert loss < 1e-1
