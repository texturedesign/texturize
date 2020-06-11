# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import pytest
import PIL.Image

from texturize.api import process_octaves


def test_gram_single(image, size=(96, 88)):
    for _, loss, images in process_octaves(
        image(size), octaves=2, size=size, mode="gram"
    ):
        assert len(images) == 1
        assert all(isinstance(img, PIL.Image.Image) for img in images)
        assert all(img.size == size for img in images)
        assert loss < 5e-2


def test_gram_variations(image, size=(72, 64)):
    for _, loss, images in process_octaves(
        image(size), variations=2, octaves=2, size=size, mode="gram"
    ):
        assert len(images) == 2
        assert all(isinstance(img, PIL.Image.Image) for img in images)
        assert all(img.size == size for img in images)
        assert loss < 5e-1
