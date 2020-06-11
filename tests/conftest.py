# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import glob
import pytest
import PIL.Image, PIL.ImageOps


@pytest.fixture(params=glob.glob("tests/data/*.png"))
def image(request):
    def build(size):
        img = PIL.Image.open(request.param)
        return PIL.ImageOps.fit(img.convert("RGB"), size)

    return build
