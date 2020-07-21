# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import glob
import random

import pytest
import PIL.Image, PIL.ImageOps


def pytest_addoption(parser):
    parser.addoption(
        "--suite",
        action="store",
        default="fast",
        help="image suite to test: fast or full",
    )


def pytest_generate_tests(metafunc):
    if "filename" not in metafunc.fixturenames:
        return

    if metafunc.config.getoption("suite") == "full":
        filenames = glob.glob("examples/*.webp")
        count = 4
    if metafunc.config.getoption("suite") == "fast":
        filenames = glob.glob("tests/data/*.png")
        count = 1

    assert len(filenames) >= count
    metafunc.parametrize("filename", random.sample(filenames, count))


@pytest.fixture()
def size(request):
    if request.config.getoption("--suite") == "full":
        return (272, 240)
    if request.config.getoption("--suite") == "fast":
        return (96, 88)
    assert False, "Invalid test suite specified."


@pytest.fixture(scope="function")
def image(request, filename, size):
    img = PIL.Image.open(filename)
    return PIL.ImageOps.fit(img.convert("RGB"), size)
