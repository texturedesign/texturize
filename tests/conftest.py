# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import glob
import random
import pathlib

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
        return (96, 80)
    assert False, "Invalid test suite specified."


@pytest.fixture(scope="function")
def image(request, filename, size):
    img = PIL.Image.open(filename)
    return PIL.ImageOps.fit(img.convert("RGB"), size)


def pytest_collect_file(path, parent):
    if not path.strpath.endswith('conftest.py'):
        return
    
    if parent.config.getoption('--suite') != "full":
        return

    from doctest import ELLIPSIS

    from sybil import Sybil
    import sybil.parsers.rest as SybilParsers
    import sybil.integration.pytest as SybilTest

    sybil = Sybil(
        parsers=[
            SybilParsers.DocTestParser(optionflags=ELLIPSIS),
            SybilParsers.PythonCodeBlockParser(),
        ],
        path='.',
        patterns=['*.rst'],
    )

    return SybilTest.SybilFile.from_parent(parent, path=pathlib.Path('README.rst'), sybil=sybil)
