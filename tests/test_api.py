# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

import PIL.ImageOps

from texturize.commands import *
from texturize.api import process_octaves


def test_api_remix(image, size):
    remix = Remix(source=image)
    for r in process_octaves(remix, size=size, quality=1):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_remake(image, size):
    remake = Remake(target=image, source=image)
    for r in process_octaves(remake, size=size, quality=1):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_mashup(image, size):
    mashup = Mashup(sources=[image, image])
    for r in process_octaves(mashup, size=size, quality=1):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_enhance(image, size):
    mashup = Enhance(target=image, source=image, zoom=2)
    for r in process_octaves(mashup, size=size, quality=1):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_expand(image, size):
    expand = Expand(target=image, source=image, factor=(2.0, 2.0))
    for r in process_octaves(expand, size=size):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_repair(image, size):
    src = PIL.ImageOps.mirror(image)
    repair = Repair(target=image.convert("RGBA"), source=src)
    for r in process_octaves(repair, size=size, quality=1):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
