# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

import PIL.ImageOps

from texturize import api, commands


def test_api_remix(image, size):
    remix = commands.Remix(source=image)
    for r in api.process_octaves(remix, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_remake(image, size):
    remake = commands.Remake(target=image, source=image)
    for r in api.process_octaves(remake, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_mashup(image, size):
    mashup = commands.Mashup(sources=[image, image])
    for r in api.process_octaves(mashup, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_enhance(image, size):
    mashup = commands.Enhance(target=image, source=image, zoom=2)
    for r in api.process_octaves(mashup, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_expand(image, size):
    expand = commands.Expand(target=image, source=image, factor=(1.5, 1.5))
    for r in api.process_octaves(expand, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)


def test_api_repair(image, size):
    src = PIL.ImageOps.mirror(image)
    repair = commands.Repair(target=image.convert("RGBA"), source=src)
    for r in api.process_octaves(repair, size=size, iterations=2):
        assert len(r.images) == 1
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape[2:] == (size[1] // r.scale, size[0] // r.scale)
