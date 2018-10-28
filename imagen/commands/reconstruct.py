# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import os

import torch
import numpy as np

from ..data import patches, colorspace, scalespace, images
from ..algos import patchmatch


def main(args):
    device = torch.device('cpu')
    content_img = images.load_from_file(args.content, device, mode='YCbCr')
    style_img = images.load_from_file(args.style, device, mode='YCbCr')

    # content_tensor = scalespace.OctaveBuilder(levels=2).build(content_img)
    # style_tensor = scalespace.OctaveBuilder(levels=2).build(style_img)
    content_tensor = colorspace.MinMeanMaxBuilder(normalize=True).build(content_img)
    style_tensor = colorspace.MinMeanMaxBuilder(normalize=True).build(style_img)

    pb = patches.PatchBuilder(patch_size=3, weights=[0.1, 0.1, 0.1, 0.1, 4.0, 0.1, 0.1, 0.1, 0.1])
    content_ptch = pb.extract(content_tensor)
    style_ptch = pb.extract(style_tensor)

    pm = patchmatch.PatchMatcher(content_ptch, style_ptch, indices='random')
    for p in range(args.passes):
        for i in range(args.iterations):
            print('pass', p, 'iter', i, pm.scores.min(), pm.scores.median(), pm.scores.mean(), pm.scores.max())
            pm.search_patches_random(radius=64)
            pm.search_patches_propagate()

        content_ptch = content_ptch * 0.5 + 0.5 * patchmatch.torch_gather_2d(style_ptch, pm.indices)
        pm.content = pm.normalize_patches(content_ptch)
        pm.scores.zero_()
        pm.improve_patches(pm.indices)

    style_orig = pb.extract(style_img)

    recons_ptch = patchmatch.torch_gather_2d(style_orig, pm.indices)
    recons_tensor = pb.reconstruct(recons_ptch)
    images.save_to_file(recons_tensor, os.path.splitext(args.content)[0]+'_recons.png', mode='YCbCr')
