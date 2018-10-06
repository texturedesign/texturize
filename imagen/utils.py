# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch


def torch_flatten_1d(a):
    return a.contiguous().view(a.nelement())

def torch_flatten_2d(a):
    return a.contiguous().view((-1, a.shape[-1]))

def torch_gather_2d(array, indices):
    """Extract the content of an array using the 2D coordinates provided.
    """
    idx = indices[:, :, 0] * array.shape[1] + indices[:, :, 1]
    x = torch.index_select(torch_flatten_2d(array), 0, torch_flatten_1d(idx))
    return x.view(indices.shape[:2] + array.shape[-1:])

def torch_pad_replicate(array, padding):
    array = array.permute(2, 0, 1)[None]
    array = torch.nn.functional.pad(array, padding, mode='replicate')
    return array[0].permute(1, 2, 0)
