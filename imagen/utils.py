# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch


def torch_flatten_1d(a):
    return a.contiguous().view(a.nelement())

def torch_flatten_2d(a):
    return a.contiguous().view(a.shape[:2] + (-1,))

def torch_gather_2d(array, indices):
    """Extract the content of an array using the 2D coordinates provided.
    """
    assert indices.shape[0] == 1 and array.shape[0] == 1

    idx = indices[0, 0, :, :] * array.shape[2] + indices[0, 1, :, :]
    x = torch.index_select(torch_flatten_2d(array), 2, torch_flatten_1d(idx))
    return x.view(array.shape[:2] + indices.shape[-2:])

def torch_pad_replicate(array, padding):
    return torch.nn.functional.pad(array, padding, mode='replicate')

def torch_interp(array, xs, ys):
    result = torch.zeros_like(array)
    array = array.clamp(xs[0], xs[-1])
    for (xn, xm), (yn, ym) in zip(zip(xs[:-1], xs[1:]), zip(ys[:-1], ys[1:])):
        if xn == xm:
            continue

        sliced = yn + ((array - xn) / (xm - xn)) * (ym - yn)
        result += torch.where(xn <= array, torch.where(array < xm, sliced, torch.tensor(0.0)), torch.tensor(0.0))
    return result

def torch_mean(array, dims):
    for d in dims:
        array = torch.mean(array, dim=d, keepdim=True)
    return array

def torch_std(array, dims):
    remaining = tuple(set(range(array.dim())) - set(dims))
    copy = array.permute(remaining + dims).contiguous()

    rem_shape = torch.gather(torch.tensor(array.shape), 0, torch.tensor(remaining))
    dim_shape = torch.gather(torch.tensor(array.shape), 0, torch.tensor(dims))
    copy = copy.view([torch.prod(rem_shape), torch.prod(dim_shape)])
    copy = torch.std(copy, dim=1)

    shape = torch.tensor(array.shape)
    for d in dims:
        shape[d] = 1
    return copy.view(tuple(shape))

def torch_min(array, dims):
    for d in dims:
        array = torch.min(array, dim=d, keepdim=True)[0]
    return array

def torch_max(array, dims):
    for d in dims:
        array = torch.max(array, dim=d, keepdim=True)[0]
    return array
