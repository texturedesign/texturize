# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from ..utils import torch_mean, torch_std, torch_interp


def square_matrix(features):
    (b, ch, h, w) = features.size()
    f_i = features.view(b, ch, w * h)
    f_t = f_i.transpose(1, 2)
    return f_i.bmm(f_t) / (ch * h * w)


def rectangular_matrix(column, row):
    (b, ch, h, w) = column.size()
    f_c = column.view(b, ch, w * h)

    (b, ch, h, w) = row.size()
    f_r = row.view(b, ch, w * h).transpose(1, 2)

    return f_c.bmm(f_r) / (ch * h * w)


def extract_histograms(data, bins=7, min=None, max=None):
    if min is None:
        min = data.min()
    if max is None:
        max = data.max()

    counts = data.new_empty(data.shape[:2] + (bins,), dtype=torch.long)
    data = ((data - min) * bins / (max - min)).clamp(0, bins - 1).long()

    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            counts[b, c] = torch.bincount(data[b, c].view(-1), minlength=bins)

    return counts.float() / (data.shape[2] * data.shape[3]), (min, max)


def match_histograms(data, histogram, same_range=False):
    target, (tmin, tmax) = histogram
    output = torch.empty_like(data)
    bins = target.shape[2]

    if same_range is True:
        min, max = tmin, tmax
    else:
        min, max = None, None
    current, (cmin, cmax) = extract_histograms(data, bins, min, max)

    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            step_c = (cmax - cmin) / bins
            step_t = (tmax - tmin) / bins
            cdf_c, cdf_t, edg_c, edg_t = (
                [torch.tensor(0.0)],
                [torch.tensor(0.0)],
                [cmin],
                [tmin],
            )
            for v in range(bins):
                cdf_c.append(current[b, c, v] + cdf_c[-1])
                cdf_t.append(target[b, c, v] + cdf_t[-1])
                edg_c.append(cmin + (v + 1) * step_c)
                edg_t.append(tmin + (v + 1) * step_t)

            tmp = torch_interp(data[b, c].view(-1), [e.item() for e in edg_c], cdf_c)
            output[b, c] = torch_interp(tmp, cdf_t, [e.item() for e in edg_t]).view(
                data.shape[-2:]
            )

    return output.detach()
