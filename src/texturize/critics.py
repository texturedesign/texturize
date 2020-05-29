# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import itertools

import torch
import torch.nn.functional as F


class GramMatrixCritic:
    """A `Critic` evaluates the features of an image to determine how it scores.

    This critic computes a 2D histogram of feature cross-correlations for the specified
    layer (e.g. "1_1") or layer pair (e.g. "1_1:2_1"), and compares it to the target
    gram matrix.
    """

    def __init__(self, layer, offset: float = -1.0):
        self.pair = tuple(layer.split(":"))
        if len(self.pair) == 1:
            self.pair = (self.pair[0], self.pair[0])
        self.offset = offset
        self.gram = None

    def evaluate(self, features):
        current = self._prepare_gram(features)
        yield 1e4 * F.mse_loss(current, self.gram.expand_as(current), reduction="mean")

    def from_features(self, features):
        self.gram = self._prepare_gram(features)

    def get_layers(self):
        return set(self.pair)

    def _gram_matrix(self, column, row):
        (b, ch, h, w) = column.size()
        f_c = column.view(b, ch, w * h)
        (b, ch, h, w) = row.size()
        f_r = row.view(b, ch, w * h)

        gram = (f_c / w).bmm((f_r / h).transpose(1, 2)) / ch
        assert not torch.isnan(gram).any()

        return gram

    def _prepare_gram(self, features):
        lower = features[self.pair[0]] + self.offset
        upper = features[self.pair[1]] + self.offset
        return self._gram_matrix(
            lower, F.interpolate(upper, size=lower.shape[2:], mode="nearest")
        )


class PatchBuilder:
    def __init__(self, patch_size=3, weights=None):
        self.min = -((patch_size - 1) // 2)
        self.max = patch_size + self.min - 1
        self.patch_size = patch_size

        if weights is None:
            weights = torch.ones(size=(patch_size ** 2,))
        else:
            weights = torch.tensor(weights, dtype=torch.float32)

        self.weights = weights / weights.sum()

    def extract(self, array):
        padded = F.pad(
            array,
            pad=(abs(self.min), self.max, abs(self.min), self.max),
            mode="replicate",
        )
        h, w = (
            padded.shape[2] - self.patch_size + 1,
            padded.shape[3] - self.patch_size + 1,
        )
        output = []
        for y, x in itertools.product(self.coords, repeat=2):
            p = padded[:, :, y : h + y, x : w + x]
            output.append(p)
        return torch.cat(output, dim=1)

    @property
    def coords(self):
        return range(self.patch_size)


def cosine_similarity_1d(source, target, eps=1e-8):
    source = source / (torch.norm(source, dim=1, keepdim=True) + eps)
    target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

    result = torch.bmm(source.permute(0, 2, 1), target)
    return torch.clamp(result, max=1.0 / eps)


def nearest_neighbors_1d(a, b, split=1, eps=1e-8):
    batch = a.shape[0]
    size = b.shape[2] // split

    score_a = a.new_full((batch, a.shape[2]), float("-inf"))
    index_a = a.new_full((batch, a.shape[2]), -1, dtype=torch.int64)
    score_b = b.new_full((batch, b.shape[2]), float("-inf"))
    index_b = b.new_full((batch, b.shape[2]), -1, dtype=torch.int64)

    for i in range(split):
        start_b, finish_b = i * size, (i + 1) * size
        bb = b[:, :, start_b:finish_b]
        sim = cosine_similarity_1d(a, bb, eps=eps)

        max_a = torch.max(sim, dim=2)
        cond_a = max_a.values > score_a
        index_a[:] = torch.where(cond_a, max_a.indices + start_b, index_a)
        score_a[:] = torch.where(cond_a, max_a.values, score_a)

        max_b = torch.max(sim, dim=1)
        slice_b = slice(start_b, finish_b)
        cond_b = max_b.values > score_b[:, slice_b]
        index_b[:, slice_b] = torch.where(cond_b, max_b.indices, index_b[:, slice_b])
        score_b[:, slice_b] = torch.where(cond_b, max_b.values, score_b[:, slice_b])

    return index_a, index_b


class PatchCritic:
    def __init__(self, layer):
        self.layer = layer
        self.patches = None
        self.builder = PatchBuilder(patch_size=2)
        self.split_hints = {}

    def get_layers(self):
        return {self.layer}

    def from_features(self, features):
        self.patches = self.prepare(features).detach()

    def prepare(self, features):
        f = features[self.layer]
        p = self.builder.extract(f)
        return p

    def evaluate(self, features):
        current = self.prepare(features)
        yield from self.bidirectional_patch_similarity(current, self.patches)

    def bidirectional_patch_similarity(self, source, target, eps=1e-8):
        assert source.shape[0] == 1 and target.shape[0] == 1
        assert source.shape[1] == target.shape[1]

        source = torch.flatten(source, 2)
        target = torch.flatten(target, 2)

        with torch.no_grad():
            for i in self.split_hints.get(source.shape[1:], range(16)):
                try:
                    ids, idt = nearest_neighbors_1d(
                        source, target, split=2 ** i, eps=eps
                    )
                    self.split_hints[source.shape[1:]] = [i]
                    break
                except RuntimeError:
                    continue

            matched_source = torch.index_select(target, dim=2, index=ids[0])

        yield 0.5 * F.mse_loss(source, matched_source)

        matched_target = torch.index_select(source, dim=2, index=idt[0])
        yield 0.5 * F.mse_loss(target, matched_target)
