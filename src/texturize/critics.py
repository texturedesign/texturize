# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

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
