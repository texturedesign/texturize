# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch

from texturize.match import (
    FeatureMatcher,
    Mapping,
    torch_scatter_2d,
    cosine_similarity_vector_1d,
)


import pytest
from hypothesis import settings, given, event, strategies as H


def make_square_tensor(size, channels):
    return torch.empty((1, channels, size, size), dtype=torch.float).uniform_(0.1, 0.9)


def Tensor(range=(4, 32), channels=None) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_tensor,
        size=H.integers(min_value=range[0], max_value=range[-1]),
        channels=H.integers(min_value=channels or 1, max_value=channels or 8),
    )


Coord = H.tuples(H.integers(), H.integers())
CoordList = H.lists(Coord, min_size=1, max_size=32)


@given(content=Tensor(channels=4), style=Tensor(channels=4))
def test_indices_random_range(content, style):
    """Determine that random indices are in range.
    """
    mapping = Mapping(content.shape)
    mapping.from_linear(style.shape)

    assert mapping.indices[:, 0, :, :].min() >= 0
    assert mapping.indices[:, 0, :, :].max() < style.shape[2]

    assert mapping.indices[:, 1, :, :].min() >= 0
    assert mapping.indices[:, 1, :, :].max() < style.shape[3]


@given(content=Tensor(channels=5), style=Tensor(channels=5))
def test_indices_linear_range(content, style):
    """Determine that random indices are in range.
    """
    mapping = Mapping(content.shape)
    mapping.from_linear(style.shape)

    assert mapping.indices[:, 0, :, :].min() >= 0
    assert mapping.indices[:, 0, :, :].max() < style.shape[2]

    assert mapping.indices[:, 1, :, :].min() >= 0
    assert mapping.indices[:, 1, :, :].max() < style.shape[3]


@given(
    target=Tensor(range=(3, 11), channels=4), source=Tensor(range=(5, 9), channels=4)
)
def test_scores_range_matrix(target, source):
    """Determine that the scores of random patches are in correct range.
    """
    matcher = FeatureMatcher(target, source)
    matcher.compare_features_matrix(split=2)
    assert matcher.repro_target.scores.min() >= 0.0
    assert matcher.repro_target.scores.max() <= 1.0
    assert matcher.repro_sources.scores.min() >= 0.0
    assert matcher.repro_sources.scores.max() <= 1.0


@given(
    target=Tensor(range=(3, 11), channels=3), source=Tensor(range=(5, 9), channels=3)
)
def test_scores_range_random(target, source):
    """Determine that the scores of random patches are in correct range.
    """
    matcher = FeatureMatcher(target, source)
    matcher.compare_features_random(split=1)

    assert matcher.repro_sources.indices.min() != -1
    assert matcher.repro_target.scores.min() >= 0.0
    assert matcher.repro_target.scores.max() <= 1.0

    assert matcher.repro_sources.scores.max() >= 0.0
    assert matcher.repro_sources.indices.max() != -1


@given(
    target=Tensor(range=(5, 11), channels=3), source=Tensor(range=(7, 9), channels=3)
)
@settings(deadline=None)
def test_compare_random_converges(target, source):
    """Determine that the scores of random patches are in correct range.
    """
    matcher1 = FeatureMatcher(target, source)
    matcher1.compare_features_matrix(split=2)

    matcher2 = FeatureMatcher(target, source)
    for _ in range(500):
        matcher2.compare_features_random(split=2)
        missing = (
            (matcher1.repro_target.indices != matcher2.repro_target.indices).sum()
            + (matcher1.repro_sources.indices != matcher2.repro_sources.indices).sum()
        )
        if missing == 0:
            break

    assert (matcher1.repro_target.indices != matcher2.repro_target.indices).sum() <= 2
    assert pytest.approx(0.0, abs=1e-6) == torch.dist(
        matcher1.repro_target.scores, matcher2.repro_target.scores
    )

    assert matcher2.repro_sources.indices.min() != -1
    assert (matcher1.repro_sources.indices != matcher2.repro_sources.indices).sum() <= 2
    assert pytest.approx(0.0, abs=1e-6) == torch.dist(
        matcher1.repro_sources.scores, matcher2.repro_sources.scores
    )


@given(content=Tensor(range=(2, 8)), style=Tensor(range=(2, 8)))
def test_indices_random(content, style):
    """Determine that random indices are indeed random in larger grids.
    """
    m = Mapping(content.shape)
    m.from_random(style.shape)

    assert m.indices[:, 0].min() != m.indices[:, 0].max(dim=2)
    assert m.indices[:, 1].min() != m.indices[:, 1].max(dim=2)


@given(array=Tensor(range=(2, 8)))
def test_indices_linear(array):
    """Indices of the indentity transformation should be linear.
    """
    m = Mapping(array.shape)
    m.from_linear(array.shape)

    assert (
        m.indices[:, 0, :, :]
        == torch.arange(start=0, end=array.shape[2]).view(1, -1, 1)
    ).all()
    assert (
        m.indices[:, 1, :, :]
        == torch.arange(start=0, end=array.shape[3]).view(1, 1, -1)
    ).all()


@given(array=Tensor(range=(4, 16)))
def test_scores_identity(array):
    """The score of the identity operation with linear indices should be one.
    """
    matcher = FeatureMatcher(array, array)
    matcher.repro_target.from_linear(array.shape)
    matcher.repro_sources.from_linear(array.shape)
    matcher.compare_features_matrix(split=2)

    assert pytest.approx(1.0) == matcher.repro_target.scores.min()
    assert pytest.approx(1.0) == matcher.repro_sources.scores.min()


@given(
    content=Tensor(range=(17, 37), channels=5), style=Tensor(range=(13, 39), channels=5)
)
@settings(deadline=None)
def test_indices_same_split(content, style):
    """The score of the identity operation with linear indices should be one.
    """
    matcher = FeatureMatcher(content, style)
    matcher.compare_features_matrix(split=1)
    target_indices = matcher.repro_target.indices.clone()
    source_indices = matcher.repro_sources.indices.clone()

    for split in [2, 4, 8]:
        matcher.update_target(content)
        matcher.compare_features_matrix(split=split)

        assert (target_indices != matcher.repro_target.indices).sum() <= 2
        assert (source_indices != matcher.repro_sources.indices).sum() <= 2


@given(
    content=Tensor(range=(17, 37), channels=5), style=Tensor(range=(13, 39), channels=5)
)
def test_indices_same_rotate(content, style):
    """The score of the identity operation with linear indices should be one.
    """
    matcher1 = FeatureMatcher(content, style)
    matcher1.compare_features_matrix(split=2)

    matcher2 = FeatureMatcher(content, style.permute(0, 1, 3, 2))
    matcher2.compare_features_matrix(split=2)

    assert (
        matcher1.repro_target.indices[:, 0] != matcher2.repro_target.indices[:, 1]
    ).sum() <= 1
    assert (
        matcher2.repro_target.indices[:, 1] != matcher1.repro_target.indices[:, 0]
    ).sum() <= 1


@given(content=Tensor(range=(5, 7), channels=2), style=Tensor(range=(3, 9), channels=2))
def test_indices_symmetry_matrix(content, style):
    """The indices of the symmerical operation must be equal.
    """
    matcher1 = FeatureMatcher(content, style)
    matcher2 = FeatureMatcher(style, content)

    matcher1.compare_features_matrix(split=2)
    matcher2.compare_features_matrix(split=2)

    assert (matcher1.repro_target.indices != matcher2.repro_sources.indices).sum() <= 2
    assert (matcher1.repro_sources.indices != matcher2.repro_target.indices).sum() <= 2


@given(content=Tensor(range=(2, 4), channels=2), style=Tensor(range=(3, 3), channels=2))
@settings(deadline=None)
def test_indices_symmetry_random(content, style):
    """The indices of the symmerical operation must be the same.
    """
    matcher1 = FeatureMatcher(content, style)
    matcher2 = FeatureMatcher(style, content)

    for _ in range(25):
        matcher1.compare_features_random()
        matcher2.compare_features_random()

        missing = sum(
            [
                (matcher1.repro_target.indices != matcher2.repro_sources.indices).sum(),
                (matcher1.repro_sources.indices != matcher2.repro_target.indices).sum(),
            ]
        )
        if missing == 0:
            break

    assert (matcher1.repro_target.indices != matcher2.repro_sources.indices).sum() <= 2
    assert (matcher1.repro_sources.indices != matcher2.repro_target.indices).sum() <= 2


@given(
    content=Tensor(range=(4, 16), channels=2), style=Tensor(range=(4, 16), channels=2)
)
def test_scores_zero(content, style):
    """Scores must be zero if inputs vary on different dimensions.
    """
    content[:, 0], style[:, 1] = 0.0, 0.0
    matcher = FeatureMatcher(content, style)
    matcher.compare_features_matrix(split=2)
    assert pytest.approx(0.0) == matcher.repro_target.scores.max()
    assert pytest.approx(0.0) == matcher.repro_sources.scores.max()


@given(
    content=Tensor(range=(4, 16), channels=2), style=Tensor(range=(4, 16), channels=2)
)
def test_scores_one(content, style):
    """Scores must be one if inputs only vary on one dimension.
    """
    content[:, 0], style[:, 0] = 0.0, 0.0
    matcher = FeatureMatcher(content, style)
    matcher.compare_features_matrix(split=2)
    assert pytest.approx(1.0) == matcher.repro_target.scores.min()
    assert pytest.approx(1.0) == matcher.repro_sources.scores.min()


@given(
    target=Tensor(range=(4, 12), channels=3), source=Tensor(range=(4, 12), channels=3)
)
def test_scores_reconstruct(target, source):
    """Scores must be one if inputs only vary on one dimension.
    """
    matcher = FeatureMatcher(target, source)
    matcher.compare_features_matrix()

    recons_target = matcher.reconstruct_target()
    score = cosine_similarity_vector_1d(target, recons_target)
    assert pytest.approx(0.0, abs=1e-6) == abs(
        score.mean() - matcher.repro_target.scores.mean()
    )

    recons_source = matcher.reconstruct_source()
    score = cosine_similarity_vector_1d(source, recons_source)
    assert pytest.approx(0.0, abs=1e-6) == abs(
        score.mean() - matcher.repro_sources.scores.mean()
    )


@given(
    content=Tensor(range=(4, 16), channels=5), style=Tensor(range=(4, 16), channels=5)
)
def test_scores_improve(content, style):
    """Scores must be one if inputs only vary on one dimension.
    """
    matcher = FeatureMatcher(content, style)
    matcher.compare_features_identity()
    before = matcher.repro_target.scores.sum()
    matcher.compare_features_random(times=1)
    after = matcher.repro_target.scores.sum()
    event("equal? %i" % int(after == before))
    assert after >= before


@given(array=Tensor(range=(9, 9), channels=5))
def test_scores_source_bias_matrix(array):
    matcher = FeatureMatcher(array, torch.cat([array, array], dim=2))

    matcher.repro_target.biases[:, :, 9:] = 1.0
    matcher.repro_target.scores.zero_()
    matcher.compare_features_matrix(split=2)
    assert (matcher.repro_target.indices[:,0] >= 9).all()

    matcher.repro_target.biases[:, :, 9:] = 0.0
    matcher.repro_target.biases[:, :, :9] = 1.0
    matcher.repro_target.scores.zero_()
    matcher.compare_features_matrix(split=2)
    assert (matcher.repro_target.indices[:,0] < 9).all()


@given(array=Tensor(range=(11, 11), channels=4))
def test_scores_target_bias_matrix(array):
    matcher = FeatureMatcher(torch.cat([array, array], dim=2), array)

    matcher.repro_sources.biases[:, :, 11:] = 1.0
    matcher.repro_sources.scores.zero_()
    matcher.compare_features_matrix(split=2)
    assert (matcher.repro_sources.indices[:,0] >= 11).all()

    matcher.repro_sources.biases[:, :, 11:] = 0.0
    matcher.repro_sources.biases[:, :, :11] = 1.0
    matcher.repro_sources.scores.zero_()
    matcher.compare_features_matrix(split=2)
    assert (matcher.repro_sources.indices[:,0] < 11).all()


@given(array=Tensor(range=(8, 8), channels=5))
def test_scores_source_bias_random(array):
    matcher = FeatureMatcher(array, torch.cat([array, array], dim=2))

    matcher.repro_target.biases[:, :, 8:] = 1.0
    matcher.repro_target.scores.fill_(-1.0)
    for _ in range(10):
        matcher.compare_features_random(split=2)
    assert (matcher.repro_target.indices[:,0] >= 8).all()

    matcher.repro_target.biases[:, :, 8:] = 0.0
    matcher.repro_target.biases[:, :, :8] = 1.0
    matcher.repro_target.scores.fill_(-1.0)
    for _ in range(10):
        matcher.compare_features_random(split=2)
    assert (matcher.repro_target.indices[:,0] < 8).all()


@given(array=Tensor(range=(12, 12), channels=3))
def test_scores_target_bias_random(array):
    matcher = FeatureMatcher(torch.cat([array, array], dim=2), array)

    matcher.repro_sources.biases[:, :, 12:] = 1.0
    matcher.repro_sources.scores.fill_(-1.0)
    for _ in range(10):
        matcher.compare_features_random(split=2)
    assert (matcher.repro_sources.indices[:,0] >= 12).all()

    matcher.repro_sources.biases[:, :, 12:] = 0.0
    matcher.repro_sources.biases[:, :, :12] = 1.0
    matcher.repro_sources.scores.fill_(-1.0)
    for _ in range(10):
        matcher.compare_features_random(split=2)
    assert (matcher.repro_sources.indices[:,0] < 12).all()


@given(array=Tensor(range=(3, 8), channels=5))
def test_propagate_down_right(array):
    """Propagating the identity transformation expects indices to propagate
    one cell at a time, this time down and towards the right.
    """
    matcher = FeatureMatcher(array, array)
    indices = matcher.repro_target.indices
    indices.zero_()

    matcher.compare_features_nearby(radius=1, split=2)
    assert (indices[:, :, 1, 0] == torch.tensor([1, 0], dtype=torch.long)).all()
    assert (indices[:, :, 0, 1] == torch.tensor([0, 1], dtype=torch.long)).all()

    matcher.compare_features_nearby(radius=1, split=2)
    assert (indices[:, :, 1, 1] == torch.tensor([1, 1], dtype=torch.long)).all()


@given(array=Tensor(range=(2, 8), channels=5))
def test_propagate_up_left(array):
    """Propagating the identity transformation expects indices to propagate
    one cell at a time, here up and towards the left.
    """
    y, x = array.shape[-2:]
    matcher = FeatureMatcher(array, array)
    indices, scores = matcher.repro_target.indices, matcher.repro_target.scores
    indices.zero_()

    indices[:, 0, -1, -1] = y - 1
    indices[:, 1, -1, -1] = x - 1
    scores[:, 0, -1, -1] = 1.0

    matcher.compare_features_nearby(radius=1, split=2)
    assert (
        indices[:, :, y - 2, x - 1] == torch.tensor([y - 2, x - 1], dtype=torch.long)
    ).all()
    assert (
        indices[:, :, y - 1, x - 2] == torch.tensor([y - 1, x - 2], dtype=torch.long)
    ).all()

    matcher.compare_features_nearby(radius=1, split=2)
    assert (
        indices[:, :, y - 2, x - 2] == torch.tensor([y - 2, x - 2], dtype=torch.long)
    ).all()


@given(content=Tensor(range=(6, 8), channels=4), style=Tensor(range=(7, 9), channels=4))
def test_compare_inverse_asymmetrical(content, style):
    """Check that doing the identity comparison also projects the inverse
    coordinates into the other buffer.
    """

    # Set corner pixel as identical, so it matches 100%.
    content[:, :, -1, -1] = style[:, :, -1, -1]

    matcher = FeatureMatcher(content, style)
    matcher.repro_target.from_linear(style.shape)
    matcher.repro_sources.indices.zero_()
    matcher.compare_features_identity()
    matcher.compare_features_inverse(split=2)

    assert matcher.repro_sources.indices.max() > 0

    matcher.repro_sources.from_linear(content.shape)
    matcher.repro_target.indices.zero_()
    matcher.repro_target.scores.zero_()
    matcher.compare_features_identity()
    matcher.compare_features_inverse(split=2)

    assert matcher.repro_target.indices.max() > 0


@given(array=Tensor(range=(3, 3), channels=3))
def test_compare_inverse_symmetrical(array):
    """Check that doing the identity comparison also projects the inverse
    coordinates into the other buffer.
    """
    matcher = FeatureMatcher(array, array)
    matcher.repro_target.from_linear(array.shape)
    matcher.repro_sources.indices.zero_()
    matcher.compare_features_identity()
    matcher.compare_features_inverse(split=1)

    assert (matcher.repro_target.indices != matcher.repro_sources.indices).sum() == 0

    matcher.repro_target.indices.zero_()
    matcher.repro_target.scores.fill_(float("-inf"))
    matcher.compare_features_identity()
    matcher.compare_features_inverse(split=1)

    assert (matcher.repro_target.indices != matcher.repro_sources.indices).sum() == 0


# ================================================================================


@given(array=Tensor(range=(3, 5), channels=3))
def test_scatter_2d_single_float(array):
    matcher = FeatureMatcher(array, array)
    matcher.repro_target.scores.zero_()

    indices = torch.ones(size=(1, 2, 1, 1), dtype=torch.int64)
    values = torch.full((1, 1, 1, 1), 2.34, dtype=torch.float32)

    torch_scatter_2d(matcher.repro_target.scores, indices, values)

    assert matcher.repro_target.scores[:, :, 1, 1] == 2.34
    assert matcher.repro_target.scores[:, :, 0, 0] == 0.0


@given(array=Tensor(range=(3, 5), channels=3))
def test_scatter_2d_single_long(array):
    matcher = FeatureMatcher(array, array)
    matcher.repro_target.indices.zero_()

    indices = torch.ones(size=(1, 2, 1, 1), dtype=torch.int64)
    values = torch.tensor([[234, 345]], dtype=torch.int64).view(1, 2, 1, 1)

    torch_scatter_2d(matcher.repro_target.indices, indices, values)

    assert (matcher.repro_target.indices == 234).sum() == 1
    assert (matcher.repro_target.indices == 345).sum() == 1
    assert matcher.repro_target.indices[:, 0, 1, 1] == 234
    assert matcher.repro_target.indices[:, 1, 1, 1] == 345
    assert matcher.repro_target.indices[:, :, 0, 0].max() == 0


def test_improve_window():
    u, v = 4, 5

    mapping = Mapping((1, -1, u, u), "cpu").from_linear((1, 1, 7+v, 7+v))
    similarity = torch.empty(size=(1, u * u, v * v), dtype=torch.float32).uniform_()

    mapping.improve_window((0, u, 0, u), (7, v, 7, v), similarity.max(dim=2))
    orig = mapping.indices.clone()

    mapping.scores.fill_(float("-inf"))
    mapping.indices.zero_()

    mapping.improve_window(
        (0, u // 2, 0, u), (7, v, 7, v), similarity[:, : u * u // 2].max(dim=2)
    )
    mapping.improve_window(
        (u // 2, u // 2, 0, u), (7, v, 7, v), similarity[:, u * u // 2 :].max(dim=2)
    )
    assert (orig != mapping.indices).sum() == 0


# ================================================================================

from texturize.match import cosine_similarity_matrix_1d


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
        sim = cosine_similarity_matrix_1d(a, bb, eps=eps)

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


@given(
    content=Tensor(range=(15, 37), channels=5), style=Tensor(range=(13, 39), channels=5)
)
@settings(deadline=None)
def test_nearest_neighbor_vs_matcher(content, style):
    """The score of the identity operation with linear indices should be one.
    """

    matcher = FeatureMatcher(content, style)
    matcher.compare_features_matrix(split=1)

    ida, idb = nearest_neighbors_1d(content.flatten(2), style.flatten(2), split=1)

    ima = (
        matcher.repro_target.indices[:, 0] * style.shape[2]
        + matcher.repro_target.indices[:, 1]
    )
    assert (ima.flatten(1) != ida).sum() == 0

    imb = (
        matcher.repro_sources.indices[:, 0] * content.shape[2]
        + matcher.repro_sources.indices[:, 1]
    )
    assert (imb.flatten(1) != idb).sum() == 0
